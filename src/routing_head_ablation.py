#!/usr/bin/env python3
"""
Head-level ablation and interchange interventions for routing localization.

Two modes:

1. **Head ablation** (--mode ablate):
   For each candidate head, project out the political direction from that
   head's output only and measure the change in behavioral metric (refusal
   rate, teacher-forced NLL, or KL).  Heads where single-head ablation
   reduces censorship are *necessary* for routing.

2. **Interchange** (--mode interchange):
   For each candidate head, swap that head's output between a CCP prompt run
   and a matched control prompt run.  Measure whether:
   - Swapping control→CCP eliminates refusal (head is necessary)
   - Swapping CCP→control induces routing (head is sufficient)

The candidate set can be specified explicitly (--heads L.H,L.H,...) or
derived from per-head DLA results (--top-k from a head_summary.csv).

Primary output: a matrix of (layer, head, metric_baseline, metric_intervened)
for the ablation mode, or (layer, head, direction, metric) for interchange.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from routing_logit_trajectory import (
    DEFAULT_REFUSAL_STRINGS,
    SEED,
    AblationSpec,
    DecisionPoint,
    PromptPair,
    _resolve_device,
    _resolve_transformer_layers,
    _timestamp,
    _to_device,
    _tokenize,
    build_answer_bundle,
    build_refusal_bundle,
    find_first_meaningful_generation,
    load_ablation_spec,
    load_model_and_tokenizer,
    resolve_final_norm,
    resolve_output_head,
    resolve_prompt_pairs,
    temporary_ablation,
    write_csv,
    nullcontext,
)
from routing_direct_logit_attribution import (
    _prepare_inputs,
    _resolve_layer_components,
    load_probe_direction,
)
from routing_head_dla import (
    _resolve_n_heads,
    _resolve_head_dim,
    _resolve_o_proj,
)


def _git_sha7() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), text=True,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Head-level intervention hooks
# ---------------------------------------------------------------------------

@contextmanager
def head_direction_ablation(
    model: Any,
    layer_idx: int,
    head_idx: int,
    direction: torch.Tensor,
    alpha: float,
    head_dim: int,
):
    """Ablate a direction from a single attention head's output.

    Hooks into the o_proj input to modify only the specified head's slice
    before the output projection.
    """
    layer = _resolve_transformer_layers(model)[layer_idx]
    attn_mod, _ = _resolve_layer_components(layer)
    o_proj = _resolve_o_proj(attn_mod)

    # We need the direction in head-output space (pre-o_proj).
    # The head's contribution to the residual is: o_proj_h @ head_h
    # To remove direction d from the residual contribution:
    #   o_proj_h @ (head_h - alpha * (head_h . inv_proj_d) * inv_proj_d)
    # where inv_proj_d = pinv(o_proj_h) @ d
    # But this is expensive. Simpler: hook the o_proj OUTPUT and subtract
    # the direction's component that came from this specific head.
    # Actually even simpler: hook the full layer output and project out
    # direction only scaled by how much this head contributed.

    # Cleanest approach: hook o_proj, intercept the input, modify just this head's slice.
    o_proj_weight = o_proj.weight.detach().float()  # [d_model, n_heads*head_dim]
    dev = o_proj_weight.device

    start = head_idx * head_dim
    end = start + head_dim
    w_h = o_proj_weight[:, start:end]  # [d_model, head_dim]

    # Project direction into head space: pseudo-inverse
    # head_dir = pinv(w_h) @ direction = (w_h^T w_h)^{-1} w_h^T @ d
    d_dev = direction.to(dev, dtype=torch.float32)
    w_h_f = w_h.float()
    head_dir = torch.linalg.lstsq(w_h_f, d_dev).solution  # [head_dim]
    head_dir = head_dir / (head_dir.norm() + 1e-8)
    head_dir = head_dir.to(dtype=o_proj.weight.dtype, device=dev)

    def hook_fn(_module: Any, inputs: Any, _output: Any) -> Any:
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        # inp shape: [batch, seq, n_heads * head_dim]
        head_slice = inp[:, :, start:end]  # [batch, seq, head_dim]
        proj = (head_slice * head_dir.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True)
        new_slice = head_slice - alpha * proj * head_dir.unsqueeze(0).unsqueeze(0)
        new_inp = inp.clone()
        new_inp[:, :, start:end] = new_slice
        # Re-run o_proj with modified input
        return _module(new_inp)

    # Need to hook as a pre-hook that modifies the input to o_proj
    # Actually, register_forward_hook gets (module, input, output).
    # We want to modify the input. Use register_forward_pre_hook instead.
    def pre_hook_fn(_module: Any, inputs: Any) -> Any:
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        head_slice = inp[:, :, start:end]
        proj = (head_slice * head_dir.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True)
        new_slice = head_slice - alpha * proj * head_dir.unsqueeze(0).unsqueeze(0)
        new_inp = inp.clone()
        new_inp[:, :, start:end] = new_slice
        if isinstance(inputs, tuple):
            return (new_inp,) + inputs[1:]
        return (new_inp,)

    handle = o_proj.register_forward_pre_hook(pre_hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def head_interchange(
    model: Any,
    layer_idx: int,
    head_idx: int,
    replacement_activation: torch.Tensor,
    head_dim: int,
):
    """Replace a single head's o_proj input with a cached activation from another run.

    replacement_activation: [seq_len, head_dim] tensor to substitute for this head's output.
    """
    layer = _resolve_transformer_layers(model)[layer_idx]
    attn_mod, _ = _resolve_layer_components(layer)
    o_proj = _resolve_o_proj(attn_mod)

    start = head_idx * head_dim
    end = start + head_dim

    def pre_hook_fn(_module: Any, inputs: Any) -> Any:
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        new_inp = inp.clone()
        seq_len = min(inp.shape[1], replacement_activation.shape[0])
        dev = inp.device
        new_inp[0, :seq_len, start:end] = replacement_activation[:seq_len].to(dev, dtype=inp.dtype)
        if isinstance(inputs, tuple):
            return (new_inp,) + inputs[1:]
        return (new_inp,)

    handle = o_proj.register_forward_pre_hook(pre_hook_fn)
    try:
        yield
    finally:
        handle.remove()


def capture_head_activations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    head_idx: int,
    head_dim: int,
    ablation_spec: Optional[AblationSpec] = None,
) -> torch.Tensor:
    """Run a forward pass and capture a specific head's o_proj input activation.

    Returns: [seq_len, head_dim] tensor.
    """
    dev = _resolve_device(model)
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)

    layer = _resolve_transformer_layers(model)[layer_idx]
    attn_mod, _ = _resolve_layer_components(layer)
    o_proj = _resolve_o_proj(attn_mod)

    start = head_idx * head_dim
    end = start + head_dim
    captured: List[torch.Tensor] = []

    def hook_fn(_module: Any, inputs: Any, _output: Any) -> None:
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        captured.append(inp[0, :, start:end].detach().float().cpu())

    handle = o_proj.register_forward_hook(hook_fn)
    ctx = (
        temporary_ablation(
            model=model, layer_idx=ablation_spec.layer_idx,
            vector=ablation_spec.vector.to(dev, dtype=torch.float32),
            alpha=ablation_spec.alpha,
        )
        if ablation_spec is not None else nullcontext()
    )
    try:
        with ctx:
            with torch.no_grad():
                model(**encoded, use_cache=False)
    finally:
        handle.remove()

    return captured[0]  # [seq_len, head_dim]


# ---------------------------------------------------------------------------
# Behavioral measurement
# ---------------------------------------------------------------------------

def measure_teacher_forced_nll(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    prompt: str,
    continuation_ids: Sequence[int],
    intervention_context: Any = None,
) -> float:
    """Measure mean NLL of continuation tokens under the prompt, with optional intervention context."""
    dev = _resolve_device(model)
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    input_ids = encoded["input_ids"]
    attn_mask = encoded.get("attention_mask")
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)

    cont_tensor = torch.tensor([list(continuation_ids)], dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids, cont_tensor], dim=-1).to(dev)
    full_mask = torch.cat([attn_mask, torch.ones_like(cont_tensor)], dim=-1).to(dev)

    prompt_len = int(attn_mask[0].sum().item())
    n_cont = len(continuation_ids)

    ctx = intervention_context if intervention_context is not None else nullcontext()
    with ctx:
        with torch.no_grad():
            out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)

    logits = out.logits[0].detach().float().cpu()  # [seq_len, vocab]
    total_nll = 0.0
    for i in range(n_cont):
        pos = prompt_len - 1 + i
        log_probs = F.log_softmax(logits[pos], dim=-1)
        total_nll -= float(log_probs[continuation_ids[i]].item())

    return total_nll / n_cont


def generate_continuation_ids(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> List[int]:
    """Generate a short continuation and return the first content token IDs."""
    dev = _resolve_device(model)
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)
    input_len = encoded["input_ids"].shape[-1]

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": 0.0}
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        gen_kwargs["pad_token_id"] = eos

    with torch.no_grad():
        out = model.generate(**encoded, **gen_kwargs)
    new_ids = out[0][input_len:].tolist()

    special = set()
    for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            special.add(v)

    content = [int(t) for t in new_ids if int(t) not in special]
    return content[:max_new_tokens] if content else [int(new_ids[0])] if new_ids else [0]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def parse_head_specs(spec_str: str) -> List[Tuple[int, int]]:
    """Parse 'L.H,L.H,...' into list of (layer, head) tuples."""
    heads = []
    for part in spec_str.split(","):
        part = part.strip()
        if "." in part:
            l, h = part.split(".", 1)
            heads.append((int(l), int(h)))
    return heads


def load_top_heads_from_summary(path: str, top_k: int) -> List[Tuple[int, int]]:
    """Load top-k heads from a head_summary.csv or run_manifest.json."""
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        entries = data.get("top_20_routing_heads", [])
        if entries:
            return [(e["layer"], e["head"]) for e in entries[:top_k]]

    # CSV: detect format and rank accordingly
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    cols = set(rows[0].keys())

    # Ablation/interchange summary format: has 'layer', 'head', 'mean_nll_reduction' or similar
    if "head" in cols and "layer" in cols:
        rank_col = None
        for candidate in ("mean_nll_reduction", "mean_ctrl_to_ccp_reduction", "mean_baseline_delta"):
            if candidate in cols:
                rank_col = candidate
                break
        if rank_col:
            scored = [(int(r["layer"]), int(r["head"]), abs(float(r[rank_col]))) for r in rows]
            scored.sort(key=lambda x: x[2], reverse=True)
            return [(l, h) for l, h, _ in scored[:top_k]]

    # DLA head summary format: has mean_delta_head_0, mean_delta_head_1, ...
    head_cols = [c for c in cols if c.startswith("mean_delta_head_")]
    if head_cols:
        n_layers = len(rows)
        mid_start = int(n_layers * 0.25)
        mid_end = int(n_layers * 0.65)
        scores: List[Tuple[int, int, float]] = []
        for row in rows:
            layer = int(row["layer"])
            if mid_start <= layer <= mid_end:
                for col in head_cols:
                    h = int(col.replace("mean_delta_head_", ""))
                    scores.append((layer, h, abs(float(row[col]))))
        scores.sort(key=lambda x: x[2], reverse=True)
        return [(l, h) for l, h, _ in scores[:top_k]]

    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--corpus",
        choices=["v1", "v2", "adversarial", "safety_v1", "safety_v2", "safety_v3"],
        default="v1",
    )
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--answer-max-new-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--mode",
        choices=["ablate", "interchange"],
        required=True,
        help="'ablate': project out political direction from individual heads. "
        "'interchange': swap head activations between CCP and control runs.",
    )
    parser.add_argument(
        "--heads",
        type=str, default=None,
        help="Explicit head specs as 'L.H,L.H,...' (e.g., '12.7,14.3,16.21')",
    )
    parser.add_argument(
        "--top-k-from",
        type=str, default=None,
        help="Path to head_summary.csv or run_manifest.json to auto-select top-k heads",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--direction-checkpoint",
        type=str, required=True,
        help="Checkpoint with the political direction vector for ablation/analysis",
    )
    parser.add_argument("--direction-layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Ablation strength (only used in ablate mode)")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    if args.heads is None and args.top_k_from is None:
        parser.error("Either --heads or --top-k-from is required")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve candidate heads
    if args.heads:
        candidate_heads = parse_head_specs(args.heads)
    else:
        candidate_heads = load_top_heads_from_summary(args.top_k_from, args.top_k)

    pairs = resolve_prompt_pairs(args.corpus, args.limit_pairs)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        sha = _git_sha7()
        short = re.sub(r"[^a-z0-9]+", "_", args.model.split("/")[-1].lower()).strip("_")
        from datetime import datetime
        from zoneinfo import ZoneInfo
        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        run_dir = Path(f"runs/{date_str}_head{args.mode}_{short}_{sha}_r01")
    run_dir.mkdir(parents=True, exist_ok=True)

    direction = load_probe_direction(args.direction_checkpoint, args.direction_layer)

    print("=" * 80)
    print(f"Head-Level {'Ablation' if args.mode == 'ablate' else 'Interchange'}")
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Pairs: {len(pairs)}")
    print(f"Candidate heads: {len(candidate_heads)}")
    print(f"Mode: {args.mode}")
    if args.mode == "ablate":
        print(f"Alpha: {args.alpha}")
    print(f"Run dir: {run_dir}")
    print("=" * 80)

    print(f"[0/3] Loading model... ({_timestamp()})")
    model, tokenizer = load_model_and_tokenizer(args.model)
    n_heads = _resolve_n_heads(model)
    head_dim = _resolve_head_dim(model)

    # Pre-generate control continuations for teacher-forced measurement
    print(f"[1/3] Generating control continuations... ({_timestamp()})")
    continuations: Dict[int, List[int]] = {}
    for idx, pair in enumerate(pairs, start=1):
        continuations[pair.pair_idx] = generate_continuation_ids(
            model, tokenizer, pair.control_prompt, args.answer_max_new_tokens,
        )
        print(f"    {idx}/{len(pairs)}", end="\r", flush=True)
    print(f"    done ({len(pairs)} pairs)          ")

    # Measure baseline NLL for each pair
    print(f"[2/3] Measuring baseline + interventions... ({_timestamp()})")
    results: List[Dict[str, Any]] = []

    for pair_idx_i, pair in enumerate(pairs):
        cont_ids = continuations[pair.pair_idx]

        # Baseline: CCP prompt NLL with no intervention
        baseline_nll = measure_teacher_forced_nll(
            model, tokenizer, resolve_final_norm(model), resolve_output_head(model),
            pair.ccp_prompt, cont_ids,
        )
        # Baseline: control prompt NLL
        ctrl_baseline_nll = measure_teacher_forced_nll(
            model, tokenizer, resolve_final_norm(model), resolve_output_head(model),
            pair.control_prompt, cont_ids,
        )

        for layer_idx, head_idx in candidate_heads:
            if args.mode == "ablate":
                ctx = head_direction_ablation(
                    model, layer_idx, head_idx, direction, args.alpha, head_dim,
                )
                intervened_nll = measure_teacher_forced_nll(
                    model, tokenizer, resolve_final_norm(model), resolve_output_head(model),
                    pair.ccp_prompt, cont_ids, intervention_context=ctx,
                )
                results.append({
                    "pair_idx": pair.pair_idx,
                    "topic": pair.topic,
                    "layer": layer_idx,
                    "head": head_idx,
                    "mode": "ablate",
                    "baseline_ccp_nll": baseline_nll,
                    "baseline_ctrl_nll": ctrl_baseline_nll,
                    "baseline_delta": baseline_nll - ctrl_baseline_nll,
                    "intervened_ccp_nll": intervened_nll,
                    "intervened_delta": intervened_nll - ctrl_baseline_nll,
                    "nll_reduction": baseline_nll - intervened_nll,
                })

            elif args.mode == "interchange":
                # Capture control head activation
                ctrl_act = capture_head_activations(
                    model, tokenizer, pair.control_prompt,
                    layer_idx, head_idx, head_dim,
                )
                # Capture CCP head activation
                ccp_act = capture_head_activations(
                    model, tokenizer, pair.ccp_prompt,
                    layer_idx, head_idx, head_dim,
                )

                # Swap control→CCP (does it reduce CCP's divergence?)
                ctx_ctrl_to_ccp = head_interchange(
                    model, layer_idx, head_idx, ctrl_act, head_dim,
                )
                nll_ctrl_to_ccp = measure_teacher_forced_nll(
                    model, tokenizer, resolve_final_norm(model), resolve_output_head(model),
                    pair.ccp_prompt, cont_ids, intervention_context=ctx_ctrl_to_ccp,
                )

                # Swap CCP→control (does it increase control's divergence?)
                ctx_ccp_to_ctrl = head_interchange(
                    model, layer_idx, head_idx, ccp_act, head_dim,
                )
                nll_ccp_to_ctrl = measure_teacher_forced_nll(
                    model, tokenizer, resolve_final_norm(model), resolve_output_head(model),
                    pair.control_prompt, cont_ids, intervention_context=ctx_ccp_to_ctrl,
                )

                results.append({
                    "pair_idx": pair.pair_idx,
                    "topic": pair.topic,
                    "layer": layer_idx,
                    "head": head_idx,
                    "mode": "interchange",
                    "baseline_ccp_nll": baseline_nll,
                    "baseline_ctrl_nll": ctrl_baseline_nll,
                    "baseline_delta": baseline_nll - ctrl_baseline_nll,
                    "ctrl_to_ccp_nll": nll_ctrl_to_ccp,
                    "ctrl_to_ccp_reduction": baseline_nll - nll_ctrl_to_ccp,
                    "ccp_to_ctrl_nll": nll_ccp_to_ctrl,
                    "ccp_to_ctrl_increase": nll_ccp_to_ctrl - ctrl_baseline_nll,
                })

        print(
            f"    pair {pair_idx_i+1}/{len(pairs)}: {pair.topic} "
            f"({len(candidate_heads)} heads)",
            end="\r", flush=True,
        )
    print(f"    done ({len(pairs)} pairs x {len(candidate_heads)} heads)          ")

    # Aggregate per-head summaries
    print(f"[3/3] Writing outputs... ({_timestamp()})")
    by_head: Dict[Tuple[int, int], List[Dict]] = {}
    for r in results:
        by_head.setdefault((r["layer"], r["head"]), []).append(r)

    summary: List[Dict[str, Any]] = []
    for (layer, head), rows in sorted(by_head.items()):
        row: Dict[str, Any] = {
            "layer": layer,
            "head": head,
            "n_pairs": len(rows),
        }
        if args.mode == "ablate":
            reductions = [r["nll_reduction"] for r in rows]
            row["mean_nll_reduction"] = float(np.mean(reductions))
            row["mean_baseline_delta"] = float(np.mean([r["baseline_delta"] for r in rows]))
            row["mean_intervened_delta"] = float(np.mean([r["intervened_delta"] for r in rows]))
        elif args.mode == "interchange":
            row["mean_ctrl_to_ccp_reduction"] = float(np.mean([r["ctrl_to_ccp_reduction"] for r in rows]))
            row["mean_ccp_to_ctrl_increase"] = float(np.mean([r["ccp_to_ctrl_increase"] for r in rows]))
            row["mean_baseline_delta"] = float(np.mean([r["baseline_delta"] for r in rows]))
        summary.append(row)

    # Rank heads
    if args.mode == "ablate":
        summary.sort(key=lambda x: x["mean_nll_reduction"], reverse=True)
        print("\nHead ranking by NLL reduction (necessary for routing):")
        for r in summary[:10]:
            print(f"  L{r['layer']:>2d}.H{r['head']:>2d}: reduction={r['mean_nll_reduction']:>+.4f}  "
                  f"baseline_delta={r['mean_baseline_delta']:.4f}  "
                  f"intervened_delta={r['mean_intervened_delta']:.4f}")
    else:
        summary.sort(key=lambda x: x["mean_ctrl_to_ccp_reduction"], reverse=True)
        print("\nHead ranking by ctrl→CCP reduction (necessary for routing):")
        for r in summary[:10]:
            print(f"  L{r['layer']:>2d}.H{r['head']:>2d}: "
                  f"ctrl→CCP reduction={r['mean_ctrl_to_ccp_reduction']:>+.4f}  "
                  f"CCP→ctrl increase={r['mean_ccp_to_ctrl_increase']:>+.4f}")

    metadata = {
        "model_id": args.model,
        "corpus": args.corpus,
        "mode": args.mode,
        "seed": args.seed,
        "candidate_heads": [{"layer": l, "head": h} for l, h in candidate_heads],
        "direction_checkpoint": args.direction_checkpoint,
        "direction_layer": args.direction_layer,
        "alpha": args.alpha if args.mode == "ablate" else None,
        "n_pairs": len(pairs),
        "run_dir": str(run_dir),
    }

    write_csv(run_dir / f"head_{args.mode}_pairwise.csv", results)
    write_csv(run_dir / f"head_{args.mode}_summary.csv", summary)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
