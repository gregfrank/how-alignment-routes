#!/usr/bin/env python3
"""
Per-head direct logit attribution for political-routing analysis.

Decomposes the attention DLA signal from `routing_direct_logit_attribution.py`
into individual attention-head contributions.  Since the output projection
(o_proj) is linear, the total attention contribution to a direction d is:

    attn_contribution = sum_h  d . norm(o_proj_h @ head_h_output)

where o_proj_h is the h-th column slice of o_proj.weight (shape [d_model, head_dim]).

This requires hooking deeper into the attention module to capture per-head
outputs before they are projected and summed.  We capture the post-attention,
pre-o_proj tensor and manually slice the projection.

Primary outputs:
- per-head contribution matrix (layer x head) for each prompt pair
- aggregate layer x head summary across all pairs
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
    bootstrap_summary,
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
)
from routing_direct_logit_attribution import (
    _component_logit_contribution,
    _linearized_norm_component,
    _prepare_inputs,
    _resolve_layer_components,
    _resolve_logit_diff_direction,
    load_probe_direction,
    nullcontext,
)


def _git_sha7() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), text=True,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


def _resolve_o_proj(attn_module: Any) -> torch.nn.Linear:
    """Find the output projection linear layer inside an attention module."""
    for name in ("o_proj", "out_proj", "dense", "output"):
        candidate = getattr(attn_module, name, None)
        if isinstance(candidate, torch.nn.Linear):
            return candidate
    raise AttributeError(
        f"Could not find output projection in {type(attn_module).__name__}. "
        f"Children: {[n for n, _ in attn_module.named_children()]}"
    )


def _resolve_config(model: Any):
    """Get the text config, handling multimodal wrappers like Gemma-3."""
    config = getattr(model, "config", None)
    if config is not None:
        # Multimodal models (Gemma-3, etc.) nest text config inside text_config
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            return text_config
    return config


def _resolve_n_heads(model: Any) -> int:
    """Get the number of attention heads from model config."""
    config = _resolve_config(model)
    if config is not None:
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            val = getattr(config, attr, None)
            if isinstance(val, int):
                return val
    raise AttributeError("Could not determine number of attention heads")


def _resolve_head_dim(model: Any) -> int:
    config = _resolve_config(model)
    if config is not None:
        hd = getattr(config, "head_dim", None)
        if isinstance(hd, int):
            return hd
        hidden = getattr(config, "hidden_size", None)
        n_heads = _resolve_n_heads(model)
        if isinstance(hidden, int):
            return hidden // n_heads
    raise AttributeError("Could not determine head dimension")


def _forward_with_head_captures(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prefix_token_ids: Sequence[int],
    ablation_spec: Optional[AblationSpec],
    n_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Forward pass capturing per-head attention outputs and MLP outputs.

    Returns:
        final_residual: [d_model] tensor
        head_outputs: list of [n_heads, d_model] tensors, one per layer
            (each head's contribution after o_proj slicing)
        mlp_outputs: list of [d_model] tensors, one per layer
    """
    dev = _resolve_device(model)
    raw_inputs, last_idx = _prepare_inputs(tokenizer, prompt, prefix_token_ids)
    model_inputs = _to_device(raw_inputs, dev)

    layers = list(_resolve_transformer_layers(model))
    n_layers = len(layers)

    # We need to capture the input to o_proj (which is the concatenated head outputs)
    # and the o_proj weight matrix, then manually decompose.
    pre_oproj_captures: List[Optional[torch.Tensor]] = [None] * n_layers
    oproj_weights: List[Optional[torch.Tensor]] = [None] * n_layers
    mlp_captures: List[Optional[torch.Tensor]] = [None] * n_layers
    handles: List[Any] = []

    for layer_idx, layer in enumerate(layers):
        attn_mod, mlp_mod = _resolve_layer_components(layer)
        o_proj = _resolve_o_proj(attn_mod)

        # Capture o_proj weight (only need to do once, but it's cheap)
        oproj_weights[layer_idx] = o_proj.weight.detach().float().cpu()

        # Hook o_proj to capture its INPUT (the concatenated head outputs)
        def make_oproj_hook(store: List, idx: int):
            def hook(_module: Any, inputs: Any, _output: Any) -> None:
                # inputs[0] is the pre-projection tensor: [batch, seq, n_heads * head_dim]
                inp = inputs[0] if isinstance(inputs, tuple) else inputs
                store[idx] = inp[0, last_idx, :].detach().float().cpu()
            return hook

        def make_mlp_hook(store: List, idx: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                store[idx] = tensor[0, last_idx, :].detach().float().cpu()
            return hook

        handles.append(o_proj.register_forward_hook(make_oproj_hook(pre_oproj_captures, layer_idx)))
        handles.append(mlp_mod.register_forward_hook(make_mlp_hook(mlp_captures, layer_idx)))

    context = (
        temporary_ablation(
            model=model,
            layer_idx=ablation_spec.layer_idx,
            vector=ablation_spec.vector.to(dev, dtype=torch.float32),
            alpha=ablation_spec.alpha,
        )
        if ablation_spec is not None
        else nullcontext()
    )

    try:
        with context:
            with torch.no_grad():
                out = model(**model_inputs, output_hidden_states=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    final_residual = out.hidden_states[-1][0, last_idx, :].detach().float().cpu()

    # Decompose pre-o_proj captures into per-head contributions
    # pre_oproj: [n_heads * head_dim]
    # o_proj.weight: [d_model, n_heads * head_dim]
    # Per-head contribution: o_proj.weight[:, h*hd:(h+1)*hd] @ pre_oproj[h*hd:(h+1)*hd]
    head_contributions: List[torch.Tensor] = []
    for layer_idx in range(n_layers):
        pre_oproj = pre_oproj_captures[layer_idx]
        w = oproj_weights[layer_idx]
        if pre_oproj is None or w is None:
            raise RuntimeError(f"Missing capture at layer {layer_idx}")

        per_head = torch.zeros(n_heads, w.shape[0])  # [n_heads, d_model]
        for h in range(n_heads):
            start = h * head_dim
            end = start + head_dim
            head_input = pre_oproj[start:end]       # [head_dim]
            w_slice = w[:, start:end]                # [d_model, head_dim]
            per_head[h] = w_slice @ head_input       # [d_model]
        head_contributions.append(per_head)

    mlp_outputs = [t for t in mlp_captures if t is not None]
    if len(mlp_outputs) != n_layers:
        raise RuntimeError(f"Missing MLP captures: got {len(mlp_outputs)}, expected {n_layers}")

    return final_residual, head_contributions, mlp_outputs


def compute_head_dla_records(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    diff_direction: torch.Tensor,
    prompt: str,
    ablation_spec: Optional[AblationSpec],
    position_mode: str,
    decision_point: Optional[DecisionPoint],
    n_heads: int,
    head_dim: int,
) -> List[Dict[str, Any]]:
    """Compute per-head DLA contributions for a single prompt."""
    prefix_token_ids: Sequence[int]
    if position_mode == "last_prompt":
        prefix_token_ids = ()
    elif position_mode == "first_meaningful":
        if decision_point is None:
            raise ValueError("decision_point is required for first_meaningful")
        prefix_token_ids = decision_point.prefix_token_ids
    else:
        raise ValueError(f"Unsupported position mode: {position_mode}")

    final_residual, head_contribs, mlp_outputs = _forward_with_head_captures(
        model=model, tokenizer=tokenizer, prompt=prompt,
        prefix_token_ids=prefix_token_ids, ablation_spec=ablation_spec,
        n_heads=n_heads, head_dim=head_dim,
    )

    rows: List[Dict[str, Any]] = []
    for layer_idx, (per_head, mlp_out) in enumerate(zip(head_contribs, mlp_outputs)):
        head_vals: Dict[str, float] = {}
        for h in range(n_heads):
            contrib = _component_logit_contribution(
                final_norm=final_norm,
                diff_direction=diff_direction,
                component=per_head[h],
                reference_residual=final_residual,
            )
            head_vals[f"head_{h}"] = contrib

        mlp_contrib = _component_logit_contribution(
            final_norm=final_norm,
            diff_direction=diff_direction,
            component=mlp_out,
            reference_residual=final_residual,
        )

        row: Dict[str, Any] = {
            "layer": layer_idx,
            "mlp_contribution": mlp_contrib,
            "attn_total": sum(head_vals.values()),
        }
        row.update(head_vals)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--corpus",
        choices=["v1", "v2", "adversarial", "safety_v1", "safety_v2", "safety_v3"],
        default="v1",
    )
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--refusal-strings", nargs="+", default=list(DEFAULT_REFUSAL_STRINGS))
    parser.add_argument("--answer-max-new-tokens", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--position",
        choices=["first_meaningful", "last_prompt"],
        default="last_prompt",
    )
    parser.add_argument(
        "--target",
        choices=["logit_diff", "probe"],
        default="logit_diff",
    )
    parser.add_argument("--probe-direction-checkpoint", type=str, default=None)
    parser.add_argument("--probe-direction-layer", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--ablation-checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.target == "probe" and args.probe_direction_checkpoint is None:
        parser.error("--probe-direction-checkpoint is required when --target probe")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs = resolve_prompt_pairs(args.corpus, args.limit_pairs)
    base_dir = Path("runs")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        sha = _git_sha7()
        short = re.sub(r"[^a-z0-9]+", "_", args.model.split("/")[-1].lower()).strip("_")
        from datetime import datetime
        from zoneinfo import ZoneInfo
        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        stem = f"{date_str}_headDLA_{short}_{args.corpus}_s{args.seed}_{sha}"
        for idx in range(1, 100):
            run_dir = base_dir / f"{stem}_r{idx:02d}"
            if not run_dir.exists():
                break
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load probe direction if needed
    probe_direction: Optional[torch.Tensor] = None
    if args.target == "probe":
        probe_direction = load_probe_direction(
            args.probe_direction_checkpoint, args.probe_direction_layer,
        )

    ablation_spec = load_ablation_spec(Path(args.ablation_checkpoint)) if args.ablation_checkpoint else None

    print("=" * 80)
    print("Per-Head Direct Logit Attribution")
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Pairs: {len(pairs)}")
    print(f"Position: {args.position}")
    print(f"Target: {args.target}")
    print(f"Run dir: {run_dir}")
    if ablation_spec is not None:
        print(f"Ablation: layer={ablation_spec.layer_idx} alpha={ablation_spec.alpha:.4f}")
    print("=" * 80)

    print(f"[0/4] Loading model... ({_timestamp()})")
    model, tokenizer = load_model_and_tokenizer(args.model)
    final_norm = resolve_final_norm(model)
    lm_head = resolve_output_head(model)
    lm_head_weight = getattr(lm_head, "weight", None)
    if not isinstance(lm_head_weight, torch.Tensor):
        raise AttributeError("LM head missing weight tensor")
    lm_head_weight = lm_head_weight.detach().float().cpu()

    n_heads = _resolve_n_heads(model)
    head_dim = _resolve_head_dim(model)
    print(f"  Heads: {n_heads}, head_dim: {head_dim}")

    # Build answer bundle for logit_diff target or first_meaningful position
    refusal_ids: List[int] = []
    answer_decisions: Dict[int, DecisionPoint] = {}
    decoded_answer_ids: Dict[int, str] = {}
    need_bundles = args.target == "logit_diff" or args.position == "first_meaningful"
    if need_bundles:
        refusal_bundle = build_refusal_bundle(tokenizer, args.refusal_strings)
        refusal_ids = refusal_bundle["token_ids"]
        answer_decisions, _, decoded_answer_ids = build_answer_bundle(
            model=model, tokenizer=tokenizer, pairs=pairs,
            ablation_spec=ablation_spec, max_new_tokens=args.answer_max_new_tokens,
        )

    ccp_decisions: Dict[int, DecisionPoint] = {}
    if args.position == "first_meaningful":
        print(f"[1/4] Resolving CCP positions... ({_timestamp()})")
        for idx, pair in enumerate(pairs, start=1):
            ccp_decisions[pair.pair_idx] = find_first_meaningful_generation(
                model=model, tokenizer=tokenizer, prompt=pair.ccp_prompt,
                ablation_spec=ablation_spec, max_new_tokens=args.answer_max_new_tokens,
            )
            print(f"    {idx}/{len(pairs)}", end="\r", flush=True)
        print(f"    done ({len(pairs)} pairs)          ")

    print(f"[2/4] Computing per-head DLA... ({_timestamp()})")
    pair_rows: List[Dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        # Resolve direction
        if probe_direction is not None:
            direction = probe_direction
        else:
            pair_answer_id = (
                answer_decisions[pair.pair_idx].meaningful_token_id
                if pair.pair_idx in answer_decisions else 0
            )
            direction = _resolve_logit_diff_direction(
                lm_head_weight=lm_head_weight,
                refusal_ids=refusal_ids,
                answer_token_id=pair_answer_id,
            )

        ccp_rows = compute_head_dla_records(
            model=model, tokenizer=tokenizer, final_norm=final_norm,
            diff_direction=direction, prompt=pair.ccp_prompt,
            ablation_spec=ablation_spec, position_mode=args.position,
            decision_point=ccp_decisions.get(pair.pair_idx),
            n_heads=n_heads, head_dim=head_dim,
        )
        ctrl_rows = compute_head_dla_records(
            model=model, tokenizer=tokenizer, final_norm=final_norm,
            diff_direction=direction, prompt=pair.control_prompt,
            ablation_spec=ablation_spec, position_mode=args.position,
            decision_point=answer_decisions.get(pair.pair_idx),
            n_heads=n_heads, head_dim=head_dim,
        )

        for ccp_row, ctrl_row in zip(ccp_rows, ctrl_rows):
            row: Dict[str, Any] = {
                "pair_idx": pair.pair_idx,
                "topic": pair.topic,
                "layer": ccp_row["layer"],
                "delta_mlp": ccp_row["mlp_contribution"] - ctrl_row["mlp_contribution"],
                "delta_attn_total": ccp_row["attn_total"] - ctrl_row["attn_total"],
            }
            for h in range(n_heads):
                key = f"head_{h}"
                row[f"delta_{key}"] = ccp_row[key] - ctrl_row[key]
            pair_rows.append(row)
        print(f"    pair {idx}/{len(pairs)}: {pair.topic}", end="\r", flush=True)
    print(f"    done ({len(pairs)} pairs)          ")

    print(f"[3/4] Aggregating summaries... ({_timestamp()})")
    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    for row in pair_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        layer_rows = by_layer[layer]
        row: Dict[str, Any] = {"layer": layer, "n_pairs": len(layer_rows)}

        mlp_vals = np.array([r["delta_mlp"] for r in layer_rows])
        row["mean_delta_mlp"] = float(mlp_vals.mean())

        attn_vals = np.array([r["delta_attn_total"] for r in layer_rows])
        row["mean_delta_attn_total"] = float(attn_vals.mean())

        for h in range(n_heads):
            vals = np.array([r[f"delta_head_{h}"] for r in layer_rows])
            row[f"mean_delta_head_{h}"] = float(vals.mean())
            ci_lo, ci_hi = bootstrap_summary(vals, args.bootstrap, args.seed + layer * 100 + h)
            row[f"head_{h}_ci_low"] = ci_lo
            row[f"head_{h}_ci_high"] = ci_hi

        summary_rows.append(row)

    # Find top routing heads (largest mean delta across layers in the effective band)
    n_layers = len(summary_rows)
    mid_start = int(n_layers * 0.25)
    mid_end = int(n_layers * 0.65)
    head_scores: List[Tuple[int, int, float]] = []
    for row in summary_rows:
        layer = row["layer"]
        if mid_start <= layer <= mid_end:
            for h in range(n_heads):
                head_scores.append((layer, h, abs(row[f"mean_delta_head_{h}"])))

    head_scores.sort(key=lambda x: x[2], reverse=True)
    top_heads = head_scores[:20]

    metadata = {
        "model_id": args.model,
        "corpus": args.corpus,
        "seed": args.seed,
        "position": args.position,
        "target": args.target,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "run_dir": str(run_dir),
        "n_pairs": len(pairs),
        "top_20_routing_heads": [
            {"layer": l, "head": h, "abs_delta": round(v, 4)} for l, h, v in top_heads
        ],
        "ablation": None if ablation_spec is None else {
            "layer_idx": ablation_spec.layer_idx,
            "alpha": ablation_spec.alpha,
            "source_path": ablation_spec.source_path,
        },
    }

    print(f"[4/4] Writing outputs... ({_timestamp()})")
    write_csv(run_dir / "head_pairwise_by_layer.csv", pair_rows)
    write_csv(run_dir / "head_summary.csv", summary_rows)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    print("\nTop 20 routing head candidates (mid-band, by |delta|):")
    for l, h, v in top_heads:
        print(f"  L{l:>2d}.H{h:>2d}: |delta| = {v:.4f}")

    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
