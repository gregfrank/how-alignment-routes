#!/usr/bin/env python3
"""
Approximate per-layer direct logit attribution for political-routing analysis.

This runner reuses the matched-pair routing setup from `routing_logit_trajectory.py`
and estimates which transformer blocks contribute to the refusal-vs-answer shift
at the decision point. The main outputs are per-pair, per-layer attention and MLP
contributions plus aggregate summaries by layer and topic.
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


def _git_sha7() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _sanitize_model_short(model_id: str) -> str:
    short = model_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", short).strip("_")


def _next_run_dir(base_dir: Path, model_id: str, corpus: str, seed: int) -> Path:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    stem = f"{date_str}_routingdla_{_sanitize_model_short(model_id)}_{corpus}_s{seed}_{_git_sha7()}"
    for idx in range(1, 100):
        candidate = base_dir / f"{stem}_r{idx:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate run dir for stem={stem}")


class nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _extract_module_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        if not output:
            raise ValueError("Encountered empty tuple output while extracting component tensor")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported module output type for DLA capture: {type(output)!r}")
    return output


def _resolve_layer_components(layer: Any) -> Tuple[torch.nn.Module, torch.nn.Module]:
    attn = None
    mlp = None
    for name in ("self_attn", "attention", "attn", "mixer", "linear_attn"):
        candidate = getattr(layer, name, None)
        if isinstance(candidate, torch.nn.Module):
            attn = candidate
            break
    for name in ("mlp", "feed_forward", "ffn"):
        candidate = getattr(layer, name, None)
        if isinstance(candidate, torch.nn.Module):
            mlp = candidate
            break
    if attn is None or mlp is None:
        raise AttributeError(
            f"Could not resolve attention/MLP components on layer type {type(layer).__name__}: "
            f"attn={type(attn).__name__ if attn is not None else None}, "
            f"mlp={type(mlp).__name__ if mlp is not None else None}"
        )
    return attn, mlp


def _prepare_inputs(
    tokenizer: Any,
    prompt: str,
    prefix_token_ids: Sequence[int],
) -> Tuple[Dict[str, Any], int]:
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    if prefix_token_ids:
        prefix = torch.tensor([list(prefix_token_ids)], dtype=input_ids.dtype)
        prefix_mask = torch.ones_like(prefix, dtype=attention_mask.dtype)
        input_ids = torch.cat([input_ids, prefix], dim=-1)
        attention_mask = torch.cat([attention_mask, prefix_mask], dim=-1)

    model_inputs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "token_type_ids" in encoded:
        token_type_ids = encoded["token_type_ids"]
        if prefix_token_ids:
            prefix_tti = torch.zeros((1, len(prefix_token_ids)), dtype=token_type_ids.dtype)
            token_type_ids = torch.cat([token_type_ids, prefix_tti], dim=-1)
        model_inputs["token_type_ids"] = token_type_ids

    last_idx = int(attention_mask[0].sum().item()) - 1
    return model_inputs, last_idx


def _forward_with_component_captures(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prefix_token_ids: Sequence[int],
    ablation_spec: Optional[AblationSpec],
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    dev = _resolve_device(model)
    raw_inputs, last_idx = _prepare_inputs(tokenizer, prompt, prefix_token_ids)
    model_inputs = _to_device(raw_inputs, dev)

    layers = list(_resolve_transformer_layers(model))
    attn_outputs: List[Optional[torch.Tensor]] = [None] * len(layers)
    mlp_outputs: List[Optional[torch.Tensor]] = [None] * len(layers)
    handles: List[Any] = []

    def make_hook(store: List[Optional[torch.Tensor]], layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = _extract_module_tensor(output)
            store[layer_idx] = tensor[0, last_idx, :].detach().float().cpu()

        return hook

    for layer_idx, layer in enumerate(layers):
        attn_mod, mlp_mod = _resolve_layer_components(layer)
        handles.append(attn_mod.register_forward_hook(make_hook(attn_outputs, layer_idx)))
        handles.append(mlp_mod.register_forward_hook(make_hook(mlp_outputs, layer_idx)))

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
        for handle in handles:
            handle.remove()

    final_residual = out.hidden_states[-1][0, last_idx, :].detach().float().cpu()
    missing_attn = [idx for idx, tensor in enumerate(attn_outputs) if tensor is None]
    missing_mlp = [idx for idx, tensor in enumerate(mlp_outputs) if tensor is None]
    if missing_attn or missing_mlp:
        raise RuntimeError(f"Missing component captures: attn={missing_attn} mlp={missing_mlp}")

    return (
        final_residual,
        [tensor for tensor in attn_outputs if tensor is not None],
        [tensor for tensor in mlp_outputs if tensor is not None],
    )


def _resolve_logit_diff_direction(
    lm_head_weight: torch.Tensor,
    refusal_ids: Sequence[int],
    answer_token_id: int,
) -> torch.Tensor:
    refusal_vec = lm_head_weight[list(refusal_ids)].mean(dim=0)
    answer_vec = lm_head_weight[int(answer_token_id)]
    return refusal_vec - answer_vec


def _linearized_norm_component(
    final_norm: torch.nn.Module,
    component: torch.Tensor,
    reference_residual: torch.Tensor,
) -> torch.Tensor:
    if type(final_norm).__name__ == "IdentityModule":
        return component

    weight = getattr(final_norm, "weight", None)
    if not isinstance(weight, torch.Tensor):
        # Fallback: apply the module directly, even though this is a weaker approximation.
        with torch.no_grad():
            return final_norm(component.view(1, 1, -1)).view(-1).detach().float().cpu()

    weight = weight.detach().float().cpu()
    eps = float(
        getattr(
            final_norm,
            "eps",
            getattr(final_norm, "variance_epsilon", 1e-5),
        )
    )
    class_name = type(final_norm).__name__.lower()
    ref = reference_residual.detach().float().cpu()
    comp = component.detach().float().cpu()

    if "rmsnorm" in class_name:
        rms = torch.sqrt(ref.pow(2).mean() + eps)
        return comp * (weight / rms)

    if "layernorm" in class_name:
        centered_comp = comp - comp.mean()
        centered_ref = ref - ref.mean()
        std = torch.sqrt(centered_ref.pow(2).mean() + eps)
        return centered_comp * (weight / std)

    with torch.no_grad():
        return final_norm(comp.view(1, 1, -1)).view(-1).detach().float().cpu()


def _component_logit_contribution(
    final_norm: torch.nn.Module,
    diff_direction: torch.Tensor,
    component: torch.Tensor,
    reference_residual: torch.Tensor,
) -> float:
    transformed = _linearized_norm_component(final_norm, component, reference_residual)
    return float(torch.dot(transformed, diff_direction).item())


def load_probe_direction(checkpoint_path: str, layer_idx: Optional[int] = None) -> torch.Tensor:
    """Load a ridge-cleaned direction vector from an ablation or ridge checkpoint.

    If *layer_idx* is given, return the ``v_clean`` for that layer.  Otherwise
    return the direction from the best candidate (ablation checkpoint) or the
    first candidate (ridge checkpoint).
    """
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"{checkpoint_path} did not contain a dict checkpoint")

    # Ablation-style checkpoint (bank + best)
    if "bank" in obj and "best" in obj:
        bank = obj["bank"]
        if layer_idx is not None:
            for entry in bank:
                if int(entry["layer_idx"]) == layer_idx:
                    return entry["v_clean"].detach().cpu().float()
            raise ValueError(f"No bank entry at layer {layer_idx} in {checkpoint_path}")
        best_idx = int(obj["best"]["candidate_idx"])
        return bank[best_idx]["v_clean"].detach().cpu().float()

    # Ridge-style checkpoint (ridge_candidates)
    if "ridge_candidates" in obj:
        candidates = obj["ridge_candidates"]
        if layer_idx is not None:
            for entry in candidates:
                if int(entry["layer_idx"]) == layer_idx:
                    return entry["v_clean"].detach().cpu().float()
            raise ValueError(f"No ridge candidate at layer {layer_idx} in {checkpoint_path}")
        return candidates[0]["v_clean"].detach().cpu().float()

    raise ValueError(f"Unsupported probe direction checkpoint format: {checkpoint_path}")


def compute_prompt_component_records(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head_weight: torch.Tensor,
    prompt: str,
    answer_token_id: int,
    refusal_ids: Sequence[int],
    ablation_spec: Optional[AblationSpec],
    position_mode: str,
    decision_point: Optional[DecisionPoint],
    override_direction: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    prefix_token_ids: Sequence[int]
    if position_mode == "last_prompt":
        prefix_token_ids = ()
    elif position_mode == "first_meaningful":
        if decision_point is None:
            raise ValueError("decision_point is required for first_meaningful DLA")
        prefix_token_ids = decision_point.prefix_token_ids
    else:
        raise ValueError(f"Unsupported position mode: {position_mode}")

    final_residual, attn_components, mlp_components = _forward_with_component_captures(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prefix_token_ids=prefix_token_ids,
        ablation_spec=ablation_spec,
    )
    if override_direction is not None:
        diff_direction = override_direction
    else:
        diff_direction = _resolve_logit_diff_direction(
            lm_head_weight=lm_head_weight,
            refusal_ids=refusal_ids,
            answer_token_id=answer_token_id,
        )
    rows: List[Dict[str, Any]] = []
    for layer_idx, (attn_component, mlp_component) in enumerate(zip(attn_components, mlp_components)):
        attn_contrib = _component_logit_contribution(
            final_norm=final_norm,
            diff_direction=diff_direction,
            component=attn_component,
            reference_residual=final_residual,
        )
        mlp_contrib = _component_logit_contribution(
            final_norm=final_norm,
            diff_direction=diff_direction,
            component=mlp_component,
            reference_residual=final_residual,
        )
        rows.append(
            {
                "layer": layer_idx,
                "attn_pair_contribution": attn_contrib,
                "mlp_pair_contribution": mlp_contrib,
                "total_pair_contribution": attn_contrib + mlp_contrib,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--corpus", choices=["v1", "adversarial"], default="v1")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--refusal-strings", nargs="+", default=list(DEFAULT_REFUSAL_STRINGS))
    parser.add_argument("--answer-max-new-tokens", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--position",
        choices=["first_meaningful", "last_prompt"],
        default="first_meaningful",
    )
    parser.add_argument(
        "--target",
        choices=["logit_diff", "probe"],
        default="logit_diff",
        help="DLA attribution target. 'logit_diff' projects onto refusal-vs-answer direction (default). "
        "'probe' projects onto a ridge-cleaned direction loaded from --probe-direction-checkpoint.",
    )
    parser.add_argument(
        "--probe-direction-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint containing v_clean direction vector (ablation or ridge checkpoint). "
        "Required when --target probe.",
    )
    parser.add_argument(
        "--probe-direction-layer",
        type=int,
        default=None,
        help="Layer index to select from the probe checkpoint. If omitted, use best/first candidate.",
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--ablation-checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.target == "probe" and args.probe_direction_checkpoint is None:
        parser.error("--probe-direction-checkpoint is required when --target probe")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs = resolve_prompt_pairs(args.corpus, args.limit_pairs)
    base_dir = Path("runs")
    run_dir = Path(args.run_dir) if args.run_dir else _next_run_dir(base_dir, args.model, args.corpus, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    ablation_spec = load_ablation_spec(Path(args.ablation_checkpoint)) if args.ablation_checkpoint else None

    # Load probe direction if in probe mode
    probe_direction: Optional[torch.Tensor] = None
    if args.target == "probe":
        probe_direction = load_probe_direction(
            args.probe_direction_checkpoint, args.probe_direction_layer,
        )
        print(f"Probe direction: shape={probe_direction.shape}, "
              f"norm={probe_direction.norm().item():.4f}, "
              f"source={args.probe_direction_checkpoint}")

    print("=" * 80)
    print("Routing Direct Logit Attribution")
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Pairs: {len(pairs)}")
    print(f"Position: {args.position}")
    print(f"Target: {args.target}")
    print(f"Run dir: {run_dir}")
    if ablation_spec is not None:
        print(
            f"Ablation: layer={ablation_spec.layer_idx} alpha={ablation_spec.alpha:.4f} "
            f"source={ablation_spec.source_path}"
        )
    print("=" * 80)

    print(f"[0/4] Loading model/tokenizer... ({_timestamp()})")
    model, tokenizer = load_model_and_tokenizer(args.model)
    final_norm = resolve_final_norm(model)
    lm_head = resolve_output_head(model)
    lm_head_weight = getattr(lm_head, "weight", None)
    if not isinstance(lm_head_weight, torch.Tensor):
        raise AttributeError("LM head does not expose a weight tensor needed for DLA projection")
    lm_head_weight = lm_head_weight.detach().float().cpu()

    # In probe mode with last_prompt, we don't need answer/refusal bundles at all.
    # But we still build them for logit_diff mode and for first_meaningful position
    # (which needs answer bundle to resolve decision points).
    refusal_bundle: Dict[str, Any] = {}
    refusal_ids: List[int] = []
    answer_decisions_by_pair: Dict[int, DecisionPoint] = {}
    decoded_answer_ids: Dict[int, str] = {}

    need_bundles = args.target == "logit_diff" or args.position == "first_meaningful"
    if need_bundles:
        refusal_bundle = build_refusal_bundle(tokenizer, args.refusal_strings)
        refusal_ids = refusal_bundle["token_ids"]
        if not refusal_ids and args.target == "logit_diff":
            raise ValueError("Refusal bundle is empty after tokenization")

        answer_decisions_by_pair, _, decoded_answer_ids = build_answer_bundle(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            ablation_spec=ablation_spec,
            max_new_tokens=args.answer_max_new_tokens,
        )

    ccp_decisions_by_pair: Dict[int, DecisionPoint] = {}
    if args.position == "first_meaningful":
        print(f"[1/4] Resolving CCP decision positions... ({_timestamp()})")
        for idx, pair in enumerate(pairs, start=1):
            decision = find_first_meaningful_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=pair.ccp_prompt,
                ablation_spec=ablation_spec,
                max_new_tokens=args.answer_max_new_tokens,
            )
            ccp_decisions_by_pair[pair.pair_idx] = decision
            decoded = tokenizer.decode([decision.meaningful_token_id], skip_special_tokens=False)
            print(
                f"    ccp token {idx}/{len(pairs)}: pair={pair.pair_idx} "
                f"id={decision.meaningful_token_id} text={decoded!r}",
                end="\r",
                flush=True,
            )
        print(f"    ccp positions: done ({len(pairs)} pairs)          ")

    print(f"[2/4] Computing layerwise DLA... ({_timestamp()})")
    pair_rows: List[Dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        pair_answer_id = (
            answer_decisions_by_pair[pair.pair_idx].meaningful_token_id
            if pair.pair_idx in answer_decisions_by_pair
            else 0
        )
        ccp_rows = compute_prompt_component_records(
            model=model,
            tokenizer=tokenizer,
            final_norm=final_norm,
            lm_head_weight=lm_head_weight,
            prompt=pair.ccp_prompt,
            answer_token_id=pair_answer_id,
            refusal_ids=refusal_ids,
            ablation_spec=ablation_spec,
            position_mode=args.position,
            decision_point=ccp_decisions_by_pair.get(pair.pair_idx),
            override_direction=probe_direction,
        )
        control_rows = compute_prompt_component_records(
            model=model,
            tokenizer=tokenizer,
            final_norm=final_norm,
            lm_head_weight=lm_head_weight,
            prompt=pair.control_prompt,
            answer_token_id=pair_answer_id,
            refusal_ids=refusal_ids,
            ablation_spec=ablation_spec,
            position_mode=args.position,
            decision_point=answer_decisions_by_pair.get(pair.pair_idx),
            override_direction=probe_direction,
        )
        answer_text = decoded_answer_ids.get(pair_answer_id, "n/a")
        for ccp_row, control_row in zip(ccp_rows, control_rows):
            pair_rows.append(
                {
                    "pair_idx": pair.pair_idx,
                    "topic": pair.topic,
                    "intensity": pair.intensity,
                    "layer": ccp_row["layer"],
                    "pair_answer_token_id": pair_answer_id,
                    "pair_answer_token_text": answer_text,
                    "ccp_attn_pair_contribution": ccp_row["attn_pair_contribution"],
                    "ccp_mlp_pair_contribution": ccp_row["mlp_pair_contribution"],
                    "ccp_total_pair_contribution": ccp_row["total_pair_contribution"],
                    "control_attn_pair_contribution": control_row["attn_pair_contribution"],
                    "control_mlp_pair_contribution": control_row["mlp_pair_contribution"],
                    "control_total_pair_contribution": control_row["total_pair_contribution"],
                    "delta_attn_pair_contribution": (
                        ccp_row["attn_pair_contribution"] - control_row["attn_pair_contribution"]
                    ),
                    "delta_mlp_pair_contribution": (
                        ccp_row["mlp_pair_contribution"] - control_row["mlp_pair_contribution"]
                    ),
                    "delta_total_pair_contribution": (
                        ccp_row["total_pair_contribution"] - control_row["total_pair_contribution"]
                    ),
                }
            )
        print(f"    pair {idx}/{len(pairs)} complete: topic={pair.topic}", end="\r", flush=True)
    print(f"    DLA: done ({len(pairs)} pairs)          ")

    print(f"[3/4] Aggregating summaries... ({_timestamp()})")
    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    by_layer_topic: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in pair_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
        by_layer_topic.setdefault((int(row["layer"]), str(row["topic"])), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        attn_vals = np.asarray([float(r["delta_attn_pair_contribution"]) for r in rows], dtype=np.float64)
        mlp_vals = np.asarray([float(r["delta_mlp_pair_contribution"]) for r in rows], dtype=np.float64)
        total_vals = np.asarray([float(r["delta_total_pair_contribution"]) for r in rows], dtype=np.float64)
        attn_low, attn_high = bootstrap_summary(attn_vals, args.bootstrap, args.seed + layer)
        mlp_low, mlp_high = bootstrap_summary(mlp_vals, args.bootstrap, args.seed + 1000 + layer)
        total_low, total_high = bootstrap_summary(total_vals, args.bootstrap, args.seed + 2000 + layer)
        summary_rows.append(
            {
                "layer": layer,
                "n_pairs": len(rows),
                "mean_delta_attn_pair_contribution": float(attn_vals.mean()),
                "attn_ci_low": attn_low,
                "attn_ci_high": attn_high,
                "mean_delta_mlp_pair_contribution": float(mlp_vals.mean()),
                "mlp_ci_low": mlp_low,
                "mlp_ci_high": mlp_high,
                "mean_delta_total_pair_contribution": float(total_vals.mean()),
                "total_ci_low": total_low,
                "total_ci_high": total_high,
            }
        )

    topic_rows: List[Dict[str, Any]] = []
    for (layer, topic), rows in sorted(by_layer_topic.items()):
        topic_rows.append(
            {
                "layer": layer,
                "topic": topic,
                "n_pairs": len(rows),
                "mean_delta_attn_pair_contribution": float(
                    np.mean([float(r["delta_attn_pair_contribution"]) for r in rows])
                ),
                "mean_delta_mlp_pair_contribution": float(
                    np.mean([float(r["delta_mlp_pair_contribution"]) for r in rows])
                ),
                "mean_delta_total_pair_contribution": float(
                    np.mean([float(r["delta_total_pair_contribution"]) for r in rows])
                ),
            }
        )

    metadata: Dict[str, Any] = {
        "model_id": args.model,
        "corpus": args.corpus,
        "seed": args.seed,
        "position": args.position,
        "target": args.target,
        "run_dir": str(run_dir),
        "n_pairs": len(pairs),
        "final_norm_type": type(final_norm).__name__,
        "ablation": None
        if ablation_spec is None
        else {
            "layer_idx": ablation_spec.layer_idx,
            "alpha": ablation_spec.alpha,
            "source_path": ablation_spec.source_path,
            "source_kind": ablation_spec.source_kind,
        },
    }
    if args.target == "logit_diff":
        metadata["refusal_bundle"] = refusal_bundle
    elif args.target == "probe":
        metadata["probe_direction_checkpoint"] = args.probe_direction_checkpoint
        metadata["probe_direction_layer"] = args.probe_direction_layer

    print(f"[4/4] Writing outputs... ({_timestamp()})")
    write_csv(run_dir / "component_pairwise_by_layer.csv", pair_rows)
    write_csv(run_dir / "component_summary.csv", summary_rows)
    write_csv(run_dir / "component_by_topic.csv", topic_rows)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"  Summary: {run_dir / 'component_summary.csv'}")
    print(f"  Pairwise: {run_dir / 'component_pairwise_by_layer.csv'}")
    print(f"  By topic: {run_dir / 'component_by_topic.csv'}")


if __name__ == "__main__":
    main()
