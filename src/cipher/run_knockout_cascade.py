#!/usr/bin/env python3
"""Knockout cascade at n=120: zero the gate head and measure amplifier DLA changes.

For each prompt pair, computes per-head DLA (via o_proj slicing) for target
amplifier heads both normally and with the gate head zeroed.  If zeroing the
gate suppresses amplifier contributions, the gate causally drives those heads.

Outputs cascade_summary.csv with columns:
  layer, head, mean_normal_delta, mean_knockout_delta, change_pct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from routing_logit_trajectory import (
    DEFAULT_REFUSAL_STRINGS,
    SEED,
    PromptPair,
    _resolve_device,
    _resolve_transformer_layers,
    build_answer_bundle,
    build_refusal_bundle,
    load_model_and_tokenizer,
    resolve_final_norm,
    resolve_output_head,
    resolve_prompt_pairs,
    write_csv,
)
from routing_direct_logit_attribution import (
    _resolve_layer_components,
    _resolve_logit_diff_direction,
)
from routing_head_dla import (
    _resolve_head_dim,
    _resolve_n_heads,
    _resolve_o_proj,
    compute_head_dla_records,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Gate zeroing hook — zeros a head's o_proj input (pre-projection space)
# ---------------------------------------------------------------------------

@contextmanager
def zero_head_hook(model: Any, layer_idx: int, head_idx: int, head_dim: int):
    """Zero a specific head's contribution by zeroing its o_proj input slice."""
    layer = _resolve_transformer_layers(model)[layer_idx]
    attn_mod, _ = _resolve_layer_components(layer)
    o_proj = _resolve_o_proj(attn_mod)

    start = head_idx * head_dim
    end = start + head_dim

    def pre_hook(_module: Any, inputs: Any) -> Any:
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        new_inp = inp.clone()
        new_inp[:, :, start:end] = 0.0
        if isinstance(inputs, tuple):
            return (new_inp,) + inputs[1:]
        return (new_inp,)

    handle = o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Main cascade logic
# ---------------------------------------------------------------------------

def parse_head(s: str) -> Tuple[int, int]:
    parts = s.strip().split(".")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--corpus", required=True,
        choices=["v1", "v2", "safety_v1", "safety_v2", "safety_v3"],
    )
    parser.add_argument("--gate", required=True, help="Gate head as L.H, e.g. 17.17")
    parser.add_argument("--amplifiers", required=True,
                        help="Amplifier heads as L.H,L.H,... e.g. 22.7,23.3,22.4")
    parser.add_argument("--run-dir", required=True, help="Output directory")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--answer-max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    gate_layer, gate_head = parse_head(args.gate)
    amplifiers = [parse_head(h) for h in args.amplifiers.split(",")]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Knockout cascade: gate L{gate_layer}.H{gate_head} ===")
    print(f"  Model: {args.model}")
    print(f"  Corpus: {args.corpus}")
    print(f"  Amplifiers: {['L%d.H%d' % (l, h) for l, h in amplifiers]}")

    # Load model
    print(f"\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    dev = _resolve_device(model)
    head_dim = _resolve_head_dim(model)
    n_heads = _resolve_n_heads(model)
    final_norm = resolve_final_norm(model)
    lm_head = resolve_output_head(model)
    W = lm_head.weight.detach().float().cpu()

    # Load pairs
    pairs = resolve_prompt_pairs(args.corpus, args.limit_pairs)
    n_pairs = len(pairs)
    print(f"Pairs: {n_pairs}")

    # Build per-pair diff directions
    print(f"\nBuilding answer bundle...")
    refusal_bundle = build_refusal_bundle(tokenizer, DEFAULT_REFUSAL_STRINGS)
    refusal_ids = refusal_bundle["token_ids"]
    answer_bundle, _, _ = build_answer_bundle(
        model, tokenizer, pairs, None, args.answer_max_new_tokens,
    )

    # Per-pair results: {(layer, head): [list of (normal_delta, ko_delta)]}
    pair_results: Dict[Tuple[int, int], List[Tuple[float, float]]] = {
        h: [] for h in amplifiers
    }

    t0 = time.time()

    for pi, pair in enumerate(pairs):
        dp = answer_bundle[pair.pair_idx]
        diff_direction = _resolve_logit_diff_direction(W, refusal_ids, dp.meaningful_token_id)

        # Normal: DLA for sensitive and control prompts
        normal_ccp = compute_head_dla_records(
            model, tokenizer, final_norm, diff_direction,
            pair.ccp_prompt, None, "last_prompt", None, n_heads, head_dim,
        )
        normal_ctrl = compute_head_dla_records(
            model, tokenizer, final_norm, diff_direction,
            pair.control_prompt, None, "last_prompt", None, n_heads, head_dim,
        )

        # Knockout: zero the gate head, then compute DLA
        with zero_head_hook(model, gate_layer, gate_head, head_dim):
            ko_ccp = compute_head_dla_records(
                model, tokenizer, final_norm, diff_direction,
                pair.ccp_prompt, None, "last_prompt", None, n_heads, head_dim,
            )
            ko_ctrl = compute_head_dla_records(
                model, tokenizer, final_norm, diff_direction,
                pair.control_prompt, None, "last_prompt", None, n_heads, head_dim,
            )

        # Extract amplifier deltas
        for (al, ah) in amplifiers:
            key = f"head_{ah}"
            normal_delta = normal_ccp[al][key] - normal_ctrl[al][key]
            ko_delta = ko_ccp[al][key] - ko_ctrl[al][key]
            pair_results[(al, ah)].append((normal_delta, ko_delta))

        elapsed = time.time() - t0
        rate = (pi + 1) / elapsed
        eta = (n_pairs - pi - 1) / rate if rate > 0 else 0
        print(f"  Pair {pi+1}/{n_pairs}  ({rate:.1f} pairs/min, ETA {eta:.0f}s)", end="\r", flush=True)

    print(f"\nDone ({time.time()-t0:.0f}s total)")

    # Aggregate into cascade_summary.csv
    print(f"\n{'Head':>10} {'Normal':>10} {'Knockout':>10} {'Change%':>10}")
    print("-" * 44)
    rows = []
    for (al, ah) in amplifiers:
        deltas = pair_results[(al, ah)]
        normals = [d[0] for d in deltas]
        knockouts = [d[1] for d in deltas]
        mean_normal = float(np.mean(normals))
        mean_knockout = float(np.mean(knockouts))
        if abs(mean_normal) > 1e-8:
            change_pct = (mean_knockout - mean_normal) / abs(mean_normal) * 100
        else:
            change_pct = 0.0
        rows.append({
            "layer": al,
            "head": ah,
            "mean_normal_delta": round(mean_normal, 6),
            "mean_knockout_delta": round(mean_knockout, 6),
            "change_pct": round(change_pct, 1),
            "n_pairs": len(deltas),
        })
        sign = "+" if change_pct > 0 else ""
        print(f"  L{al:>2}.H{ah:<2}  {mean_normal:>+10.4f} {mean_knockout:>+10.4f}  {sign}{change_pct:.1f}%")

    write_csv(run_dir / "cascade_summary.csv", rows)

    # Also write per-pair raw data for bootstrap analysis
    raw_rows = []
    for (al, ah) in amplifiers:
        for pi, (nd, kd) in enumerate(pair_results[(al, ah)]):
            raw_rows.append({
                "pair_idx": pi,
                "layer": al,
                "head": ah,
                "normal_delta": round(nd, 6),
                "knockout_delta": round(kd, 6),
            })
    write_csv(run_dir / "cascade_raw.csv", raw_rows)

    # Write manifest
    manifest = {
        "experiment": "knockout_cascade_n120",
        "model": args.model,
        "corpus": args.corpus,
        "n_pairs": n_pairs,
        "gate_head": {"layer": gate_layer, "head": gate_head},
        "amplifier_heads": [{"layer": l, "head": h} for l, h in amplifiers],
        "method": "zero_head_o_proj_pre_hook",
        "dla_target": "logit_diff",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults written to {run_dir}")


if __name__ == "__main__":
    main()
