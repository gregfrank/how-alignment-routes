#!/usr/bin/env python3
"""M97b: Knockout cascade null distribution.

Knocks out random non-gate heads and measures downstream amplifier suppression.
If the gate is special, its knockout should produce much stronger suppression
than random heads at similar depths.

Tests 10 random heads per model, same amplifier set as the real cascade.

Usage:
  python scripts/run_knockout_null_m97.py \
      --model Qwen/Qwen3-8B --corpus v2 \
      --gate 17.17 --amplifiers 22.7,23.3,22.4,22.5,22.6,23.2 \
      --null-heads 15.3,16.8,14.12,18.5,19.10,16.2,15.20,14.7,18.15,13.9 \
      --run-dir results/m97_knockout_null/qwen3_8b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from routing_logit_trajectory import (
    DEFAULT_REFUSAL_STRINGS, SEED, _resolve_device,
    _resolve_transformer_layers, build_answer_bundle, build_refusal_bundle,
    load_model_and_tokenizer, resolve_final_norm, resolve_output_head,
    resolve_prompt_pairs, write_csv,
)
from routing_direct_logit_attribution import (
    _resolve_layer_components, _resolve_logit_diff_direction,
)
from routing_head_dla import (
    _resolve_head_dim, _resolve_n_heads, _resolve_o_proj,
    compute_head_dla_records,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


@contextmanager
def zero_head_hook(model, layer_idx, head_idx, head_dim):
    layer = _resolve_transformer_layers(model)[layer_idx]
    attn_mod, _ = _resolve_layer_components(layer)
    o_proj = _resolve_o_proj(attn_mod)
    start = head_idx * head_dim
    end = start + head_dim

    def pre_hook(_module, inputs):
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


def parse_head(s):
    parts = s.strip().split(".")
    return int(parts[0]), int(parts[1])


def run_cascade_for_head(model, tokenizer, final_norm, W, refusal_ids,
                          answer_bundle, pairs, knockout_layer, knockout_head,
                          amplifiers, head_dim, n_heads, limit=20):
    """Run a mini knockout cascade for a single knocked-out head."""
    results = {h: [] for h in amplifiers}

    for pi, pair in enumerate(pairs[:limit]):
        dp = answer_bundle[pair.pair_idx]
        diff_direction = _resolve_logit_diff_direction(W, refusal_ids, dp.meaningful_token_id)

        # Normal DLA
        normal_ccp = compute_head_dla_records(
            model, tokenizer, final_norm, diff_direction,
            pair.ccp_prompt, None, "last_prompt", None, n_heads, head_dim,
        )
        normal_ctrl = compute_head_dla_records(
            model, tokenizer, final_norm, diff_direction,
            pair.control_prompt, None, "last_prompt", None, n_heads, head_dim,
        )

        # Knockout DLA
        with zero_head_hook(model, knockout_layer, knockout_head, head_dim):
            ko_ccp = compute_head_dla_records(
                model, tokenizer, final_norm, diff_direction,
                pair.ccp_prompt, None, "last_prompt", None, n_heads, head_dim,
            )
            ko_ctrl = compute_head_dla_records(
                model, tokenizer, final_norm, diff_direction,
                pair.control_prompt, None, "last_prompt", None, n_heads, head_dim,
            )

        for (al, ah) in amplifiers:
            key = f"head_{ah}"
            normal_delta = normal_ccp[al][key] - normal_ctrl[al][key]
            ko_delta = ko_ccp[al][key] - ko_ctrl[al][key]
            results[(al, ah)].append((normal_delta, ko_delta))

    # Aggregate
    head_changes = []
    for (al, ah) in amplifiers:
        deltas = results[(al, ah)]
        mn = float(np.mean([d[0] for d in deltas]))
        mk = float(np.mean([d[1] for d in deltas]))
        change = (mk - mn) / abs(mn) * 100 if abs(mn) > 1e-8 else 0.0
        head_changes.append(change)

    return float(np.mean([abs(c) for c in head_changes]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--amplifiers", required=True)
    parser.add_argument("--null-heads", required=True,
                        help="Comma-separated L.H specs for null distribution")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit-pairs", type=int, default=20,
                        help="Pairs per head (fewer for speed)")
    args = parser.parse_args()

    gate_layer, gate_head = parse_head(args.gate)
    amplifiers = [parse_head(h) for h in args.amplifiers.split(",")]
    null_heads = [parse_head(h) for h in args.null_heads.split(",")]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Knockout null distribution ===")
    print(f"Gate: L{gate_layer}.H{gate_head}")
    print(f"Null heads: {len(null_heads)}")
    print(f"Pairs per head: {args.limit_pairs}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    dev = _resolve_device(model)
    head_dim = _resolve_head_dim(model)
    n_heads = _resolve_n_heads(model)
    final_norm = resolve_final_norm(model)
    W = resolve_output_head(model).weight.detach().float().cpu()

    pairs = resolve_prompt_pairs(args.corpus, None)
    refusal_bundle = build_refusal_bundle(tokenizer, DEFAULT_REFUSAL_STRINGS)
    refusal_ids = refusal_bundle["token_ids"]
    answer_bundle, _, _ = build_answer_bundle(model, tokenizer, pairs, None, 5)

    # Run gate knockout
    print(f"\nGate L{gate_layer}.H{gate_head}...")
    gate_effect = run_cascade_for_head(
        model, tokenizer, final_norm, W, refusal_ids, answer_bundle,
        pairs, gate_layer, gate_head, amplifiers, head_dim, n_heads,
        limit=args.limit_pairs,
    )
    print(f"  Gate mean |change|: {gate_effect:.1f}%")

    # Run null heads
    null_effects = []
    for i, (nl, nh) in enumerate(null_heads):
        print(f"Null L{nl}.H{nh} ({i+1}/{len(null_heads)})...")
        effect = run_cascade_for_head(
            model, tokenizer, final_norm, W, refusal_ids, answer_bundle,
            pairs, nl, nh, amplifiers, head_dim, n_heads,
            limit=args.limit_pairs,
        )
        null_effects.append(effect)
        print(f"  Mean |change|: {effect:.1f}%")

    # Summary
    null_mean = float(np.mean(null_effects))
    null_std = float(np.std(null_effects))
    null_max = float(np.max(null_effects))

    print(f"\n=== Summary ===")
    print(f"Gate effect: {gate_effect:.1f}%")
    print(f"Null mean:   {null_mean:.1f}% +/- {null_std:.1f}%")
    print(f"Null max:    {null_max:.1f}%")
    print(f"Gate / null_mean: {gate_effect/null_mean:.1f}x" if null_mean > 0 else "Null mean = 0")

    rows = [{"head": f"GATE_L{gate_layer}.H{gate_head}", "mean_abs_change_pct": round(gate_effect, 2), "role": "gate"}]
    for (nl, nh), eff in zip(null_heads, null_effects):
        rows.append({"head": f"L{nl}.H{nh}", "mean_abs_change_pct": round(eff, 2), "role": "null"})

    write_csv(run_dir / "knockout_null_distribution.csv", rows)

    summary = {
        "model": args.model,
        "gate": f"L{gate_layer}.H{gate_head}",
        "gate_effect": round(gate_effect, 2),
        "null_mean": round(null_mean, 2),
        "null_std": round(null_std, 2),
        "null_max": round(null_max, 2),
        "n_null_heads": len(null_heads),
        "limit_pairs": args.limit_pairs,
    }
    with open(run_dir / "knockout_null_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
