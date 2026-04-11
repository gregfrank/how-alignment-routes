#!/usr/bin/env python3
"""M98: Intermediate-layer DLA — measure each head's contribution to the
residual stream at the amplifier layer, not the final output.

If the gate ranks highly at L22 (amplifier layer) but rank 292 at the output,
this confirms the gate writes a routing signal that amplifiers read and boost.

No GPU needed if we already have the per-head o_proj outputs saved.
Since we don't, this requires a forward pass.

Usage:
  python scripts/run_intermediate_dla_m98.py \
      --model Qwen/Qwen3-8B --corpus v2 --gate-layer 17 \
      --measure-at-layers 18,20,22,25,30,35 \
      --checkpoint runs/qwen3_8b_ablation/checkpoint.pt \
      --run-dir results/m98_intermediate_dla/qwen3_8b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from routing_logit_trajectory import (
    SEED, load_model_and_tokenizer, _resolve_device,
    _resolve_transformer_layers, _tokenize, _to_device,
    resolve_prompt_pairs, write_csv,
)
from routing_direct_logit_attribution import (
    load_probe_direction, _resolve_layer_components,
)
from routing_head_dla import (
    _resolve_head_dim, _resolve_n_heads, _resolve_o_proj,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_head_contributions_at_layer(
    model, tokenizer, prompt, measure_layer, n_heads, head_dim,
):
    """Compute each head's contribution to the residual stream, evaluated
    at a specific intermediate layer (not the final output).

    Returns dict mapping (layer, head) -> contribution vector [d_model].
    Only includes heads at layers BEFORE measure_layer.
    """
    dev = _resolve_device(model)
    enc = _tokenize(tokenizer, prompt, max_length=512, padding=False, chat_template=True)
    enc = _to_device(enc, dev)

    layers = list(_resolve_transformer_layers(model))

    # Hook every o_proj to capture per-head outputs at last token
    captures = {}
    weights = {}
    handles = []

    for li in range(min(measure_layer, len(layers))):
        layer = layers[li]
        attn_mod, _ = _resolve_layer_components(layer)
        o_proj = _resolve_o_proj(attn_mod)

        weights[li] = o_proj.weight.detach().float().cpu()

        def make_hook(idx):
            def hook(_module, inputs, _output):
                inp = inputs[0] if isinstance(inputs, tuple) else inputs
                captures[idx] = inp[0, -1, :].detach().float().cpu()
            return hook

        handles.append(o_proj.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**enc, output_hidden_states=False, use_cache=False)

    for h in handles:
        h.remove()

    # Decompose: per-head contribution = W_o[:, h_start:h_end] @ pre_oproj[h_start:h_end]
    head_contribs = {}
    for li in captures:
        pre_oproj = captures[li]
        w = weights[li]
        for h in range(n_heads):
            start = h * head_dim
            end = start + head_dim
            contrib = w[:, start:end] @ pre_oproj[start:end]  # [d_model]
            head_contribs[(li, h)] = contrib

    return head_contribs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--gate-layer", type=int, required=True)
    parser.add_argument("--measure-at-layers", required=True,
                        help="Comma-separated layer indices to measure at")
    parser.add_argument("--checkpoint", required=True,
                        help="Probe direction checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit-pairs", type=int, default=30)
    args = parser.parse_args()

    measure_layers = [int(x) for x in args.measure_at_layers.split(",")]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model)
    dev = _resolve_device(model)
    head_dim = _resolve_head_dim(model)
    n_heads = _resolve_n_heads(model)
    direction = load_probe_direction(args.checkpoint)

    pairs = resolve_prompt_pairs(args.corpus, args.limit_pairs)
    print(f"=== Intermediate DLA: {args.model} ===")
    print(f"Measuring at layers: {measure_layers}")
    print(f"Pairs: {len(pairs)}")

    # For each measurement layer, compute head ranking by DLA
    results = {ml: {} for ml in measure_layers}

    for pi, pair in enumerate(pairs):
        for ml in measure_layers:
            # Sensitive prompt
            s_contribs = compute_head_contributions_at_layer(
                model, tokenizer, pair.ccp_prompt, ml, n_heads, head_dim,
            )
            # Control prompt
            c_contribs = compute_head_contributions_at_layer(
                model, tokenizer, pair.control_prompt, ml, n_heads, head_dim,
            )
            # Delta projected onto probe direction
            for (li, h), s_vec in s_contribs.items():
                c_vec = c_contribs.get((li, h), torch.zeros_like(s_vec))
                delta_vec = s_vec - c_vec
                dla = float(torch.dot(delta_vec, direction))
                key = (li, h)
                if key not in results[ml]:
                    results[ml][key] = []
                results[ml][key].append(dla)

        print(f"  Pair {pi+1}/{len(pairs)}", end="\r", flush=True)

    print(f"\nDone")

    # For each measurement layer, rank heads and find gate
    all_rows = []
    print(f"\n{'Meas layer':>10} {'Gate rank':>10} {'Gate DLA':>10} {'Top-1':>12} {'Top-1 DLA':>10}")
    print("-" * 56)

    for ml in measure_layers:
        head_means = {}
        for (li, h), vals in results[ml].items():
            head_means[(li, h)] = float(np.mean(vals))

        ranked = sorted(head_means.items(), key=lambda x: abs(x[1]), reverse=True)
        total_abs = sum(abs(v) for _, v in ranked)

        # Find gate
        gate_rank = None
        gate_dla = 0
        for i, ((li, h), v) in enumerate(ranked):
            if li == args.gate_layer:
                # Find the gate head (highest-ranked head at gate layer)
                if gate_rank is None:
                    gate_rank = i + 1
                    gate_dla = v

        top1_head = ranked[0][0]
        top1_dla = ranked[0][1]

        print(f"{ml:>10} {gate_rank:>10} {gate_dla:>+10.4f} L{top1_head[0]}.H{top1_head[1]:>12} {top1_dla:>+10.4f}")

        for i, ((li, h), v) in enumerate(ranked[:20]):
            is_gate_layer = "GATE_LAYER" if li == args.gate_layer else ""
            all_rows.append({
                "measure_at_layer": ml,
                "rank": i + 1,
                "head_layer": li,
                "head_idx": h,
                "mean_dla_delta": round(v, 6),
                "pct_of_total": round(abs(v) / total_abs * 100, 2) if total_abs > 0 else 0,
                "note": is_gate_layer,
            })

    with open(run_dir / "intermediate_dla_rankings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    summary = {
        "model": args.model,
        "gate_layer": args.gate_layer,
        "measure_layers": measure_layers,
        "n_pairs": len(pairs),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run_dir / "intermediate_dla_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
