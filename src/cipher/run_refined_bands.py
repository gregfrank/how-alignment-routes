#!/usr/bin/env python3
"""M109: Refined band interchange — pro-routing only, cross-layer, and per-prompt correlation.

Tests multiple grouping strategies for multi-head interchange:
  1. Pro-routing-only layer bands (positive plaintext DLA only)
  2. Cross-layer top-K group (best K pro-routing heads regardless of layer)
  3. Counter-routing-only bands (test if they suppress routing when swapped)
  4. Per-prompt DLA collection for offline correlation clustering

Usage:
  python scripts/run_refined_bands_m109.py \
      --model microsoft/Phi-4-mini-instruct \
      --checkpoint results/m85_phi4_mini/direction.pt \
      --corpus safety_v3 \
      --diagnostic results/m106_cipher_diagnostic/phi4_mini/cipher_diagnostic_all_heads.csv \
      --n-pairs 120 \
      --run-dir results/m109_refined_bands/phi4_mini
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from routing_logit_trajectory import (
    SEED, load_model_and_tokenizer, _resolve_device,
    resolve_final_norm, resolve_prompt_pairs,
    DEFAULT_REFUSAL_STRINGS, _timestamp, write_csv, nullcontext,
)
from routing_head_dla import (
    _resolve_head_dim, _resolve_n_heads,
    compute_head_dla_records,
)
from routing_head_ablation import (
    capture_head_activations, head_interchange,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


@contextmanager
def multi_head_interchange(model, head_specs, replacement_activations, head_dim):
    with ExitStack() as stack:
        for (layer_idx, head_idx) in head_specs:
            act = replacement_activations[(layer_idx, head_idx)]
            ctx = head_interchange(model, layer_idx, head_idx, act, head_dim)
            stack.enter_context(ctx)
        yield


def compute_total_dla(model, tokenizer, final_norm, diff_direction,
                      prompt, n_heads, head_dim, intervention_context=None):
    ctx = intervention_context if intervention_context is not None else nullcontext()
    with ctx:
        rows = compute_head_dla_records(
            model=model, tokenizer=tokenizer, final_norm=final_norm,
            diff_direction=diff_direction, prompt=prompt,
            ablation_spec=None, position_mode="last_prompt",
            decision_point=None, n_heads=n_heads, head_dim=head_dim,
        )
    return sum(row["mlp_contribution"] + row["attn_total"] for row in rows)


def run_group_interchange(model, tokenizer, final_norm, diff_direction,
                          harmful_prompt, benign_prompt,
                          group_heads, n_heads, head_dim):
    """Run multi-head interchange for a group of heads."""
    harm_acts = {}
    benign_acts = {}
    for (layer, head) in group_heads:
        harm_acts[(layer, head)] = capture_head_activations(
            model, tokenizer, harmful_prompt, layer, head, head_dim)
        benign_acts[(layer, head)] = capture_head_activations(
            model, tokenizer, benign_prompt, layer, head, head_dim)

    baseline_harm = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        harmful_prompt, n_heads, head_dim)
    baseline_benign = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        benign_prompt, n_heads, head_dim)

    ctx_nec = multi_head_interchange(model, group_heads, benign_acts, head_dim)
    swapped_harm = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        harmful_prompt, n_heads, head_dim, intervention_context=ctx_nec)

    ctx_suf = multi_head_interchange(model, group_heads, harm_acts, head_dim)
    swapped_benign = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        benign_prompt, n_heads, head_dim, intervention_context=ctx_suf)

    return {
        "necessity": baseline_harm - swapped_harm,
        "sufficiency": swapped_benign - baseline_benign,
    }


def collect_per_prompt_dla(model, tokenizer, final_norm, diff_direction,
                           pairs, target_heads, n_heads, head_dim):
    """Collect per-prompt DLA for target heads under plaintext and cipher conditions.
    Returns dict of (layer,head) -> [dla_pair_0, dla_pair_1, ...] for plaintext harmful."""
    from cipher.run_cipher_interchange import CIPHER_PREFIX, encode_latin

    per_head_plain = {h: [] for h in target_heads}
    per_head_cipher = {h: [] for h in target_heads}
    per_head_benign = {h: [] for h in target_heads}

    for i, pair in enumerate(pairs):
        harmful = pair.ccp_prompt
        benign = pair.control_prompt
        cipher = CIPHER_PREFIX + encode_latin(harmful)

        for condition, prompt, store in [
            ("plain", harmful, per_head_plain),
            ("cipher", cipher, per_head_cipher),
            ("benign", benign, per_head_benign),
        ]:
            rows = compute_head_dla_records(
                model=model, tokenizer=tokenizer, final_norm=final_norm,
                diff_direction=diff_direction, prompt=prompt,
                ablation_spec=None, position_mode="last_prompt",
                decision_point=None, n_heads=n_heads, head_dim=head_dim,
            )
            for (layer, head) in target_heads:
                store[(layer, head)].append(rows[layer][f"head_{head}"])

        print(f"  per-prompt DLA: pair {i+1}/{len(pairs)}", end="\r", flush=True)

    print()
    return per_head_plain, per_head_cipher, per_head_benign


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", default="safety_v3")
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load diagnostic data and build groups
    diag_rows = list(csv.DictReader(open(args.diagnostic)))
    pro_heads = []  # (layer, head, routing, sens)
    counter_heads = []
    for r in diag_rows:
        sens = float(r['cipher_sensitivity'])
        plain = float(r['mean_plain_dla'])
        routing = float(r['plain_minus_benign'])
        if sens > 0.1 and abs(plain) >= 0.05:
            entry = (int(r['layer']), int(r['head']), routing, sens)
            if plain > 0:
                pro_heads.append(entry)
            else:
                counter_heads.append(entry)

    pro_heads.sort(key=lambda h: -h[2])  # sort by routing
    counter_heads.sort(key=lambda h: h[2])  # sort by most negative

    # Build groups to test
    groups = {}

    # 1. Pro-routing layer bands (top 3 layers by pro-routing count)
    by_layer_pro = defaultdict(list)
    for h in pro_heads:
        by_layer_pro[h[0]].append((h[0], h[1]))
    top_layers = sorted(by_layer_pro.keys(),
                        key=lambda l: sum(h[2] for h in pro_heads if h[0] == l), reverse=True)
    for i, layer in enumerate(top_layers[:3]):
        groups[f"pro_L{layer}"] = by_layer_pro[layer]

    # 2. Cross-layer top-K pro-routing groups
    for k in [3, 6, 10]:
        groups[f"cross_top{k}"] = [(h[0], h[1]) for h in pro_heads[:k]]

    # 3. Counter-routing layer bands (top 2 layers)
    by_layer_counter = defaultdict(list)
    for h in counter_heads:
        by_layer_counter[h[0]].append((h[0], h[1]))
    counter_layers = sorted(by_layer_counter.keys(),
                            key=lambda l: len(by_layer_counter[l]), reverse=True)
    for layer in counter_layers[:2]:
        if len(by_layer_counter[layer]) >= 2:
            groups[f"counter_L{layer}"] = by_layer_counter[layer]

    # 4. Single best head (for comparison)
    groups["single_best"] = [(pro_heads[0][0], pro_heads[0][1])]

    print(f"=== M109: Refined Band Interchange ===")
    print(f"Model: {args.model}")
    print(f"Pro-routing heads: {len(pro_heads)}, Counter-routing: {len(counter_heads)}")
    print(f"\nGroups to test:")
    for name, heads in sorted(groups.items()):
        layers = sorted(set(h[0] for h in heads))
        print(f"  {name:20s}: {len(heads)} heads at layers {layers}")

    # Load model
    print(f"\n[1/5] Loading model... ({_timestamp()})")
    model, tokenizer = load_model_and_tokenizer(args.model)
    n_heads = _resolve_n_heads(model)
    head_dim = _resolve_head_dim(model)
    final_norm = resolve_final_norm(model)

    # Load direction
    print(f"[2/5] Loading direction... ({_timestamp()})")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "bank" in ckpt and "best" in ckpt:
        direction = ckpt["bank"][ckpt["best"]["candidate_idx"]]["v_clean"].float().cpu()
    else:
        direction = ckpt["direction"].float().cpu()

    # Load prompts
    print(f"[3/5] Loading prompts... ({_timestamp()})")
    pairs = resolve_prompt_pairs(args.corpus, args.n_pairs)
    print(f"  {len(pairs)} pairs")

    # Run group interchange
    print(f"[4/5] Running group interchange ({len(groups)} groups)... ({_timestamp()})")
    all_results = {}

    for group_name, group_heads in sorted(groups.items()):
        print(f"\n  {group_name} ({len(group_heads)} heads):")
        necs = []
        sufs = []
        for i, pair in enumerate(pairs):
            scores = run_group_interchange(
                model, tokenizer, final_norm, direction,
                pair.ccp_prompt, pair.control_prompt,
                group_heads, n_heads, head_dim,
            )
            necs.append(abs(scores['necessity']))
            sufs.append(abs(scores['sufficiency']))
            print(f"    pair {i+1}/{len(pairs)}", end="\r", flush=True)

        mean_nec = float(np.mean(necs))
        mean_suf = float(np.mean(sufs))
        print(f"    |necessity|={mean_nec:.4f}, |sufficiency|={mean_suf:.4f}")

        all_results[group_name] = {
            "heads": [f"L{h[0]}.H{h[1]}" for h in group_heads],
            "n_heads": len(group_heads),
            "layers": sorted(set(h[0] for h in group_heads)),
            "mean_abs_necessity": mean_nec,
            "mean_abs_sufficiency": mean_suf,
        }

    # Compute ratios vs single best
    single_nec = all_results["single_best"]["mean_abs_necessity"]
    for name, data in all_results.items():
        data["ratio_vs_single"] = data["mean_abs_necessity"] / single_nec if single_nec > 0 else 0

    # Collect per-prompt DLA for correlation analysis
    print(f"\n[5/5] Collecting per-prompt DLA for correlation clustering... ({_timestamp()})")
    # Use top 20 pro + top 10 counter heads
    corr_heads = [(h[0], h[1]) for h in pro_heads[:20]] + [(h[0], h[1]) for h in counter_heads[:10]]
    per_prompt_plain, per_prompt_cipher, per_prompt_benign = collect_per_prompt_dla(
        model, tokenizer, final_norm, direction, pairs, corr_heads, n_heads, head_dim)

    # Compute correlation matrix
    head_labels = [f"L{h[0]}.H{h[1]}" for h in corr_heads]
    plain_matrix = np.array([per_prompt_plain[h] for h in corr_heads])  # [n_heads, n_pairs]
    corr = np.corrcoef(plain_matrix)

    # Save correlation data
    corr_data = []
    for i, h1 in enumerate(head_labels):
        for j, h2 in enumerate(head_labels):
            if i < j:
                corr_data.append({"head1": h1, "head2": h2, "correlation": float(corr[i, j])})
    corr_data.sort(key=lambda x: -abs(x["correlation"]))
    write_csv(run_dir / "head_correlation_pairs.csv", corr_data)

    # Find clusters: heads with >0.5 positive correlation
    print(f"\nHighly correlated head pairs (r > 0.5):")
    for d in corr_data[:20]:
        if d["correlation"] > 0.5:
            print(f"  {d['head1']:10s} ↔ {d['head2']:10s}  r={d['correlation']:.3f}")

    print(f"\nHighly anti-correlated pairs (r < -0.5):")
    for d in reversed(corr_data):
        if d["correlation"] < -0.5:
            print(f"  {d['head1']:10s} ↔ {d['head2']:10s}  r={d['correlation']:.3f}")

    # Save per-prompt DLA for further offline analysis
    per_prompt_rows = []
    for h in corr_heads:
        label = f"L{h[0]}.H{h[1]}"
        for i in range(len(pairs)):
            per_prompt_rows.append({
                "head": label, "pair_idx": i,
                "plain_dla": per_prompt_plain[h][i],
                "cipher_dla": per_prompt_cipher[h][i],
                "benign_dla": per_prompt_benign[h][i],
            })
    write_csv(run_dir / "per_prompt_head_dla.csv", per_prompt_rows)

    # Write summary
    summary = {
        "model": args.model,
        "corpus": args.corpus,
        "n_pairs": len(pairs),
        "n_pro_heads": len(pro_heads),
        "n_counter_heads": len(counter_heads),
        "groups": all_results,
        "timestamp": _timestamp(),
    }
    (run_dir / "refined_bands_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Group':>22s}  {'Heads':>5s}  {'Layers':>15s}  {'|Nec|':>8s}  {'|Suf|':>8s}  {'Ratio':>6s}")
    print(f"{'-'*70}")
    for name in sorted(all_results.keys()):
        d = all_results[name]
        layers_str = ",".join(str(l) for l in d["layers"])
        print(f"{name:>22s}  {d['n_heads']:>5d}  {layers_str:>15s}  "
              f"{d['mean_abs_necessity']:>8.4f}  {d['mean_abs_sufficiency']:>8.4f}  "
              f"{d['ratio_vs_single']:>6.2f}x")
    print(f"{'='*70}")
    print(f"\nResults in {run_dir}")


if __name__ == "__main__":
    main()
