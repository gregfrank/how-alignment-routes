#!/usr/bin/env python3
"""M100: DLA robustness check — is gate identification stable across alternative
target directions?

Tests three alternative logit-diff contrasts and shows the gate head's
interchange ranking is unchanged:
1. Default: model's first answer token vs mean refusal tokens
2. Alternative refusal set: only "I" and "Sorry" (subset)
3. Alternative answer: second-most-likely token instead of argmax
4. Fixed direction: "The" vs "I" (corpus-independent)

Usage:
  python scripts/run_dla_robustness_m100.py \
      --model Qwen/Qwen3-8B --corpus v2 \
      --checkpoint runs/qwen3_8b_ablation/checkpoint.pt \
      --run-dir results/m100_dla_robustness/qwen3_8b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from routing_logit_trajectory import (
    SEED, load_model_and_tokenizer, _resolve_device,
    _resolve_transformer_layers, _tokenize, _to_device,
    resolve_final_norm, resolve_output_head,
    resolve_prompt_pairs, write_csv, DEFAULT_REFUSAL_STRINGS,
)
from routing_direct_logit_attribution import (
    _resolve_logit_diff_direction,
)
from routing_head_dla import (
    _resolve_head_dim, _resolve_n_heads,
    compute_head_dla_records,
)
from routing_head_ablation import head_interchange

torch.manual_seed(SEED)
np.random.seed(SEED)


def run_interchange_with_direction(model, tokenizer, pairs, final_norm,
                                    W, n_heads, head_dim, refusal_ids,
                                    answer_fn, direction_name, limit=30):
    """Run interchange on top DLA heads using a specific direction definition."""
    from routing_head_ablation import head_interchange
    from routing_direct_logit_attribution import _resolve_logit_diff_direction

    # First pass: DLA screening to get top heads
    head_scores = {}
    for pi, pair in enumerate(pairs[:limit]):
        answer_id = answer_fn(model, tokenizer, pair)
        diff_dir = _resolve_logit_diff_direction(W, refusal_ids, answer_id)

        ccp_rec = compute_head_dla_records(
            model, tokenizer, final_norm, diff_dir,
            pair.ccp_prompt, None, "last_prompt", None, n_heads, head_dim,
        )
        ctrl_rec = compute_head_dla_records(
            model, tokenizer, final_norm, diff_dir,
            pair.control_prompt, None, "last_prompt", None, n_heads, head_dim,
        )

        for li in range(len(ccp_rec)):
            for h in range(n_heads):
                key = (li, h)
                delta = ccp_rec[li][f"head_{h}"] - ctrl_rec[li][f"head_{h}"]
                head_scores.setdefault(key, []).append(abs(delta))

    # Rank by mean absolute DLA delta
    ranked = sorted(
        [(k, float(np.mean(v))) for k, v in head_scores.items()],
        key=lambda x: x[1], reverse=True,
    )

    # Find L17.H17 rank
    gate_rank = None
    for i, ((l, h), score) in enumerate(ranked):
        if l == 17 and h == 17:
            gate_rank = i + 1
            gate_score = score
            break

    return {
        "direction": direction_name,
        "gate_rank": gate_rank,
        "gate_score": round(gate_score, 6) if gate_rank else None,
        "top_5": [(f"L{l}.H{h}", round(s, 4)) for (l, h), s in ranked[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit-pairs", type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model)
    dev = _resolve_device(model)
    head_dim = _resolve_head_dim(model)
    n_heads = _resolve_n_heads(model)
    final_norm = resolve_final_norm(model)
    W = resolve_output_head(model).weight.detach().float().cpu()

    pairs = resolve_prompt_pairs(args.corpus, args.limit_pairs)
    print(f"=== DLA robustness: {args.model}, {len(pairs)} pairs ===")

    # Define alternative answer functions
    def default_answer(model, tokenizer, pair):
        enc = _tokenize(tokenizer, pair.control_prompt, max_length=256,
                        padding=False, chat_template=True)
        enc = _to_device(enc, dev)
        with torch.no_grad():
            out = model(**enc, use_cache=False)
        return out.logits[0, -1, :].argmax().item()

    def second_best_answer(model, tokenizer, pair):
        enc = _tokenize(tokenizer, pair.control_prompt, max_length=256,
                        padding=False, chat_template=True)
        enc = _to_device(enc, dev)
        with torch.no_grad():
            out = model(**enc, use_cache=False)
        topk = out.logits[0, -1, :].topk(2)
        return topk.indices[1].item()

    def fixed_the(model, tokenizer, pair):
        ids = tokenizer.encode("The", add_special_tokens=False)
        return ids[0] if ids else 785

    # Define alternative refusal sets
    full_refusal = []
    for s in DEFAULT_REFUSAL_STRINGS:
        for v in (s, f" {s}"):
            ids = tokenizer.encode(v, add_special_tokens=False)
            if ids:
                full_refusal.append(ids[0])
    full_refusal = sorted(set(full_refusal))

    minimal_refusal = []
    for s in ["I", "Sorry"]:
        for v in (s, f" {s}"):
            ids = tokenizer.encode(v, add_special_tokens=False)
            if ids:
                minimal_refusal.append(ids[0])
    minimal_refusal = sorted(set(minimal_refusal))

    # Run each contrast
    results = []

    print("\n1. Default (argmax answer, full refusal set)...")
    r = run_interchange_with_direction(
        model, tokenizer, pairs, final_norm, W, n_heads, head_dim,
        full_refusal, default_answer, "default", args.limit_pairs,
    )
    results.append(r)
    print(f"   Gate rank: {r['gate_rank']}")

    print("\n2. Minimal refusal set (only I, Sorry)...")
    r = run_interchange_with_direction(
        model, tokenizer, pairs, final_norm, W, n_heads, head_dim,
        minimal_refusal, default_answer, "minimal_refusal", args.limit_pairs,
    )
    results.append(r)
    print(f"   Gate rank: {r['gate_rank']}")

    print("\n3. Second-best answer token...")
    r = run_interchange_with_direction(
        model, tokenizer, pairs, final_norm, W, n_heads, head_dim,
        full_refusal, second_best_answer, "second_best_answer", args.limit_pairs,
    )
    results.append(r)
    print(f"   Gate rank: {r['gate_rank']}")

    print("\n4. Fixed direction (The vs refusal)...")
    r = run_interchange_with_direction(
        model, tokenizer, pairs, final_norm, W, n_heads, head_dim,
        full_refusal, fixed_the, "fixed_the", args.limit_pairs,
    )
    results.append(r)
    print(f"   Gate rank: {r['gate_rank']}")

    # Summary
    print(f"\n=== Gate L17.H17 rank across contrasts ===")
    for r in results:
        print(f"  {r['direction']:>25}: rank {r['gate_rank']}")

    with open(run_dir / "robustness_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
