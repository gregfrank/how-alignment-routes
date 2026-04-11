#!/usr/bin/env python3
"""Compute a mean-difference probe direction from paired prompts.

Quick alternative to ridge regression for models without a pre-computed
direction checkpoint. Uses the mean hidden-state difference between
harmful and control prompts at the middle layer.
"""
from __future__ import annotations
import argparse, sys, pathlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import torch
from routing_logit_trajectory import (
    load_model_and_tokenizer, resolve_prompt_pairs,
    _resolve_device, _tokenize,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="safety_v3")
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    dev = _resolve_device(model)
    pairs = resolve_prompt_pairs(args.corpus, args.n_pairs)
    print(f"Computing direction from {len(pairs)} pairs...")

    diffs = []
    for i, pair in enumerate(pairs):
        enc_h = _tokenize(tokenizer, pair.ccp_prompt, max_length=256, padding=False, chat_template=True)
        enc_c = _tokenize(tokenizer, pair.control_prompt, max_length=256, padding=False, chat_template=True)
        enc_h = {k: v.to(dev) for k, v in enc_h.items()}
        enc_c = {k: v.to(dev) for k, v in enc_c.items()}
        with torch.no_grad():
            h_out = model(**enc_h, output_hidden_states=True, use_cache=False)
            c_out = model(**enc_c, output_hidden_states=True, use_cache=False)
        n = len(h_out.hidden_states)
        mid = n // 2
        diffs.append(
            h_out.hidden_states[mid][0, -1, :].float().cpu()
            - c_out.hidden_states[mid][0, -1, :].float().cpu()
        )
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)}")

    direction = torch.stack(diffs).mean(dim=0)
    direction = direction / direction.norm()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_id": args.model,
        "bank": [{"layer_idx": mid, "v_clean": direction}],
        "best": {"candidate_idx": 0, "layer_idx": mid},
        "baseline": {},
        "version": "mean_diff",
    }, out)
    print(f"Saved direction to {out} (layer {mid}, shape {direction.shape})")

if __name__ == "__main__":
    main()
