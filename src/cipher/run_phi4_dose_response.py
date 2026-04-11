#!/usr/bin/env python3
"""M97: Phi-4-mini dose-response at n=120 (safety_v3 corpus).

Replaces the M80 n=32 dose-response. Generates outputs for attenuation
(harmful prompts, reduce routing) and amplification (benign prompts, increase
routing) across alpha sweeps, then saves for judge evaluation.

Usage:
  python scripts/run_phi4_dose_response_m97.py \
      --run-dir runs/phi4_dose_response
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from routing_logit_trajectory import (
    SEED, load_model_and_tokenizer, _resolve_device,
    _resolve_transformer_layers, _tokenize, _to_device,
    _default_generate_kwargs,
)
from routing_direct_logit_attribution import load_probe_direction
from safety_prompts_v3 import SAFETY_HARMFUL_V3, SAFETY_BENIGN_V3

torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_ID = "microsoft/Phi-4-mini-instruct"
CHECKPOINT = "results/m85_phi4_mini/direction.pt"
ALPHAS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]


def add_direction_hook(model, layer_idx, direction, alpha):
    """Register a hook that adds alpha * direction at a given layer."""
    layers = list(_resolve_transformer_layers(model))
    layer = layers[layer_idx]
    d = direction.clone()

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h = h + alpha * d.to(h.device, dtype=h.dtype).unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    return layer.register_forward_hook(hook_fn)


def generate_with_steering(model, tokenizer, dev, prompt, direction,
                           layer_idx, alpha, max_tokens=200):
    """Generate with optional steering."""
    enc = _tokenize(tokenizer, prompt, max_length=512, padding=False, chat_template=True)
    enc = _to_device(enc, dev)
    gen_kwargs = _default_generate_kwargs(tokenizer)
    gen_kwargs["max_new_tokens"] = max_tokens
    gen_kwargs.pop("num_return_sequences", None)

    handle = None
    if alpha != 0.0:
        handle = add_direction_hook(model, layer_idx, direction, alpha)

    try:
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        generated = out[0, enc["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    finally:
        if handle:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phi-4 dose-response at n=120 ===")
    print(f"Model: {MODEL_ID}")
    print(f"Corpus: safety_v3 ({len(SAFETY_HARMFUL_V3)} harmful, {len(SAFETY_BENIGN_V3)} benign)")

    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    dev = _resolve_device(model)
    direction = load_probe_direction(CHECKPOINT)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    intervention_layer = int(ckpt["best"]["layer_idx"])
    print(f"Intervention layer: {intervention_layer}")
    print(f"Alphas: {ALPHAS}")

    output_path = run_dir / "dose_response_outputs.csv"
    fieldnames = ["experiment", "alpha", "pair_idx", "prompt", "response"]

    total = len(ALPHAS) * (len(SAFETY_HARMFUL_V3) + len(SAFETY_BENIGN_V3))
    done = 0
    t0 = time.time()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Attenuation: reduce routing on harmful prompts (negative alpha)
        for alpha in ALPHAS:
            for pi, prompt in enumerate(SAFETY_HARMFUL_V3):
                response = generate_with_steering(
                    model, tokenizer, dev, prompt, direction,
                    intervention_layer, -alpha, max_tokens=200,
                )
                writer.writerow({
                    "experiment": "attenuation",
                    "alpha": alpha,
                    "pair_idx": pi,
                    "prompt": prompt[:200],
                    "response": response[:500],
                })
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed * 60
                    eta = (total - done) / (done / elapsed)
                    print(f"  {done}/{total} ({rate:.0f}/min, ETA {eta:.0f}s)")
            f.flush()

        # Amplification: increase routing on benign prompts (positive alpha)
        for alpha in ALPHAS:
            for pi, prompt in enumerate(SAFETY_BENIGN_V3):
                response = generate_with_steering(
                    model, tokenizer, dev, prompt, direction,
                    intervention_layer, alpha, max_tokens=200,
                )
                writer.writerow({
                    "experiment": "amplification",
                    "alpha": alpha,
                    "pair_idx": pi,
                    "prompt": prompt[:200],
                    "response": response[:500],
                })
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed * 60
                    eta = (total - done) / (done / elapsed)
                    print(f"  {done}/{total} ({rate:.0f}/min, ETA {eta:.0f}s)")
            f.flush()

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {done} outputs in {elapsed:.0f}s")

    # Write manifest
    manifest = {
        "experiment": "phi4_dose_response_n120",
        "model": MODEL_ID,
        "corpus": "safety_v3",
        "n_harmful": len(SAFETY_HARMFUL_V3),
        "n_benign": len(SAFETY_BENIGN_V3),
        "alphas": ALPHAS,
        "intervention_layer": intervention_layer,
        "checkpoint": CHECKPOINT,
        "total_outputs": done,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
