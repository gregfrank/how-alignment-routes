#!/usr/bin/env python3
"""M83: Expanded dose-response on v2 political corpus (n=120) for Qwen3-8B.

Runs bidirectional alpha sweeps:
- Attenuation on all v2 politically sensitive prompts (n=120)
- Amplification on the matched control prompts (n=120)

At each alpha, generates the model's response and records it for later
judge evaluation. Also records teacher-forced NLL for quantitative analysis.

The judge evaluation step runs separately after generation.
"""

from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import csv
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List

from routing_logit_trajectory import (
    SEED, load_model_and_tokenizer, _resolve_device,
    _resolve_transformer_layers, _tokenize, _to_device,
    _default_generate_kwargs, write_csv,
)
from routing_direct_logit_attribution import load_probe_direction
from political_prompts_v2 import POLITICAL_SENSITIVE as POLITICAL_PROMPTS_V2, POLITICAL_CONTROL as CONTROL_PROMPTS_V2

torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("runs/dose_response")
QWEN_MODEL = "Qwen/Qwen3-8B"
QWEN_CKPT = "runs/qwen3_8b_ablation/checkpoint.pt"
ALPHAS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
MAX_NEW_TOKENS = 200


def add_direction_hook(model, layer_idx: int, direction: torch.Tensor, alpha: float):
    """Hook that adds alpha * direction to the residual stream at the specified layer."""
    layers = _resolve_transformer_layers(model)
    layer = layers[layer_idx]
    dev = next(model.parameters()).device

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # Match direction dtype to hidden states
        d = direction.to(h.device).to(h.dtype)
        # Add direction at all positions (broadcasting)
        h = h + alpha * d.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = layer.register_forward_hook(hook_fn)
    return handle


def generate_with_intervention(model, tokenizer, prompt, direction, layer_idx, alpha, max_tokens=200):
    """Generate text with a steering direction added at the specified layer."""
    handle = None
    if alpha != 0:
        handle = add_direction_hook(model, layer_idx, direction, alpha)

    try:
        enc = _tokenize(tokenizer, prompt, max_length=512, padding=False, chat_template=True)
        enc = _to_device(enc, next(model.parameters()).device)
        gen_kwargs = _default_generate_kwargs(tokenizer)
        gen_kwargs["max_new_tokens"] = max_tokens
        gen_kwargs.pop("num_return_sequences", None)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        # Decode only the generated tokens
        generated_ids = out[0, enc["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()
    finally:
        if handle is not None:
            handle.remove()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and direction...")
    direction = load_probe_direction(QWEN_CKPT)
    model, tokenizer = load_model_and_tokenizer(QWEN_MODEL)
    dev = _resolve_device(model)

    # Determine intervention layer from checkpoint
    ckpt = torch.load(QWEN_CKPT, map_location="cpu", weights_only=False)
    intervention_layer = int(ckpt["best"]["layer_idx"])
    print(f"Intervention layer: {intervention_layer}")

    # Build prompt lists
    political = POLITICAL_PROMPTS_V2  # sensitive prompts
    controls = CONTROL_PROMPTS_V2    # matched controls

    n_political = len(political)
    n_controls = len(controls)
    print(f"Political prompts: {n_political}, Control prompts: {n_controls}")

    # ===== ATTENUATION: subtract direction from political prompts =====
    print(f"\n=== Attenuation: political prompts (n={n_political}) ===")
    attenuation_rows = []

    for alpha in ALPHAS:
        print(f"\n  Alpha = {alpha} (attenuation)")
        for i, prompt in enumerate(political):
            # Attenuation = subtract direction (negative alpha)
            response = generate_with_intervention(
                model, tokenizer, prompt, direction, intervention_layer,
                -alpha, MAX_NEW_TOKENS
            )
            attenuation_rows.append({
                "experiment": "attenuation",
                "alpha": alpha,
                "pair_idx": i,
                "prompt": prompt[:200],
                "response": response[:500],
            })
            print(f"    {i+1}/{n_political}", end="\r", flush=True)
        print(f"    alpha={alpha}: done ({n_political} prompts)          ")

    # ===== AMPLIFICATION: add direction to control prompts =====
    print(f"\n=== Amplification: control prompts (n={n_controls}) ===")
    amplification_rows = []

    for alpha in ALPHAS:
        print(f"\n  Alpha = {alpha} (amplification)")
        for i, prompt in enumerate(controls):
            response = generate_with_intervention(
                model, tokenizer, prompt, direction, intervention_layer,
                alpha, MAX_NEW_TOKENS
            )
            amplification_rows.append({
                "experiment": "amplification",
                "alpha": alpha,
                "pair_idx": i,
                "prompt": prompt[:200],
                "response": response[:500],
            })
            print(f"    {i+1}/{n_controls}", end="\r", flush=True)
        print(f"    alpha={alpha}: done ({n_controls} prompts)          ")

    # Write outputs
    all_rows = attenuation_rows + amplification_rows
    write_csv(RESULTS_DIR / "dose_response_outputs.csv", all_rows)

    with open(RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump({
            "experiment": "expanded_dose_response",
            "model": QWEN_MODEL,
            "corpus": "political_v2",
            "intervention_layer": intervention_layer,
            "alphas": ALPHAS,
            "n_political": n_political,
            "n_controls": n_controls,
            "max_new_tokens": MAX_NEW_TOKENS,
            "total_outputs": len(all_rows),
        }, f, indent=2)

    print(f"\nTotal outputs generated: {len(all_rows)}")
    print(f"  Attenuation: {len(attenuation_rows)} ({n_political} prompts x {len(ALPHAS)} alphas)")
    print(f"  Amplification: {len(amplification_rows)} ({n_controls} prompts x {len(ALPHAS)} alphas)")
    print(f"Wrote results to {RESULTS_DIR}")
    print("\nNext step: run judge evaluation on these outputs")


if __name__ == "__main__":
    main()
