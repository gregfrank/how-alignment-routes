#!/usr/bin/env python3
"""M107: Multi-head band interchange — test whether cipher-sensitive layer bands
function as collective gates.

For each model, we:
1. Load the cipher diagnostic results to identify layer bands
2. Run multi-head interchange: swap ALL heads in a band simultaneously
3. Compare collective necessity to single-head necessity
4. Test whether the band's collective trigger function is stronger than
   any individual head

This tests David Evans' hypothesis that the gate is a subnetwork, not
a single head, and that the subnetwork's size scales with model size.

Usage (GPU):
  python scripts/run_band_interchange_m107.py \
      --model microsoft/Phi-4-mini-instruct \
      --checkpoint results/m85_phi4_mini/direction.pt \
      --corpus safety_v3 \
      --diagnostic results/m106_cipher_diagnostic/phi4_mini/cipher_diagnostic_all_heads.csv \
      --n-pairs 120 \
      --run-dir results/m107_band_interchange/phi4_mini
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


def identify_bands(diagnostic_path: str, sens_threshold: float = 0.1,
                   dla_threshold: float = 0.05, min_band_size: int = 2,
                   ) -> List[Dict]:
    """Identify layer bands from cipher diagnostic data.

    A band is a group of cipher-sensitive heads at the same layer (or
    adjacent layers within ±1).
    """
    rows = list(csv.DictReader(open(diagnostic_path)))

    # Filter to cipher-sensitive heads with non-negligible DLA
    sensitive = []
    for r in rows:
        if (float(r['cipher_sensitivity']) > sens_threshold and
                abs(float(r['mean_plain_dla'])) >= dla_threshold):
            sensitive.append({
                'layer': int(r['layer']),
                'head': int(r['head']),
                'sensitivity': float(r['cipher_sensitivity']),
                'plain_dla': float(r['mean_plain_dla']),
                'routing': float(r['plain_minus_benign']),
            })

    # Group by layer
    by_layer = defaultdict(list)
    for h in sensitive:
        by_layer[h['layer']].append(h)

    # Each layer with enough sensitive heads is its own band (no merging)
    bands = []
    for layer in sorted(by_layer.keys()):
        heads = by_layer[layer]
        if len(heads) >= min_band_size:
            bands.append({
                'layers': [layer],
                'heads': [(h['layer'], h['head']) for h in heads],
                'total_sensitivity': sum(h['sensitivity'] for h in heads),
                'n_heads': len(heads),
            })

    bands.sort(key=lambda b: b['total_sensitivity'], reverse=True)
    return bands


@contextmanager
def multi_head_interchange(
    model: Any,
    head_specs: List[Tuple[int, int]],
    replacement_activations: Dict[Tuple[int, int], torch.Tensor],
    head_dim: int,
):
    """Swap multiple heads simultaneously."""
    with ExitStack() as stack:
        for (layer_idx, head_idx) in head_specs:
            act = replacement_activations[(layer_idx, head_idx)]
            ctx = head_interchange(model, layer_idx, head_idx, act, head_dim)
            stack.enter_context(ctx)
        yield


def compute_total_dla(
    model, tokenizer, final_norm, diff_direction,
    prompt, n_heads, head_dim, intervention_context=None,
) -> float:
    ctx = intervention_context if intervention_context is not None else nullcontext()
    with ctx:
        rows = compute_head_dla_records(
            model=model, tokenizer=tokenizer, final_norm=final_norm,
            diff_direction=diff_direction, prompt=prompt,
            ablation_spec=None, position_mode="last_prompt",
            decision_point=None, n_heads=n_heads, head_dim=head_dim,
        )
    return sum(row["mlp_contribution"] + row["attn_total"] for row in rows)


def run_band_interchange(
    model, tokenizer, final_norm, diff_direction,
    harmful_prompt, benign_prompt,
    band_heads: List[Tuple[int, int]],
    n_heads, head_dim,
) -> Dict[str, float]:
    """Run collective interchange for a band of heads."""

    # Capture activations for all heads in the band
    harm_acts = {}
    benign_acts = {}
    for (layer, head) in band_heads:
        harm_acts[(layer, head)] = capture_head_activations(
            model, tokenizer, harmful_prompt, layer, head, head_dim,
        )
        benign_acts[(layer, head)] = capture_head_activations(
            model, tokenizer, benign_prompt, layer, head, head_dim,
        )

    # Baseline DLA
    baseline_harm = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        harmful_prompt, n_heads, head_dim,
    )
    baseline_benign = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        benign_prompt, n_heads, head_dim,
    )

    # Collective necessity: run harmful, replace ALL band heads with benign
    ctx_nec = multi_head_interchange(model, band_heads, benign_acts, head_dim)
    swapped_harm = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        harmful_prompt, n_heads, head_dim,
        intervention_context=ctx_nec,
    )

    # Collective sufficiency: run benign, inject ALL band heads from harmful
    ctx_suf = multi_head_interchange(model, band_heads, harm_acts, head_dim)
    swapped_benign = compute_total_dla(
        model, tokenizer, final_norm, diff_direction,
        benign_prompt, n_heads, head_dim,
        intervention_context=ctx_suf,
    )

    return {
        "baseline_harm_dla": baseline_harm,
        "baseline_benign_dla": baseline_benign,
        "swapped_harm_dla": swapped_harm,
        "swapped_benign_dla": swapped_benign,
        "necessity": baseline_harm - swapped_harm,
        "sufficiency": swapped_benign - baseline_benign,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", default="safety_v3")
    parser.add_argument("--diagnostic", required=True,
                        help="Path to cipher_diagnostic_all_heads.csv from M106")
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sens-threshold", type=float, default=0.1)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Identify bands
    print(f"=== M107: Band Interchange ===")
    bands = identify_bands(args.diagnostic, sens_threshold=args.sens_threshold)
    print(f"Found {len(bands)} bands:")
    for i, band in enumerate(bands):
        layers_str = ",".join(str(l) for l in band['layers'])
        print(f"  Band {i}: layers [{layers_str}], {band['n_heads']} heads, "
              f"total sensitivity {band['total_sensitivity']:.3f}")

    # Load model
    print(f"\n[1/4] Loading model... ({_timestamp()})")
    model, tokenizer = load_model_and_tokenizer(args.model)
    n_heads = _resolve_n_heads(model)
    head_dim = _resolve_head_dim(model)
    final_norm = resolve_final_norm(model)

    # Load direction
    print(f"[2/4] Loading direction... ({_timestamp()})")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "bank" in ckpt and "best" in ckpt:
        direction = ckpt["bank"][ckpt["best"]["candidate_idx"]]["v_clean"].float().cpu()
    elif "direction" in ckpt:
        direction = ckpt["direction"].float().cpu()
    else:
        raise KeyError(f"Unknown checkpoint: {list(ckpt.keys())}")

    # Load prompts
    print(f"[3/4] Loading prompts... ({_timestamp()})")
    pairs = resolve_prompt_pairs(args.corpus, args.n_pairs)
    print(f"  {len(pairs)} pairs")

    # Run band interchange for each band + individual heads for comparison
    print(f"[4/4] Running band interchange... ({_timestamp()})")

    all_results = []

    for band_idx, band in enumerate(bands[:3]):  # Top 3 bands
        band_heads = band['heads']
        layers_str = ",".join(str(l) for l in band['layers'])
        print(f"\n  Band {band_idx} (layers [{layers_str}], {len(band_heads)} heads):")

        band_results = []
        for i, pair in enumerate(pairs):
            scores = run_band_interchange(
                model, tokenizer, final_norm, direction,
                pair.ccp_prompt, pair.control_prompt,
                band_heads, n_heads, head_dim,
            )
            band_results.append({
                "pair_idx": pair.pair_idx,
                "topic": pair.topic,
                "band": band_idx,
                "band_layers": layers_str,
                "band_n_heads": len(band_heads),
                **scores,
            })
            print(f"    pair {i+1}/{len(pairs)}: nec={scores['necessity']:+.4f}", end="\r", flush=True)

        all_results.extend(band_results)

        # Summary for this band
        necs = [abs(r['necessity']) for r in band_results]
        sufs = [abs(r['sufficiency']) for r in band_results]
        print(f"\n    Band {band_idx} mean |necessity|={np.mean(necs):.4f}, "
              f"mean |sufficiency|={np.mean(sufs):.4f}")

        # Also run single-head interchange on the top head in this band for comparison
        top_head = max(band['heads'],
                       key=lambda h: next(
                           (float(r['cipher_sensitivity'])
                            for r in csv.DictReader(open(args.diagnostic))
                            if int(r['layer']) == h[0] and int(r['head']) == h[1]),
                           0))

        single_results = []
        for i, pair in enumerate(pairs):
            scores = run_band_interchange(
                model, tokenizer, final_norm, direction,
                pair.ccp_prompt, pair.control_prompt,
                [top_head], n_heads, head_dim,
            )
            single_results.append(scores)

        single_necs = [abs(r['necessity']) for r in single_results]
        single_sufs = [abs(r['sufficiency']) for r in single_results]
        ratio_nec = np.mean(necs) / np.mean(single_necs) if np.mean(single_necs) > 0 else float('inf')
        print(f"    Top single head L{top_head[0]}.H{top_head[1]}: "
              f"|necessity|={np.mean(single_necs):.4f}")
        print(f"    Band/single ratio: {ratio_nec:.2f}x")

    # Write results
    write_csv(run_dir / "band_interchange_pairwise.csv", all_results)

    summary = {
        "model": args.model,
        "corpus": args.corpus,
        "n_pairs": len(pairs),
        "bands": [],
        "timestamp": _timestamp(),
    }
    for band_idx, band in enumerate(bands[:3]):
        band_rows = [r for r in all_results if r["band"] == band_idx]
        necs = [abs(r['necessity']) for r in band_rows]
        sufs = [abs(r['sufficiency']) for r in band_rows]
        summary["bands"].append({
            "band": band_idx,
            "layers": band['layers'],
            "n_heads": band['n_heads'],
            "heads": [f"L{l}.H{h}" for l, h in band['heads']],
            "mean_abs_necessity": float(np.mean(necs)),
            "mean_abs_sufficiency": float(np.mean(sufs)),
        })

    (run_dir / "band_interchange_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"\nResults in {run_dir}")


if __name__ == "__main__":
    main()
