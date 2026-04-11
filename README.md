# How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models

**Paper:** [arXiv:2604.04385](https://arxiv.org/abs/2604.04385)

## Abstract

This paper localizes the policy routing mechanism in alignment-trained language models. An intermediate-layer attention gate reads detected content and triggers deeper amplifier heads that boost the signal toward refusal. In smaller models the gate and amplifier are single heads; at larger scale they become bands of heads across adjacent layers. The gate contributes under 1% of output DLA, but interchange testing (p<0.001) and knockout cascade confirm it is causally necessary. Interchange screening at n≥120 detects the same motif in twelve models from six labs (2B to 72B), though specific heads differ by lab. Per-head ablation weakens up to 58× at 72B and misses gates that interchange identifies; interchange is the only reliable audit at scale. Modulating the detection-layer signal continuously controls policy from hard refusal through evasion to factual answering. On safety prompts the same intervention turns refusal into harmful guidance, showing the safety-trained capability is gated by routing rather than removed. Thresholds vary by topic and by input language, and the circuit relocates across generations within a family while behavioral benchmarks register no change. Routing is early-commitment: the gate commits at its own layer before deeper layers finish processing the input. Under an in-context substitution cipher, gate interchange necessity collapses 70 to 99% across three models and the model switches to puzzle-solving. Injecting the plaintext gate activation into the cipher forward pass restores 48% of refusals in Phi-4-mini, localizing the bypass to the routing interface. A second method, cipher contrast analysis, uses plain/cipher DLA differences to map the full cipher-sensitive routing circuit in O(3n) forward passes. Any encoding that defeats detection-layer pattern matching bypasses the policy regardless of whether deeper layers reconstruct the content.

## Hardware

| Model size | GPU requirement |
|-----------|----------------|
| ≤9B (Qwen3-8B, Phi-4-mini, Gemma-2-2B, etc.) | 1× 16 GB (RTX 4090, A5000, etc.) |
| 14B–32B (Phi-4, Qwen3-32B) | 1× 24–48 GB (A100-40GB) |
| 70B–72B (Llama-3.3-70B, Qwen2.5-72B) | 2× A100-80GB, fp16, `device_map="auto"` |

Figure generation requires CPU only. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for compute time estimates.

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/gregfrank/how-alignment-routes.git
cd how-alignment-routes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run per-head DLA on Qwen3-8B (requires GPU with ≥16GB VRAM)
cd src
python routing_head_dla.py --model Qwen/Qwen3-8B --corpus v2 --run-dir ../runs/demo_dla

# Run head ablation (requires a direction checkpoint)
python compute_mean_diff_direction.py --model Qwen/Qwen3-8B --corpus v2 --n-pairs 120 --out ../runs/demo_direction.pt
python routing_head_ablation.py --model Qwen/Qwen3-8B --corpus v2 --mode ablate \
    --heads "17.17,22.7,23.2" --direction-checkpoint ../runs/demo_direction.pt \
    --run-dir ../runs/demo_ablation

# Run interchange intervention
python routing_head_ablation.py --model Qwen/Qwen3-8B --corpus v2 --mode interchange \
    --heads "17.17,22.7,23.2" --direction-checkpoint ../runs/demo_direction.pt \
    --run-dir ../runs/demo_interchange
```

## Repository Structure

```
how-alignment-routes/
├── README.md
├── REPRODUCIBILITY.md          # Step-by-step reproduction guide
├── LICENSE                     # MIT
├── requirements.txt
│
├── src/                        # Core experiment code
│   ├── routing_logit_trajectory.py      # Shared utilities, model loading, prompt resolution
│   ├── routing_direct_logit_attribution.py  # Per-layer DLA decomposition
│   ├── routing_head_dla.py              # Per-head DLA decomposition
│   ├── routing_head_ablation.py         # Head ablation + interchange interventions
│   ├── compute_mean_diff_direction.py   # Quick direction computation from paired prompts
│   ├── political_prompts_v2.py          # Political corpus (120 pairs, 15 categories)
│   ├── safety_prompts_v3.py             # Safety corpus (120 pairs, HarmBench-derived)
│   ├── political_prompts_v1.py          # Political corpus v1 (24 pairs)
│   ├── political_prompts_adversarial.py # Adversarial political corpus (32 pairs)
│   ├── safety_prompts.py               # Safety corpus v1 (16 pairs)
│   ├── safety_prompts_v2.py            # Safety corpus v2 (32 pairs)
│   ├── probe_cross_validation.py        # Category definitions for v1 corpus
│   └── cipher/                          # Experiment runner scripts (18 total)
│       ├── run_cipher_interchange.py    # Gate necessity under cipher encoding
│       ├── run_cipher_diagnostic.py     # Per-head cipher sensitivity analysis
│       ├── run_cipher_intent_separation.py  # Intent separation DLA + logit lens
│       ├── run_cipher_rescue.py         # Gate activation rescue under cipher
│       ├── run_band_interchange.py      # Layer-band interchange
│       ├── run_band_behavioral.py       # Behavioral validation of band ablation
│       ├── run_refined_bands.py         # Coalition analysis with correlation clustering
│       ├── run_knockout_cascade.py      # Gate→amplifier knockout cascade
│       ├── run_knockout_null.py         # Knockout null distribution
│       ├── run_bijection_bypass.py      # Cipher bijection bypass
│       ├── run_bijection_multimodel.py  # Multi-model bijection comparison
│       ├── run_dose_response.py         # Dose-response generation (Qwen)
│       ├── judge_dose_response.py       # Dose-response judging (Qwen)
│       ├── run_phi4_dose_response.py    # Dose-response generation (Phi-4)
│       ├── judge_phi4_dose_response.py  # Dose-response judging (Phi-4)
│       ├── run_language_routing.py      # Language-conditioned routing (EN vs CN)
│       ├── run_intermediate_dla.py      # Intermediate-layer DLA rankings
│       └── run_dla_robustness.py        # Direction robustness check
│
├── figures/                    # Figure generation scripts
│   ├── generate_fig_scaling.py
│   ├── generate_fig_knockout_cascade.py
│   ├── generate_fig_cipher_interchange.py
│   ├── generate_fig_cipher_diagnostic_scatter.py
│   ├── generate_fig_discovery_pipeline.py
│   ├── generate_fig_intent_separation.py
│   ├── generate_fig_dose_response_v3.py
│   ├── generate_fig_logit_lens.py
│   └── ... (19 scripts total)
│
├── results/                    # Experiment result summaries
│   ├── EXPERIMENT_INDEX.md     # Master index of all experiments
│   ├── panel/                  # Per-model DLA, ablation, interchange summaries
│   │   ├── qwen3_8b/          # Qwen3-8B (political corpus, n=120)
│   │   ├── phi4_mini/          # Phi-4-mini (safety corpus, n=120)
│   │   ├── gemma2_2b/          # Gemma-2-2B
│   │   ├── ... (12 models)
│   │   └── llama33_70b/        # Llama-3.3-70B (largest model)
│   ├── knockout/               # Knockout cascade data (3 models)
│   ├── cipher_interchange/     # M105: cipher interchange (3 models)
│   ├── cipher_contrast/        # M106: cipher diagnostic (4 models)
│   ├── band_interchange/       # M107: band interchange (2 models)
│   ├── refined_bands/          # M109: coalition analysis (Phi-4-mini)
│   ├── rescue/                 # M99b: gate activation rescue (n=120)
│   ├── dose_response/          # Judge panel data
│   ├── intent_separation/      # M94: DLA under cipher + logit lens
│   ├── bijection/              # Bijection bypass summaries (3 models)
│   ├── dla_robustness/         # M100: direction robustness check
│   ├── band_behavioral/        # M108: behavioral validation of band ablation
│   ├── corpus_robustness/      # M81: corpus robustness (v2 + adversarial DLA)
│   ├── intermediate_dla/       # M98: intermediate-layer DLA rankings
│   ├── language_routing/       # M95: language-conditioned routing (EN vs CN)
│   ├── discovery_pipeline/     # M63: headDLA, ablation, interchange (Qwen3-8B, n=24)
│   ├── m75_contextual_detection/   # Contextual gradient data (Figure 2)
│   └── m62_routing_localization/   # Position comparison data (Figure 2)
```

## Model Panel

| Model | Lab | Params | Corpus | Gate Head | Top Necessity |
|-------|-----|--------|--------|-----------|---------------|
| Gemma-2-2B | Google | 2B | safety_v3 | L13.H2 | 8.4% |
| Llama-3.2-3B | Meta | 3B | safety_v3 | L27.H1 | 3.0% |
| Phi-4-mini | Microsoft | 3.8B | safety_v3 | L13.H7 | 3.4% |
| Qwen2.5-7B | Alibaba | 7B | safety_v3 | L25.H1 | 2.4% |
| Mistral-7B | Mistral | 7B | safety_v3 | L31.H22 | 1.0% |
| Qwen3-8B | Alibaba | 8B | v2 | L17.H17 | 1.1% |
| Gemma-2-9B | Google | 9B | safety_v3 | L38.H14 | 1.9% |
| GLM-Z1-9B | Zhipu | 9B | safety_v3 | L19.H23 | 4.7% |
| Phi-4 | Microsoft | 14B | safety_v3 | L38.H25 | 2.6% |
| Qwen3-32B | Alibaba | 32B | v2 | L56.H3 | 3.2% |
| Llama-3.3-70B | Meta | 70B | safety_v3 | L26.H40 | 2.0% |
| Qwen2.5-72B | Alibaba | 72B | safety_v3 | L79.H11 | 1.3% |

## Related Work

The companion paper ["Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails"](https://arxiv.org/abs/2603.18280) (code: [routing-is-learned](https://github.com/gregfrank/routing-is-learned)) establishes the representational basis for political sensitivity detection that this paper's mechanistic analysis builds upon.

## Citation

```bibtex
@article{frank2026howalignmentroutes,
  title={How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models},
  author={Frank, Gregory N.},
  journal={arXiv preprint arXiv:2604.04385},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
