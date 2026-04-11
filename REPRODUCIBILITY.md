# Reproducibility Guide

Step-by-step instructions for reproducing all experiments in "How Alignment Routes" (arXiv:2604.04385).

## Hardware Requirements

| Experiment | GPU VRAM | Approx. Time |
|-----------|----------|------------|
| DLA + ablation (≤9B models) | 16 GB (e.g., RTX 4090, A5000) | 30–60 min per model |
| DLA + ablation (14B–32B) | 24–48 GB (e.g., A100-40GB) | 1–3 hours per model |
| DLA + ablation (70B–72B) | 2× A100-80GB (fp16) | 4–8 hours per model |
| Cipher interchange (M105) | Same as base model | 1–2 hours per model |
| Band interchange (M107/M109) | Same as base model | 2–4 hours |
| Figure generation | CPU only | < 1 min each |

**Total compute for full replication:** ~80–120 A100-GPU-hours.

## Software Requirements

```bash
Python 3.10+
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Tested versions:** torch 2.4+, transformers 4.45+. Gemma-2 models require transformers ≥ 4.42.

## Authentication

Some models are gated and require a Hugging Face token:

```bash
export HF_TOKEN=<your_huggingface_token>
# Or: huggingface-cli login
```

Gated models in the panel: Gemma-2-2B, Gemma-2-9B, Llama-3.2-3B, Llama-3.3-70B, Mistral-7B.

## Experiment Reproduction

All commands should be run from the `src/` directory:

```bash
cd src
```

### Step 1: Compute Direction Checkpoint

Before running DLA or ablation, compute a mean-difference direction for the model:

```bash
# For Chinese-origin models (political corpus):
python compute_mean_diff_direction.py --model Qwen/Qwen3-8B --corpus v2 --n-pairs 120 \
    --out ../runs/qwen3_8b/direction.pt

# For Western models (safety corpus):
python compute_mean_diff_direction.py --model microsoft/Phi-4-mini-instruct --corpus safety_v3 --n-pairs 120 \
    --out ../runs/phi4_mini/direction.pt
```

**Expected output:** A `.pt` checkpoint file (~16 KB for 8B models).

### Step 2: Per-Head DLA (Discovery)

```bash
python routing_head_dla.py --model Qwen/Qwen3-8B --corpus v2 \
    --run-dir ../runs/qwen3_8b/headDLA
```

**Expected output:** `head_summary.csv` with per-head DLA scores, `run_manifest.json`.

### Step 3: Head Ablation

```bash
python routing_head_ablation.py --model Qwen/Qwen3-8B --corpus v2 --mode ablate \
    --heads "22.7,23.2,17.17,22.5,23.0" \
    --direction-checkpoint ../runs/qwen3_8b/direction.pt \
    --run-dir ../runs/qwen3_8b/headablate
```

**Expected output:** `head_ablate_summary.csv` ranking heads by NLL reduction.

### Step 4: Interchange Intervention

```bash
python routing_head_ablation.py --model Qwen/Qwen3-8B --corpus v2 --mode interchange \
    --heads "22.7,23.2,17.17,22.5,23.0" \
    --direction-checkpoint ../runs/qwen3_8b/direction.pt \
    --run-dir ../runs/qwen3_8b/interchange
```

**Expected output:** `head_interchange_summary.csv` with necessity (ctrl→CCP reduction) and sufficiency (CCP→ctrl increase) per head.

### Step 5: Knockout Cascade (§3.3)

The knockout cascade runner tests whether ablating the gate suppresses downstream amplifiers:

```bash
python cipher/run_knockout_cascade.py --model Qwen/Qwen3-8B \
    --corpus v2 --gate 17.17 \
    --amplifiers "22.7,23.2,22.5,23.0,22.4,23.3" \
    --run-dir ../runs/qwen3_8b/knockout
```

**Expected:** Amplifier necessity drops 5–26% when gate is ablated (Qwen3-8B).

### Step 6: Cipher Interchange (§6.1, M105)

```bash
python cipher/run_cipher_interchange.py --model microsoft/Phi-4-mini-instruct \
    --checkpoint ../runs/phi4_mini/direction.pt \
    --corpus safety_v3 --gate 13.7 --n-pairs 120 \
    --run-dir ../runs/cipher_interchange/phi4_mini
```

**Expected:** Gate necessity collapses under cipher encoding. The cipher interchange experiment measures total-DLA necessity (plaintext ~14.5%, cipher ~0.3% for Phi-4-mini). The 3.4% figure in the panel table uses the per-head interchange method, which measures a different quantity.

### Step 7: Cipher Diagnostic (§6.1, M106)

```bash
python cipher/run_cipher_diagnostic.py --model microsoft/Phi-4-mini-instruct \
    --checkpoint ../runs/phi4_mini/direction.pt \
    --corpus safety_v3 --n-pairs 120 \
    --run-dir ../runs/cipher_diagnostic/phi4_mini
```

**Expected:** Known circuit heads rank in top 5% by cipher sensitivity.

### Step 8: Band Interchange (§4.3, M107)

```bash
python cipher/run_band_interchange.py --model microsoft/Phi-4-mini-instruct \
    --checkpoint ../runs/phi4_mini/direction.pt \
    --corpus safety_v3 --n-pairs 120 \
    --diagnostic ../runs/cipher_diagnostic/phi4_mini/cipher_diagnostic_all_heads.csv \
    --run-dir ../runs/band_interchange/phi4_mini
```

**Expected:** Gate band (6 heads) achieves 1.88× single-head necessity.

### Step 9: Refined Bands / Coalition Analysis (§4.3, M109)

```bash
python cipher/run_refined_bands.py --model microsoft/Phi-4-mini-instruct \
    --checkpoint ../runs/phi4_mini/direction.pt \
    --corpus safety_v3 --n-pairs 120 \
    --diagnostic ../runs/cipher_diagnostic/phi4_mini/cipher_diagnostic_all_heads.csv \
    --run-dir ../runs/refined_bands/phi4_mini
```

**Expected:** Two opposing coalitions with r = −0.86 anti-correlation.

### Step 10: Generate Figures

All figure scripts should be run from the **repo root** (not from `figures/`):

```bash
cd ..  # back to repo root
python figures/generate_fig_scaling.py
python figures/generate_fig_knockout_cascade.py
python figures/generate_fig_cipher_interchange.py
# ... etc. All scripts output to figures/output/
```

**Note:** Most figure scripts read from `results/` (pre-computed summaries included in this repo). A few scripts embed their data directly. Two supplementary figure scripts (`generate_fig_dla_heatmaps.py`, `generate_fig_prompt_time.py`) require pre-redraft experiment data not included here — these produce appendix figures superseded by later analyses in the main paper.

## Full Panel Replication

To replicate the full 12-model panel (Table 2 in the paper), repeat Steps 1–4 for each model:

| Model ID | Corpus | Candidate Heads |
|----------|--------|----------------|
| `Qwen/Qwen3-8B` | v2 | 17.17, 22.7, 23.2, 22.5, 23.0 |
| `Qwen/Qwen3-32B` | v2 | Top-20 from DLA |
| `microsoft/Phi-4-mini-instruct` | safety_v3 | 13.7, 16.9, 26.9, 14.16, 29.9 |
| `microsoft/phi-4` | safety_v3 | Top-20 from DLA |
| `meta-llama/Llama-3.2-3B-Instruct` | safety_v3 | Top-20 from DLA |
| `THUDM/glm-z1-9b-0414` | safety_v3 | Top-20 from DLA |
| `mistralai/Mistral-7B-Instruct-v0.3` | safety_v3 | Top-20 from DLA |
| `google/gemma-2-2b-it` | safety_v3 | Top-20 from DLA |
| `google/gemma-2-9b-it` | safety_v3 | Top-20 from DLA |
| `Qwen/Qwen2.5-7B-Instruct` | safety_v3 | Top-20 from DLA |
| `Qwen/Qwen2.5-72B-Instruct` | safety_v3 | Top-20 from DLA |
| `meta-llama/Llama-3.3-70B-Instruct` | safety_v3 | Top-20 from DLA |

## Known Issues

1. **Gemma-3 models** produce NaN in DLA due to multimodal architecture incompatibility. Use Gemma-2 instead.
2. **MiniCPM4.1-8B** requires `attn_implementation="eager"` and may still fail with custom attention code.
3. **70B+ models** require multi-GPU setups (2× A100-80GB) with `device_map="auto"`.
4. **Transformers version**: Some models require specific minimum versions. If loading fails, try updating: `pip install --upgrade transformers`.

## Verifying Results

Each experiment produces a `run_manifest.json` recording the model, corpus, seed, and configuration. Compare your summary CSVs against the provided `results/` files. Key metrics to check:

- **Gate head necessity** (interchange ctrl→CCP reduction): should match Table 2 within ±0.5%
- **Knockout suppression**: 5–26% for Qwen, 6–16% for Phi-4-mini
- **Cipher bypass**: 66–88% collapse in gate necessity
