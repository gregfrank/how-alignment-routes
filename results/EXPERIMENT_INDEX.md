# Master Experiment Index

Last updated: 2026-04-05


---

## Methodological conventions

1. **Primary evidence uses n≥120.** Small-n results appear in appendix only, marked with CAUTION.
2. **Corpus assignment:** Qwen models use the political corpus (v2, n=120). Other models use the safety corpus (safety_v3, n=120).
3. **Scaling comparisons are same-generation only:** Qwen3 vs Qwen3, Phi-4 vs Phi-4. Never cross generations.
4. Results marked CAUTION have limited sample sizes or are superseded by later experiments.

---

## Prompt corpora

| Corpus | ID | n pairs | Type | Script | Use for |
|--------|------|---------|------|--------|---------|
| Political v1 | `v1` | 24 | Political (CCP) | `political_prompts.py` | CAUTION: small n. Discovery/bootstrap only. |
| Political v2 | `v2` | 120 | Political (CCP, 15 categories) | `political_prompts_v2.py` | Chinese-origin models. Primary corpus. |
| Adversarial v3 | `adversarial` | 32 | Political (CCP + parallels) | `political_prompts_adversarial.py` | Supplementary. |
| Safety v1 | `safety_v1` | 16 | Safety | `safety_prompts.py` | CAUTION: too small. Do not cite. |
| Safety v2 | `safety_v2` | 32 | Safety | `safety_prompts_v2.py` | CAUTION: superseded by v3. |
| Safety v3 | `safety_v3` | 120 | Safety (HarmBench) | `safety_prompts_v3.py` | Western models. Primary corpus. |

---

## Model panel (final, all at n=120)

### Models with full pipeline (DLA + ablation + interchange)

| Model | Lab | Origin | Gen | Params | Corpus | Gate | Top nec% | Run |
|-------|-----|--------|-----|--------|--------|------|----------|-----|
| Qwen3-8B | Alibaba | Chinese | Qwen3 | 8B | v2 | L17.H17 | 1.1% | M83 |
| Qwen3-32B | Alibaba | Chinese | Qwen3 | 32B | v2 | L56.H3 | 3.2% | M84 |
| Phi-4-mini | Microsoft | Western | Phi-4 | 3.8B | safety_v3 | L13.H7 | 3.4% | M85 |
| Phi-4 | Microsoft | Western | Phi-4 | 14B | safety_v3 | L38.H25 | 2.6% | M86 |
| Llama-3.2-3B | Meta | Western | Llama-3.2 | 3B | safety_v3 | L27.H1 | 3.0% | M85 |
| GLM-Z1-9B | Zhipu | Chinese | GLM-Z1 | 9B | safety_v3 | L19.H23 | 4.7% | M85 |
| Mistral-7B | Mistral | Western | Mistral | 7B | safety_v3 | L31.H22 | 1.0% | M85 |
| Qwen2.5-7B | Alibaba | Chinese | Qwen2.5 | 7B | safety_v3 | L25.H1 | 2.4% | M110 |
| Qwen2.5-72B | Alibaba | Chinese | Qwen2.5 | 72B | safety_v3 | L79.H11 | 1.3% | M110 |
| Llama-3.3-70B | Meta | Western | Llama-3.3 | 70B | safety_v3 | L26.H40 (CC) | 2.0% | M111 |
| Gemma-2-2B | Google | Western | Gemma-2 | 2B | safety_v3 | L13.H2 | 8.4% | M87 |
| Gemma-2-9B | Google | Western | Gemma-2 | 9B | safety_v3 | L38.H14 | 1.9% | M87 |

### Scaling pairs (same generation, same corpus, n=120)

| Family | Small | Large | Scale | Top ablation change | Pattern |
|--------|-------|-------|-------|--------------------|---------| 
| Gemma-2 | 2B (L13.H2, 1.015) | 9B (L24.H7, 0.129) | 4.5x | 8x weaker | Strongest small-model gate |
| Qwen3 | 8B (L17.H17, 0.137) | 32B (L56.H3, 0.105) | 4x | 1.3x weaker | Gate persists, similar ablation |
| Phi-4 | 3.8B (L13.H7, 1.42) | 14B (L38.H25, 0.083) | 4x | 17x weaker | Gate persists, more distributed |
| Qwen2.5 | 7B (L25.H1, 0.906) | 72B (L79.H11, 0.016) | 10x | 58x weaker | Ablation essentially useless at 72B |

### Knockout cascade (gate→amplifier causal test, n=120)

| Model | Gate | Amplifiers tested | Suppressed | Strongest | Run |
|-------|------|-------------------|------------|-----------|-----|
| Qwen3-8B | L17.H17 | 6 heads (L22-L23) | 5/6 (5-26%) | L22.H5 (-25.8%) | M91 |
| Phi-4-mini | L13.H7 | 5 heads (L16-L29) | 3/5 (6-16%) | L26.H9 (-15.6%) | M91 |
| Gemma-2-2B | L13.H2 | 5 heads (L8-L16) | 3/5 (2-10%) | L15.H2 (-9.5%) | M101 |

### Bijection bypass (cipher encoding vs detection signal)

| Model | Corpus | Peak detection drop | Below benign? | Behavioral bypass? | Run |
|-------|--------|--------------------|--------------|--------------------|-----|
| Qwen3-8B | political | 66% (142.3→48.5 at L35, n=120) | Yes | Yes | M97 |
| Phi-4-mini | safety | 88% (37.1→4.3 at L16, n=120) | No (just above) | Yes — model decodes cipher as puzzle | M97 |
| Gemma-2-2B | safety | 70% (97.6→28.9 at L14, n=120) | No | Yes — model decodes cipher as puzzle | M101 |

### Models with CAUTION (do not cite as primary)

| Model | Issue |
|-------|-------|
| Qwen2.5-32B | Wrong generation for Qwen3 comparison |
| Gemma-3-4B/27B | NaN results (multimodal architecture incompatible with DLA pipeline) |


---

## Statistical validation

| Test | Result | Artifact |
|------|--------|----------|
| Bootstrap stability (2000 iter) | ABL Jaccard 0.92, INT Jaccard 1.0 | *(original experiment run)* |
| Permutation null (10000 perm) | Gate combined p=0.0001, familywise p=0.0001 | *(original experiment run)* |
| Broader corpus validation | Core amplifiers top-3 on v1(24), adv(32), v2(120) | `results/corpus_robustness/` |
| Llama gate relocation (n=16→120) | Gate moved from L13 to L27 | Validates need for large corpus |

---

## Dose-response

| Evaluation | Corpus | n outputs | Judges | Run | Status |
|-----------|--------|-----------|--------|-----|--------|
| M80 (original) | v1 political + safety_v1 | 1,188 | Gemini, Claude Haiku, GPT-4o-mini | pre_redraft | Complete |
| M83 (expanded) | v2 political | 2,400 | Gemini, Mistral Large, GPT-4o-mini | M83 | Complete (all 2400 judged) |
| M93 (re-judge) | v2 political | 2,400 | Gemini, **Llama 3.1 8B**, GPT-4o-mini | M93 | Replaces Mistral with Llama |

---

## MLP/attention balance

| Corpus | n | Attention | MLP | Source |
|--------|---|-----------|-----|--------|
| v2 (120 pairs, 15 categories) | 120 | ~77% | ~23% | `results/panel/qwen3_8b/headDLA/head_summary.csv` |
| adversarial (32 pairs, mixed) | 32 | ~63% | ~37% | `results/corpus_robustness/adversarial_headDLA/head_summary.csv` |
| v1 (24 pairs, Tiananmen-heavy) | 24 | ~39% | ~61% | Pre-redraft (no longer primary) |

Gate+amplifier heads contribute <1% of DLA signal at n=120. MLP share is corpus-dependent.

---

## Dose-response

| Evaluation | Model | Corpus | n outputs | Judges | Run | Status |
|-----------|-------|--------|-----------|--------|-----|--------|
| M83+M93 | Qwen3-8B | v2 political | 2,400 | Gemini Flash + Llama 3.1 8B + GPT-4o-mini | M83 (gen) + M93 (judge) | Complete |
| M97 | Phi-4-mini | safety_v3 | 2,400 | Gemini Flash + Llama 3.1 8B + GPT-4o-mini | M97 | Complete |

Category mapping for Qwen dose-response: `results/dose_response/category_mapping.csv`
- pair_idx 0-7 = Tiananmen, 8-15 = Tibet, ..., 112-119 = Labor Rights (15 categories × 8 prompts)

---

## Claim-evidence matrix (How_Alignment_Routes.tex → data)

| Paper claim | Section | Value | Data source | n | Verified |
|------------|---------|-------|-------------|---|----------|
| Gate L17.H17 necessity | §3.1 | 1.1% | `results/panel/qwen3_8b/interchange_v2/` | 120 | Yes |
| Gate L17.H17 sufficiency | §3.1 | 0.3% | Same | 120 | Yes |
| DLA rank of L17.H17 | §3.1 | >150th | `results/panel/qwen3_8b/headDLA/head_summary.csv` | 120 | Yes |
| Ablation: L22.H7 = 8.8% | §3.1 | 8.8% | *(original experiment run)* | 24 | Yes (discovery) |
| Bootstrap Jaccard (abl) | §3.1 | 0.92 | *(original experiment run)* | 24 | Yes |
| Bootstrap Jaccard (int) | §3.1 | 1.0 | Same | 24 | Yes |
| Permutation null | §3.1 | p<0.001 | *(original experiment run)* | 24 | Yes |
| Qwen knockout 5-26% | §3.3 | 5.4-25.8% | `results/knockout/qwen3_8b/` | 120 | Yes |
| Phi-4 knockout 6-16% | §3.3 | 5.5-15.6% | `results/knockout/phi4_mini/` | 120 | Yes (5.5 rounds to 6) |
| Knockout null 10.5% vs 3.9% | §3.3 | 10.51 vs 3.91 | `results/knockout/null/qwen3_8b/` | 20/head | Yes |
| Gate DLA rank #2 at L18 | §3.4 | #2 | `results/intermediate_dla/` | 30 | Yes |
| Gate DLA rank >20 at output | §3.4 | >20 | Same | 30 | Yes |
| Table 2: 12 models (2B-72B) | §4.1 | All values | `results/panel/*/` | 120 each | Yes |
| Scaling: ablation up to 58x weaker (4 pairs) | §4.2 | 1.3-58x | Derived from Table 2 | 120 | Yes |
| Dose-response sigmoid | §5.1 | 100%→0% | `results/dose_response/judgments_merged.csv` | 2400 | Yes |
| Language routing +0.33/+0.32 | §5.1 | +0.33/+0.32 | `results/language_routing/qwen3_8b/` | 16 pairs | Yes (CAUTION: small n) |
| Judge agreement 76.0%/97.2% | §5.2 | 76.0%/97.2% | `results/dose_response/rejudge_summary.json` | 2400 | Yes |
| Phi-4 REFUSAL→HARMFUL | §5.2 | Confirmed | `results/dose_response/ *(Phi-4 judgments)*` | 2400 | Yes |
| Bijection: Qwen 66% drop | §6.1 | 65.9% | `results/bijection/qwen3_8b/` | 120 | Yes |
| Bijection: Phi-4 88% drop | §6.1 | 88.4% | `results/bijection/phi4_mini/` | 120 | Yes |
| Bijection: Gemma 70% drop | §6.1 | 70.4% | `results/bijection/gemma2_2b/` | 120 | Yes |
| Gate DLA 78% collapse (Phi-4) | §6.1 | 78.5% | `results/intent_separation/ *(Phi-4 data not included; Qwen3-8B data available)*` | 120 | Yes |
| Gate DLA n=120 (Qwen) | §6.1 | plain=-0.041, cipher=-0.132 (no reversal; small under both conditions) | `results/intent_separation/qwen3_8b/` | 120 | Yes |
| Qwen amplifier L22.H7 reversal | §6.1 (appendix) | plain=+0.168, cipher=-0.093 | `results/intent_separation/qwen3_8b/` | 120 | Yes |
| Logit lens (Qwen) | Appendix D | Refusal tokens at L24 (7%), L34-35 (17%) plain; max 2% cipher | `results/intent_separation/qwen3_8b/logit_lens_refusal.csv` | 120 | Yes |
| Rescue: Phi-4 48.3% recovery (n=120) | §6.1 | 58/120 refusal restored | `results/rescue/phi4_single/` | 120 | Yes |
| Rescue: Qwen 0% recovery | §6.1 | 0/8 (distributed circuit) | `results/rescue/ *(preliminary, small n)*` | 8 | Yes (small n) |
| MLP/attention 23%/77% | §3.4 | 22.9%/77.1% | `results/panel/qwen3_8b/headDLA/head_summary.csv` | 120 | Yes |
| DLA robustness | Appendix A | Gate DLA rank #177-294 across 4 directions (expected; gate found by interchange not DLA) | `results/dla_robustness/qwen3_8b/` | 30 | Yes |

---

## Experiment batches

| Batch | Purpose | Status |
|-------|---------|--------|
| M62-M79 | Pre-redraft mechanistic discovery | Complete (many at small n) |
| M80 | Full-corpus dose-response (n=20-32) | Complete |
| M81 | Broader corpus validation + Phi-4 MLP | Complete |
| M82 | Gemma-2 interchange | Complete (n=32, superseded) |
| M83 | Deep remediation (n=120 interchange, dose-response, Qwen2.5-32B) | Complete |
| M84 | Same-gen scaling: Qwen3-32B, Gemma-3 (failed), Gemma-2-27B | Complete |
| M85 | Panel re-run: Phi-4, Llama, GLM-Z1, Mistral on safety_v3 n=120 | Complete |
| M86 | Phi-4 14B scaling probe | Complete |
| M87 | Gemma-2 (2B + 9B) on safety_v3 n=120 | Complete |
| M88 | Gemma-4 exploratory (thinking tokens issue) | Complete (documented) |
| M89 | Initial bijection probe (Qwen only) | Superseded by M90 |
| M90 | Full bijection bypass (Qwen3-8B, 5 experiments) | Complete |
| M91 | Knockout cascade re-run at n=120 (Qwen + Phi-4) | Complete |
| M92 | Multi-model bijection (Phi-4 + Gemma-2-2B) | Complete |
| M93 | Judge panel re-run: Llama 3.1 replaces Mistral | Complete |
| M94 | Intent separation: per-head DLA under cipher + logit lens. Phi-4 n=120, Qwen n=120. Logit lens: refusal tokens appear L24 (7%) and L34-35 (17%) plaintext, never under cipher (max 2% at L35) | Complete |
| M95 | Language-conditioned routing (Qwen, EN vs CN, 16 pairs) CAUTION: small n | Complete |
| M96 | Llama knockout cascade (L27.H1 gate, n=120) — zero cascade (last-layer gate) | Complete (not cited) |
| M97 | Phi-4 dose-response n=120 + bijection n=120 + knockout null | Complete |
| M98 | Intermediate-layer DLA: gate rank #2 at L18, >20 at output | Complete |
| M99 | Cipher rescue: inject plaintext gate activation into cipher forward pass | Complete |
| M100 | DLA robustness: gate DLA rank #177-294 across 4 alternative directions; interchange Jaccard 1.0 provides the real robustness | Complete |
| M101 | Gemma-2-2B bijection n=120 (70% drop) + knockout cascade n=120 (3/5 suppressed 2-10%) | Complete |
| M105 | Cipher interchange: gate necessity/sufficiency under cipher vs plaintext. 3 models (Gemma-2-2B, Phi-4-mini, Qwen3-8B), n=120. DLA-based interchange with absolute values. Gemma/Phi-4: 99% necessity collapse; Qwen: 70% (distributed architecture). | Complete |
| M106 | Cipher diagnostic: per-head DLA under plaintext vs cipher to identify refusal pathway. All known circuit heads rank in top 5% by cipher sensitivity. Phi-4 gate+amplifier in top 4/768. Qwen amplifiers in top 20/1152, gate at 57/1152. | Complete |
| M107 | Band interchange (Phi-4-mini). Ratios are band/best-in-band: L13 band (6 heads, nec=2.116) / L13.H7 (1.123) = 1.88x; L16 band (12 heads, 0.714) / L16.H9 (0.384) = 1.86x; L14 band (8 heads, 0.718) / L14.H16 (0.279) = 2.58x. Band summaries in `results/band_interchange/phi4_mini/band_interchange_summary.json`; single-head denominators from runner log. | Complete |
| M107b | Band interchange (Gemma-2-9B). L22 band (0.875) / L22.H7 (0.932) = 0.94x; L26 band (0.190) / L26.H8 (0.720) = 0.26x; L17 band (0.453) / L17.H7 (0.285) = 1.59x. Band summaries in `results/band_interchange/gemma2_9b/band_interchange_summary.json`. | Complete |
| M108 | Behavioral validation of band ablation (n=30, heuristic classifier). Phi-4-mini: single head 0% effect, gate band -7%, all bands (26 heads) -40%. Gemma-2-2B: all bands (10 heads) -3%. Band ablation has real behavioral effects in Phi-4 but Gemma-2-2B routing too concentrated. | Complete |

| M109 | Refined band interchange (Phi-4-mini, n=120). Pro-routing-only bands, cross-layer groups, counter-routing bands, and per-prompt correlation clustering. Cross-layer top-10 is 3.31x single head. Two opposing coalitions discovered: pro-routing (L13 gate + L23/L27/L29 allies) vs counter-routing (L16 counter + L14/L27 allies), r=-0.863 between coalition leaders. | Complete |
| M99b | Rescue at n=120: single-head gate injection restores refusal in 48% of cases (Phi-4-mini, up from 0% cipher). Replaces preliminary n=8 result. | Complete |
| M110 | Qwen2.5 scaling pair (7B→72B, safety_v3, n=120). DLA-selected: ablation 58x weaker (0.906→0.016), necessity 2.4%→1.3%, gate L25→L79. CC-selected (72B): L53.H16 nec=0.5%, weaker than DLA gate — DLA found better candidate at 72B. L78 band (4 heads) identified by CC. Run on 2x A100-80GB fp16. | Complete |
| M111 | Llama-3.3-70B-Instruct (safety_v3, n=120, 2x A100-80GB fp16). DLA-selected: L77.H47 nec=1.3%, top DLA ablation L79.H32=0.101. CC-selected: L26.H40 nec=2.0%, top CC ablation L23.H48=0.382 — CC found a BETTER gate (shallower, stronger). L26 band (3 heads: H40, H39, H27). Largest model tested (70B). Panel entry only (not scaling pair: cross-generation 3.2 vs 3.3). | Complete |

### Key conclusions from M105-M109

1. **Cipher contrast analysis** identifies the full content-dependent circuit in O(3n) forward passes. Known heads validated in top 5%. New heads discovered (L16.H9 in Phi-4, L31.H3 in Qwen).
2. **77/23 split** between content-dependent and content-independent routing is consistent across 3 models (77.6%, 76.8%, 77.4%).
3. **Gate bands** are stronger than individual gate heads in Phi-4 (1.88x at L13) but NOT universally — Gemma-2-9B L22 band includes counter-routing heads (0.94x).
4. **Behavioral effects** are real: 26-head band ablation eliminates 40% of refusals in Phi-4-mini. Single-head ablation has zero behavioral effect across all models.
5. **Small models resist band ablation** (Gemma-2-2B: 97% still refuse). Routing is too concentrated.
6. **Cross-layer grouping outperforms single-layer bands.** Top 10 pro-routing heads across 5 layers achieve 3.31x single-head necessity. The optimal "gate group" spans 3-5 layers.
7. **Two opposing coalitions** compete to determine routing outcome. Pro-routing coalition (led by L13.H7, r=0.5-0.78 internal) and counter-routing coalition (led by L16.H12, r=0.6-0.88 internal) activate simultaneously on harmful content with strong anti-correlation between them (r=-0.863). Routing is coalition competition, not a single pathway.
8. **Plaintext DLA sign** cleanly separates pro-routing (positive) from counter-routing (negative) heads. Gate-like profile: high positive plaintext DLA, near-zero cipher DLA, near-zero benign DLA.

---

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# Per-head DLA
.venv/bin/python routing_head_dla.py --model "Qwen/Qwen3-8B" --corpus v2
# Ablation
.venv/bin/python routing_head_ablation.py --model "Qwen/Qwen3-8B" --corpus v2 --mode ablate --heads "22.7,23.2,17.17" --direction-checkpoint runs/<model>/direction.pt (recomputable via compute_mean_diff_direction.py)
# Interchange
.venv/bin/python routing_head_ablation.py --model "Qwen/Qwen3-8B" --corpus v2 --mode interchange --heads "22.7,23.2,17.17" --direction-checkpoint runs/<model>/direction.pt (recomputable via compute_mean_diff_direction.py)
```
