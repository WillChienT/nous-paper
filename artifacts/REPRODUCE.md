# Reproduction Guide

Each experiment below lists: which frozen files it reads, which analysis script to run, 
and which table/section of the paper the output corresponds to.

## Setup

```bash
pip install -r requirements.txt
```

The analysis scripts in `recoverability/analysis/` are self-contained: they read only
from `recoverability/profiles/` (frozen JSON) and `recoverability/results/` (canonical).
No access to the original extractor code or raw wallet data is required.

## Scope of reproduction

Two levels of artifact are released, and they support different things:

- **Independently runnable from this package (CPU only):** the recoverability analyses
  (Test A/B/C), the heterogeneity four-group analysis and its power (MDE) analysis, the
  difficulty-subset reanalysis, and the Mantic leakage scan and top-vs-mid comparison.
  Running the listed scripts reproduces the reported numbers directly from the released
  frozen files. The prompt-space diversity analysis is also runnable from the package
  but additionally requires a local Ollama instance with `nomic-embed-text` for the
  embeddings; its canonical results are included either way.
- **Released as data plus computed results, not regenerable here:** the injectability
  ablation, the temperature-sweep, and the mock-population experiments. Generating their
  model outputs required local model inference (GPU) and, for the ablation and mock
  experiments, the unreleased structure-to-narrative translator, which sits outside the
  release boundary (see `README.md`). We ship the input profiles/prompts, the model
  outputs, and the computed result files; their reported statistics can be read directly
  from those result files.

---

## Section 4 — Recoverability

### Test A: Temporal Stability (Table 2)

**Inputs:** `recoverability/profiles/profiles_testa_half1.json`, `profiles_testa_half2.json`  
**Script:** `recoverability/analysis/verify_test_a_icc.py`  
**Output:** ICC(2,1) for 14 dimension parameters across 100 wallets

```bash
python recoverability/analysis/verify_test_a_icc.py
```

Expected key values (paper Table 2):

| Dimension.Parameter | ICC |
|---|---|
| independence_index.contrarian_score | 0.9015 |
| attention_allocation.entropy | 0.7638 |
| time_scale_preference.horizon_bias | 0.7517 |
| independence_index.crowd_sensitivity | 0.7572 |
| cognitive_style.analytical_ratio | 0.7342 |

Pass criterion: ICC ≥ 0.50 for ≥ 5 of 8 dimensions → **9 of 14 parameters pass**.

Canonical result file: `recoverability/results/results_test_a.json`

### Test B: Wallet Identification (Table 3)

**Inputs:** same split-half profiles as Test A  
**Script:** `recoverability/analysis/verify_test_b_retrieval.py`  
**Output:** top-1/top-5 retrieval accuracy and within-vs-between distance ratio

```bash
python recoverability/analysis/verify_test_b_retrieval.py
```

Expected key values (reliable_high subset, 6 dims):

| Metric | Value |
|---|---|
| top-1 | 0.17 (vs random 0.01) |
| top-5 | 0.49 (vs random 0.05) |
| within/between ratio | 0.42 |
| p (permutation, 2000 iter) | 0.000 |

Canonical result file: `recoverability/results/results_test_b_retrieval_mmd.json`

### Test C: Out-of-Sample PnL Correlation (Table 4)

**Inputs:** `recoverability/profiles/profiles_testc_early.json` (60% early-window profiles),  
`recoverability/profiles/profiles_testc_late_pnl.json` (40% late-window PnL)  
**Script:** `recoverability/analysis/verify_test_c_oos.py`  
**Output:** Spearman rho and Bonferroni-corrected p-values for 4 pre-registered dimensions

```bash
python recoverability/analysis/verify_test_c_oos.py
```

Expected key values (paper Table 4):

| Dimension | rho | p_perm | Bonferroni |
|---|---|---|---|
| independence_index.contrarian_score | 0.58 | 0.0005 | PASS |
| cognitive_style.analytical_ratio | 0.43 | 0.0005 | PASS |
| belief_update_inertia.update_rate | 0.19 | ~0.05 | fail |
| loss_response.sunk_cost_sensitivity | −0.14 | ~0.15 | fail |

Canonical result file: `recoverability/results/results_test_c_oos.json`

---

## Section 5 — Injectability (Ablation)

Released as data plus computed results (see Scope of reproduction).

**Inputs / results:** `injectability/prompts/*.json` (generated prompts for all conditions),  
`injectability/outputs/results_ablation_ols_ci_embedding.json` (canonical embedding-OLS result),  
`injectability/outputs/results_ablation.json`, `injectability/outputs/results/cross_model_summary_ablation.json`,  
`injectability/outputs/results_v2_vs_control_embedding.json` (v2 vs no-instruction control, per-scenario)

The embedding-cosine OLS was computed with a local Ollama model (nomic-embed-text) over the
generated prompts and model outputs; regenerating it requires that model and is outside the
release boundary. The reported coefficients and the overall conclusion are in the result files.

Key finding (read from `results_ablation_ols_ci_embedding.json`): no model shows a significant
v2 − length_controlled contrast on embedding diversity; the most favorable case (qwen3-32b,
contrast +0.065) has p = 0.28 with a 95% CI of [−0.05, +0.18]. The overall null holds across
all models.

Supporting result (read from `results_v2_vs_control_embedding.json`, paper Section 5.2): on the
three models whose divergence metric discriminates, v2 inter-response divergence exceeds the
no-instruction control on 5/5 scenarios (e.g., 0.16 vs 0.05 on qwen3-32b); on qwen3.5-122b the
embedding metric saturates (control = 1.00) and cannot adjudicate.

---

## Section 6 + Appendix A — Usefulness

### Heterogeneity (Table 5 / Figure 3)

**Inputs:** `usefulness/heterogeneity/results_heterogeneity.json` (treatment group model outputs),
`results_heterogeneity_controlb.json`, `results_heterogeneity_placebo.json`,
`eval_set_mantic_baseline.jsonl`  
**Script:** `usefulness/heterogeneity/evaluate_heterogeneity_v3.py`

Key finding: pure_cognitive JSD Δ = +0.0035, p = 0.010, d = 0.729; Brier not improved (p > 0.9).

Note: run from the experiment directory so the script resolves its inputs (the three group
output files plus `data/placebo_preamble.txt` and `data/eval_set_mantic_baseline.jsonl`):
```bash
cd usefulness/heterogeneity && python evaluate_heterogeneity_v3.py
```

### Heterogeneity power analysis (Section 6.2 limitations)

**Inputs:** same group output files as above  
**Script:** `usefulness/heterogeneity/mde_heterogeneity.py` (run from the same directory)  
**Canonical result:** `usefulness/heterogeneity/results_mde_heterogeneity.json`

Key finding: at N = 28, the 80%-power minimum detectable effect on the pure-cognitive
ensemble-Brier contrast is ≈ 0.10 (Welch, as reported) / ≈ 0.066 (paired by question),
against a Control-B baseline of 0.086 — the null is informative only against effects of
that magnitude. The script also verifies the exact identity
ensemble Brier = 0.1 × mean individual Brier + 0.9 × mean pairwise error product.

### Temperature Sweep

**Inputs:** `usefulness/tempsweep/results_heterogeneity_temp{00,03,07,10}_{treatment,controlb}.json`  
**Canonical result:** `usefulness/tempsweep/results_heterogeneity_tempsweep.json`  
**Generation script (requires local Ollama models + GPU; not runnable from this package alone):** `usefulness/tempsweep/run_tempsweep.py`

Key finding (read from the released result files): JSD elevated at all four temperatures (0.0, 0.3, 0.7, 1.0); Brier not improved at any (p > 0.89).

### Mock-Population Null

**Inputs:** `usefulness/mock_population/mock_profiles_lhs.json`, `mock_persona_prompts.json`,  
`results_mock_temp03.json`, `results_mock_temp07.json`  
Generation required the unreleased translator plus local model inference; released here as data plus computed result.  
Key finding (read from the released result files): LHS-synthetic mock profiles produce the same null JSD distribution as real profiles, 
confirming the null is not an artifact of the specific 10 real profiles.

### Prompt-Space Diversity (Section 6.2 robustness / translator compression)

**Inputs:** `usefulness/heterogeneity/persona_prompts_10.json`, `mock_persona_prompts.json`,
`mock_profiles_lhs.json`  
**Script:** `usefulness/mock_population/prompt_space_diversity.py` (requires a local Ollama
instance with `nomic-embed-text`; embeddings are deterministic, so re-running reproduces the
canonical numbers)  
**Canonical result:** `usefulness/mock_population/results_prompt_space_diversity.json`

Key finding: the mock population's persona prompts are no more semantically spread than the
real population's (mean pairwise cosine distance 0.0162 vs 0.0160, permutation p = 0.93; max
pair ratio 0.84 despite an input-space max ratio of 1.46), and prompt spread is near-uniform
in absolute terms — the structure-to-narrative translator compresses profile diversity before
the model sees the prompt.

### Mantic Baseline (Appendix A)

**Inputs:** `usefulness/mantic_baseline/eval_set_mantic_baseline.jsonl`,  
`results_mantic_baseline.json`, `results_mantic_baseline_midvol.json`, `research_cache/` (50 files)  
**Key finding:** Qwen Metaculus score 38.45 (all 50 Qs); leak-resistant subset (fed + eurovision + other, n=23): 74.32.  
0 dated post-horizon hits in the leak-resistant subset (confirmed by `scan_leakage.py`).

Leakage scan (run from the experiment directory):
```bash
cd usefulness/mantic_baseline
python scan_leakage.py --eval-set eval_set_mantic_baseline.jsonl --cache-dir research_cache
```

Top-volume vs mid-volume difficulty comparison:
```bash
cd usefulness/mantic_baseline && python compare_eval_sets.py
```

---

## Reproducibility Notes

- **Permutation tests** (Test B, Test C): seed=42 throughout. Results should be exact except for
  minor floating-point differences (<0.002 in p-values) due to permutation draw order differences
  between Python versions.
- **Bootstrap CIs** (Test C): N_BOOT=5000, seed=42. CIs should agree within ±0.01.
- **OLS** (Injectability): fully deterministic given the frozen prompts and diversity values.
  Rerunning embedding computation requires Ollama (outside release boundary).
- **Heterogeneity analysis**: deterministic given the model output JSON files.
