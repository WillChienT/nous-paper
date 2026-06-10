# Nous V2 Paper Artifacts

Supplementary artifacts for the paper:

> **Nous: Extracting Cognitive Profiles from On-Chain Prediction Market Behavior**  
> Mywill HK Limited, 2026

These materials allow reproduction of all reported analyses **from the extracted profiles onward**. 
Independent replication of the extraction pipeline from raw wallet histories requires the unreleased 
production extractor parameters and is outside the release boundary (see Section 3.3 of the paper).

## Release Boundary

Consistent with the grey-box design of Section 3.3, we release:

- **Evaluation and analysis code** — scripts that compute ICC, retrieval accuracy, Spearman correlations, OLS regressions, JSD/Brier metrics
- **Extracted profiles** — frozen JSON snapshots of per-wallet behavioral profiles (split-half for Test A/B; 60/40 temporal split for Test C); wallet addresses pseudonymized to W001–W100
- **Generated prompts** — the persona prompts fed to language models in all experiments
- **Model outputs** — complete prediction records (probabilities, parse status) for all experiments

We do **not** release:

- Production extraction parameters fitted to live user behavior (the grey-box parameter λ = 2.73 and related constants)
- The `translator_v2.py` source code (its output prompts are released)
- The `engine/extractors/` source code
- Raw wallet trade histories (`trades-*.csv`)

The released artifacts allow reproduction of all reported analyses from the extracted profiles onward.

## Directory Structure

```
recoverability/         Section 4 (Test A, B, C)
  profiles/             Frozen extracted profiles (pseudonymized)
  analysis/             Analysis scripts (ICC, retrieval, OOS Spearman)
  results/              Canonical result files

injectability/          Section 5 (ablation)
  prompts/              Generated prompts for ablation conditions
  analysis/             OLS analysis scripts
  outputs/              Ablation results

usefulness/             Section 6 + Appendix A
  heterogeneity/        Four-group JSD/Brier analysis
  tempsweep/            Temperature sweep (0.0 to 1.0)
  mock_population/      Mock-population null experiment
  difficulty_subset/    Difficulty-stratified re-analysis
  mantic_baseline/      Pure-AI baseline + leakage stratification
```

## Reproducing Key Results

See `REPRODUCE.md` for step-by-step commands.

```bash
pip install -r requirements.txt

# Test A — Temporal Stability (ICC)
python recoverability/analysis/verify_test_a_icc.py

# Test B — Wallet Identification (retrieval)
python recoverability/analysis/verify_test_b_retrieval.py

# Test C — Out-of-Sample PnL Correlation
python recoverability/analysis/verify_test_c_oos.py

# Heterogeneity power analysis (MDE for the usefulness null)
cd usefulness/heterogeneity && python mde_heterogeneity.py && cd ../..

# Prompt-space diversity (translator compression test; needs local Ollama
# with nomic-embed-text — canonical results are included if you skip this)
cd usefulness/mock_population && python prompt_space_diversity.py && cd ../..
```

## Pseudonymization

All wallet addresses in released files are replaced with pseudonyms (W001–W100).
The address-to-pseudonym mapping is internal and not included in this release,
consistent with the grey-box approach and user privacy.

## License

CC BY-NC-ND 4.0 — see `LICENSE`.
