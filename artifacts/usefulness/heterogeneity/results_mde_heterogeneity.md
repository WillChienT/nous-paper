# MDE analysis for the heterogeneity null (pure-cognitive contrast)

Listwise-complete questions: **28**; agents per group: 10; alpha = 0.05, target power = 0.8.

## Minimum detectable effects (80% power)

| Metric | Test | SE | MDE (80%) | Observed diff |
|---|---|---|---|---|
| Ensemble Brier | Welch (as reported) | 0.03608 | 0.1029 | -0.00229 |
| Ensemble Brier | Paired by question | 0.02263 | 0.0658 | -0.00229 |
| Pairwise error covariance | Welch (as reported) | 0.03606 | 0.1029 | -0.00237 |
| Pairwise error covariance | Paired by question | 0.02263 | 0.0658 | -0.00237 |

## Decomposition (identity: ensemble = 0.1 x indiv + 0.9 x errprod)

| Group | Ensemble Brier | Mean indiv Brier | Mean pairwise err-prod | Identity gap |
|---|---|---|---|---|
| control_a | 0.09327 | 0.09327 | 0.09327 | 0.0 |
| control_b | 0.08569 | 0.08644 | 0.08561 | 0.0 |
| treatment | 0.08884 | 0.09049 | 0.08866 | 0.0 |
| placebo | 0.09113 | 0.09204 | 0.09103 | 0.0 |

Interpretation: the null on the pure-cognitive contrast is informative only
for effects at least as large as the MDE; smaller real improvements would not
have been detected by this design.
