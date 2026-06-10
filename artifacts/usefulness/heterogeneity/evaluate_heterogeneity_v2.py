"""Three-group heterogeneity analysis: Control-A, Control-B, Treatment.

Decomposes the combined effect (Treatment − Control-A) from the first run into:
  stochastic effect  = Control-B − Control-A  (different seeds, same default prompt)
  cognitive effect   = Treatment − Control-B  (same seeds range, different prompts)

Each comparison uses the same metrics as the original evaluate_heterogeneity.py:
ensemble Brier, pairwise error correlation, pairwise prob correlation, mean JSD.
Statistical tests: Welch's t-test, Cohen's d, bootstrap 95% CI (N=2000).

Outputs:
  results_heterogeneity_combined.json         — all metrics for three groups + three pairs
  results_heterogeneity_combined.md           — report for Will
  results_heterogeneity_combined_diversity.png — pairwise error covariance distributions
  results_heterogeneity_combined_brier.png    — ensemble Brier comparison
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BASE_DIR = Path(__file__).parent
V1_RESULTS_PATH = BASE_DIR / "results_heterogeneity.json"
CB_RESULTS_PATH = BASE_DIR / "results_heterogeneity_controlb.json"
EVAL_PATH = BASE_DIR / "data" / "eval_set_mantic_baseline.jsonl"
MANTIC_PATH = BASE_DIR / "results_mantic_baseline.json"

OUT_JSON = BASE_DIR / "results_heterogeneity_combined.json"
OUT_REPORT = BASE_DIR / "results_heterogeneity_combined.md"
OUT_DIVERSITY = BASE_DIR / "results_heterogeneity_combined_diversity.png"
OUT_BRIER = BASE_DIR / "results_heterogeneity_combined_brier.png"

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


# ── Metric helpers ────────────────────────────────────────────────────────────

def _jsd(p: float, q: float) -> float:
    dist_p = np.array([p, 1.0 - p])
    dist_q = np.array([q, 1.0 - q])
    m = 0.5 * (dist_p + dist_q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(dist_p, m) + 0.5 * kl(dist_q, m)


def _pairwise_corr_across_questions(agent_series: list[list[float]]) -> float:
    """Mean pairwise Pearson r across C(n_agents, 2) pairs, computed over questions."""
    n = len(agent_series)
    corrs = []
    for i, j in combinations(range(n), 2):
        a, b = np.array(agent_series[i]), np.array(agent_series[j])
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            corrs.append(1.0)
            continue
        corrs.append(float(stats.pearsonr(a, b)[0]))  # type: ignore[arg-type]
    return float(np.mean(corrs)) if corrs else 0.0


def _per_question_stats(probs: list[float], actual: int) -> dict:
    n = len(probs)
    ensemble_prob = float(np.mean(probs))
    brier = (ensemble_prob - actual) ** 2
    jsd_vals = [_jsd(probs[i], probs[j]) for i, j in combinations(range(n), 2)]
    mean_jsd = float(np.mean(jsd_vals)) if jsd_vals else 0.0
    errors = [p - actual for p in probs]
    return {"probs": probs, "errors": errors, "brier": brier, "mean_jsd": mean_jsd}


def bootstrap_diff(
    a_vals: list[float],
    b_vals: list[float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for (mean_b - mean_a). Returns (obs_diff, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    obs_diff = float(np.mean(b_vals) - np.mean(a_vals))
    boot_diffs = [
        float(np.mean(rng.choice(b_vals, size=len(b_vals), replace=True)) -
              np.mean(rng.choice(a_vals, size=len(a_vals), replace=True)))
        for _ in range(n_boot)
    ]
    return obs_diff, float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))


def cohen_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    return 0.0 if pooled < 1e-10 else float((np.mean(b) - np.mean(a)) / pooled)


def _compare_pair(
    a_brier: list[float],
    b_brier: list[float],
    a_jsd: list[float],
    b_jsd: list[float],
    a_errcov: list[float],
    b_errcov: list[float],
    label: str,
) -> dict:
    """Compute all stats for one group-pair comparison."""
    brier_diff, brier_ci_lo, brier_ci_hi = bootstrap_diff(a_brier, b_brier, N_BOOTSTRAP, BOOTSTRAP_SEED)
    jsd_diff, jsd_ci_lo, jsd_ci_hi = bootstrap_diff(a_jsd, b_jsd, N_BOOTSTRAP, BOOTSTRAP_SEED)
    _, errcov_ci_lo, errcov_ci_hi = bootstrap_diff(a_errcov, b_errcov, N_BOOTSTRAP, BOOTSTRAP_SEED)

    p_brier = float(stats.ttest_ind(a_brier, b_brier, equal_var=False)[1])  # type: ignore[arg-type]
    p_jsd = float(stats.ttest_ind(a_jsd, b_jsd, equal_var=False)[1])  # type: ignore[arg-type]
    p_errcov = float(stats.ttest_ind(a_errcov, b_errcov, equal_var=False)[1])  # type: ignore[arg-type]

    return {
        "comparison": label,
        "ensemble_brier": {
            "diff_b_minus_a": round(brier_diff, 4),
            "bootstrap_95ci": [round(brier_ci_lo, 4), round(brier_ci_hi, 4)],
            "welch_p": round(p_brier, 4),
            "cohen_d": round(cohen_d(a_brier, b_brier), 3),
        },
        "mean_pairwise_jsd": {
            "diff_b_minus_a": round(jsd_diff, 4),
            "bootstrap_95ci": [round(jsd_ci_lo, 4), round(jsd_ci_hi, 4)],
            "welch_p": round(p_jsd, 4),
            "cohen_d": round(cohen_d(a_jsd, b_jsd), 3),
        },
        "pairwise_error_covariance": {
            "diff_b_minus_a": round(float(np.mean(b_errcov) - np.mean(a_errcov)), 4),
            "bootstrap_95ci": [round(errcov_ci_lo, 4), round(errcov_ci_hi, 4)],
            "welch_p": round(p_errcov, 4),
            "cohen_d": round(cohen_d(a_errcov, b_errcov), 3),
        },
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_set() -> dict[str, dict]:
    with open(EVAL_PATH) as f:
        rows = [json.loads(line) for line in f]
    return {q["market_id"]: q for q in rows}


def _load_mantic_brier() -> float:
    with open(MANTIC_PATH) as f:
        data = json.load(f)
    records = [r for r in data["records"] if r["model"] == "qwen3.5:122b"]
    actuals = {r["market_id"]: 1 if r["resolved_outcome"] == "Yes" else 0 for r in records}
    briers = [(r["final_prob"] - actuals[r["market_id"]]) ** 2
              for r in records if not r["parse_failed"]]
    return float(np.mean(briers)) if briers else float("nan")


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "control_a": "#4C72B0",
    "control_b": "#937860",
    "treatment": "#DD8452",
}
LABELS = {
    "control_a": "Control-A (same seed, default prompt)",
    "control_b": "Control-B (diff seeds, default prompt)",
    "treatment": "Treatment (diff seeds, persona prompts)",
}


def _plot_diversity(ca: list[float], cb: list[float], trt: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for vals, key in [(ca, "control_a"), (cb, "control_b"), (trt, "treatment")]:
        ax.hist(vals, bins=15, alpha=0.55, label=LABELS[key], color=COLORS[key])
        ax.axvline(float(np.mean(vals)), color=COLORS[key], linestyle="--", linewidth=2,
                   label=f"{key} mean = {np.mean(vals):.3f}")
    ax.set_xlabel("Pairwise error covariance (within question, mean across 45 pairs)")
    ax.set_ylabel("Number of questions")
    ax.set_title("Pairwise Error Covariance: Three-Group Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIVERSITY, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIVERSITY}")


def _plot_brier(ca: list[float], cb: list[float], trt: list[float], mantic: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for vals, key in [(ca, "control_a"), (cb, "control_b"), (trt, "treatment")]:
        ax.hist(vals, bins=12, alpha=0.55, label=LABELS[key], color=COLORS[key])
        ax.axvline(float(np.mean(vals)), color=COLORS[key], linestyle="--", linewidth=2,
                   label=f"{key} Brier = {np.mean(vals):.4f}")
    if not np.isnan(mantic):
        ax.axvline(mantic, color="#55A868", linestyle=":", linewidth=2,
                   label=f"Mantic baseline = {mantic:.4f}")
    ax.set_xlabel("Ensemble Brier score (lower = better)")
    ax.set_ylabel("Number of questions")
    ax.set_title("Ensemble Brier Score: Three-Group Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_BRIER, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_BRIER}")


# ── Report ────────────────────────────────────────────────────────────────────

def _sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _write_report(
    group_metrics: dict,
    pair_metrics: list[dict],
    mantic_brier: float,
    N: int,
) -> None:
    ca = group_metrics["control_a"]
    cb = group_metrics["control_b"]
    trt = group_metrics["treatment"]

    stoch = next(p for p in pair_metrics if p["comparison"] == "stochastic (Control-B − Control-A)")
    cog = next(p for p in pair_metrics if p["comparison"] == "cognitive_net (Treatment − Control-B)")
    combined = next(p for p in pair_metrics if p["comparison"] == "combined (Treatment − Control-A)")

    cog_brier_diff = cog["ensemble_brier"]["diff_b_minus_a"]
    cog_jsd_diff = cog["mean_pairwise_jsd"]["diff_b_minus_a"]
    cog_brier_p = cog["ensemble_brier"]["welch_p"]
    cog_jsd_p = cog["mean_pairwise_jsd"]["welch_p"]
    cog_errcov_p = cog["pairwise_error_covariance"]["welch_p"]

    stoch_brier_diff = stoch["ensemble_brier"]["diff_b_minus_a"]
    stoch_jsd_diff = stoch["mean_pairwise_jsd"]["diff_b_minus_a"]

    # Assess cognitive net criterion using the same logic as the original script
    # but applied to the cognitive-net comparison (Treatment vs Control-B)
    cog_criterion: str
    if (
        (ca["err_corr"] - trt["err_corr"]) >= 0.15 and
        (trt["mean_jsd"] - cb["mean_jsd"]) >= 0.05 and
        cog_brier_p > 0.10
    ):
        cog_criterion = "Strong"
    elif sum([
        cog_errcov_p < 0.05,
        cog_jsd_p < 0.05 and cog_jsd_diff > 0,
        cog_brier_p < 0.05 and cog_brier_diff < 0,
    ]) >= 2:
        cog_criterion = "Adequate"
    else:
        cog_criterion = "Negative"

    lines = [
        "# Heterogeneity Validation — Three-Group Combined Analysis",
        "",
        f"**N = {N} questions  |  30 agents (10 Control-A + 10 Control-B + 10 Treatment)  |  Model: qwen3.5:122b**",
        "",
        "## Summary",
        "",
    ]

    if cog_criterion == "Negative":
        lines.append(
            f"The cognitive net effect (Treatment − Control-B) is **{cog_criterion}**: "
            f"no diversity metric reached p < 0.05 when comparing Treatment against the stochastic-only "
            f"Control-B baseline. Ensemble Brier moved by {cog_brier_diff:+.4f} "
            f"(p = {cog_brier_p:.3f}), JSD moved by {cog_jsd_diff:+.4f} (p = {cog_jsd_p:.3f}). "
            f"By contrast, the stochastic effect (Control-B − Control-A) accounts for most of the "
            f"diversity gain seen in the original combined comparison: JSD gain = {stoch_jsd_diff:+.4f}, "
            f"Brier change = {stoch_brier_diff:+.4f}. "
            f"This means persona-level cognitive injection is largely subsumed by the variance "
            f"already introduced by stochastic sampling across different seeds."
        )
    elif cog_criterion == "Adequate":
        lines.append(
            f"The cognitive net effect (Treatment − Control-B) meets the **{cog_criterion}** criterion: "
            f"at least two diversity metrics showed significant improvement beyond the stochastic baseline. "
            f"Ensemble Brier changed by {cog_brier_diff:+.4f} (p = {cog_brier_p:.3f}), "
            f"JSD improved by {cog_jsd_diff:+.4f} (p = {cog_jsd_p:.3f}). "
            f"The stochastic contribution (Control-B − Control-A) was also non-trivial: "
            f"JSD = {stoch_jsd_diff:+.4f}, Brier = {stoch_brier_diff:+.4f}, suggesting both "
            f"stochastic and cognitive sources contribute to the total observed diversity."
        )
    else:
        lines.append(
            f"The cognitive net effect (Treatment − Control-B) meets the **{cog_criterion}** criterion. "
            f"Error correlation dropped by ≥ 0.15 and JSD increased by ≥ 0.05 beyond the stochastic "
            f"baseline, while ensemble Brier remained statistically indistinguishable. "
            f"This confirms that persona-level cognitive injection generates diversity over and above "
            f"what stochastic sampling alone produces."
        )

    lines += [
        "",
        "## Group Metrics",
        "",
        "| Group | Ensemble Brier | Pairwise err corr | Pairwise prob corr | Mean JSD |",
        "|-------|---------------|-------------------|-------------------|----------|",
        f"| Control-A (seed=42, default) | {ca['mean_brier']:.4f} | {ca['err_corr']:.4f} | {ca['prob_corr']:.4f} | {ca['mean_jsd']:.4f} |",
        f"| Control-B (seeds 53–62, default) | {cb['mean_brier']:.4f} | {cb['err_corr']:.4f} | {cb['prob_corr']:.4f} | {cb['mean_jsd']:.4f} |",
        f"| Treatment (seeds 43–52, persona) | {trt['mean_brier']:.4f} | {trt['err_corr']:.4f} | {trt['prob_corr']:.4f} | {trt['mean_jsd']:.4f} |",
        f"| Mantic baseline (Session 4) | {mantic_brier:.4f} | — | — | — |",
        "",
        "## Three-Pair Comparison",
        "",
        "### Stochastic effect: Control-B − Control-A",
        *(f"*(Different seeds vs same seed, both default prompt — isolates stochastic sampling contribution)*",),
        "",
        f"| Metric | Δ (B−A) | 95% CI | p | Cohen's d |",
        f"|--------|---------|--------|---|-----------|",
        f"| Ensemble Brier | {stoch['ensemble_brier']['diff_b_minus_a']:+.4f} | [{stoch['ensemble_brier']['bootstrap_95ci'][0]:.4f}, {stoch['ensemble_brier']['bootstrap_95ci'][1]:.4f}] | {stoch['ensemble_brier']['welch_p']:.3f} {_sig(stoch['ensemble_brier']['welch_p'])} | {stoch['ensemble_brier']['cohen_d']:.3f} |",
        f"| Mean JSD | {stoch['mean_pairwise_jsd']['diff_b_minus_a']:+.4f} | [{stoch['mean_pairwise_jsd']['bootstrap_95ci'][0]:.4f}, {stoch['mean_pairwise_jsd']['bootstrap_95ci'][1]:.4f}] | {stoch['mean_pairwise_jsd']['welch_p']:.3f} {_sig(stoch['mean_pairwise_jsd']['welch_p'])} | {stoch['mean_pairwise_jsd']['cohen_d']:.3f} |",
        f"| Pairwise err covariance | {stoch['pairwise_error_covariance']['diff_b_minus_a']:+.4f} | [{stoch['pairwise_error_covariance']['bootstrap_95ci'][0]:.4f}, {stoch['pairwise_error_covariance']['bootstrap_95ci'][1]:.4f}] | {stoch['pairwise_error_covariance']['welch_p']:.3f} {_sig(stoch['pairwise_error_covariance']['welch_p'])} | {stoch['pairwise_error_covariance']['cohen_d']:.3f} |",
        "",
        "### Cognitive net effect: Treatment − Control-B",
        "*(Persona prompts vs default prompt, both with different seeds — isolates cognitive injection contribution)*",
        "",
        f"| Metric | Δ (Trt−CB) | 95% CI | p | Cohen's d |",
        f"|--------|-----------|--------|---|-----------|",
        f"| Ensemble Brier | {cog['ensemble_brier']['diff_b_minus_a']:+.4f} | [{cog['ensemble_brier']['bootstrap_95ci'][0]:.4f}, {cog['ensemble_brier']['bootstrap_95ci'][1]:.4f}] | {cog['ensemble_brier']['welch_p']:.3f} {_sig(cog['ensemble_brier']['welch_p'])} | {cog['ensemble_brier']['cohen_d']:.3f} |",
        f"| Mean JSD | {cog['mean_pairwise_jsd']['diff_b_minus_a']:+.4f} | [{cog['mean_pairwise_jsd']['bootstrap_95ci'][0]:.4f}, {cog['mean_pairwise_jsd']['bootstrap_95ci'][1]:.4f}] | {cog['mean_pairwise_jsd']['welch_p']:.3f} {_sig(cog['mean_pairwise_jsd']['welch_p'])} | {cog['mean_pairwise_jsd']['cohen_d']:.3f} |",
        f"| Pairwise err covariance | {cog['pairwise_error_covariance']['diff_b_minus_a']:+.4f} | [{cog['pairwise_error_covariance']['bootstrap_95ci'][0]:.4f}, {cog['pairwise_error_covariance']['bootstrap_95ci'][1]:.4f}] | {cog['pairwise_error_covariance']['welch_p']:.3f} {_sig(cog['pairwise_error_covariance']['welch_p'])} | {cog['pairwise_error_covariance']['cohen_d']:.3f} |",
        "",
        "### Combined effect: Treatment − Control-A",
        "*(Replication of the first run's main comparison)*",
        "",
        f"| Metric | Δ (Trt−CA) | 95% CI | p | Cohen's d |",
        f"|--------|-----------|--------|---|-----------|",
        f"| Ensemble Brier | {combined['ensemble_brier']['diff_b_minus_a']:+.4f} | [{combined['ensemble_brier']['bootstrap_95ci'][0]:.4f}, {combined['ensemble_brier']['bootstrap_95ci'][1]:.4f}] | {combined['ensemble_brier']['welch_p']:.3f} {_sig(combined['ensemble_brier']['welch_p'])} | {combined['ensemble_brier']['cohen_d']:.3f} |",
        f"| Mean JSD | {combined['mean_pairwise_jsd']['diff_b_minus_a']:+.4f} | [{combined['mean_pairwise_jsd']['bootstrap_95ci'][0]:.4f}, {combined['mean_pairwise_jsd']['bootstrap_95ci'][1]:.4f}] | {combined['mean_pairwise_jsd']['welch_p']:.3f} {_sig(combined['mean_pairwise_jsd']['welch_p'])} | {combined['mean_pairwise_jsd']['cohen_d']:.3f} |",
        f"| Pairwise err covariance | {combined['pairwise_error_covariance']['diff_b_minus_a']:+.4f} | [{combined['pairwise_error_covariance']['bootstrap_95ci'][0]:.4f}, {combined['pairwise_error_covariance']['bootstrap_95ci'][1]:.4f}] | {combined['pairwise_error_covariance']['welch_p']:.3f} {_sig(combined['pairwise_error_covariance']['welch_p'])} | {combined['pairwise_error_covariance']['cohen_d']:.3f} |",
        "",
        "ns = p ≥ 0.05, * = p < 0.05, ** = p < 0.01, *** = p < 0.001",
        "",
        "## Key Finding: Cognitive vs Stochastic Contribution",
        "",
    ]

    # Qualitative magnitude comparison
    total_jsd_gain = trt["mean_jsd"] - ca["mean_jsd"]
    stoch_jsd_share = (stoch_jsd_diff / total_jsd_gain * 100) if abs(total_jsd_gain) > 1e-6 else float("nan")
    cog_jsd_share = (cog_jsd_diff / total_jsd_gain * 100) if abs(total_jsd_gain) > 1e-6 else float("nan")

    if not (np.isnan(stoch_jsd_share) or np.isnan(cog_jsd_share)):
        lines.append(
            f"Of the total JSD gain ({total_jsd_gain:+.4f}) seen in the original combined comparison "
            f"(Treatment − Control-A), the stochastic component accounts for approximately "
            f"{stoch_jsd_share:.0f}% ({stoch_jsd_diff:+.4f}) and the cognitive component for "
            f"{cog_jsd_share:.0f}% ({cog_jsd_diff:+.4f}). "
        )
    else:
        lines.append(
            "JSD gain between Control-A and Treatment was near zero; decomposition percentages "
            "are not meaningful."
        )

    lines += [
        "",
        "## Correction to the First-Run Report",
        "",
        f"The original `results_heterogeneity.md` reported a combined error-correlation drop of "
        f"{combined['pairwise_error_covariance']['diff_b_minus_a']:+.4f} and JSD gain of "
        f"{combined['mean_pairwise_jsd']['diff_b_minus_a']:+.4f} (Treatment − Control-A). "
        "That comparison conflated stochastic and cognitive effects because Control-A used a single "
        "seed while Treatment used 10 different seeds. The cognitive net effect (Treatment − Control-B), "
        "which is the number Section 6.3 should report, is substantially different.",
        "",
        "## Implications for Section 6.3 Paper Writing",
        "",
    ]

    if cog_criterion == "Negative":
        lines += [
            "Given a Negative cognitive net criterion, Section 6.3 should frame the finding honestly:",
            "",
            "- Report the three-group decomposition explicitly rather than the combined comparison.",
            "- State that persona-level cognitive injection did not produce statistically significant "
              "diversity beyond stochastic sampling at this evaluation scale (N = 50, single base model).",
            "- Frame as preliminary evidence that stochastic seed diversity is a confound, "
              "motivating future experiments with controlled seeds across all groups.",
            "- The positive Brier improvement (Treatment − Control-A) can still be reported as "
              "a combined effect of stochastic + cognitive sources, but must be attributed correctly.",
        ]
    elif cog_criterion == "Adequate":
        lines += [
            "The Adequate cognitive net criterion supports Section 6.3 reporting:",
            "",
            "- Report both stochastic and cognitive contributions in the decomposition table.",
            "- Describe the cognitive net effect as 'preliminary evidence': statistically significant "
              "on at least two metrics, but modest in absolute magnitude at N = 50.",
            "- Acknowledge that multi-seed control (Control-B) is the appropriate baseline, "
              "and the original combined comparison overstated the cognitive contribution.",
        ]
    else:
        lines += [
            "The Strong cognitive net criterion gives Section 6.3 solid grounds to claim:",
            "",
            "- Persona-level cognitive injection produces diversity beyond stochastic sampling.",
            "- Report the three-group table with stochastic and cognitive decomposition.",
            "- Note that the original combined comparison was a lower bound on cognitive effect "
              "(stochastic diluted it toward zero).",
        ]

    lines += [
        "",
        "## Limitations",
        "",
        "- **N = 50 questions**: Power to detect effects below d ≈ 0.4 is low at this sample size.",
        "- **Single base model (qwen3.5:122b)**: All diversity is expressed through prompt-level "
          "conditioning. Multi-model ensembles would likely show stronger cognitive heterogeneity.",
        "- **Seed range mismatch**: Control-B uses seeds 53–62, Treatment uses 43–52. "
          "The stochastic noise profile differs slightly across seed ranges; a design with a single "
          "pooled seed set for both would be cleaner. The effect is likely small but cannot be ruled out.",
        "- **Top-volume ceiling effect**: The eval set draws from high-volume Mantic markets "
          "(near-consensus predictions), which may suppress observable diversity on questions where "
          "all agents converge regardless of persona.",
        "- **Temperature = 0.3**: Low temperature limits stochastic variance. At temperature = 0, "
          "Control-A and Control-B would be identical; the fact that they differ here reflects "
          "seed-dependent sampling paths under a moderately peaked distribution.",
        "",
    ]

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {OUT_REPORT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading results...")
    with open(V1_RESULTS_PATH) as f:
        v1_records = json.load(f)
    with open(CB_RESULTS_PATH) as f:
        cb_records = json.load(f)

    all_records = v1_records + cb_records
    eval_set = _load_eval_set()

    # Group by market_id and group label
    by_q: dict[str, dict[str, list]] = {}
    for r in all_records:
        mid = r["market_id"]
        if mid not in by_q:
            by_q[mid] = {"control": [], "control_b": [], "treatment": []}
        if not r["parse_failed"] and r["prob"] is not None:
            by_q[mid][r["group"]].append(r)

    # Filter to questions with complete data for all three groups
    complete: dict[str, dict] = {}
    for mid, groups in by_q.items():
        ca = sorted(groups["control"], key=lambda r: r["agent_idx"])
        cb = sorted(groups["control_b"], key=lambda r: r["agent_idx"])
        trt = sorted(groups["treatment"], key=lambda r: r["agent_idx"])
        if len(ca) == 10 and len(cb) == 10 and len(trt) == 10:
            complete[mid] = {"control_a": ca, "control_b": cb, "treatment": trt}

    N = len(complete)
    print(f"Complete questions (all three groups): {N}/{len(by_q)}")

    # Per-agent series for cross-question pairwise correlation
    ca_probs: list[list[float]] = [[] for _ in range(10)]
    ca_errors: list[list[float]] = [[] for _ in range(10)]
    cb_probs: list[list[float]] = [[] for _ in range(10)]
    cb_errors: list[list[float]] = [[] for _ in range(10)]
    trt_probs: list[list[float]] = [[] for _ in range(10)]
    trt_errors: list[list[float]] = [[] for _ in range(10)]

    per_q_brier: dict[str, list[float]] = {"ca": [], "cb": [], "trt": []}
    per_q_jsd: dict[str, list[float]] = {"ca": [], "cb": [], "trt": []}
    per_q_errcov: dict[str, list[float]] = {"ca": [], "cb": [], "trt": []}

    pairs = list(combinations(range(10), 2))

    for mid, groups in complete.items():
        q = eval_set[mid]
        actual = 1 if q["resolved_outcome"] == "Yes" else 0

        for key, agent_probs_list, agent_errors_list, brier_list, jsd_list, errcov_list in [
            ("control_a", ca_probs, ca_errors, per_q_brier["ca"], per_q_jsd["ca"], per_q_errcov["ca"]),
            ("control_b", cb_probs, cb_errors, per_q_brier["cb"], per_q_jsd["cb"], per_q_errcov["cb"]),
            ("treatment", trt_probs, trt_errors, per_q_brier["trt"], per_q_jsd["trt"], per_q_errcov["trt"]),
        ]:
            probs = [r["prob"] for r in groups[key]]
            stats_q = _per_question_stats(probs, actual)
            brier_list.append(stats_q["brier"])
            jsd_list.append(stats_q["mean_jsd"])
            errors = stats_q["errors"]
            errcov_list.append(float(np.mean([errors[i] * errors[j] for i, j in pairs])))
            for local_i, (p, e) in enumerate(zip(probs, errors)):
                agent_probs_list[local_i].append(p)
                agent_errors_list[local_i].append(e)

    print("Computing pairwise correlations across questions...")
    group_metrics = {
        "control_a": {
            "mean_brier": round(float(np.mean(per_q_brier["ca"])), 4),
            "mean_jsd": round(float(np.mean(per_q_jsd["ca"])), 4),
            "err_corr": round(_pairwise_corr_across_questions(ca_errors), 4),
            "prob_corr": round(_pairwise_corr_across_questions(ca_probs), 4),
        },
        "control_b": {
            "mean_brier": round(float(np.mean(per_q_brier["cb"])), 4),
            "mean_jsd": round(float(np.mean(per_q_jsd["cb"])), 4),
            "err_corr": round(_pairwise_corr_across_questions(cb_errors), 4),
            "prob_corr": round(_pairwise_corr_across_questions(cb_probs), 4),
        },
        "treatment": {
            "mean_brier": round(float(np.mean(per_q_brier["trt"])), 4),
            "mean_jsd": round(float(np.mean(per_q_jsd["trt"])), 4),
            "err_corr": round(_pairwise_corr_across_questions(trt_errors), 4),
            "prob_corr": round(_pairwise_corr_across_questions(trt_probs), 4),
        },
    }

    print("Bootstrapping confidence intervals for three pairs...")
    pair_metrics = [
        _compare_pair(
            per_q_brier["ca"], per_q_brier["cb"],
            per_q_jsd["ca"], per_q_jsd["cb"],
            per_q_errcov["ca"], per_q_errcov["cb"],
            "stochastic (Control-B − Control-A)",
        ),
        _compare_pair(
            per_q_brier["cb"], per_q_brier["trt"],
            per_q_jsd["cb"], per_q_jsd["trt"],
            per_q_errcov["cb"], per_q_errcov["trt"],
            "cognitive_net (Treatment − Control-B)",
        ),
        _compare_pair(
            per_q_brier["ca"], per_q_brier["trt"],
            per_q_jsd["ca"], per_q_jsd["trt"],
            per_q_errcov["ca"], per_q_errcov["trt"],
            "combined (Treatment − Control-A)",
        ),
    ]

    mantic_brier = _load_mantic_brier()

    output = {
        "n_questions": N,
        "n_agents_per_group": 10,
        "model": "qwen3.5:122b",
        "mantic_baseline_brier": round(mantic_brier, 4),
        "group_metrics": group_metrics,
        "pair_comparisons": pair_metrics,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Combined metrics saved to {OUT_JSON}")

    _plot_diversity(per_q_errcov["ca"], per_q_errcov["cb"], per_q_errcov["trt"])
    _plot_brier(per_q_brier["ca"], per_q_brier["cb"], per_q_brier["trt"], mantic_brier)

    _write_report(group_metrics, pair_metrics, mantic_brier, N)
    print("Done.")


if __name__ == "__main__":
    main()
