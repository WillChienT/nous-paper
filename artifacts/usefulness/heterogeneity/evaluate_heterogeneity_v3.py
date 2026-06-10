"""Four-group heterogeneity analysis: Control-A, Control-B, Placebo, Treatment.

Adds a length-matched Placebo group (generic preamble, same seeds as Treatment)
to isolate prompt length effects from cognitive content effects.

Decomposition:
  stochastic effect   = Control-B − Control-A   (diff seeds, default ~720-char prompt)
  length effect       = Placebo   − Control-B   (generic 4387-char preamble vs short default)
  pure cognitive      = Treatment − Placebo     (persona prompts vs length-matched generic)
  old cognitive_net   = Treatment − Control-B   (original confounded estimate, kept for reference)

The key question: is pure cognitive (Treatment − Placebo) significantly > 0?
  - If yes: cognitive content has a genuine effect beyond length alone
  - If no:  the original +0.0034 was largely a length artifact

Outputs:
  results_heterogeneity_v3.json    — all metrics for four groups + four pairs
  results_heterogeneity_v3.md      — report for Will with four-group decomposition table
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

# Existing three-group data files
V1_RESULTS_PATH = BASE_DIR / "results_heterogeneity.json"
CB_RESULTS_PATH = BASE_DIR / "results_heterogeneity_controlb.json"
# New placebo group
PLACEBO_RESULTS_PATH = BASE_DIR / "results_heterogeneity_placebo.json"

EVAL_PATH = BASE_DIR / "data" / "eval_set_mantic_baseline.jsonl"
PLACEBO_PREAMBLE_PATH = BASE_DIR / "data" / "placebo_preamble.txt"

OUT_JSON = BASE_DIR / "results_heterogeneity_v3.json"
OUT_REPORT = BASE_DIR / "results_heterogeneity_v3.md"

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
    brier_diff, brier_ci_lo, brier_ci_hi = bootstrap_diff(a_brier, b_brier, N_BOOTSTRAP, BOOTSTRAP_SEED)
    jsd_diff, jsd_ci_lo, jsd_ci_hi = bootstrap_diff(a_jsd, b_jsd, N_BOOTSTRAP, BOOTSTRAP_SEED)
    _, errcov_ci_lo, errcov_ci_hi = bootstrap_diff(a_errcov, b_errcov, N_BOOTSTRAP, BOOTSTRAP_SEED)

    p_brier = float(stats.ttest_ind(a_brier, b_brier, equal_var=False)[1])
    p_jsd = float(stats.ttest_ind(a_jsd, b_jsd, equal_var=False)[1])
    p_errcov = float(stats.ttest_ind(a_errcov, b_errcov, equal_var=False)[1])

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


def _sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(
    group_metrics: dict,
    pair_metrics: list[dict],
    N: int,
    placebo_preamble_len: int,
) -> None:
    ca = group_metrics["control_a"]
    cb = group_metrics["control_b"]
    pl = group_metrics["placebo"]
    trt = group_metrics["treatment"]

    stoch = next(p for p in pair_metrics if "stochastic" in p["comparison"])
    length_eff = next(p for p in pair_metrics if "length_effect" in p["comparison"])
    pure_cog = next(p for p in pair_metrics if "pure_cognitive" in p["comparison"])
    old_cog = next(p for p in pair_metrics if "cognitive_net" in p["comparison"])

    pure_jsd = pure_cog["mean_pairwise_jsd"]["diff_b_minus_a"]
    pure_jsd_p = pure_cog["mean_pairwise_jsd"]["welch_p"]
    pure_jsd_d = pure_cog["mean_pairwise_jsd"]["cohen_d"]
    pure_jsd_ci = pure_cog["mean_pairwise_jsd"]["bootstrap_95ci"]

    length_jsd = length_eff["mean_pairwise_jsd"]["diff_b_minus_a"]
    length_jsd_p = length_eff["mean_pairwise_jsd"]["welch_p"]

    stoch_jsd = stoch["mean_pairwise_jsd"]["diff_b_minus_a"]

    # Determine verdict
    if pure_jsd_p < 0.05 and pure_jsd > 0:
        verdict = "GENUINE_EFFECT"
        verdict_text = (
            f"Pure cognitive effect (Treatment − Placebo) is statistically significant: "
            f"JSD Δ = {pure_jsd:+.4f} (p = {pure_jsd_p:.3f}, d = {pure_jsd_d:.3f}). "
            f"Cognitive content contributes genuine diversity beyond length alone. "
            f"Section 6.3 should report this more conservative number ({pure_jsd:+.4f}) "
            f"rather than the confounded estimate ({old_cog['mean_pairwise_jsd']['diff_b_minus_a']:+.4f})."
        )
    else:
        verdict = "LENGTH_ARTIFACT"
        verdict_text = (
            f"Pure cognitive effect (Treatment − Placebo) is NOT statistically significant: "
            f"JSD Δ = {pure_jsd:+.4f} (p = {pure_jsd_p:.3f}, d = {pure_jsd_d:.3f}). "
            f"The original cognitive_net estimate of "
            f"{old_cog['mean_pairwise_jsd']['diff_b_minus_a']:+.4f} appears to be substantially "
            f"a prompt length artifact. Length effect (Placebo − Control-B) = {length_jsd:+.4f} "
            f"(p = {length_jsd_p:.3f}). Section 6.3 must be reframed: cognitive content does not "
            f"produce detectable diversity beyond what equal-length generic text produces."
        )

    lines = [
        "# Heterogeneity Analysis V3 — Four-Group Decomposition (Audit 2026-05-27)",
        "",
        "## Audit Note",
        "",
        "This analysis adds a **Placebo** group to the original three-group design to isolate "
        "prompt length effects from cognitive content effects. The original cognitive_net estimate "
        "(Treatment − Control-B, JSD +0.0034) was confounded because Treatment prompts are ~4350 "
        "chars while Control-B uses a ~720-char default prompt — a 6x length difference. "
        "The Placebo group uses the same seeds as Treatment (43–52) but replaces the persona "
        "prompt with a length-matched generic forecasting methodology guide "
        f"({placebo_preamble_len} chars, cognitively neutral). "
        "Pure cognitive effect = Treatment − Placebo isolates cognitive content from length.",
        "",
        "## Verdict",
        "",
        verdict_text,
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Four-Group Metrics",
        "",
        "| Group | Prompt | Seeds | Ensemble Brier | Mean JSD |",
        "|-------|--------|-------|---------------|----------|",
        f"| Control-A | default ~720 chars | seed=42 (same) | {ca['mean_brier']:.4f} | {ca['mean_jsd']:.4f} |",
        f"| Control-B | default ~720 chars | seeds 53–62 (diff) | {cb['mean_brier']:.4f} | {cb['mean_jsd']:.4f} |",
        f"| Placebo | generic {placebo_preamble_len}-char preamble | seeds 43–52 (diff) | {pl['mean_brier']:.4f} | {pl['mean_jsd']:.4f} |",
        f"| Treatment | persona 4043–4673 chars | seeds 43–52 (diff) | {trt['mean_brier']:.4f} | {trt['mean_jsd']:.4f} |",
        "",
        "## Four-Pair Decomposition",
        "",
        "| Effect | Comparison | JSD Δ | 95% CI | p | Cohen's d | Brier Δ | p |",
        "|--------|-----------|-------|--------|---|-----------|---------|---|",
    ]

    def _row(label: str, pair: dict) -> str:
        jsd = pair["mean_pairwise_jsd"]
        brier = pair["ensemble_brier"]
        return (
            f"| {label} | {pair['comparison']} "
            f"| {jsd['diff_b_minus_a']:+.4f} "
            f"| [{jsd['bootstrap_95ci'][0]:.4f}, {jsd['bootstrap_95ci'][1]:.4f}] "
            f"| {jsd['welch_p']:.3f} {_sig(jsd['welch_p'])} "
            f"| {jsd['cohen_d']:.3f} "
            f"| {brier['diff_b_minus_a']:+.4f} "
            f"| {brier['welch_p']:.3f} {_sig(brier['welch_p'])} |"
        )

    lines.append(_row("stochastic", stoch))
    lines.append(_row("length effect", length_eff))
    lines.append(_row("**pure cognitive**", pure_cog))
    lines.append(_row("old cognitive_net (confounded)", old_cog))
    lines += [
        "",
        "ns = p ≥ 0.05, * = p < 0.05, ** = p < 0.01, *** = p < 0.001",
        "",
        "## Decomposition Summary",
        "",
    ]

    total_jsd = trt["mean_jsd"] - ca["mean_jsd"]
    stoch_pct = (stoch_jsd / total_jsd * 100) if abs(total_jsd) > 1e-6 else float("nan")
    length_pct = (length_jsd / total_jsd * 100) if abs(total_jsd) > 1e-6 else float("nan")
    pure_pct = (pure_jsd / total_jsd * 100) if abs(total_jsd) > 1e-6 else float("nan")

    lines.append(
        f"Total JSD gain (Treatment − Control-A): {total_jsd:+.4f}. "
        f"Approximate attribution: "
        f"stochastic {stoch_pct:.0f}% ({stoch_jsd:+.4f}), "
        f"length {length_pct:.0f}% ({length_jsd:+.4f}), "
        f"pure cognitive {pure_pct:.0f}% ({pure_jsd:+.4f}). "
        f"Note: these components add up approximately but not exactly because "
        f"Placebo−ControlB and Treatment−Placebo use different seed sets from ControlB−ControlA."
    )

    lines += [
        "",
        "## Implications for Section 6.3",
        "",
    ]

    if verdict == "GENUINE_EFFECT":
        lines += [
            "Cognitive content has a genuine effect beyond prompt length. Section 6.3 should:",
            "",
            "- Report the four-group decomposition table.",
            "- State the pure cognitive effect as the primary claim: "
              f"Treatment − Placebo JSD = {pure_jsd:+.4f} (p = {pure_jsd_p:.3f}, d = {pure_jsd_d:.3f}).",
            "- Note that this is more conservative than the original cognitive_net estimate "
              f"({old_cog['mean_pairwise_jsd']['diff_b_minus_a']:+.4f}), because the placebo control "
              "absorbs the length contribution.",
            "- Acknowledge that length effect accounts for "
              f"approximately {length_pct:.0f}% of the total JSD gain.",
        ]
    else:
        lines += [
            "The original +0.0034 cognitive_net was substantially a length artifact. Section 6.3 must:",
            "",
            "- Report the four-group decomposition table.",
            "- Acknowledge honestly that the length-matched control eliminates the apparent cognitive effect.",
            "- State that prompt length (not cognitive content structure) is the primary driver "
              "of output diversity in this experiment.",
            "- Frame as a negative result with methodological value: the experiment identifies "
              "prompt length as a confound that future studies must control for.",
            "- The stochastic baseline (Control-B − Control-A) finding is unaffected.",
        ]

    lines += [
        "",
        "## N and Limitations Note",
        "",
        f"N = {N} questions after listwise deletion (not 50 — audit HIGH#12). "
        "All four groups used 10 agents per question. "
        "Placebo preamble: generic forecasting methodology guide, "
        f"{placebo_preamble_len} chars (within the 4043–4673 char range of persona prompts). "
        "Placebo and Treatment use the same seeds (43–52) to make them directly comparable.",
        "",
    ]

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {OUT_REPORT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading results (three existing groups + placebo)...")
    with open(V1_RESULTS_PATH) as f:
        v1_records = json.load(f)
    with open(CB_RESULTS_PATH) as f:
        cb_records = json.load(f)
    with open(PLACEBO_RESULTS_PATH) as f:
        placebo_records = json.load(f)

    with open(PLACEBO_PREAMBLE_PATH) as f:
        placebo_preamble_len = len(f.read().strip())

    all_records = v1_records + cb_records + placebo_records
    eval_set = _load_eval_set()

    # Group by market_id and group label
    by_q: dict[str, dict[str, list]] = {}
    for r in all_records:
        mid = r["market_id"]
        if mid not in by_q:
            by_q[mid] = {"control": [], "control_b": [], "treatment": [], "placebo": []}
        group = r["group"]
        if group not in by_q[mid]:
            by_q[mid][group] = []
        if not r["parse_failed"] and r["prob"] is not None:
            by_q[mid][group].append(r)

    # Filter to questions with complete data for all four groups (10 agents each)
    complete: dict[str, dict] = {}
    for mid, groups in by_q.items():
        ca = sorted(groups.get("control", []), key=lambda r: r["agent_idx"])
        cb = sorted(groups.get("control_b", []), key=lambda r: r["agent_idx"])
        trt = sorted(groups.get("treatment", []), key=lambda r: r["agent_idx"])
        pl = sorted(groups.get("placebo", []), key=lambda r: r["agent_idx"])
        if len(ca) == 10 and len(cb) == 10 and len(trt) == 10 and len(pl) == 10:
            complete[mid] = {"control_a": ca, "control_b": cb, "treatment": trt, "placebo": pl}

    N = len(complete)
    print(f"Complete questions (all four groups): {N}/{len(by_q)}")

    pairs = list(combinations(range(10), 2))

    per_q_brier: dict[str, list[float]] = {"ca": [], "cb": [], "trt": [], "pl": []}
    per_q_jsd: dict[str, list[float]] = {"ca": [], "cb": [], "trt": [], "pl": []}
    per_q_errcov: dict[str, list[float]] = {"ca": [], "cb": [], "trt": [], "pl": []}

    group_agent_probs: dict[str, list[list[float]]] = {
        "ca": [[] for _ in range(10)],
        "cb": [[] for _ in range(10)],
        "trt": [[] for _ in range(10)],
        "pl": [[] for _ in range(10)],
    }
    group_agent_errors: dict[str, list[list[float]]] = {
        "ca": [[] for _ in range(10)],
        "cb": [[] for _ in range(10)],
        "trt": [[] for _ in range(10)],
        "pl": [[] for _ in range(10)],
    }

    for mid, groups in complete.items():
        q = eval_set[mid]
        actual = 1 if q["resolved_outcome"] == "Yes" else 0

        for key_label, short_key in [
            ("control_a", "ca"),
            ("control_b", "cb"),
            ("treatment", "trt"),
            ("placebo", "pl"),
        ]:
            probs = [r["prob"] for r in groups[key_label]]
            stats_q = _per_question_stats(probs, actual)
            per_q_brier[short_key].append(stats_q["brier"])
            per_q_jsd[short_key].append(stats_q["mean_jsd"])
            errors = stats_q["errors"]
            per_q_errcov[short_key].append(
                float(np.mean([errors[i] * errors[j] for i, j in pairs]))
            )
            for local_i, (p, e) in enumerate(zip(probs, errors)):
                group_agent_probs[short_key][local_i].append(p)
                group_agent_errors[short_key][local_i].append(e)

    print("Computing group metrics...")
    from evaluate_heterogeneity_v2 import _pairwise_corr_across_questions

    group_metrics = {}
    for label, short_key in [("control_a", "ca"), ("control_b", "cb"),
                               ("treatment", "trt"), ("placebo", "pl")]:
        group_metrics[label] = {
            "mean_brier": round(float(np.mean(per_q_brier[short_key])), 4),
            "mean_jsd": round(float(np.mean(per_q_jsd[short_key])), 4),
            "err_corr": round(_pairwise_corr_across_questions(group_agent_errors[short_key]), 4),
            "prob_corr": round(_pairwise_corr_across_questions(group_agent_probs[short_key]), 4),
        }

    print("Bootstrapping confidence intervals for four pairs...")
    pair_metrics = [
        _compare_pair(
            per_q_brier["ca"], per_q_brier["cb"],
            per_q_jsd["ca"], per_q_jsd["cb"],
            per_q_errcov["ca"], per_q_errcov["cb"],
            "stochastic (Control-B − Control-A)",
        ),
        _compare_pair(
            per_q_brier["cb"], per_q_brier["pl"],
            per_q_jsd["cb"], per_q_jsd["pl"],
            per_q_errcov["cb"], per_q_errcov["pl"],
            "length_effect (Placebo − Control-B)",
        ),
        _compare_pair(
            per_q_brier["pl"], per_q_brier["trt"],
            per_q_jsd["pl"], per_q_jsd["trt"],
            per_q_errcov["pl"], per_q_errcov["trt"],
            "pure_cognitive (Treatment − Placebo)",
        ),
        _compare_pair(
            per_q_brier["cb"], per_q_brier["trt"],
            per_q_jsd["cb"], per_q_jsd["trt"],
            per_q_errcov["cb"], per_q_errcov["trt"],
            "cognitive_net (Treatment − Control-B) [confounded reference]",
        ),
    ]

    output = {
        "n_questions": N,
        "n_agents_per_group": 10,
        "model": "qwen3.5:122b",
        "placebo_preamble_chars": placebo_preamble_len,
        "group_metrics": group_metrics,
        "pair_comparisons": pair_metrics,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"V3 metrics saved to {OUT_JSON}")

    _write_report(group_metrics, pair_metrics, N, placebo_preamble_len)
    print("Done.")


if __name__ == "__main__":
    main()
