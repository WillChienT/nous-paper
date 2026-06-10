"""
Difficulty-subset re-analysis for Nous Paper V2.

PRE-REGISTERED DIFFICULTY DEFINITIONS (defined before computing any treatment effect):

Primary proxy (no regression-to-mean risk):
  "model uncertainty" = distance of Control-B ensemble mean probability from 0.5
  Uncertain group: distance < 0.2 (i.e., ensemble mean in [0.3, 0.7])
  Certain group:   distance >= 0.2

Secondary proxy (used ONLY for error-correlation outcome, NOT for Brier — regression-to-mean warning):
  "baseline Brier difficulty" = Control-B ensemble Brier, top-half quantile = "hard"

Metric functions are imported/copied from evaluate_heterogeneity_v3.py and
evaluate_heterogeneity_v2.py to ensure comparability with paper numbers.

Outputs:
  injection/v2/results_heterogeneity_difficulty_subset.json
  injection/v2/results_heterogeneity_difficulty_subset.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).parent
EVAL_PATH = BASE_DIR / "data" / "eval_set_mantic_baseline.jsonl"

OUT_JSON = BASE_DIR / "results_heterogeneity_difficulty_subset.json"
OUT_MD   = BASE_DIR / "results_heterogeneity_difficulty_subset.md"

N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 42

# ── Metric helpers (copied from evaluate_heterogeneity_v3.py / v2.py) ─────────

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


def _pairwise_corr_across_questions(agent_series: list[list[float]]) -> float:
    """Mean pairwise Pearson r across C(n_agents, 2) pairs, computed over questions."""
    n = len(agent_series)
    corrs = []
    for i, j in combinations(range(n), 2):
        a, b = np.array(agent_series[i]), np.array(agent_series[j])
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            corrs.append(1.0)
            continue
        corrs.append(float(stats.pearsonr(a, b)[0]))
    return float(np.mean(corrs)) if corrs else 0.0


def cluster_bootstrap_diff(
    a_vals: list[float],
    b_vals: list[float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Cluster-bootstrap by question (paired: same question in both groups).
    Returns (obs_diff, ci_lo, ci_hi, p_value).
    a_vals and b_vals must be same length (one value per question).
    Resamples questions with replacement.
    """
    assert len(a_vals) == len(b_vals), "Must be paired by question"
    n = len(a_vals)
    if n < 6:
        # Not enough for bootstrap — return observed diff with nan CI
        obs = float(np.mean(b_vals) - np.mean(a_vals))
        return obs, float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    a = np.array(a_vals)
    b = np.array(b_vals)
    obs_diff = float(np.mean(b) - np.mean(a))

    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs.append(float(np.mean(b[idx]) - np.mean(a[idx])))

    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))

    # Bootstrap p-value (two-sided): shift distribution to null, then measure
    # fraction as extreme as observed. Clamp to [0, 1].
    boot_arr = np.array(boot_diffs)
    centered_null = boot_arr - float(np.mean(boot_arr))  # shift to H0: mean = 0
    p_val = float(np.mean(np.abs(centered_null) >= abs(obs_diff)))
    p_val = float(min(max(p_val, 0.0), 1.0))

    return obs_diff, ci_lo, ci_hi, p_val


# ── Data loading ───────────────────────────────────────────────────────────────

def load_eval_set() -> dict[str, dict]:
    with open(EVAL_PATH) as f:
        rows = [json.loads(line) for line in f]
    return {q["market_id"]: q for q in rows}


def load_and_group_complete(
    fname_cb: str,
    fname_trt: str,
    eval_set: dict[str, dict],
) -> dict[str, dict]:
    """Load control-B and treatment records; return dict of complete questions
    (both groups have exactly 10 non-failed agents)."""
    with open(fname_cb) as f:
        cb = json.load(f)
    with open(fname_trt) as f:
        trt = json.load(f)

    cb_by_market: dict[str, list] = defaultdict(list)
    for r in cb:
        if not r["parse_failed"] and r["prob"] is not None:
            cb_by_market[r["market_id"]].append(r)

    trt_by_market: dict[str, list] = defaultdict(list)
    for r in trt:
        if not r["parse_failed"] and r["prob"] is not None:
            trt_by_market[r["market_id"]].append(r)

    complete: dict[str, dict] = {}
    for mid in eval_set:
        cb_recs = sorted(cb_by_market[mid], key=lambda r: r["agent_idx"])
        trt_recs = sorted(trt_by_market[mid], key=lambda r: r["agent_idx"])
        if len(cb_recs) == 10 and len(trt_recs) == 10:
            actual = 1 if eval_set[mid]["resolved_outcome"] == "Yes" else 0
            complete[mid] = {
                "controlb": cb_recs,
                "treatment": trt_recs,
                "eval": eval_set[mid],
                "actual": actual,
            }
    return complete


# ── Per-question metric computation ───────────────────────────────────────────

def compute_per_question_metrics(
    complete: dict[str, dict],
) -> dict[str, dict]:
    """For each complete question, compute:
    - cb_ensemble_mean: mean of Control-B probs (used for difficulty proxy)
    - cb_brier: Control-B ensemble Brier
    - cb_dist_from_half: |cb_ensemble_mean - 0.5| (primary difficulty proxy)
    - For each group: brier, jsd, errcov, errcorr_agent_series
    """
    pairs10 = list(combinations(range(10), 2))
    per_q: dict[str, dict] = {}

    for mid, data in complete.items():
        actual = data["actual"]
        cb_probs = [r["prob"] for r in data["controlb"]]
        trt_probs = [r["prob"] for r in data["treatment"]]

        # Control-B stats
        cb_stats = _per_question_stats(cb_probs, actual)
        cb_ensemble_mean = float(np.mean(cb_probs))
        cb_dist_from_half = abs(cb_ensemble_mean - 0.5)

        # Treatment stats
        trt_stats = _per_question_stats(trt_probs, actual)

        # Per-question error covariance (mean of all pairwise e_i * e_j)
        cb_errcov = float(np.mean([
            cb_stats["errors"][i] * cb_stats["errors"][j]
            for i, j in pairs10
        ]))
        trt_errcov = float(np.mean([
            trt_stats["errors"][i] * trt_stats["errors"][j]
            for i, j in pairs10
        ]))

        per_q[mid] = {
            "market_id": mid,
            "question": data["eval"]["question"],
            "category": data["eval"]["category"],
            "actual": actual,
            "cb_ensemble_mean": cb_ensemble_mean,
            "cb_dist_from_half": cb_dist_from_half,
            "cb_brier": cb_stats["brier"],
            "trt_brier": trt_stats["brier"],
            "cb_jsd": cb_stats["mean_jsd"],
            "trt_jsd": trt_stats["mean_jsd"],
            "cb_errcov": cb_errcov,
            "trt_errcov": trt_errcov,
            "cb_errors": cb_stats["errors"],
            "trt_errors": trt_stats["errors"],
        }

    return per_q


def compute_errcorr(errors_list: list[list[float]]) -> float:
    """Agent-level error series: 10 agents x N questions. Return mean pairwise Pearson r."""
    n_agents = len(errors_list[0])  # 10
    n_q = len(errors_list)
    agent_series = [[] for _ in range(n_agents)]
    for q_errors in errors_list:
        for i, e in enumerate(q_errors):
            agent_series[i].append(e)
    return _pairwise_corr_across_questions(agent_series)


# ── Subset analysis ────────────────────────────────────────────────────────────

def analyze_subset(
    subset_ids: list[str],
    per_q: dict[str, dict],
    label: str,
    seed_offset: int = 0,
) -> dict:
    """Compute Treatment − Control-B deltas with cluster-bootstrap CI for a subset."""
    n = len(subset_ids)
    if n == 0:
        return {"n": 0, "error": "empty subset"}

    cb_brier = [per_q[mid]["cb_brier"] for mid in subset_ids]
    trt_brier = [per_q[mid]["trt_brier"] for mid in subset_ids]
    cb_jsd = [per_q[mid]["cb_jsd"] for mid in subset_ids]
    trt_jsd = [per_q[mid]["trt_jsd"] for mid in subset_ids]
    cb_errcov = [per_q[mid]["cb_errcov"] for mid in subset_ids]
    trt_errcov = [per_q[mid]["trt_errcov"] for mid in subset_ids]

    # Error correlation: computed from agent x question matrix across subset
    cb_errors_list = [per_q[mid]["cb_errors"] for mid in subset_ids]
    trt_errors_list = [per_q[mid]["trt_errors"] for mid in subset_ids]

    cb_errcorr = compute_errcorr(cb_errors_list) if n >= 2 else float("nan")
    trt_errcorr = compute_errcorr(trt_errors_list) if n >= 2 else float("nan")
    delta_errcorr = trt_errcorr - cb_errcorr

    # Deltas with cluster-bootstrap CI
    def bs(a, b, offset=0):
        return cluster_bootstrap_diff(a, b, N_BOOTSTRAP, BOOTSTRAP_SEED + seed_offset + offset)

    brier_d, brier_lo, brier_hi, brier_p = bs(cb_brier, trt_brier, 0)
    jsd_d, jsd_lo, jsd_hi, jsd_p = bs(cb_jsd, trt_jsd, 1)
    errcov_d, errcov_lo, errcov_hi, errcov_p = bs(cb_errcov, trt_errcov, 2)

    def fmt(v):
        return round(v, 4) if not np.isnan(v) else None

    result = {
        "subset_label": label,
        "n_questions": n,
        "subset_market_ids": subset_ids,
        "group_means": {
            "cb_brier_mean": round(float(np.mean(cb_brier)), 4),
            "trt_brier_mean": round(float(np.mean(trt_brier)), 4),
            "cb_jsd_mean": round(float(np.mean(cb_jsd)), 4),
            "trt_jsd_mean": round(float(np.mean(trt_jsd)), 4),
            "cb_errcov_mean": round(float(np.mean(cb_errcov)), 4),
            "trt_errcov_mean": round(float(np.mean(trt_errcov)), 4),
            "cb_errcorr": fmt(cb_errcorr),
            "trt_errcorr": fmt(trt_errcorr),
        },
        "treatment_minus_controlb": {
            "brier": {
                "delta": fmt(brier_d), "ci_lo": fmt(brier_lo), "ci_hi": fmt(brier_hi),
                "p": fmt(brier_p),
                "note": "Regression-to-mean risk if uncertain-group was selected by high CB brier"
            },
            "jsd": {
                "delta": fmt(jsd_d), "ci_lo": fmt(jsd_lo), "ci_hi": fmt(jsd_hi),
                "p": fmt(jsd_p),
            },
            "error_covariance": {
                "delta": fmt(errcov_d), "ci_lo": fmt(errcov_lo), "ci_hi": fmt(errcov_hi),
                "p": fmt(errcov_p),
            },
            "error_correlation": {
                "delta": fmt(delta_errcorr),
                "ci_lo": None, "ci_hi": None, "p": None,
                "note": "Scalar computed across all questions in subset; no per-question CI available for corr"
            },
        },
    }

    if n < 6:
        result["bootstrap_warning"] = f"N={n} < 6; bootstrap CIs unreliable, not computed"

    return result


# ── Continuous difficulty-interaction regression ───────────────────────────────

def continuous_interaction(per_q: dict[str, dict]) -> dict:
    """Regress Treatment-minus-ControlB delta on difficulty score (cb_dist_from_half).
    Difficulty score: smaller = harder (more uncertain).
    Outcome: delta_errcov = trt_errcov - cb_errcov
             delta_brier  = trt_brier  - cb_brier  (use dist_from_half difficulty only)
             delta_jsd    = trt_jsd    - cb_jsd
    """
    mids = list(per_q.keys())
    difficulty = np.array([per_q[mid]["cb_dist_from_half"] for mid in mids])
    delta_errcov = np.array([per_q[mid]["trt_errcov"] - per_q[mid]["cb_errcov"] for mid in mids])
    delta_brier  = np.array([per_q[mid]["trt_brier"]  - per_q[mid]["cb_brier"]  for mid in mids])
    delta_jsd    = np.array([per_q[mid]["trt_jsd"]    - per_q[mid]["cb_jsd"]    for mid in mids])

    def ols_with_ci(x, y):
        n = len(x)
        slope, intercept, r, p, se = stats.linregress(x, y)
        t_crit = stats.t.ppf(0.975, df=n - 2)
        ci_lo = slope - t_crit * se
        ci_hi = slope + t_crit * se
        return {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 6),
            "r": round(float(r), 4),
            "r_sq": round(float(r**2), 4),
            "p": round(float(p), 4),
            "se": round(float(se), 6),
            "ci_lo_95": round(float(ci_lo), 6),
            "ci_hi_95": round(float(ci_hi), 6),
            "n": n,
            "interpretation": (
                "slope < 0 means harder questions (small dist_from_half) show larger improvement; "
                "slope > 0 means easier questions benefit more"
            ),
        }

    return {
        "difficulty_proxy": "cb_dist_from_half (|CB_ensemble_mean - 0.5|; smaller = more uncertain/harder)",
        "note_on_brier": (
            "Brier regression uses model-uncertainty difficulty (cb_dist_from_half), "
            "NOT cb_brier, to avoid regression-to-mean artifact"
        ),
        "regression_delta_errcov_on_difficulty": ols_with_ci(difficulty, delta_errcov),
        "regression_delta_brier_on_difficulty":  ols_with_ci(difficulty, delta_brier),
        "regression_delta_jsd_on_difficulty":    ols_with_ci(difficulty, delta_jsd),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_temp(
    fname_cb: str,
    fname_trt: str,
    temp_label: str,
    eval_set: dict[str, dict],
    seed_offset: int = 0,
) -> dict:
    print(f"\n=== {temp_label} ===")
    complete = load_and_group_complete(fname_cb, fname_trt, eval_set)
    n_complete = len(complete)
    print(f"  Complete questions: {n_complete}")

    per_q = compute_per_question_metrics(complete)

    # ── PRE-REGISTER difficulty definitions ────────────────────────────────────
    # Primary proxy: cb_dist_from_half < 0.2 => uncertain (harder)
    UNCERTAIN_THRESHOLD = 0.2

    uncertain_ids = [mid for mid, q in per_q.items() if q["cb_dist_from_half"] < UNCERTAIN_THRESHOLD]
    certain_ids   = [mid for mid, q in per_q.items() if q["cb_dist_from_half"] >= UNCERTAIN_THRESHOLD]
    print(f"  Uncertain group (dist < {UNCERTAIN_THRESHOLD}): N={len(uncertain_ids)}")
    print(f"  Certain group (dist >= {UNCERTAIN_THRESHOLD}): N={len(certain_ids)}")

    # Secondary proxy: brier-based (hard = top half of CB brier)
    brier_vals = [(mid, per_q[mid]["cb_brier"]) for mid in per_q]
    brier_vals_sorted = sorted(brier_vals, key=lambda x: x[1], reverse=True)
    median_brier = float(np.median([v for _, v in brier_vals]))
    hard_brier_ids = [mid for mid, v in brier_vals if v >= median_brier]
    easy_brier_ids = [mid for mid, v in brier_vals if v < median_brier]
    print(f"  Hard-by-brier (CB Brier >= median={median_brier:.4f}): N={len(hard_brier_ids)}")
    print(f"  Easy-by-brier (CB Brier < median): N={len(easy_brier_ids)}")

    # ── Subset analyses ────────────────────────────────────────────────────────
    results_uncertain = analyze_subset(uncertain_ids, per_q, f"uncertain_primary_{temp_label}", seed_offset)
    results_certain   = analyze_subset(certain_ids,   per_q, f"certain_primary_{temp_label}", seed_offset + 100)
    results_hard_brier = analyze_subset(hard_brier_ids, per_q, f"hard_by_brier_{temp_label}_errcov_only", seed_offset + 200)
    results_easy_brier = analyze_subset(easy_brier_ids, per_q, f"easy_by_brier_{temp_label}_errcov_only", seed_offset + 300)
    results_all = analyze_subset(list(per_q.keys()), per_q, f"all_complete_{temp_label}", seed_offset + 400)

    # ── Continuous interaction ─────────────────────────────────────────────────
    interaction = continuous_interaction(per_q)

    # Collect per-question details for output
    per_q_out = []
    for mid in sorted(per_q.keys()):
        q = per_q[mid]
        per_q_out.append({
            "market_id": mid,
            "question": q["question"],
            "category": q["category"],
            "actual": q["actual"],
            "cb_ensemble_mean": round(q["cb_ensemble_mean"], 4),
            "cb_dist_from_half": round(q["cb_dist_from_half"], 4),
            "cb_brier": round(q["cb_brier"], 4),
            "trt_brier": round(q["trt_brier"], 4),
            "delta_brier": round(q["trt_brier"] - q["cb_brier"], 4),
            "cb_jsd": round(q["cb_jsd"], 4),
            "trt_jsd": round(q["trt_jsd"], 4),
            "delta_jsd": round(q["trt_jsd"] - q["cb_jsd"], 4),
            "cb_errcov": round(q["cb_errcov"], 4),
            "trt_errcov": round(q["trt_errcov"], 4),
            "delta_errcov": round(q["trt_errcov"] - q["cb_errcov"], 4),
            "uncertain_primary": mid in uncertain_ids,
            "hard_by_brier": mid in hard_brier_ids,
        })

    return {
        "temperature": temp_label,
        "n_complete": n_complete,
        "difficulty_definitions_preregistered": {
            "primary_proxy": "cb_dist_from_half = |mean(CB_probs) - 0.5|",
            "uncertain_threshold": UNCERTAIN_THRESHOLD,
            "uncertain_definition": f"cb_dist_from_half < {UNCERTAIN_THRESHOLD} (CB ensemble mean in (0.3, 0.7))",
            "certain_definition": f"cb_dist_from_half >= {UNCERTAIN_THRESHOLD}",
            "secondary_proxy_errcov_only": "cb_brier >= median(all_cb_brier) = hard",
            "secondary_proxy_warning": (
                "REGRESSION-TO-MEAN WARNING: Using cb_brier to select hard questions, "
                "then observing Brier improvement, creates spurious improvement artifact. "
                "Secondary proxy is ONLY used for error-covariance outcome."
            ),
            "n_uncertain": len(uncertain_ids),
            "n_certain": len(certain_ids),
            "n_hard_brier": len(hard_brier_ids),
            "n_easy_brier": len(easy_brier_ids),
            "median_cb_brier": round(median_brier, 4),
        },
        "subset_analyses": {
            "uncertain_primary": results_uncertain,
            "certain_primary": results_certain,
            "hard_by_brier_errcov_only": results_hard_brier,
            "easy_by_brier_errcov_only": results_easy_brier,
            "all_complete": results_all,
        },
        "continuous_interaction": interaction,
        "per_question_details": per_q_out,
    }


def main():
    print("Loading eval set...")
    eval_set = load_eval_set()
    print(f"Eval set: {len(eval_set)} questions")

    temp03 = run_temp(
        str(BASE_DIR / "results_heterogeneity_temp03_controlb.json"),
        str(BASE_DIR / "results_heterogeneity_temp03_treatment.json"),
        "temp03",
        eval_set,
        seed_offset=0,
    )

    temp07 = run_temp(
        str(BASE_DIR / "results_heterogeneity_temp07_controlb.json"),
        str(BASE_DIR / "results_heterogeneity_temp07_treatment.json"),
        "temp07",
        eval_set,
        seed_offset=1000,
    )

    output = {
        "analysis": "difficulty_subset_reanalysis_2026-06-03",
        "description": (
            "Exploratory analysis: does cognitive injection show a signal on high-uncertainty "
            "questions that is washed out in the full set?"
        ),
        "metric_source": (
            "JSD, ensemble Brier, pairwise error covariance copied from evaluate_heterogeneity_v3.py; "
            "pairwise_corr_across_questions copied from evaluate_heterogeneity_v2.py"
        ),
        "bootstrap_method": "cluster-bootstrap-by-question (paired, 4000 iterations)",
        "temp03": temp03,
        "temp07": temp07,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")

    write_md_report(output)
    print(f"Saved: {OUT_MD}")


def write_md_report(output: dict) -> None:
    t03 = output["temp03"]
    t07 = output["temp07"]

    def fmt_ci(d, lo, hi, p):
        if d is None:
            return "N/A"
        p_str = f"p={p:.3f}" if p is not None and not (isinstance(p, float) and np.isnan(p)) else "p=NA"
        if lo is None or (isinstance(lo, float) and np.isnan(lo)):
            return f"Δ={d:+.4f} [CI unavail] {p_str}"
        return f"Δ={d:+.4f} [{lo:+.4f}, {hi:+.4f}] {p_str}"

    def subset_row(label, s):
        n = s["n_questions"]
        if n < 6:
            return f"| {label} | N={n} (< 6, CI suppressed) | — | — | — | — |"
        b = s["treatment_minus_controlb"]["brier"]
        j = s["treatment_minus_controlb"]["jsd"]
        ec = s["treatment_minus_controlb"]["error_covariance"]
        ecorr = s["treatment_minus_controlb"]["error_correlation"]
        return (
            f"| {label} (N={n}) "
            f"| JSD {fmt_ci(j['delta'], j['ci_lo'], j['ci_hi'], j['p'])} "
            f"| Brier {fmt_ci(b['delta'], b['ci_lo'], b['ci_hi'], b['p'])} "
            f"| ErrCov {fmt_ci(ec['delta'], ec['ci_lo'], ec['ci_hi'], ec['p'])} "
            f"| ErrCorr Δ={ecorr['delta']:+.4f} (no per-q CI) |"
        )

    def interaction_block(t):
        c = t["continuous_interaction"]
        e = c["regression_delta_errcov_on_difficulty"]
        b = c["regression_delta_brier_on_difficulty"]
        j = c["regression_delta_jsd_on_difficulty"]
        lines = [
            f"Regressor: cb_dist_from_half (smaller = harder/more uncertain). N={e['n']}.",
            "",
            f"- delta_errcov ~ difficulty: slope={e['slope']:+.6f}, 95% CI [{e['ci_lo_95']:+.6f}, {e['ci_hi_95']:+.6f}], p={e['p']:.4f}, r²={e['r_sq']:.4f}",
            f"- delta_brier ~ difficulty:  slope={b['slope']:+.6f}, 95% CI [{b['ci_lo_95']:+.6f}, {b['ci_hi_95']:+.6f}], p={b['p']:.4f}, r²={b['r_sq']:.4f}",
            f"- delta_jsd ~ difficulty:    slope={j['slope']:+.6f}, 95% CI [{j['ci_lo_95']:+.6f}, {j['ci_hi_95']:+.6f}], p={j['p']:.4f}, r²={j['r_sq']:.4f}",
        ]
        return "\n".join(lines)

    lines = [
        "# Difficulty-Subset Re-analysis — Nous Paper V2 (Exploratory)",
        "",
        "**Date**: 2026-06-03  ",
        "**Status**: EXPLORATORY / hypothesis-generating. Not confirmatory.",
        "",
        "## 1. Pre-registered Difficulty Definitions",
        "",
        "The following difficulty proxies were defined before computing any treatment effect:",
        "",
        "**Primary proxy (no regression-to-mean risk):**",
        "cb_dist_from_half = |mean(Control-B probabilities) - 0.5|.",
        "Uncertain group: cb_dist_from_half < 0.2 (ensemble mean in (0.3, 0.7)).",
        "Certain group: cb_dist_from_half >= 0.2.",
        "",
        "**Secondary proxy (used ONLY for error-covariance outcome, NOT Brier):**",
        "Control-B ensemble Brier >= median of the complete set = 'hard-by-brier'.",
        "REGRESSION-TO-MEAN WARNING: selecting hard questions by high CB Brier and then",
        "observing Treatment Brier improvement creates a spurious artifact because CB Brier",
        "is high partly by chance; Treatment regresses toward the mean. This proxy is",
        "therefore restricted to the error-covariance outcome only.",
        "",
        "## 2. Sample Sizes",
        "",
        f"**temp=0.3**: {t03['n_complete']} complete questions (both groups 10 agents each).",
        f"  Uncertain (primary): N={t03['difficulty_definitions_preregistered']['n_uncertain']}",
        f"  Certain (primary): N={t03['difficulty_definitions_preregistered']['n_certain']}",
        f"  Hard-by-brier (secondary): N={t03['difficulty_definitions_preregistered']['n_hard_brier']}",
        f"  Easy-by-brier (secondary): N={t03['difficulty_definitions_preregistered']['n_easy_brier']}",
        "",
        f"**temp=0.7**: {t07['n_complete']} complete questions.",
        f"  Uncertain (primary): N={t07['difficulty_definitions_preregistered']['n_uncertain']}",
        f"  Certain (primary): N={t07['difficulty_definitions_preregistered']['n_certain']}",
        "",
        "These are small samples from an already-small experiment (50 questions,",
        "33 complete at temp=0.3). All findings below are exploratory.",
        "",
        "## 3. Results — temp=0.3 (Primary)",
        "",
        "Treatment − Control-B, cluster-bootstrap 95% CI (4000 iterations, by question).",
        "",
        "| Subset | JSD | Brier | ErrCov | ErrCorr |",
        "|--------|-----|-------|--------|---------|",
        subset_row("Uncertain (primary)", t03["subset_analyses"]["uncertain_primary"]),
        subset_row("Certain (primary)", t03["subset_analyses"]["certain_primary"]),
        subset_row("Hard-by-Brier (errcov only)", t03["subset_analyses"]["hard_by_brier_errcov_only"]),
        subset_row("Easy-by-Brier (errcov only)", t03["subset_analyses"]["easy_by_brier_errcov_only"]),
        subset_row("All complete", t03["subset_analyses"]["all_complete"]),
        "",
        "### Continuous Interaction (temp=0.3)",
        "",
        interaction_block(t03),
        "",
        "## 4. Robustness — temp=0.7",
        "",
        "| Subset | JSD | Brier | ErrCov | ErrCorr |",
        "|--------|-----|-------|--------|---------|",
        subset_row("Uncertain (primary)", t07["subset_analyses"]["uncertain_primary"]),
        subset_row("Certain (primary)", t07["subset_analyses"]["certain_primary"]),
        subset_row("Hard-by-Brier (errcov only)", t07["subset_analyses"]["hard_by_brier_errcov_only"]),
        subset_row("Easy-by-Brier (errcov only)", t07["subset_analyses"]["easy_by_brier_errcov_only"]),
        subset_row("All complete", t07["subset_analyses"]["all_complete"]),
        "",
        "### Continuous Interaction (temp=0.7)",
        "",
        interaction_block(t07),
        "",
        "## 5. Verdict",
        "",
        "See main report for narrative.",
        "",
        "---",
        "*Metric functions copied from evaluate_heterogeneity_v3.py and evaluate_heterogeneity_v2.py.",
        "Input files read-only. New outputs: results_heterogeneity_difficulty_subset.json + .md*",
    ]

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
