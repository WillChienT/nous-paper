"""Minimum detectable effect (MDE) analysis for the heterogeneity null.

The four-group experiment (evaluate_heterogeneity_v3.py) reports a null on the
pure-cognitive contrast (Treatment - Placebo) for ensemble Brier and pairwise
error covariance. A null is only informative relative to the effect size the
design could have detected. This script quantifies that: the smallest contrast
the design had 80% power to detect at alpha = 0.05, under both the Welch
two-sample test actually reported and the more powerful paired test available
because Treatment and Placebo share seeds and questions.

Also computes mean individual (per-agent) Brier per group, and verifies the
exact algebraic decomposition that links the two reported decision-quality
metrics with N = 10 agents:

    ensemble_brier = (1/10) * mean_individual_brier + (9/10) * mean_pairwise_errprod

so that the paper can disclose that ensemble Brier and pairwise error
covariance are not independent evidence.

Reads the same canonical inputs as evaluate_heterogeneity_v3.py; writes new
side files only (results_mde_heterogeneity.json / .md). Does not modify any
canonical result file.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from evaluate_heterogeneity_v3 import (
    CB_RESULTS_PATH,
    PLACEBO_RESULTS_PATH,
    V1_RESULTS_PATH,
    _load_eval_set,
    _per_question_stats,
)

BASE_DIR = Path(__file__).parent
OUT_JSON = BASE_DIR / "results_mde_heterogeneity.json"
OUT_REPORT = BASE_DIR / "results_mde_heterogeneity.md"

ALPHA = 0.05
POWER = 0.80
N_AGENTS = 10


def _load_complete_groups() -> dict[str, dict[str, list]]:
    """Replicate the v3 listwise-complete filter: questions with 10 parsed
    agents in every one of the four groups."""
    with open(V1_RESULTS_PATH) as f:
        v1_records = json.load(f)
    with open(CB_RESULTS_PATH) as f:
        cb_records = json.load(f)
    with open(PLACEBO_RESULTS_PATH) as f:
        placebo_records = json.load(f)

    by_q: dict[str, dict[str, list]] = {}
    for r in v1_records + cb_records + placebo_records:
        mid = r["market_id"]
        if mid not in by_q:
            by_q[mid] = {"control": [], "control_b": [], "treatment": [], "placebo": []}
        group = r["group"]
        if group not in by_q[mid]:
            by_q[mid][group] = []
        if not r["parse_failed"] and r["prob"] is not None:
            by_q[mid][group].append(r)

    complete: dict[str, dict[str, list]] = {}
    for mid, groups in by_q.items():
        ca = sorted(groups.get("control", []), key=lambda r: r["agent_idx"])
        cb = sorted(groups.get("control_b", []), key=lambda r: r["agent_idx"])
        trt = sorted(groups.get("treatment", []), key=lambda r: r["agent_idx"])
        pl = sorted(groups.get("placebo", []), key=lambda r: r["agent_idx"])
        if len(ca) == len(cb) == len(trt) == len(pl) == N_AGENTS:
            complete[mid] = {"control_a": ca, "control_b": cb, "treatment": trt, "placebo": pl}
    return complete


def _per_question_metrics(complete: dict, eval_set: dict) -> dict[str, dict[str, list[float]]]:
    pairs = list(combinations(range(N_AGENTS), 2))
    out: dict[str, dict[str, list[float]]] = {
        g: {"ensemble_brier": [], "errcov": [], "indiv_brier": []}
        for g in ("control_a", "control_b", "treatment", "placebo")
    }
    for mid, groups in complete.items():
        actual = 1 if eval_set[mid]["resolved_outcome"] == "Yes" else 0
        for g, recs in groups.items():
            probs = [r["prob"] for r in recs]
            stats_q = _per_question_stats(probs, actual)
            errors = stats_q["errors"]
            out[g]["ensemble_brier"].append(stats_q["brier"])
            out[g]["errcov"].append(float(np.mean([errors[i] * errors[j] for i, j in pairs])))
            out[g]["indiv_brier"].append(float(np.mean([e * e for e in errors])))
    return out


def _welch_mde(a: list[float], b: list[float]) -> dict:
    """Smallest |mean difference| detectable with 80% power at alpha=0.05 under
    the Welch two-sample t-test (the test reported in the paper)."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    na, nb = len(a_arr), len(b_arr)
    va, vb = a_arr.var(ddof=1), b_arr.var(ddof=1)
    se = float(np.sqrt(va / na + vb / nb))
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    mde = (stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(POWER, df)) * se
    return {"se": round(se, 5), "df": round(float(df), 1), "mde_80": round(float(mde), 4)}


def _paired_mde(a: list[float], b: list[float]) -> dict:
    """Same, under the paired t-test (Treatment and Placebo share seeds and
    questions, so pairing by question is legitimate and more powerful)."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n))
    df = n - 1
    mde = (stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(POWER, df)) * se
    t_obs, p_obs = stats.ttest_rel(a, b)
    return {
        "n_pairs": n,
        "sd_diff": round(float(np.std(d, ddof=1)), 5),
        "se": round(se, 5),
        "mde_80": round(float(mde), 4),
        "observed_diff": round(float(d.mean()), 5),
        "paired_t": round(float(t_obs), 3),
        "paired_p": round(float(p_obs), 4),
    }


def main() -> None:
    complete = _load_complete_groups()
    eval_set = _load_eval_set()
    n_q = len(complete)
    print(f"Listwise-complete questions: {n_q}")

    metrics = _per_question_metrics(complete, eval_set)

    # Decomposition check: ensemble = 0.1*indiv + 0.9*errcov (exact identity)
    decomposition = {}
    for g, m in metrics.items():
        lhs = float(np.mean(m["ensemble_brier"]))
        rhs = 0.1 * float(np.mean(m["indiv_brier"])) + 0.9 * float(np.mean(m["errcov"]))
        decomposition[g] = {
            "ensemble_brier": round(lhs, 5),
            "mean_indiv_brier": round(float(np.mean(m["indiv_brier"])), 5),
            "mean_pairwise_errprod": round(float(np.mean(m["errcov"])), 5),
            "identity_rhs": round(rhs, 5),
            "identity_abs_gap": round(abs(lhs - rhs), 8),
        }

    trt, pl = metrics["treatment"], metrics["placebo"]
    results = {
        "n_questions": n_q,
        "n_agents": N_AGENTS,
        "alpha": ALPHA,
        "power": POWER,
        "contrast": "pure cognitive (Treatment - Placebo)",
        "ensemble_brier": {
            "welch": _welch_mde(trt["ensemble_brier"], pl["ensemble_brier"]),
            "paired": _paired_mde(trt["ensemble_brier"], pl["ensemble_brier"]),
        },
        "pairwise_error_covariance": {
            "welch": _welch_mde(trt["errcov"], pl["errcov"]),
            "paired": _paired_mde(trt["errcov"], pl["errcov"]),
        },
        "indiv_brier": {
            "paired": _paired_mde(trt["indiv_brier"], pl["indiv_brier"]),
        },
        "group_decomposition": decomposition,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# MDE analysis for the heterogeneity null (pure-cognitive contrast)",
        "",
        f"Listwise-complete questions: **{n_q}**; agents per group: {N_AGENTS}; "
        f"alpha = {ALPHA}, target power = {POWER}.",
        "",
        "## Minimum detectable effects (80% power)",
        "",
        "| Metric | Test | SE | MDE (80%) | Observed diff |",
        "|---|---|---|---|---|",
    ]
    for metric_key, label in [
        ("ensemble_brier", "Ensemble Brier"),
        ("pairwise_error_covariance", "Pairwise error covariance"),
    ]:
        w = results[metric_key]["welch"]
        p = results[metric_key]["paired"]
        lines.append(
            f"| {label} | Welch (as reported) | {w['se']} | {w['mde_80']} | {p['observed_diff']} |"
        )
        lines.append(
            f"| {label} | Paired by question | {p['se']} | {p['mde_80']} | {p['observed_diff']} |"
        )
    lines += [
        "",
        "## Decomposition (identity: ensemble = 0.1 x indiv + 0.9 x errprod)",
        "",
        "| Group | Ensemble Brier | Mean indiv Brier | Mean pairwise err-prod | Identity gap |",
        "|---|---|---|---|---|",
    ]
    for g, d in decomposition.items():
        lines.append(
            f"| {g} | {d['ensemble_brier']} | {d['mean_indiv_brier']} | "
            f"{d['mean_pairwise_errprod']} | {d['identity_abs_gap']} |"
        )
    lines += [
        "",
        "Interpretation: the null on the pure-cognitive contrast is informative only",
        "for effects at least as large as the MDE; smaller real improvements would not",
        "have been detected by this design.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_REPORT}")
    print(json.dumps(results, indent=2)[:2000])


if __name__ == "__main__":
    main()
