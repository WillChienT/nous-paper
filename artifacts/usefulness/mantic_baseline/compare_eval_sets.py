"""Paired comparison between Session 4 (top-volume) and Session C (mid-volume) eval sets.

Loads both result JSON files, computes metrics for each, and produces a
structured markdown report covering:
  - Overall Brier / log loss / Metaculus score per model per eval set
  - Bootstrap 95% CI for the Brier difference (unpaired, since sets differ)
  - Model degradation: which model degrades more on mid-volume
  - Per-category breakdown (keyword-inferred from question text)
  - Worst-5 questions per eval set comparison
  - Section 6.1 paper writing recommendation

Usage:
    python injection/v2/compare_eval_sets.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent

TOPVOL_JSON = BASE_DIR / "results_mantic_baseline.json"
MIDVOL_JSON = BASE_DIR / "results_mantic_baseline_midvol.json"
OUTPUT_MD = BASE_DIR / "results_eval_set_comparison.md"

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

MANTIC_PUBLIC_SCORE = 45.8


# === Category inference ===

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Monetary Policy", ["fed ", "federal reserve", "interest rate", "fomc", "bps", "basis point"]),
    ("Sports", ["fc ", " fc", "liverpool", "arsenal", "barcelona", "psg", "dortmund", "borussia",
                "paris saint-germain", "win on 20", "soccer", "football", "nfl", "nba", "mlb",
                "super bowl", "champions league", "premier league", "la liga", "bundesliga"]),
    ("Crypto", ["bitcoin", "btc", "eth", "ethereum", "crypto", "solana", "sol ", "defi",
                "stablecoin", "usdc", "usdt"]),
    ("Politics / Elections", ["trump", "biden", "election", "democrat", "republican", "senate",
                               "congress", "president", "vote", "ballot", "white house", "cabinet",
                               "confirmed as", "nomination"]),
    ("Economics / Markets", ["gdp", "cpi", "inflation", "unemployment", "recession", "s&p",
                              "nasdaq", "dow jones", "stock market", "oil price", "gold price",
                              "yield", "treasury"]),
    ("Geopolitics", ["ukraine", "russia", "china", "taiwan", "nato", "sanction", "war",
                     "ceasefire", "conflict", "iran", "israel", "middle east", "north korea"]),
    ("AI / Tech", ["openai", "anthropic", "google", "gpt", "llm", "ai model", "deepseek",
                   "apple", "microsoft", "meta ", "amazon", "nvidia"]),
]


def infer_category(question: str) -> str:
    q_lower = question.lower()
    for label, keywords in CATEGORY_RULES:
        if any(kw in q_lower for kw in keywords):
            return label
    return "Other"


# === Metric helpers ===

def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def log_loss(p: float, y: int) -> float:
    p_clip = max(0.01, min(0.99, p))
    return -math.log(p_clip) if y == 1 else -math.log(1 - p_clip)


def metaculus_score(log_losses: list[float]) -> float:
    if not log_losses:
        return 0.0
    uniform = -math.log(0.5)
    return 100.0 * (1.0 - sum(log_losses) / len(log_losses) / uniform)


def compute(records: list[dict]) -> dict:
    briers = [brier(r["final_prob"], r["resolved_outcome_int"]) for r in records]
    lls = [log_loss(r["final_prob"], r["resolved_outcome_int"]) for r in records]
    n = len(records)
    return {
        "n": n,
        "mean_brier": round(sum(briers) / n, 4) if n else 0,
        "mean_log_loss": round(sum(lls) / n, 4) if n else 0,
        "metaculus_score": round(metaculus_score(lls), 2) if n else 0,
        "briers": briers,
    }


def bootstrap_ci(
    briers_a: list[float],
    briers_b: list[float],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """95% CI for (mean_brier_b - mean_brier_a) via unpaired bootstrap.

    Since the two eval sets have different questions, we resample each
    independently and compute the difference distribution.
    """
    rng = random.Random(seed)
    diffs = []
    na, nb = len(briers_a), len(briers_b)
    for _ in range(n_bootstrap):
        sample_a = [rng.choice(briers_a) for _ in range(na)]
        sample_b = [rng.choice(briers_b) for _ in range(nb)]
        diffs.append(sum(sample_b) / nb - sum(sample_a) / na)
    diffs.sort()
    lo = diffs[int(0.025 * n_bootstrap)]
    hi = diffs[int(0.975 * n_bootstrap)]
    point = sum(briers_b) / nb - sum(briers_a) / na
    return {"point_estimate": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


def worst_n(records: list[dict], n: int = 5) -> list[dict]:
    sorted_recs = sorted(records, key=lambda r: brier(r["final_prob"], r["resolved_outcome_int"]), reverse=True)
    return sorted_recs[:n]


def category_breakdown(records: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        cat = infer_category(r["question"])
        by_cat.setdefault(cat, []).append(r)
    result = {}
    for cat, recs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        m = compute(recs)
        result[cat] = {"n": m["n"], "mean_brier": m["mean_brier"], "metaculus_score": m["metaculus_score"]}
    return result


def naive_baseline_metrics(records: list[dict]) -> dict:
    outcomes = [r["resolved_outcome_int"] for r in records]
    n = len(outcomes)
    if n == 0:
        return {}
    yes_rate = sum(outcomes) / n
    yes_rate_c = max(0.01, min(0.99, yes_rate))
    b05 = sum(brier(0.5, y) for y in outcomes) / n
    ll05 = sum(log_loss(0.5, y) for y in outcomes) / n
    b_yr = sum(brier(yes_rate_c, y) for y in outcomes) / n
    ll_yr = sum(log_loss(yes_rate_c, y) for y in outcomes) / n
    return {
        "constant_0.5": {
            "mean_brier": round(b05, 4),
            "metaculus_score": round(metaculus_score([ll05] * n), 2),
        },
        "historical_yes_rate": {
            "yes_rate": round(yes_rate, 4),
            "mean_brier": round(b_yr, 4),
            "metaculus_score": round(metaculus_score([ll_yr] * n), 2),
        },
    }


def section_6_recommendation(
    qw_topvol: dict, qw_midvol: dict,
    ds_topvol: dict, ds_midvol: dict,
    ci_qw: dict, ci_ds: dict,
) -> str:
    """Generate a concrete recommendation for Section 6.1 paper writing."""
    qw_deg = qw_midvol["mean_brier"] - qw_topvol["mean_brier"]
    ds_deg = ds_midvol["mean_brier"] - ds_topvol["mean_brier"]

    qw_sig = not (ci_qw["ci_lo"] < 0 < ci_qw["ci_hi"])
    ds_sig = not (ci_ds["ci_lo"] < 0 < ci_ds["ci_hi"])

    if abs(qw_deg) < 0.03 and abs(ds_deg) < 0.03:
        return (
            "**Recommendation: write as 'two-point evaluation, baseline robust across question difficulty'.**\n\n"
            f"Both models show negligible degradation on mid-volume questions "
            f"(Qwen: ΔBrier={qw_deg:+.4f}, DeepSeek: ΔBrier={ds_deg:+.4f}), "
            "and neither CI excludes zero. "
            "Section 6.1 can claim the baseline holds across a volume-stratified question set, "
            "supporting generalizability of the main result."
        )
    elif qw_sig or ds_sig:
        worse_model = "Qwen3.5:122b" if qw_deg > ds_deg else "DeepSeek-R1:32b"
        return (
            "**Recommendation: write as 'baseline shows non-trivial degradation on uncertain events, "
            "warranting future evaluation on harder subsets'.**\n\n"
            f"At least one model shows significant degradation on mid-volume questions "
            f"(Qwen: ΔBrier={qw_deg:+.4f}, DeepSeek: ΔBrier={ds_deg:+.4f}). "
            f"{worse_model} degrades more. "
            "Section 6.1 should note that top-volume performance overstates capability on ambiguous events, "
            "and that the gap is a substantive finding about AI forecasting limits rather than an artifact."
        )
    else:
        return (
            "**Recommendation: write as 'suggestive degradation on uncertain events, not conclusive at N=50'.**\n\n"
            f"Degradation direction is present (Qwen: ΔBrier={qw_deg:+.4f}, DeepSeek: ΔBrier={ds_deg:+.4f}) "
            "but CIs cross zero. Section 6.1 can report the direction as consistent with the hypothesis "
            "that high-volume markets are easier, while noting that N=50 per eval set limits power. "
            "Frame as motivation for a larger robustness study rather than a settled finding."
        )


def main() -> None:
    if not TOPVOL_JSON.exists():
        print(f"Session 4 results not found: {TOPVOL_JSON}")
        return
    if not MIDVOL_JSON.exists():
        print(f"Mid-volume results not found: {MIDVOL_JSON}")
        print("Run mantic_baseline_midvol.py first.")
        return

    print("Loading results...")
    topvol_data = json.loads(TOPVOL_JSON.read_text())
    midvol_data = json.loads(MIDVOL_JSON.read_text())

    topvol_records = topvol_data["records"]
    midvol_records = midvol_data["records"]

    # Split by model
    tv_qw = [r for r in topvol_records if r["model_key"] == "qwen"]
    tv_ds = [r for r in topvol_records if r["model_key"] == "deepseek"]
    mv_qw = [r for r in midvol_records if r["model_key"] == "qwen"]
    mv_ds = [r for r in midvol_records if r["model_key"] == "deepseek"]

    m_tv_qw = compute(tv_qw)
    m_tv_ds = compute(tv_ds)
    m_mv_qw = compute(mv_qw)
    m_mv_ds = compute(mv_ds)

    ci_qw = bootstrap_ci(m_tv_qw["briers"], m_mv_qw["briers"])
    ci_ds = bootstrap_ci(m_tv_ds["briers"], m_mv_ds["briers"])

    nb_tv = naive_baseline_metrics(tv_qw)
    nb_mv = naive_baseline_metrics(mv_qw)

    cat_tv = category_breakdown(tv_qw)
    cat_mv = category_breakdown(mv_qw)

    worst_tv_qw = worst_n(tv_qw, 5)
    worst_mv_qw = worst_n(mv_qw, 5)
    worst_tv_ds = worst_n(tv_ds, 5)
    worst_mv_ds = worst_n(mv_ds, 5)

    sec6_rec = section_6_recommendation(m_tv_qw, m_mv_qw, m_tv_ds, m_mv_ds, ci_qw, ci_ds)

    print("Building comparison report...")

    lines = [
        "# Eval Set Paired Comparison: Top-Volume vs Mid-Volume",
        "",
        "**Session 4 (top-volume)**: N=50 markets by highest volume (≥$4.9M USDC)",
        "**Session C (mid-volume)**: N=50 markets from 25th–75th percentile of volume distribution",
        "**Hypothesis**: High-volume markets are 'easier' (market consensus is clearer), so top-volume results may overstate AI accuracy on genuinely uncertain events.",
        "",
        "---",
        "",
        "## 1. Overall Metrics",
        "",
        "| Eval Set | Model | N | Brier | Log Loss | Metaculus Score |",
        "|----------|-------|---|-------|----------|-----------------|",
        f"| Top-volume (S4) | Qwen3.5:122b | {m_tv_qw['n']} | {m_tv_qw['mean_brier']:.4f} | {m_tv_qw['mean_log_loss']:.4f} | {m_tv_qw['metaculus_score']:.2f} |",
        f"| Top-volume (S4) | DeepSeek-R1:32b | {m_tv_ds['n']} | {m_tv_ds['mean_brier']:.4f} | {m_tv_ds['mean_log_loss']:.4f} | {m_tv_ds['metaculus_score']:.2f} |",
        f"| Mid-volume (SC) | Qwen3.5:122b | {m_mv_qw['n']} | {m_mv_qw['mean_brier']:.4f} | {m_mv_qw['mean_log_loss']:.4f} | {m_mv_qw['metaculus_score']:.2f} |",
        f"| Mid-volume (SC) | DeepSeek-R1:32b | {m_mv_ds['n']} | {m_mv_ds['mean_brier']:.4f} | {m_mv_ds['mean_log_loss']:.4f} | {m_mv_ds['metaculus_score']:.2f} |",
        "",
        "### Naive Baselines",
        "",
        "| Eval Set | Baseline | Brier | Metaculus Score |",
        "|----------|----------|-------|-----------------|",
    ]

    for bname, bdata in nb_tv.items():
        lines.append(f"| Top-volume | {bname} | {bdata['mean_brier']:.4f} | {bdata['metaculus_score']:.2f} |")
    for bname, bdata in nb_mv.items():
        lines.append(f"| Mid-volume | {bname} | {bdata['mean_brier']:.4f} | {bdata['metaculus_score']:.2f} |")

    qw_deg = m_mv_qw["mean_brier"] - m_tv_qw["mean_brier"]
    ds_deg = m_mv_ds["mean_brier"] - m_tv_ds["mean_brier"]

    lines += [
        "",
        "---",
        "",
        "## 2. Degradation Analysis",
        "",
        "ΔBrier = mid-volume Brier − top-volume Brier. Positive = mid-volume is harder for the model.",
        "",
        "| Model | Top-vol Brier | Mid-vol Brier | ΔBrier | 95% CI (bootstrap) | Significant? |",
        "|-------|---------------|---------------|--------|---------------------|-------------|",
        f"| Qwen3.5:122b | {m_tv_qw['mean_brier']:.4f} | {m_mv_qw['mean_brier']:.4f} | {qw_deg:+.4f} | [{ci_qw['ci_lo']:+.4f}, {ci_qw['ci_hi']:+.4f}] | {'Yes' if not (ci_qw['ci_lo'] < 0 < ci_qw['ci_hi']) else 'No (CI crosses 0)'} |",
        f"| DeepSeek-R1:32b | {m_tv_ds['mean_brier']:.4f} | {m_mv_ds['mean_brier']:.4f} | {ds_deg:+.4f} | [{ci_ds['ci_lo']:+.4f}, {ci_ds['ci_hi']:+.4f}] | {'Yes' if not (ci_ds['ci_lo'] < 0 < ci_ds['ci_hi']) else 'No (CI crosses 0)'} |",
        "",
        f"Bootstrap N={BOOTSTRAP_N:,}, seed={BOOTSTRAP_SEED}. Unpaired resampling (two independent eval sets).",
        "",
        f"**Model with larger degradation**: {'Qwen3.5:122b' if qw_deg > ds_deg else 'DeepSeek-R1:32b'} "
        f"(ΔBrier={max(qw_deg, ds_deg):+.4f} vs {min(qw_deg, ds_deg):+.4f})",
        "",
        "---",
        "",
        "## 3. Per-Category Breakdown",
        "",
        "Categories inferred from question text via keyword matching.",
        "",
        "### Top-Volume (Session 4)",
        "",
        "| Category | N | Brier | Metaculus Score |",
        "|----------|---|-------|-----------------|",
    ]

    for cat, data in cat_tv.items():
        lines.append(f"| {cat} | {data['n']} | {data['mean_brier']:.4f} | {data['metaculus_score']:.2f} |")

    lines += [
        "",
        "### Mid-Volume (Session C)",
        "",
        "| Category | N | Brier | Metaculus Score |",
        "|----------|---|-------|-----------------|",
    ]

    for cat, data in cat_mv.items():
        lines.append(f"| {cat} | {data['n']} | {data['mean_brier']:.4f} | {data['metaculus_score']:.2f} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Worst-5 Questions Comparison",
        "",
        "Questions where the model was most wrong (highest Brier), compared across eval sets.",
        "",
        "### Top-Volume — Worst 5 (Qwen3.5:122b)",
        "",
    ]

    for i, r in enumerate(worst_tv_qw, 1):
        b = brier(r["final_prob"], r["resolved_outcome_int"])
        lines.append(f"{i}. **Brier {b:.4f}** | pred={r['final_prob']:.3f}, outcome={r['resolved_outcome']}  ")
        lines.append(f"   {r['question'][:120]}")
        lines.append("")

    lines += [
        "### Mid-Volume — Worst 5 (Qwen3.5:122b)",
        "",
    ]

    for i, r in enumerate(worst_mv_qw, 1):
        b = brier(r["final_prob"], r["resolved_outcome_int"])
        lines.append(f"{i}. **Brier {b:.4f}** | pred={r['final_prob']:.3f}, outcome={r['resolved_outcome']}  ")
        lines.append(f"   {r['question'][:120]}")
        lines.append("")

    lines += [
        "### Top-Volume — Worst 5 (DeepSeek-R1:32b)",
        "",
    ]

    for i, r in enumerate(worst_tv_ds, 1):
        b = brier(r["final_prob"], r["resolved_outcome_int"])
        lines.append(f"{i}. **Brier {b:.4f}** | pred={r['final_prob']:.3f}, outcome={r['resolved_outcome']}  ")
        lines.append(f"   {r['question'][:120]}")
        lines.append("")

    lines += [
        "### Mid-Volume — Worst 5 (DeepSeek-R1:32b)",
        "",
    ]

    for i, r in enumerate(worst_mv_ds, 1):
        b = brier(r["final_prob"], r["resolved_outcome_int"])
        lines.append(f"{i}. **Brier {b:.4f}** | pred={r['final_prob']:.3f}, outcome={r['resolved_outcome']}  ")
        lines.append(f"   {r['question'][:120]}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 5. Section 6.1 Paper Writing Recommendation",
        "",
        sec6_rec,
        "",
        "---",
        "",
        "## 6. Appendix: Eval Set Volume Statistics",
        "",
        "| | Top-Volume (S4) | Mid-Volume (SC) | Ratio |",
        "|--|-----------------|-----------------|-------|",
    ]

    tv_vols = sorted([r["volume"] for r in tv_qw])
    mv_vols = sorted([r["volume"] for r in mv_qw])

    if tv_vols and mv_vols:
        tv_med = tv_vols[len(tv_vols) // 2]
        mv_med = mv_vols[len(mv_vols) // 2]
        lines += [
            f"| Min volume | ${min(tv_vols):,.0f} | ${min(mv_vols):,.0f} | {min(tv_vols)/max(min(mv_vols),1):.1f}x |",
            f"| Median volume | ${tv_med:,.0f} | ${mv_med:,.0f} | {tv_med/max(mv_med,1):.1f}x |",
            f"| Max volume | ${max(tv_vols):,.0f} | ${max(mv_vols):,.0f} | {max(tv_vols)/max(max(mv_vols),1):.1f}x |",
        ]

    lines.append("")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison report written to {OUTPUT_MD}")

    # Console summary
    print("\n" + "=" * 60)
    print("PAIRED COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Qwen3.5:122b:    ΔBrier={qw_deg:+.4f}  95% CI [{ci_qw['ci_lo']:+.4f}, {ci_qw['ci_hi']:+.4f}]")
    print(f"DeepSeek-R1:32b: ΔBrier={ds_deg:+.4f}  95% CI [{ci_ds['ci_lo']:+.4f}, {ci_ds['ci_hi']:+.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
