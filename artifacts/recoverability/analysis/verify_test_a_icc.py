"""Verify Test A: Temporal Stability — ICC from frozen profiles.

Reads profiles_testa_half1.json and profiles_testa_half2.json (frozen,
pseudonymized profiles from the first and second half of each wallet's
trade history) and computes ICC(2,1) for each of the 14 dimension parameters.

This script reproduces Table 2 / Section 4.1 of the paper.

Usage:
    python analysis/verify_test_a_icc.py

Expected key results (from paper):
  independence_index.contrarian_score : ICC ≈ 0.90
  attention_allocation.entropy        : ICC ≈ 0.76
  cognitive_style.analytical_ratio    : ICC ≈ 0.73
  8 of 14 parameters pass ICC ≥ 0.50
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROFILES_DIR = Path(__file__).parent.parent / "profiles"
HALF1_PATH = PROFILES_DIR / "profiles_testa_half1.json"
HALF2_PATH = PROFILES_DIR / "profiles_testa_half2.json"

ALL_PARAMS = [
    ("risk_perception",       "alpha"),
    ("risk_perception",       "beta"),
    ("risk_perception",       "lambda_"),
    ("time_scale_preference", "horizon_bias"),
    ("time_scale_preference", "patience_index"),
    ("independence_index",    "contrarian_score"),
    ("independence_index",    "crowd_sensitivity"),
    ("belief_update_inertia", "update_rate"),
    ("belief_update_inertia", "confirmation_bias"),
    ("cognitive_style",       "analytical_ratio"),
    ("cognitive_style",       "hedgehog_fox_index"),
    ("loss_response",         "sunk_cost_sensitivity"),
    ("loss_response",         "tilt_factor"),
    ("attention_allocation",  "entropy"),
]


def icc_2_1(half1: np.ndarray, half2: np.ndarray) -> float:
    """Compute ICC(2,1) — two-way random, single measures."""
    n = len(half1)
    if n < 3:
        return 0.0
    data = np.column_stack([half1, half2])
    k = 2
    grand_mean = np.mean(data)
    subject_means = np.mean(data, axis=1)
    judge_means = np.mean(data, axis=0)
    ss_between = k * np.sum((subject_means - grand_mean) ** 2)
    ss_judges = n * np.sum((judge_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_between - ss_judges
    df_between = n - 1
    df_judges = k - 1
    df_error = (n - 1) * (k - 1)
    bms = ss_between / max(df_between, 1)
    jms = ss_judges / max(df_judges, 1)
    ems = ss_error / max(df_error, 1)
    denom = bms + ems + 2 * (jms - ems) / n
    if abs(denom) < 1e-10:
        return 0.0
    return float((bms - ems) / denom)


def main() -> list[dict]:
    half1 = json.loads(HALF1_PATH.read_text())
    half2 = json.loads(HALF2_PATH.read_text())

    common_wallets = sorted(set(half1.keys()) & set(half2.keys()))
    print(f"Common wallets: {len(common_wallets)}")

    results = []
    pass_count = 0

    print(f"\n{'Dimension.Parameter':<45} {'ICC':>6}  {'N':>4}  Verdict")
    print("-" * 70)

    for dim, param in ALL_PARAMS:
        key = f"{dim}.{param}"
        vals1, vals2 = [], []
        for wid in common_wallets:
            v1 = half1[wid]["profile"].get(key)
            v2 = half2[wid]["profile"].get(key)
            if v1 is not None and v2 is not None:
                vals1.append(v1)
                vals2.append(v2)

        if len(vals1) < 10:
            print(f"  {key:<43}   N/A  (insufficient data, n={len(vals1)})")
            continue

        icc = icc_2_1(np.array(vals1), np.array(vals2))
        verdict = "PASS" if icc >= 0.50 else "fail"
        if icc >= 0.50:
            pass_count += 1
        print(f"  {key:<43} {icc:6.4f}  {len(vals1):4d}  {verdict}")
        results.append({"dimension": dim, "parameter": param, "icc": round(icc, 4), "n": len(vals1)})

    print("-" * 70)
    print(f"\nParameters ICC ≥ 0.50: {pass_count}/14")
    print(f"Test A pass criterion: ICC ≥ 0.50 for ≥ 5 of 8 dimensions → {'PASS' if pass_count >= 5 else 'FAIL'}")

    return results


if __name__ == "__main__":
    main()
