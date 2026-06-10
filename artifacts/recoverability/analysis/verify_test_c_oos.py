"""Verify Test C: Out-of-Sample Predictive Validity.

Reads profiles_testc_early.json (60% early-window extracted profile) and
profiles_testc_late_pnl.json (40% late-window PnL) and computes Spearman
rank correlations between each pre-registered dimension and late-window PnL.

This script reproduces Table 4 / Section 4.3 of the paper.

Usage:
    python analysis/verify_test_c_oos.py

Expected key results (from paper):
  independence_index.contrarian_score : rho ≈ 0.58, p ≈ 0.0005 (Bonferroni pass)
  cognitive_style.analytical_ratio    : rho ≈ 0.43, p ≈ 0.0005 (Bonferroni pass)
  2 of 4 pre-registered dims survive Bonferroni correction
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROFILES_DIR = Path(__file__).parent.parent / "profiles"
EARLY_PATH = PROFILES_DIR / "profiles_testc_early.json"
LATE_PATH = PROFILES_DIR / "profiles_testc_late_pnl.json"

N_BOOT = 5000
N_PERM = 2000
SEED = 42
BONFERRONI_ALPHA = 0.05 / 4  # 0.0125

PRE_REGISTERED = [
    ("independence_index",    "contrarian_score",    +1),
    ("belief_update_inertia", "update_rate",         +1),
    ("cognitive_style",       "analytical_ratio",    +1),
    ("loss_response",         "sunk_cost_sensitivity", -1),
]


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho as a plain float. scipy's SignificanceResult is under-typed
    in the bundled stubs, so we read .statistic and coerce here in one place."""
    result = stats.spearmanr(a, b)
    return float(result.statistic)  # type: ignore[union-attr]


def permutation_p(dim_vals: np.ndarray, pnl_vals: np.ndarray, observed_rho: float, n_perm: int, seed: int) -> float:
    """Two-sided permutation p-value for Spearman rho."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(pnl_vals))
        rho = _spearman_rho(dim_vals, pnl_vals[perm])
        null.append(abs(rho))
    return float(np.mean(np.array(null) >= abs(observed_rho)))


def bootstrap_ci(dim_vals: np.ndarray, pnl_vals: np.ndarray, n_boot: int, seed: int):
    """95% bootstrap CI for Spearman rho (by-wallet resampling)."""
    rng = np.random.default_rng(seed)
    n = len(dim_vals)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho = _spearman_rho(dim_vals[idx], pnl_vals[idx])
        boot_rhos.append(rho)
    boot_rhos = np.array(boot_rhos)
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def main() -> None:
    early = json.loads(EARLY_PATH.read_text())
    late = json.loads(LATE_PATH.read_text())

    common_wallets = sorted(set(early.keys()) & set(late.keys()))
    n = len(common_wallets)
    print(f"Common wallets: {n}")

    late_pnl = np.array([late[wid]["late_pnl_usdc"] for wid in common_wallets])
    print(f"Late PnL stats: min={late_pnl.min():.2f}  median={np.median(late_pnl):.2f}  max={late_pnl.max():.2f}")

    print(f"\n{'Dimension.Parameter':<45} {'rho':>6}  {'p_perm':>8}  {'CI_lo':>6}  {'CI_hi':>6}  {'Bonf':>5}")
    print("-" * 90)

    bonf_pass = 0
    for dim, param, direction in PRE_REGISTERED:
        key = f"{dim}.{param}"
        dim_vals = np.array([early[wid]["profile"].get(key) for wid in common_wallets], dtype=float)
        valid_mask = ~np.isnan(dim_vals)
        if valid_mask.sum() < 20:
            print(f"  {key:<43}  (insufficient data, n={valid_mask.sum()})")
            continue

        dv = dim_vals[valid_mask]
        pv = late_pnl[valid_mask]

        rho = _spearman_rho(dv, pv)
        p = permutation_p(dv, pv, rho, N_PERM, SEED)
        ci_lo, ci_hi = bootstrap_ci(dv, pv, N_BOOT, SEED)
        survives = p < BONFERRONI_ALPHA and (direction * rho > 0)
        if survives:
            bonf_pass += 1
        survives_str = "PASS" if survives else "----"
        print(f"  {key:<43} {rho:6.4f}  {p:8.4f}  {ci_lo:6.4f}  {ci_hi:6.4f}  {survives_str}")

    print("-" * 90)
    print(f"\nBonferroni survivors: {bonf_pass}/4  (alpha={BONFERRONI_ALPHA})")
    print("Test C pass criterion: ≥ 1 of 4 survives Bonferroni in predicted direction")
    print(f"→ {'PASS' if bonf_pass >= 1 else 'FAIL'}")


if __name__ == "__main__":
    main()
