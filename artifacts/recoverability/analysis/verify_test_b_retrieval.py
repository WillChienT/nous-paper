"""Verify Test B: Wallet Identification via split-half retrieval.

Reads profiles_testa_half1.json and profiles_testa_half2.json and runs
wallet identification (top-1 / top-5 retrieval) + within-vs-between distance
analysis on three dimension subsets (reliable_high, reliable_med, all_dims).

The analysis follows canonical test_b_retrieval_mmd.py exactly:
  1. Exclude wallets where any dimension in the subset returns None
  2. Z-score normalize across wallets per dimension (joint pool of half1 + half2)
  3. Retrieve using L2 on normalized vectors
  4. Run permutation null (2000 iterations, seed=42)

This script reproduces Table 3 / Section 4.2 of the paper.

Usage:
    python analysis/verify_test_b_retrieval.py

Expected key results (reliable_high subset, from canonical results_test_b_retrieval_mmd.json):
  top-1 = 0.17,  top-5 = 0.49
  within/between ratio ≈ 0.42
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

PROFILES_DIR = Path(__file__).parent.parent / "profiles"
HALF1_PATH = PROFILES_DIR / "profiles_testa_half1.json"
HALF2_PATH = PROFILES_DIR / "profiles_testa_half2.json"

N_PERMUTATIONS = 2000
SEED = 42

SUBSET_DEFS = {
    "reliable_high": [
        "time_scale_preference.horizon_bias",
        "independence_index.contrarian_score",
        "independence_index.crowd_sensitivity",
        "belief_update_inertia.update_rate",
        "cognitive_style.analytical_ratio",
        "attention_allocation.entropy",
    ],
    "reliable_med": [
        "risk_perception.lambda_",
        "time_scale_preference.horizon_bias",
        "time_scale_preference.patience_index",
        "independence_index.contrarian_score",
        "independence_index.crowd_sensitivity",
        "belief_update_inertia.update_rate",
        "cognitive_style.analytical_ratio",
        "cognitive_style.hedgehog_fox_index",
        "attention_allocation.entropy",
    ],
    "all_dims": [
        "risk_perception.alpha",
        "risk_perception.beta",
        "risk_perception.lambda_",
        "time_scale_preference.horizon_bias",
        "time_scale_preference.patience_index",
        "independence_index.contrarian_score",
        "independence_index.crowd_sensitivity",
        "belief_update_inertia.update_rate",
        "belief_update_inertia.confirmation_bias",
        "cognitive_style.analytical_ratio",
        "cognitive_style.hedgehog_fox_index",
        "loss_response.sunk_cost_sensitivity",
        "loss_response.tilt_factor",
        "attention_allocation.entropy",
    ],
}


def build_valid_profiles(
    half1_raw: dict, half2_raw: dict, common_wallets: list, dims: list
) -> tuple[dict, dict, list]:
    """Build h1 and h2 profile dicts, dropping wallets where any dim is None."""
    h1_profiles: dict[str, np.ndarray] = {}
    h2_profiles: dict[str, np.ndarray] = {}
    valid_wallets = []

    for wid in common_wallets:
        vec1, vec2 = [], []
        valid = True
        for d in dims:
            v1 = half1_raw[wid]["profile"].get(d)
            v2 = half2_raw[wid]["profile"].get(d)
            if v1 is None or v2 is None:
                valid = False
                break
            vec1.append(float(v1))
            vec2.append(float(v2))
        if valid:
            h1_profiles[wid] = np.array(vec1)
            h2_profiles[wid] = np.array(vec2)
            valid_wallets.append(wid)

    return h1_profiles, h2_profiles, valid_wallets


def zscore_normalize(
    h1_profiles: dict, h2_profiles: dict
) -> tuple[dict, dict]:
    """Z-score normalize across wallets per dimension (joint pool of h1 + h2)."""
    wallets = list(h1_profiles.keys())
    X_all = np.vstack([np.vstack([h1_profiles[w], h2_profiles[w]]) for w in wallets])
    means = X_all.mean(axis=0)
    stds = X_all.std(axis=0)
    stds[stds < 1e-10] = 1.0
    h1_norm = {w: (h1_profiles[w] - means) / stds for w in wallets}
    h2_norm = {w: (h2_profiles[w] - means) / stds for w in wallets}
    return h1_norm, h2_norm


def wallet_identification(
    h1_norm: dict, h2_norm: dict, n_permutations: int = N_PERMUTATIONS, seed: int = SEED
) -> dict:
    wallets = list(h1_norm.keys())
    N = len(wallets)
    h2_matrix = np.array([h2_norm[w] for w in wallets])

    ranks = []
    for i, w in enumerate(wallets):
        query = h1_norm[w]
        dists = np.sqrt(np.sum((h2_matrix - query) ** 2, axis=1))
        rank = int(np.sum(dists <= dists[i]))
        ranks.append(rank)

    ranks_arr = np.array(ranks)
    top1 = float(np.mean(ranks_arr == 1))
    top5 = float(np.mean(ranks_arr <= 5))
    mrr = float(np.mean(1.0 / ranks_arr))

    rng = random.Random(seed)
    null_top1, null_top5, null_mrr = [], [], []
    for _ in range(n_permutations):
        perm_wallets = wallets[:]
        rng.shuffle(perm_wallets)
        perm_h2 = np.array([h2_norm[w] for w in perm_wallets])
        perm_ranks = []
        for i, w in enumerate(wallets):
            dists = np.sqrt(np.sum((perm_h2 - h1_norm[w]) ** 2, axis=1))
            perm_ranks.append(int(np.sum(dists <= dists[i])))
        perm_arr = np.array(perm_ranks)
        null_top1.append(float(np.mean(perm_arr == 1)))
        null_top5.append(float(np.mean(perm_arr <= 5)))
        null_mrr.append(float(np.mean(1.0 / perm_arr)))

    p_top1 = float(np.mean(np.array(null_top1) >= top1))
    p_top5 = float(np.mean(np.array(null_top5) >= top5))

    return {
        "N": N, "top1": round(top1, 4), "top5": round(top5, 4), "mrr": round(mrr, 4),
        "null_top1_mean": round(float(np.mean(null_top1)), 4),
        "null_top5_mean": round(float(np.mean(null_top5)), 4),
        "p_top1": round(p_top1, 4), "p_top5": round(p_top5, 4),
        "pass_top1": p_top1 < 0.05, "pass_top5": p_top5 < 0.05,
    }


def within_vs_between(h1_norm: dict, h2_norm: dict, n_permutations: int = N_PERMUTATIONS, seed: int = SEED) -> dict:
    wallets = list(h1_norm.keys())
    N = len(wallets)
    h1_mat = np.array([h1_norm[w] for w in wallets])
    h2_mat = np.array([h2_norm[w] for w in wallets])
    dist_mat = np.sqrt(np.sum((h1_mat[:, None, :] - h2_mat[None, :, :]) ** 2, axis=2))

    within_d = np.array([dist_mat[i, i] for i in range(N)])
    mask = ~np.eye(N, dtype=bool)
    between_d = dist_mat[mask]
    ratio = float(np.median(within_d) / np.median(between_d))

    rng = np.random.RandomState(seed)
    null_within = []
    for _ in range(n_permutations):
        perm = rng.permutation(N)
        h2_perm = h2_mat[perm]
        dm = np.sqrt(np.sum((h1_mat[:, None, :] - h2_perm[None, :, :]) ** 2, axis=2))
        null_within.append(np.median(np.array([dm[i, i] for i in range(N)])))
    p_within = float(np.mean(np.array(null_within) <= np.median(within_d)))

    return {
        "within_median": round(float(np.median(within_d)), 4),
        "between_median": round(float(np.median(between_d)), 4),
        "ratio_within_to_between": round(ratio, 4),
        "p_within_lt_between": round(p_within, 4),
        "pass": p_within < 0.05,
    }


def main() -> None:
    half1_raw = json.loads(HALF1_PATH.read_text())
    half2_raw = json.loads(HALF2_PATH.read_text())

    common_wallets = sorted(set(half1_raw.keys()) & set(half2_raw.keys()))
    print(f"Total wallets in frozen profiles: {len(common_wallets)}")

    for subset_name, dims in SUBSET_DEFS.items():
        h1_p, h2_p, valid_wallets = build_valid_profiles(half1_raw, half2_raw, common_wallets, dims)
        h1_norm, h2_norm = zscore_normalize(h1_p, h2_p)

        print(f"\n=== Subset: {subset_name} ({len(dims)} dims, {len(valid_wallets)} wallets) ===")

        ident = wallet_identification(h1_norm, h2_norm)
        print(f"  top-1={ident['top1']:.2f}  top-5={ident['top5']:.2f}  mrr={ident['mrr']:.4f}")
        print(f"  p(top-1)={ident['p_top1']:.4f}  p(top-5)={ident['p_top5']:.4f}")
        print(f"  pass_top1={ident['pass_top1']}  pass_top5={ident['pass_top5']}")

        wb = within_vs_between(h1_norm, h2_norm)
        print(f"  within_median={wb['within_median']:.4f}  between_median={wb['between_median']:.4f}  "
              f"ratio={wb['ratio_within_to_between']:.4f}")
        print(f"  p(within<between)={wb['p_within_lt_between']:.4f}  pass={wb['pass']}")


if __name__ == "__main__":
    main()
