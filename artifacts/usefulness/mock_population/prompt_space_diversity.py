"""Prompt-space diversity: does the structure-to-narrative translator compress
profile diversity before the model ever sees it?

The paper's channel-bottleneck reading rests on the observation that a
more-diverse mock profile population produced no more output JSD than the
real ten profiles. That observation is equally consistent with the model
flattening diversity at decoding time. The two stories are distinguishable
by measuring diversity at the layer in between: the translator's own output
text (the persona prompts released in this package).

This script embeds every persona prompt with nomic-embed-text (a local
Ollama instance is required; see REPRODUCE.md) and compares pairwise
distance distributions:

  input space   : standardized 5-dim profile vectors (same space and
                  standardization as results_mock_population.json "spread")
  prompt space  : pairwise cosine distance between persona prompt embeddings,
                  with a char-trigram Jaccard distance as a lexical
                  sensitivity check

for the real-10 set vs the mock-14 set (Latin hypercube + archetype
anchors), plus the within-set rank correlation between input distance and
prompt distance.

If mock prompts are no more spread out than real prompts (despite more
spread-out inputs), the compression happens in the translator; if mock
prompts were much more spread out while outputs were not, it would happen
in the model.

Inputs (all released in this package):
  ../heterogeneity/persona_prompts_10.json   real-10 prompts + profile vectors
  mock_persona_prompts.json                  mock-14 prompts
  mock_profiles_lhs.json                     mock-14 profile parameters

Outputs: results_prompt_space_diversity.json / .md (canonical copies of both
are included in the package; re-running reproduces them).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).parent
REAL10_PERSONA_PATH = BASE_DIR.parent / "heterogeneity" / "persona_prompts_10.json"
MOCK_PROMPTS_PATH = BASE_DIR / "mock_persona_prompts.json"
MOCK_PROFILES_PATH = BASE_DIR / "mock_profiles_lhs.json"
OUT_JSON = BASE_DIR / "results_prompt_space_diversity.json"
OUT_REPORT = BASE_DIR / "results_prompt_space_diversity.md"

N_PERMUTATIONS = 10000
PERM_SEED = 42

# Same 5-dim comparable space and standardization as the "spread" block of
# results_mock_population.json. The (prior mean, prior std) pairs below are
# the population cold-start defaults of the schema (generic midpoints, not
# parameters fitted to live user data; the same standardization underlies the
# published spread figures).
DIM_STANDARDIZATION = {
    "contrarian_score": (0.0, 0.3),
    "crowd_sensitivity": (0.5, 0.2),
    "analytical_ratio": (0.5, 0.2),
    "hedgehog_fox_index": (0.0, 0.4),
    "update_rate": (0.5, 0.18),
}


def _ollama_embed(text: str, model: str = "nomic-embed-text") -> list[float] | None:
    import subprocess

    payload = json.dumps({"model": model, "prompt": text})
    try:
        out = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/embeddings", "-d", payload],
            capture_output=True, text=True, timeout=60,
        )
        return json.loads(out.stdout).get("embedding")
    except Exception:
        return None


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _trigram_dist(a: str, b: str) -> float:
    ta = {a[i : i + 3] for i in range(len(a) - 2)}
    tb = {b[i : i + 3] for i in range(len(b) - 2)}
    if not ta or not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def _standardized_vec(values: dict) -> np.ndarray:
    vec = []
    for dim, (mean, std) in DIM_STANDARDIZATION.items():
        v = values.get(dim, mean)
        vec.append((v - mean) / std if std > 0 else 0.0)
    return np.array(vec)


def _pairwise(items: list, dist_fn) -> list[float]:
    return [dist_fn(items[i], items[j]) for i, j in combinations(range(len(items)), 2)]


def _summary(dists: list[float]) -> dict:
    return {
        "n_pairs": len(dists),
        "mean": round(float(np.mean(dists)), 4),
        "min": round(float(np.min(dists)), 4),
        "max": round(float(np.max(dists)), 4),
        "std": round(float(np.std(dists)), 4),
    }


def _perm_mean_diff(set_a: list[np.ndarray], set_b: list[np.ndarray]) -> dict:
    """Permutation test for the difference in within-set mean pairwise cosine
    distance (pool the prompts, reassign set labels)."""
    rng = np.random.default_rng(PERM_SEED)
    pooled = set_a + set_b
    na = len(set_a)
    observed = float(
        np.mean(_pairwise(set_a, _cosine_dist)) - np.mean(_pairwise(set_b, _cosine_dist))
    )
    count = 0
    for _ in range(N_PERMUTATIONS):
        idx = rng.permutation(len(pooled))
        pa = [pooled[i] for i in idx[:na]]
        pb = [pooled[i] for i in idx[na:]]
        stat = float(np.mean(_pairwise(pa, _cosine_dist)) - np.mean(_pairwise(pb, _cosine_dist)))
        if abs(stat) >= abs(observed):
            count += 1
    return {"observed_diff": round(observed, 4), "p_two_sided": round(count / N_PERMUTATIONS, 4)}


def main() -> None:
    real = json.load(open(REAL10_PERSONA_PATH))
    mock_prompts = json.load(open(MOCK_PROMPTS_PATH))
    mock_profiles = {p["profile_id"]: p for p in json.load(open(MOCK_PROFILES_PATH))}

    real_texts = [e["persona_prompt_text"] for e in real]
    mock_texts = [e["persona_prompt_text"] for e in mock_prompts]
    real_inputs = [_standardized_vec(e["profile_vector"]) for e in real]
    mock_inputs = [_standardized_vec(mock_profiles[e["profile_id"]]) for e in mock_prompts]

    print(f"Embedding {len(real_texts)} real + {len(mock_texts)} mock prompts...")
    real_embs = [np.array(_ollama_embed(t)) for t in real_texts]
    mock_embs = [np.array(_ollama_embed(t)) for t in mock_texts]
    assert all(e.size > 1 for e in real_embs + mock_embs), \
        "embedding failure: is a local Ollama with nomic-embed-text running?"

    input_real = _pairwise(real_inputs, lambda a, b: float(np.linalg.norm(a - b)))
    input_mock = _pairwise(mock_inputs, lambda a, b: float(np.linalg.norm(a - b)))
    prompt_real = _pairwise(real_embs, _cosine_dist)
    prompt_mock = _pairwise(mock_embs, _cosine_dist)
    tri_real = _pairwise(real_texts, _trigram_dist)
    tri_mock = _pairwise(mock_texts, _trigram_dist)

    # Within-set rank correlation: does prompt distance track profile distance?
    rho_real = stats.spearmanr(input_real, prompt_real)
    rho_mock = stats.spearmanr(input_mock, prompt_mock)
    rho_pooled = stats.spearmanr(input_real + input_mock, prompt_real + prompt_mock)

    results = {
        "embedding_model": "nomic-embed-text",
        "n_real": len(real_texts),
        "n_mock": len(mock_texts),
        "prompt_chars": {
            "real_mean": round(float(np.mean([len(t) for t in real_texts])), 0),
            "mock_mean": round(float(np.mean([len(t) for t in mock_texts])), 0),
        },
        "input_space_5dim": {"real": _summary(input_real), "mock": _summary(input_mock)},
        "prompt_space_embedding": {"real": _summary(prompt_real), "mock": _summary(prompt_mock)},
        "prompt_space_trigram": {"real": _summary(tri_real), "mock": _summary(tri_mock)},
        "ratios_mock_over_real": {
            "input_mean": round(float(np.mean(input_mock) / np.mean(input_real)), 3),
            "input_max": round(float(np.max(input_mock) / np.max(input_real)), 3),
            "prompt_embedding_mean": round(float(np.mean(prompt_mock) / np.mean(prompt_real)), 3),
            "prompt_embedding_max": round(float(np.max(prompt_mock) / np.max(prompt_real)), 3),
            "prompt_trigram_mean": round(float(np.mean(tri_mock) / np.mean(tri_real)), 3),
        },
        "input_vs_prompt_rank_correlation": {
            "real": {"spearman_rho": round(float(rho_real.statistic), 3),
                     "p": round(float(rho_real.pvalue), 4)},
            "mock": {"spearman_rho": round(float(rho_mock.statistic), 3),
                     "p": round(float(rho_mock.pvalue), 4)},
            "pooled": {"spearman_rho": round(float(rho_pooled.statistic), 3),
                       "p": round(float(rho_pooled.pvalue), 6)},
        },
        "perm_test_mock_vs_real_prompt_mean": _perm_mean_diff(mock_embs, real_embs),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    r = results
    lines = [
        "# Prompt-space diversity: translator compression test",
        "",
        f"Real-10 vs mock-14 persona prompts; embedding = {r['embedding_model']}.",
        "",
        "| Layer | Real mean (max) | Mock mean (max) | Mock/Real mean ratio |",
        "|---|---|---|---|",
        f"| Input (5-dim std.) | {r['input_space_5dim']['real']['mean']} ({r['input_space_5dim']['real']['max']}) "
        f"| {r['input_space_5dim']['mock']['mean']} ({r['input_space_5dim']['mock']['max']}) "
        f"| {r['ratios_mock_over_real']['input_mean']} |",
        f"| Prompt (embedding cosine) | {r['prompt_space_embedding']['real']['mean']} ({r['prompt_space_embedding']['real']['max']}) "
        f"| {r['prompt_space_embedding']['mock']['mean']} ({r['prompt_space_embedding']['mock']['max']}) "
        f"| {r['ratios_mock_over_real']['prompt_embedding_mean']} |",
        f"| Prompt (trigram Jaccard) | {r['prompt_space_trigram']['real']['mean']} ({r['prompt_space_trigram']['real']['max']}) "
        f"| {r['prompt_space_trigram']['mock']['mean']} ({r['prompt_space_trigram']['mock']['max']}) "
        f"| {r['ratios_mock_over_real']['prompt_trigram_mean']} |",
        "",
        "Within-set Spearman (input distance vs prompt embedding distance):",
        f"- real: rho = {r['input_vs_prompt_rank_correlation']['real']['spearman_rho']} "
        f"(p = {r['input_vs_prompt_rank_correlation']['real']['p']})",
        f"- mock: rho = {r['input_vs_prompt_rank_correlation']['mock']['spearman_rho']} "
        f"(p = {r['input_vs_prompt_rank_correlation']['mock']['p']})",
        f"- pooled: rho = {r['input_vs_prompt_rank_correlation']['pooled']['spearman_rho']} "
        f"(p = {r['input_vs_prompt_rank_correlation']['pooled']['p']})",
        "",
        f"Permutation test (mock vs real mean pairwise prompt distance): "
        f"diff = {r['perm_test_mock_vs_real_prompt_mean']['observed_diff']}, "
        f"p = {r['perm_test_mock_vs_real_prompt_mean']['p_two_sided']}",
        "",
        "Reading: if the prompt-space mock/real ratio stays near (or below) the",
        "input-space ratio and the input-prompt correlation is weak, the translator",
        "is the compressing stage; if prompts spread but outputs did not, the model is.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_REPORT}")
    print(json.dumps(results["ratios_mock_over_real"], indent=2))
    print(json.dumps(results["perm_test_mock_vs_real_prompt_mean"], indent=2))


if __name__ == "__main__":
    main()
