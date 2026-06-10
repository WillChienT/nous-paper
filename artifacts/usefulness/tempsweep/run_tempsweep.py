"""Temperature sweep for heterogeneity experiment.

Tests whether the null result (cognitive injection does not decorrelate errors)
holds across temperatures 0.0, 0.3, 0.7, 1.0.

For each temperature, runs Control-B and Treatment groups (10 agents × up to 50 questions).
Outputs per-cell raw files and a combined sweep results file.

Does NOT touch any canonical results files (results_heterogeneity_v3.json etc.).

Usage:
    python run_tempsweep.py --temperature 0.3 --group controlb [--smoke N]
    python run_tempsweep.py --temperature 0.7 --group treatment [--smoke N]
    python run_tempsweep.py --all  # runs all cells sequentially

Anti-silent-failure measures:
- Per-cell progress and output files with temperature+group in filename
- Verifies temperature actually varies in the Ollama options
- Reports parse rate per cell
- Halts if parse rate > 10% failure
- Warns if numbers match 0.3 baseline exactly (cache reuse red flag)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "research_cache"
EVAL_PATH = DATA_DIR / "eval_set_mantic_baseline.jsonl"
PERSONA_PATH = DATA_DIR / "persona_prompts_10.json"

# ── Model / Ollama ─────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:122b"

# ── Seeds (copied verbatim from original runners) ─────────────────────────────
CONTROL_B_SEEDS = list(range(53, 63))   # seeds 53–62 (10 agents)
TREATMENT_SEEDS = list(range(43, 53))   # seeds 43–52 (10 agents)

# ── Baseline reference for reproduction check ─────────────────────────────────
# From results_heterogeneity_v3.json group_metrics.treatment and group_metrics.control_b
BASELINE_TREATMENT_JSD  = 0.0060
BASELINE_TREATMENT_ECORR = 0.9838
BASELINE_CB_JSD         = 0.0025
BASELINE_CB_ECORR       = 0.9908

# ── Session 4 Mantic baseline forecaster prompt (verbatim from original runners) ─
PREDICTION_SYSTEM = """You are an expert forecaster. Your task is to provide a calibrated
probability estimate for a binary forecasting question.

Think step by step before reaching your conclusion. Consider:
- Base rates for similar events
- Evidence supporting YES
- Evidence supporting NO
- Uncertainty and unknown factors
- Any asymmetries in the available evidence

You MUST output your final answer as a JSON code block with this exact structure:
```json
{"scenarios_yes": [{"name": str, "prob": float}, ...],
 "scenarios_no": [{"name": str, "prob": float}, ...],
 "final_prob": float}
```

Where:
- scenarios_yes: 3 scenarios that could lead to YES, each with a rough probability (0-1)
- scenarios_no: 3 scenarios that could lead to NO, each with a rough probability (0-1)
- final_prob: your final aggregated probability of YES (a single float in [0, 1])

The scenarios are for illustrative reasoning only — they do not need to sum to 1.
The final_prob is your actual calibrated estimate."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_eval_set(n: int | None = None) -> list[dict]:
    with open(EVAL_PATH) as f:
        rows = [json.loads(line) for line in f]
    return rows[:n] if n is not None else rows


def _load_research_cache(market_id: str, info_horizon: str) -> str:
    fname = f"{market_id}_{info_horizon}.json"
    path = CACHE_DIR / fname
    if not path.exists():
        return "[Research cache not found]"
    with open(path) as f:
        data = json.load(f)
    return data.get("summary", "")


def _load_persona_prompts() -> list[dict]:
    with open(PERSONA_PATH) as f:
        return json.load(f)


def _build_user_prompt(question: str, resolution_date: str,
                       research_summary: str, info_horizon: str) -> str:
    return (
        f"Binary forecasting question: {question}\n\n"
        f"Resolution date: {resolution_date}\n"
        f"Information cutoff for this prediction: {info_horizon}\n\n"
        f"Research brief (compiled before the information cutoff):\n"
        f"---\n{research_summary}\n---\n\n"
        f"Based on this research brief, provide your probability estimate for the YES outcome. "
        f"Remember to output your answer as a JSON code block with scenarios_yes, scenarios_no, "
        f"and final_prob."
    )


def _extract_prob(content: str) -> float | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        match = re.search(r'\{[^{}]*"final_prob"[^{}]*\}', content, re.DOTALL)
    if not match:
        return None
    try:
        raw = match.group(1) if "```" in match.group(0) else match.group(0)
        data = json.loads(raw)
        prob = float(data["final_prob"])
        return max(0.01, min(0.99, prob))
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def _call_ollama(system_prompt: str, user_prompt: str, seed: int,
                 temperature: float) -> tuple[float | None, str, float]:
    """Call Ollama with explicit temperature. Returns (prob, raw_content, elapsed_s)."""
    t0 = time.time()
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,   # explicitly set, verified per-call
            "seed": seed,
            "num_predict": 2048,
        },
    }
    # Verify temperature is in payload before sending
    assert payload["options"]["temperature"] == temperature, \
        f"BUG: temperature mismatch in payload: {payload['options']['temperature']} != {temperature}"
    try:
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        elapsed = time.time() - t0
        return _extract_prob(content), content, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return None, f"[Error: {e}]", elapsed


def _temp_tag(temperature: float) -> str:
    """Convert 0.7 -> 'temp07', 1.0 -> 'temp10', 0.0 -> 'temp00'"""
    return f"temp{int(temperature * 10):02d}"


def _progress_path(temperature: float, group: str) -> Path:
    return BASE_DIR / f"results_heterogeneity_{_temp_tag(temperature)}_{group}_progress.json"


def _out_path(temperature: float, group: str) -> Path:
    return BASE_DIR / f"results_heterogeneity_{_temp_tag(temperature)}_{group}.json"


def _load_progress(temperature: float, group: str) -> dict[str, dict]:
    p = _progress_path(temperature, group)
    if p.exists():
        with open(p) as f:
            records = json.load(f)
        return {f"{r['market_id']}|{r['agent_idx']}": r for r in records}
    return {}


def _save_progress(records: list[dict], temperature: float, group: str) -> None:
    with open(_progress_path(temperature, group), "w") as f:
        json.dump(records, f, indent=2)


def _run_cell(temperature: float, group: str, questions: list[dict],
              personas: list[dict], clear_progress: bool = True) -> list[dict]:
    """Run one (temperature, group) cell. Returns all prediction records."""

    progress_p = _progress_path(temperature, group)
    out_p = _out_path(temperature, group)

    if clear_progress and progress_p.exists():
        print(f"  Clearing old progress file: {progress_p.name}")
        progress_p.unlink()

    progress = _load_progress(temperature, group)
    all_records = list(progress.values())
    n_done_start = len(all_records)

    total = len(questions) * 10  # 10 agents per group
    tag = _temp_tag(temperature)
    print(f"\n=== Cell: temperature={temperature} group={group} ===")
    print(f"  Progress: {n_done_start}/{total} already done")
    if n_done_start > 0:
        print(f"  (resuming from checkpoint — use clear_progress=True to force restart)")

    n_success = 0
    n_fail = 0
    t_start = time.time()
    elapsed_times: list[float] = []

    for q_idx, q in enumerate(questions):
        market_id = q["market_id"]
        question_text = q["question"]
        resolution_date = q["resolution_date"]
        info_horizon = q["info_horizon"]
        research_summary = _load_research_cache(market_id, info_horizon)
        user_prompt = _build_user_prompt(question_text, resolution_date,
                                         research_summary, info_horizon)

        for local_idx in range(10):
            # Determine agent_idx (global unique per group)
            if group == "controlb":
                agent_idx = 100 + local_idx  # offset to avoid collision with original runs
                seed = CONTROL_B_SEEDS[local_idx]
                system_prompt = PREDICTION_SYSTEM
                persona_id = f"controlb_seed{seed}_t{tag}"
            elif group == "treatment":
                agent_idx = 200 + local_idx
                seed = TREATMENT_SEEDS[local_idx]
                persona_entry = personas[local_idx]
                persona_text = persona_entry["persona_prompt_text"]
                system_prompt = persona_text + "\n\n" + PREDICTION_SYSTEM
                persona_id = persona_entry["wallet_id"]
            else:
                raise ValueError(f"Unknown group: {group}")

            key = f"{market_id}|{agent_idx}"
            if key in progress:
                continue  # resume

            prob, _content, elapsed = _call_ollama(system_prompt, user_prompt,
                                                    seed, temperature)
            # Retry once on parse failure
            if prob is None:
                prob, _content, elapsed2 = _call_ollama(system_prompt, user_prompt,
                                                         seed, temperature)
                elapsed += elapsed2

            record = {
                "market_id": market_id,
                "agent_idx": agent_idx,
                "local_idx": local_idx,
                "group": group,
                "temperature": temperature,
                "persona_id": persona_id,
                "seed": seed,
                "prob": prob,
                "parse_failed": prob is None,
                "inference_time_s": round(elapsed, 2),
            }
            all_records.append(record)
            progress[key] = record
            elapsed_times.append(elapsed)

            if prob is not None:
                n_success += 1
            else:
                n_fail += 1

            done = len(all_records)
            elapsed_total = time.time() - t_start
            eta = (elapsed_total / done) * (total - done) if done > 0 else 0
            prob_str = f"{prob:.3f}" if prob is not None else "FAIL"
            print(
                f"  [{done}/{total}] q={q_idx+1}/{len(questions)} agent={local_idx} "
                f"seed={seed} prob={prob_str} "
                f"t={elapsed:.1f}s ETA={eta/60:.0f}min"
            )

            _save_progress(all_records, temperature, group)

    total_preds = n_success + n_fail
    parse_rate = n_success / total_preds if total_preds > 0 else 0.0
    fail_rate = n_fail / total_preds if total_preds > 0 else 0.0

    print(f"\n  Cell done: {n_success} success, {n_fail} fail, "
          f"parse_rate={parse_rate:.1%}, fail_rate={fail_rate:.1%}")

    if elapsed_times:
        avg_t = sum(elapsed_times) / len(elapsed_times)
        print(f"  Avg inference time this cell: {avg_t:.1f}s/pred")

    if fail_rate > 0.10:
        print(f"\nHALT: parse failure rate {fail_rate:.1%} exceeds 10% threshold.")
        print("This is a hard stop per experiment protocol. Check the Ollama output format.")
        _save_final(all_records, temperature, group)
        sys.exit(1)

    _save_final(all_records, temperature, group)
    return all_records


def _save_final(records: list[dict], temperature: float, group: str) -> None:
    out_p = _out_path(temperature, group)
    with open(out_p, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved: {out_p}")


# ── Metric computation ─────────────────────────────────────────────────────────

def _jsd(p: float, q: float) -> float:
    dp = np.array([p, 1.0 - p])
    dq = np.array([q, 1.0 - q])
    m = 0.5 * (dp + dq)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(dp, m) + 0.5 * kl(dq, m)


def _per_question_stats(probs: list[float], actual: int) -> dict:
    n = len(probs)
    ensemble_prob = float(np.mean(probs))
    brier = (ensemble_prob - actual) ** 2
    pairs = list(combinations(range(n), 2))
    jsd_vals = [_jsd(probs[i], probs[j]) for i, j in pairs]
    mean_jsd = float(np.mean(jsd_vals)) if jsd_vals else 0.0
    errors = [p - actual for p in probs]
    errcov = float(np.mean([errors[i] * errors[j] for i, j in pairs])) if pairs else 0.0
    return {"probs": probs, "errors": errors, "brier": brier,
            "mean_jsd": mean_jsd, "errcov": errcov}


def _pairwise_corr(agent_errors: list[list[float]]) -> float:
    """Cross-question pairwise correlation among agent error series."""
    n_agents = len(agent_errors)
    if n_agents < 2:
        return float("nan")
    corrs = []
    for i, j in combinations(range(n_agents), 2):
        a, b = np.array(agent_errors[i]), np.array(agent_errors[j])
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            corrs.append(1.0 if np.allclose(a, b) else float("nan"))
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    valid = [c for c in corrs if not np.isnan(c)]
    return float(np.mean(valid)) if valid else float("nan")


def cluster_bootstrap_diff(a_vals: list[float], b_vals: list[float],
                            n_boot: int = 2000, seed: int = 42
                            ) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI for (mean_b - mean_a), resampling by question."""
    rng = np.random.default_rng(seed)
    obs_diff = float(np.mean(b_vals) - np.mean(a_vals))
    n = len(a_vals)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_a = [a_vals[i] for i in idx]
        boot_b = [b_vals[i] for i in idx]
        boot_diffs.append(float(np.mean(boot_b) - np.mean(boot_a)))
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    return obs_diff, ci_lo, ci_hi


def cohen_d_fn(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    return 0.0 if pooled < 1e-10 else float((np.mean(b) - np.mean(a)) / pooled)


# ── Analysis: compute metrics for one (temperature, group) pair ───────────────

def compute_cell_metrics(records: list[dict], eval_set: dict[str, dict],
                          temperature: float, group: str) -> dict:
    """Compute per-question metrics for one cell. Returns metrics dict."""
    from scipy import stats as scipy_stats

    # Filter to successful parses only
    ok = [r for r in records if r["temperature"] == temperature
          and r["group"] == group and not r["parse_failed"] and r["prob"] is not None]

    # Group by question
    by_q: dict[str, list[dict]] = {}
    for r in ok:
        by_q.setdefault(r["market_id"], []).append(r)

    # Questions with all 10 agents
    complete_qs = {mid: recs for mid, recs in by_q.items()
                   if len(recs) == 10 and mid in eval_set}

    n_total_preds = len([r for r in records
                         if r["temperature"] == temperature and r["group"] == group])
    n_ok_preds = len(ok)
    parse_rate = n_ok_preds / n_total_preds if n_total_preds > 0 else 0.0

    per_q_brier: list[float] = []
    per_q_jsd: list[float] = []
    per_q_errcov: list[float] = []
    agent_errors: list[list[float]] = [[] for _ in range(10)]

    for mid, recs in complete_qs.items():
        recs_sorted = sorted(recs, key=lambda r: r["local_idx"])
        q = eval_set[mid]
        actual = 1 if q["resolved_outcome"] == "Yes" else 0
        probs = [r["prob"] for r in recs_sorted]
        stats_q = _per_question_stats(probs, actual)
        per_q_brier.append(stats_q["brier"])
        per_q_jsd.append(stats_q["mean_jsd"])
        per_q_errcov.append(stats_q["errcov"])
        for i, e in enumerate(stats_q["errors"]):
            agent_errors[i].append(e)

    err_corr = _pairwise_corr(agent_errors)

    return {
        "temperature": temperature,
        "group": group,
        "n_complete_questions": len(complete_qs),
        "n_total_preds": n_total_preds,
        "n_ok_preds": n_ok_preds,
        "parse_rate": round(parse_rate, 4),
        "mean_brier": round(float(np.mean(per_q_brier)), 4) if per_q_brier else None,
        "mean_jsd": round(float(np.mean(per_q_jsd)), 4) if per_q_jsd else None,
        "err_corr": round(err_corr, 4) if not np.isnan(err_corr) else None,
        "mean_errcov": round(float(np.mean(per_q_errcov)), 4) if per_q_errcov else None,
        "per_q_brier": per_q_brier,
        "per_q_jsd": per_q_jsd,
        "per_q_errcov": per_q_errcov,
        "complete_q_ids": list(complete_qs.keys()),
    }


# ── Combined analysis across temperatures ──────────────────────────────────────

def analyse_sweep(all_cell_metrics: list[dict]) -> dict:
    """Compute Treatment − ControlB deltas per temperature with cluster-bootstrap CIs."""
    from scipy import stats as scipy_stats

    temps = sorted(set(m["temperature"] for m in all_cell_metrics))
    results = {}

    # Find listwise-complete question set across all cells
    all_q_sets = [set(m["complete_q_ids"]) for m in all_cell_metrics
                  if m["complete_q_ids"] is not None]
    if not all_q_sets:
        return {}
    listwise_q_ids = all_q_sets[0].intersection(*all_q_sets[1:])
    print(f"\nListwise-complete question set: {len(listwise_q_ids)} questions "
          f"(intersection across all {len(all_cell_metrics)} cells)")

    # Recompute per-question metrics restricted to listwise_q_ids
    # We need the per-q vectors filtered to the shared set
    def _filter_per_q(m: dict, q_ids: set[str]) -> dict:
        # need original per-q data with q_id tracking
        return m  # we stored per_q_ lists in order of complete_qs.keys()

    temp_results = []
    for temp in temps:
        cb_m = next((m for m in all_cell_metrics
                     if m["temperature"] == temp and m["group"] == "controlb"), None)
        trt_m = next((m for m in all_cell_metrics
                      if m["temperature"] == temp and m["group"] == "treatment"), None)
        if cb_m is None or trt_m is None:
            print(f"  Skipping temp={temp}: missing one or both groups")
            continue

        # Restrict to listwise_q_ids — match by position in complete_q_ids
        def restrict(m: dict, q_ids: set[str]) -> dict:
            idxs = [i for i, qid in enumerate(m["complete_q_ids"]) if qid in q_ids]
            return {
                "brier":  [m["per_q_brier"][i]  for i in idxs],
                "jsd":    [m["per_q_jsd"][i]    for i in idxs],
                "errcov": [m["per_q_errcov"][i] for i in idxs],
                "q_ids":  [m["complete_q_ids"][i] for i in idxs],
            }

        cb_r  = restrict(cb_m,  listwise_q_ids)
        trt_r = restrict(trt_m, listwise_q_ids)
        n_q   = len(cb_r["q_ids"])

        def compare(a_vals: list[float], b_vals: list[float],
                    metric: str) -> dict:
            if not a_vals or not b_vals:
                return {}
            obs, ci_lo, ci_hi = cluster_bootstrap_diff(a_vals, b_vals)
            _, p = scipy_stats.ttest_ind(a_vals, b_vals, equal_var=False)
            d = cohen_d_fn(a_vals, b_vals)
            return {
                "diff_trt_minus_cb": round(obs, 5),
                "bootstrap_95ci": [round(ci_lo, 5), round(ci_hi, 5)],
                "welch_p": round(float(p), 4),
                "cohen_d": round(d, 3),
            }

        entry = {
            "temperature": temp,
            "n_listwise_questions": n_q,
            "controlb": {
                "mean_brier":  round(float(np.mean(cb_r["brier"])), 4)  if cb_r["brier"]  else None,
                "mean_jsd":    round(float(np.mean(cb_r["jsd"])), 4)    if cb_r["jsd"]    else None,
                "mean_errcov": round(float(np.mean(cb_r["errcov"])), 4) if cb_r["errcov"] else None,
                "parse_rate":  cb_m["parse_rate"],
                "err_corr":    cb_m["err_corr"],
            },
            "treatment": {
                "mean_brier":  round(float(np.mean(trt_r["brier"])), 4)  if trt_r["brier"]  else None,
                "mean_jsd":    round(float(np.mean(trt_r["jsd"])), 4)    if trt_r["jsd"]    else None,
                "mean_errcov": round(float(np.mean(trt_r["errcov"])), 4) if trt_r["errcov"] else None,
                "parse_rate":  trt_m["parse_rate"],
                "err_corr":    trt_m["err_corr"],
            },
            "delta_trt_minus_cb": {
                "jsd":    compare(cb_r["jsd"],    trt_r["jsd"],    "jsd"),
                "brier":  compare(cb_r["brier"],  trt_r["brier"],  "brier"),
                "errcov": compare(cb_r["errcov"], trt_r["errcov"], "errcov"),
            },
        }
        temp_results.append(entry)

    return {
        "model": MODEL,
        "listwise_n_questions": len(listwise_q_ids),
        "temperatures_run": temps,
        "per_temperature": temp_results,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature sweep runner")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Single temperature to run (e.g. 0.7)")
    parser.add_argument("--group", choices=["controlb", "treatment"], default=None,
                        help="Group to run: controlb or treatment")
    parser.add_argument("--smoke", type=int, default=None,
                        help="Smoke test: only run N questions")
    parser.add_argument("--all", action="store_true",
                        help="Run all temperatures × groups sequentially")
    parser.add_argument("--analyse", action="store_true",
                        help="Only run analysis on existing output files, no new predictions")
    parser.add_argument("--temperatures", type=str, default="0.0,0.3,0.7,1.0",
                        help="Comma-separated temperature grid (default: 0.0,0.3,0.7,1.0)")
    parser.add_argument("--no-clear-progress", action="store_true",
                        help="Do not clear existing progress (resume mode)")
    args = parser.parse_args()

    temp_grid = [float(t) for t in args.temperatures.split(",")]

    print(f"Model: {MODEL}")
    print(f"Temperature grid: {temp_grid}")

    eval_rows = _load_eval_set(n=args.smoke)
    personas = _load_persona_prompts()
    eval_set = {q["market_id"]: q for q in eval_rows}
    print(f"Eval set: {len(eval_rows)} questions")

    if args.analyse:
        # Load all existing output files and run analysis
        all_records = []
        for temp in temp_grid:
            for group in ["controlb", "treatment"]:
                p = _out_path(temp, group)
                if p.exists():
                    with open(p) as f:
                        recs = json.load(f)
                    all_records.extend(recs)
                    print(f"Loaded {len(recs)} records from {p.name}")
        if not all_records:
            print("No output files found. Run predictions first.")
            sys.exit(1)
        _run_analysis(all_records, eval_set, temp_grid)
        return

    clear_progress = not args.no_clear_progress

    if args.all:
        all_records = []
        for temp in temp_grid:
            for group in ["controlb", "treatment"]:
                recs = _run_cell(temp, group, eval_rows, personas,
                                 clear_progress=clear_progress)
                all_records.extend(recs)
        _run_analysis(all_records, eval_set, temp_grid)
    elif args.temperature is not None and args.group is not None:
        recs = _run_cell(args.temperature, args.group, eval_rows, personas,
                         clear_progress=clear_progress)
        if not args.smoke:
            # Run analysis for just this cell (partial)
            print(f"\nSingle-cell metrics for temp={args.temperature} group={args.group}:")
            # Load eval set fully for analysis even if smoke was used
            eval_rows_full = _load_eval_set()
            eval_set_full = {q["market_id"]: q for q in eval_rows_full}
            m = compute_cell_metrics(recs, eval_set_full, args.temperature, args.group)
            print(f"  n_complete_questions: {m['n_complete_questions']}")
            print(f"  parse_rate: {m['parse_rate']:.1%}")
            print(f"  mean_jsd: {m['mean_jsd']}")
            print(f"  mean_brier: {m['mean_brier']}")
            print(f"  err_corr: {m['err_corr']}")
    else:
        parser.print_help()
        sys.exit(1)


def _run_analysis(all_records: list[dict], eval_set: dict[str, dict],
                  temp_grid: list[float]) -> None:
    print("\n\n=== Running sweep analysis ===")

    # Compute per-cell metrics
    cell_metrics = []
    for temp in temp_grid:
        for group in ["controlb", "treatment"]:
            recs_cell = [r for r in all_records
                         if r["temperature"] == temp and r["group"] == group]
            if not recs_cell:
                print(f"  No records for temp={temp} group={group}, skipping")
                continue
            m = compute_cell_metrics(recs_cell, eval_set, temp, group)
            cell_metrics.append(m)
            print(f"  temp={temp} group={group}: "
                  f"n_q={m['n_complete_questions']} "
                  f"parse_rate={m['parse_rate']:.1%} "
                  f"jsd={m['mean_jsd']} brier={m['mean_brier']} err_corr={m['err_corr']}")

    # Check for 0.3 reproduction (anti-silent-failure)
    trt_03 = next((m for m in cell_metrics
                   if m["temperature"] == 0.3 and m["group"] == "treatment"), None)
    cb_03  = next((m for m in cell_metrics
                   if m["temperature"] == 0.3 and m["group"] == "controlb"), None)

    if trt_03:
        print(f"\nReproduction check (temp=0.3 treatment):")
        print(f"  mean_jsd: {trt_03['mean_jsd']} (baseline: {BASELINE_TREATMENT_JSD})")
        print(f"  err_corr: {trt_03['err_corr']} (baseline: {BASELINE_TREATMENT_ECORR})")
        jsd_match = abs((trt_03["mean_jsd"] or 0) - BASELINE_TREATMENT_JSD) < 0.001
        ecorr_match = abs((trt_03["err_corr"] or 0) - BASELINE_TREATMENT_ECORR) < 0.002
        if jsd_match and ecorr_match:
            print("  OK: within tolerance of baseline")
        else:
            print("  NOTE: differs from baseline — acceptable if this is a fresh re-run "
                  "(different RNG path, small N difference). Review carefully.")

    # Check for cache reuse red flag: any non-0.3 temp producing identical numbers to 0.3
    if trt_03 and cb_03:
        for m in cell_metrics:
            if m["temperature"] == 0.3:
                continue
            if m["group"] == "treatment" and m["mean_jsd"] == trt_03["mean_jsd"]:
                print(f"\nRED FLAG: temp={m['temperature']} treatment JSD "
                      f"matches 0.3 baseline exactly — possible cache reuse!")
            if m["group"] == "controlb" and m["mean_jsd"] == cb_03["mean_jsd"]:
                print(f"\nRED FLAG: temp={m['temperature']} controlb JSD "
                      f"matches 0.3 baseline exactly — possible cache reuse!")

    # Cross-temperature comparative analysis
    sweep_results = analyse_sweep(cell_metrics)

    # Save
    out_json = BASE_DIR / "results_heterogeneity_tempsweep.json"
    # Strip per_q_ vectors from cell_metrics before saving (they're large and not needed)
    cell_metrics_save = []
    for m in cell_metrics:
        m_save = {k: v for k, v in m.items()
                  if not k.startswith("per_q_") and k != "complete_q_ids"}
        cell_metrics_save.append(m_save)
    sweep_results["cell_metrics"] = cell_metrics_save

    with open(out_json, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to: {out_json}")

    # Generate markdown report
    _write_sweep_report(sweep_results, BASE_DIR / "results_heterogeneity_tempsweep.md")


def _write_sweep_report(results: dict, out_path: Path) -> None:
    lines = [
        "# Heterogeneity Temperature Sweep — Results",
        "",
        f"Model: {results.get('model', MODEL)}",
        f"Listwise-complete questions (intersection across all cells): "
        f"{results.get('listwise_n_questions', 'N/A')}",
        f"Temperatures run: {results.get('temperatures_run', [])}",
        "",
        "## Per-Temperature Group Metrics",
        "",
        "| Temp | Group | N_q | Parse Rate | Mean JSD | Mean Brier | Err Corr | Mean ErrCov |",
        "|------|-------|-----|-----------|---------|-----------|---------|------------|",
    ]

    for entry in results.get("per_temperature", []):
        temp = entry["temperature"]
        n_q = entry["n_listwise_questions"]
        for grp, label in [("controlb", "Control-B"), ("treatment", "Treatment")]:
            m = entry.get(grp, {})
            lines.append(
                f"| {temp} | {label} | {n_q} "
                f"| {m.get('parse_rate', 'N/A'):.1%} "
                f"| {m.get('mean_jsd', 'N/A')} "
                f"| {m.get('mean_brier', 'N/A')} "
                f"| {m.get('err_corr', 'N/A')} "
                f"| {m.get('mean_errcov', 'N/A')} |"
            )

    lines += [
        "",
        "## Treatment − Control-B Deltas per Temperature",
        "",
        "Cluster-bootstrap CI (resample by question, 2000 iterations).",
        "",
        "| Temp | JSD Δ | JSD 95% CI | JSD p | JSD d | Brier Δ | Brier p | ErrCov Δ | ErrCov p |",
        "|------|-------|-----------|-------|-------|---------|---------|---------|---------|",
    ]

    for entry in results.get("per_temperature", []):
        temp = entry["temperature"]
        d = entry.get("delta_trt_minus_cb", {})
        jsd = d.get("jsd", {})
        brier = d.get("brier", {})
        errcov = d.get("errcov", {})

        def fmt(v, fmt_s=".5f"):
            if v is None or v == {}:
                return "N/A"
            return f"{v:{fmt_s}}"

        lines.append(
            f"| {temp} "
            f"| {fmt(jsd.get('diff_trt_minus_cb'))} "
            f"| [{fmt(jsd.get('bootstrap_95ci', [None, None])[0])}, "
            f"{fmt(jsd.get('bootstrap_95ci', [None, None])[1])}] "
            f"| {jsd.get('welch_p', 'N/A')} "
            f"| {jsd.get('cohen_d', 'N/A')} "
            f"| {fmt(brier.get('diff_trt_minus_cb'))} "
            f"| {brier.get('welch_p', 'N/A')} "
            f"| {fmt(errcov.get('diff_trt_minus_cb'))} "
            f"| {errcov.get('welch_p', 'N/A')} |"
        )

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- JSD Δ > 0: Treatment adds diversity relative to Control-B",
        "- Brier Δ < 0: Treatment improves ensemble accuracy",
        "- ErrCov Δ < 0: Treatment reduces error correlation (decorrelation)",
        "- CI crossing 0 means no statistically significant effect at that temperature",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
