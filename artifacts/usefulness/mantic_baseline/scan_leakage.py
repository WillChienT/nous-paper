"""Exhaustive leakage scan over the Mantic research_cache.

The audit lesson (2026-05-27): never judge "leakage is gone" from one or two
cache files plus a single smoke test. This script surfaces EVERY fingerprint hit
across ALL cache files with its full sentence, the file's ``info_horizon`` and the
question's ``resolution_date``, so the human reads every hit rather than sampling.

For each fingerprint hit it best-effort extracts any date in the sentence and
classifies:
  - POST_HORIZON : a date in the sentence is strictly after info_horizon  -> real leak
  - PRE_HORIZON  : all dates in the sentence are <= info_horizon          -> fine
  - DATELESS     : an outcome-style statement with no date                -> RESIDUAL leak
                   (the fundamental limitation regex date-filtering cannot fix)
Already-redacted spans ("[REDACTED") are skipped (the content filter caught them).

Auto-classification is a HINT, not a verdict. Read POST_HORIZON and DATELESS hits.

Usage:
    python scan_leakage.py            # scans research_cache/*.json
    python scan_leakage.py --strong   # only the high-precision fingerprint tier
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_CACHE_GLOB = str(BASE_DIR / "research_cache" / "*.json")
DEFAULT_EVAL_SET = BASE_DIR / "data" / "eval_set_mantic_baseline.jsonl"

# High-precision: phrases that almost always describe the realised outcome of THE event.
STRONG = re.compile(
    r"(final score|full[- ]?time|ended in a|ended with|the match ended|was won by|"
    r"concluded in|defeated .{0,30}\b\d+\s*[-–]\s*\d+|won \d+\s*[-–]\s*\d+|"
    r"actual (outcome|result)|resolved (yes|no)|the outcome was|retrospective)",
    re.I,
)
# Lower precision: outcome verbs that also appear in pre-game prose (many false positives).
WEAK = re.compile(
    r"\b(won|lost|beat|defeated|drew|draw|victory|defeat|thrashed|edged|knocked out|"
    r"advanced to|eliminated|clinched|secured the|clean sheet)\b",
    re.I,
)
# Bare scoreline like "2-1", "3–0" (strong leak signal in a sports cache).
SCORELINE = re.compile(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b")

ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_FULL = ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]
_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
MONTHS = {m: i for i, m in enumerate(_FULL, start=1)}
MONTHS.update({a: i for i, a in zip(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12], _ABBR)})
_MO = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|" \
      r"sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
NAT = re.compile(rf"\b{_MO}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b", re.I)
NAT2 = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MO}\.?\s+(20\d{{2}})\b", re.I)
# Month + year with no day (e.g. "November 2025"). End-of-month upper bound for comparison.
MONYEAR = re.compile(rf"\b{_MO}\.?\s+(20\d{{2}})\b", re.I)


def _mon(token: str) -> int:
    return MONTHS[token.lower().rstrip(".")]


def extract_dates(s: str) -> list[date]:
    """Day-precision dates found in the sentence (abbreviated months supported)."""
    out = []
    for y, m, d in ISO.findall(s):
        try:
            out.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    for mo, d, y in NAT.findall(s):
        try:
            out.append(date(int(y), _mon(mo), int(d)))
        except ValueError:
            pass
    for d, mo, y in NAT2.findall(s):
        try:
            out.append(date(int(y), _mon(mo), int(d)))
        except ValueError:
            pass
    return out


def extract_month_years(s: str) -> list[tuple[int, int]]:
    """(year, month) pairs from day-less 'November 2025' style references."""
    return [(int(y), _mon(mo)) for mo, y in MONYEAR.findall(s)]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\n])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def load_labels(eval_set: Path) -> dict[str, dict]:
    labels = {}
    for line in eval_set.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            labels[str(row["market_id"])] = row
    return labels


def classify(sent: str, horizon: date) -> tuple[str, list[date]]:
    dates = extract_dates(sent)
    if dates:
        if any(d > horizon for d in dates):
            return "POST_HORIZON", dates
        return "PRE_HORIZON", dates
    # Fall back to day-less month-year references ("November 2025").
    mys = extract_month_years(sent)
    if mys:
        hz_my = (horizon.year, horizon.month)
        if any(my > hz_my for my in mys):
            return "POST_HORIZON", []  # month strictly after horizon's month
        if all(my < hz_my for my in mys):
            return "PRE_HORIZON", []    # whole month precedes horizon
        return "DATELESS", []           # same month as horizon: genuinely ambiguous
    return "DATELESS", []


def scan(strong_only: bool = False, cache_glob: str = DEFAULT_CACHE_GLOB,
         eval_set: Path = DEFAULT_EVAL_SET) -> None:
    labels = load_labels(eval_set)
    files = sorted(glob.glob(cache_glob))
    print(f"Scanning {len(files)} cache files from {cache_glob} (strong_only={strong_only})\n")

    tally = {"POST_HORIZON": 0, "DATELESS": 0, "PRE_HORIZON": 0}
    files_with_strong = set()
    files_with_dateless = set()

    for f in files:
        name = Path(f).stem  # {market_id}_{horizon}
        market_id, _, hz = name.partition("_")
        try:
            horizon = date.fromisoformat(hz)
        except ValueError:
            horizon = None
        lab = labels.get(market_id, {})
        cat = lab.get("category", "?")
        res_date = lab.get("resolution_date", "?")
        question = lab.get("question", "")[:80]

        d = json.loads(Path(f).read_text())
        # Only scan text that actually reached the model / prediction.
        blob = "\n".join(str(d.get(k, "")) for k in ("summary", "raw_content"))

        hits = []
        seen_sents = set()
        for sent in sentences(blob):
            if "[REDACTED" in sent or sent in seen_sents:
                continue
            seen_sents.add(sent)
            is_strong = bool(STRONG.search(sent))
            is_weak = (not strong_only) and bool(WEAK.search(sent) or SCORELINE.search(sent))
            if not (is_strong or is_weak):
                continue
            kind, dates = classify(sent, horizon) if horizon else ("DATELESS", [])
            hits.append((("STRONG" if is_strong else "weak"), kind, dates, sent[:200]))

        if not hits:
            continue

        strong_hits = [h for h in hits if h[0] == "STRONG"]
        if strong_hits:
            files_with_strong.add(market_id)
        if any(h[1] == "DATELESS" and h[0] == "STRONG" for h in hits):
            files_with_dateless.add(market_id)

        print(f"=== {market_id} [{cat}] horizon={hz} resolves={res_date}")
        print(f"    Q: {question}")
        for tier, kind, dates, sent in hits:
            if strong_only and tier != "STRONG":
                continue
            tally[kind] = tally.get(kind, 0) + 1
            ds = ",".join(d.isoformat() for d in dates) if dates else "-"
            flag = "  <<<" if (tier == "STRONG" and kind in ("POST_HORIZON", "DATELESS")) else ""
            print(f"    [{tier:6}/{kind:12} dates={ds}]{flag} {sent}")
        print()

    print("=" * 70)
    print(f"Files with STRONG fingerprint hits: {len(files_with_strong)}/{len(files)}")
    print(f"  ids: {sorted(files_with_strong)}")
    print(f"Files with STRONG + DATELESS (residual-leak candidates): {len(files_with_dateless)}/{len(files)}")
    print(f"  ids: {sorted(files_with_dateless)}")
    print(f"Hit tally by class: {tally}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exhaustive leakage scan over a research cache.")
    ap.add_argument("--strong", action="store_true", help="only high-precision fingerprint tier")
    ap.add_argument("--cache-dir", default=None,
                    help="cache dir to scan (default: research_cache/); files matched as <dir>/*.json")
    ap.add_argument("--eval-set", default=None, help="eval-set jsonl for market_id labels")
    a = ap.parse_args()
    cache_glob = str(Path(a.cache_dir) / "*.json") if a.cache_dir else DEFAULT_CACHE_GLOB
    if a.cache_dir and not Path(a.cache_dir).is_absolute():
        cache_glob = str(BASE_DIR / a.cache_dir / "*.json")
    eval_set = Path(a.eval_set) if a.eval_set else DEFAULT_EVAL_SET
    if a.eval_set and not eval_set.is_absolute():
        eval_set = BASE_DIR / a.eval_set
    scan(strong_only=a.strong, cache_glob=cache_glob, eval_set=eval_set)
