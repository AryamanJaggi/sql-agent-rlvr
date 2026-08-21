"""Loads Spider (question, gold_sql, db) triples for the environment.

Combine two seperate Hugging face repos since each was missing something

  - xlangai/spider: (db_id, question, query) pairs. No .sqlite files.
  - target-benchmark/spider-corpus: the actual .sqlite files, written as
    train_database/<db_id>/<db_id>.sqlite (also validation_databse and 
    test_databse)

Everything downloaded gets cached under data/spider_cache/ (gitignored)
so this doesn't re-download 200 sqlite files on every run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

QA_REPO = "xlangai/spider"
DB_REPO = "target-benchmark/spider-corpus"

CACHE_DIR = Path(__file__).parent / "spider_cache"

Difficulty = Literal["easy", "medium", "hard", "extra"]

_SPLIT_TO_DB_DIR = {
    "train": "train_database",
    "validation": "validation_database",
}

_AGG_FUNCS = ("count(", "sum(", "avg(", "min(", "max(")
_SET_OPS = (" union ", " intersect ", " except ")


def estimate_difficulty(sql: str) -> Difficulty:
    """Replacement for Spider's official hardness metric. Good enough
    for purpose of this project.

    Their eval.py derives difficulty from much more detailed component
    count (aggregations, nested subqueries, set ops, WHERE/HAVING counts, and
    hand-tuned thresholds). Don't need paper-exact numbers here, just a
    usable difficulty knob for headroom calibration.
    """
    s = f" {sql.lower()} "

    components = 0
    components += s.count(" where ")
    components += s.count(" group by ")
    components += s.count(" having ")
    components += s.count(" order by ")
    components += s.count(" join ")
    components += sum(s.count(op) for op in _SET_OPS)
    components += s.count("(select")  # nested subquery
    components += 1 if any(fn in s for fn in _AGG_FUNCS) else 0

    if components == 0:
        return "easy"
    if components <= 2:
        return "medium"
    if components <= 4:
        return "hard"
    return "extra"


@dataclass(frozen=True)
class SpiderExample:
    db_id: str
    question: str
    gold_sql: str
    difficulty: Difficulty
    db_path: Path


def _db_dir_for_split(split: str) -> str:
    if split not in _SPLIT_TO_DB_DIR:
        raise ValueError(f"split must be one of {list(_SPLIT_TO_DB_DIR)}, got {split!r}")
    return _SPLIT_TO_DB_DIR[split]


_TRANSIENT_HF_ERRORS = (HfHubHTTPError, LocalEntryNotFoundError)


def _download_db(
    split: str,
    db_id: str,
    max_retries: int = 5,
    retry_delay_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download one Spider .sqlite file, retrying transient Hub failures.

    HF Hub rate-limits (429) a burst of many small-file downloads - hit
    for real after ~28 successful downloads in a row during data
    collection. hf_hub_download surfaces that as LocalEntryNotFoundError
    when there's no cached copy to fall back to yet, so both that and
    the underlying HTTP error are retried with exponential backoff
    rather than killing the whole run over one rate-limit window.
    """
    db_dir = _db_dir_for_split(split)
    filename = f"{db_dir}/{db_id}/{db_id}.sqlite"

    for attempt in range(max_retries):
        try:
            local_path = hf_hub_download(
                repo_id=DB_REPO,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(CACHE_DIR),
            )
            return Path(local_path)
        except _TRANSIENT_HF_ERRORS:
            if attempt == max_retries - 1:
                raise
            sleep(retry_delay_s * (2**attempt))


def load_spider(
    split: Literal["train", "validation"] = "train",
    difficulty: Difficulty | None = None,
    limit: int | None = None,
) -> list[SpiderExample]:
    """Load Spider examples, optionally filtered to one difficulty tier
    and/or capped at limit rows. DB files download lazily (and cache)
    only for db_ids actually present in the returned examples.
    """
    raw = load_dataset(QA_REPO, split=split)

    examples: list[SpiderExample] = []
    for row in raw:
        est_difficulty = estimate_difficulty(row["query"])
        if difficulty is not None and est_difficulty != difficulty:
            continue

        db_path = _download_db(split, row["db_id"])
        examples.append(
            SpiderExample(
                db_id=row["db_id"],
                question=row["question"],
                gold_sql=row["query"],
                difficulty=est_difficulty,
                db_path=db_path,
            )
        )

        if limit is not None and len(examples) >= limit:
            break

    return examples