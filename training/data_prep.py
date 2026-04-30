# data_prep.py  (UPDATED — Multi-domain version)
#
# Step 1 of the BLAIR training pipeline.
# NEW: load_and_split_multi() loads multiple JSONL files,
#      merges them, re-sorts by timestamp, then splits 8:1:1.
#      This matches the paper's approach of training across all 33 domains.
#
# Paper reference (Section 3.1):
#   "We split the reviews into training, validation, and test sets by absolute
#    timestamps ... in a ratio of 8:1:1. These two timestamps are used to split
#    data for both pretraining and all downstream evaluation tasks."

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

# ── Field name maps ────────────────────────────────────────────────────────────
_TEXT_FIELDS   = ["reviewText", "text", "review", "content", "comment", "body"]
_RATING_FIELDS = ["overall", "rating", "stars", "score"]
_ASIN_FIELDS   = ["parent_asin", "parentAsin", "asin"]
_TITLE_FIELDS  = ["title", "product_title", "name"]
_TIME_FIELDS   = ["unixReviewTime", "timestamp", "unix_time", "time"]


def _parse_record(data: dict, source_domain: str = "") -> Optional[dict]:
    """
    Parse one JSONL line into a standardised record dict.
    Returns None if no review text can be found.
    """
    text = next((str(data[f]).strip() for f in _TEXT_FIELDS if data.get(f)), None)
    if not text:
        return None

    rating      = next((data[f] for f in _RATING_FIELDS if f in data), None)
    asin        = data.get("asin", "N/A")
    parent_asin = next((data[f] for f in _ASIN_FIELDS if data.get(f)), asin)
    title       = next((str(data[f]).strip() for f in _TITLE_FIELDS if data.get(f)), "")
    timestamp   = next((data[f] for f in _TIME_FIELDS if f in data), 0)
    reviewer    = data.get("reviewerName", "Anonymous")
    date_str    = data.get("reviewTime", "")
    summary     = data.get("summary", "")

    features    = data.get("feature", "") or data.get("features", "")
    description = data.get("description", "")
    if isinstance(features, list):
        features = " ".join(features)
    if isinstance(description, list):
        description = " ".join(description)

    item_metadata = " ".join(filter(None, [title, features, description])).strip()
    if not item_metadata:
        item_metadata = title or f"Product {parent_asin}"

    return {
        "text":          text,
        "item_metadata": item_metadata,
        "rating":        rating,
        "asin":          asin,
        "parent_asin":   parent_asin,
        "item_id":       parent_asin if parent_asin != "N/A" else asin,
        "title":         title,
        "reviewer":      reviewer,
        "date":          date_str,
        "summary":       summary,
        "timestamp":     int(timestamp) if timestamp else 0,
        "domain":        source_domain,   # NEW: track which category this came from
    }


def _load_one_file(
    jsonl_path:  str,
    max_records: int = 0,
    domain_name: str = "",
) -> list[dict]:
    """
    Load records from a single JSONL file.
    Returns list of standardised record dicts.
    """
    if not domain_name:
        domain_name = Path(jsonl_path).stem  # filename without extension

    records: list[dict] = []

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            record = _parse_record(data, source_domain=domain_name)
            if record:
                records.append(record)

            if max_records and len(records) >= max_records:
                break

    return records


def load_and_split_multi(
    jsonl_paths:  list[str],
    output_dir:   str   = "data_splits",
    max_per_file: int   = 2000,
    train_ratio:  float = 0.8,
    val_ratio:    float = 0.1,
    shuffle_seed: int   = 42,
) -> dict[str, list[dict]]:
    """
    Load MULTIPLE JSONL files, merge them, sort by timestamp, split 8:1:1.

    This is the multi-domain version matching the paper's approach.
    The paper merged all 33 Amazon categories before splitting.

    Parameters
    ----------
    jsonl_paths  : list of paths to JSONL files (one per domain)
    output_dir   : where to write train/val/test.jsonl
    max_per_file : max records to load from each file (0 = unlimited)
    train_ratio  : fraction for training (0.8)
    val_ratio    : fraction for validation (0.1)
    shuffle_seed : random seed for shuffling within same timestamp

    Returns
    -------
    dict with keys "train", "val", "test"
    """
    print(f"\nLoading {len(jsonl_paths)} datasets ...")
    print("-" * 50)

    all_records: list[dict] = []
    domain_counts: dict[str, int] = {}

    for path in jsonl_paths:
        domain = Path(path).stem
        print(f"  Loading {domain} from {path} ...")

        if not os.path.exists(path):
            print(f"    [WARNING] File not found: {path} — skipping.")
            continue

        records = _load_one_file(path, max_records=max_per_file, domain_name=domain)
        domain_counts[domain] = len(records)
        all_records.extend(records)
        print(f"    Loaded {len(records):,} records.")

    print(f"\n  Total records loaded: {len(all_records):,}")
    print(f"  Domains: {list(domain_counts.keys())}")

    # ── Domain distribution ────────────────────────────────────────────────
    print("\n  Records per domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 100, 30)
        print(f"    {domain:<35} {count:>6,}  {bar}")

    # ── Sort by timestamp (paper §3.1) ─────────────────────────────────────
    # Within the same timestamp, shuffle randomly so domains are interleaved
    random.seed(shuffle_seed)
    all_records.sort(key=lambda r: (r["timestamp"], random.random()))

    no_ts = sum(1 for r in all_records if r["timestamp"] == 0)
    if no_ts:
        print(f"\n  Note: {no_ts:,} records have no timestamp — placed at start.")

    # ── Split ─────────────────────────────────────────────────────────────
    n       = len(all_records)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = all_records[:n_train]
    val   = all_records[n_train : n_train + n_val]
    test  = all_records[n_train + n_val :]

    print(f"\n  Split → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in split:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Saved {out_path}")

    _print_stats("train", train)
    _print_stats("val",   val)
    _print_stats("test",  test)

    return {"train": train, "val": val, "test": test}


def load_and_split(
    jsonl_path:  str,
    output_dir:  str   = "data_splits",
    max_records: int   = 0,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
) -> dict[str, list[dict]]:
    """
    Single-file version (kept for backward compatibility).
    Calls load_and_split_multi internally.
    """
    return load_and_split_multi(
        jsonl_paths  = [jsonl_path],
        output_dir   = output_dir,
        max_per_file = max_records,
        train_ratio  = train_ratio,
        val_ratio    = val_ratio,
    )


def _print_stats(name: str, records: list[dict]) -> None:
    if not records:
        return
    ratings = [float(r["rating"]) for r in records if r.get("rating") is not None]
    avg_r   = sum(ratings) / len(ratings) if ratings else 0.0
    n_items = len({r["item_id"] for r in records})
    domains = len({r.get("domain","") for r in records})
    print(f"\n  [{name}]  records: {len(records):,}  |  "
          f"items: {n_items:,}  |  domains: {domains}  |  avg rating: {avg_r:.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare multi-domain BLAIR training data.")
    parser.add_argument("jsonl", nargs="+", help="One or more .jsonl files")
    parser.add_argument("--output",  default="data_splits", help="Output directory")
    parser.add_argument("--max",     type=int, default=2000, help="Max records per file (0=all)")
    args = parser.parse_args()

    load_and_split_multi(
        jsonl_paths  = args.jsonl,
        output_dir   = args.output,
        max_per_file = args.max,
    )
    print("\nData preparation complete.")
