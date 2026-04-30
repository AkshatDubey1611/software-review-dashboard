# pair_builder.py
#
# Step 2 of the BLAIR training pipeline.
#
# Paper reference (Section 2.3 — Training Objective):
#   "One instance of c and m in the collected AMAZON REVIEWS 2023 dataset is
#    the pair of user reviews (language context) and item features (item metadata).
#    The metadata m is a single sentence that contains a concise description of
#    an item."
#
# This file builds (context, metadata) pairs from the split JSONL files.
# Each pair is one training instance:
#   context  c  =  review title + review text           (what user wrote)
#   metadata m  =  product title + features + description  (what the item is)
#
# Paper data filtering (§3.1):
#   "We filter out training instances with context or item metadata less than
#    30 characters."

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# ── Constants matching the paper ─────────────────────────────────────────────
MIN_CHARS      = 30      # paper §3.1: filter out pairs shorter than 30 chars
MAX_CONTEXT_LEN = 512    # truncate very long reviews
MAX_META_LEN    = 512    # truncate very long metadata


def _build_context(record: dict) -> str:
    """
    Build the language context c from a review record.
    = review summary (title) + review text
    Mirrors paper §3.1: "concatenating the title and content of user reviews"
    """
    parts = []
    summary = (record.get("summary") or "").strip()
    text    = (record.get("text")    or "").strip()
    if summary:
        parts.append(summary)
    if text:
        parts.append(text)
    context = " ".join(parts)
    return context[:MAX_CONTEXT_LEN]


def _build_metadata(record: dict) -> str:
    """
    Build the item metadata m from a review record.
    = product title + item_metadata field (already contains features+description)
    Mirrors paper §3.1: "concatenation of the title, features, and description"
    """
    meta = (record.get("item_metadata") or "").strip()
    if not meta:
        title = (record.get("title") or "").strip()
        meta  = title or f"Product {record.get('item_id','unknown')}"
    return meta[:MAX_META_LEN]


def build_pairs(
    split_dir:  str = "data_splits",
    output_dir: str = "training_pairs",
    splits:     list[str] = None,
) -> dict[str, int]:
    """
    Build (context, metadata) pairs from JSONL split files.

    Parameters
    ----------
    split_dir  : directory containing train.jsonl / val.jsonl / test.jsonl
    output_dir : directory to write pairs_{split}.jsonl
    splits     : which splits to process (default: train, val, test)

    Returns
    -------
    dict mapping split name → number of pairs written
    """
    if splits is None:
        splits = ["train", "val", "test"]

    os.makedirs(output_dir, exist_ok=True)
    counts: dict[str, int] = {}

    for split in splits:
        in_path  = os.path.join(split_dir,  f"{split}.jsonl")
        out_path = os.path.join(output_dir, f"pairs_{split}.jsonl")

        if not os.path.exists(in_path):
            print(f"  [{split}] File not found: {in_path}  — skipping.")
            continue

        n_written = n_skipped = 0

        with open(in_path,  "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                context  = _build_context(record)
                metadata = _build_metadata(record)

                # Paper filter: skip pairs shorter than MIN_CHARS
                if len(context) < MIN_CHARS or len(metadata) < MIN_CHARS:
                    n_skipped += 1
                    continue

                pair = {
                    "context":     context,    # c  — review text
                    "metadata":    metadata,   # m  — item description
                    "item_id":     record.get("item_id", ""),
                    "rating":      record.get("rating"),
                    "asin":        record.get("asin", ""),
                    "parent_asin": record.get("parent_asin", ""),
                }
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n_written += 1

        counts[split] = n_written
        print(f"  [{split}]  pairs written: {n_written:,}  |  "
              f"filtered (too short): {n_skipped:,}  →  {out_path}")

    return counts


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BLAIR training pairs.")
    parser.add_argument("--splits",  default="data_splits",   help="Split JSONL directory")
    parser.add_argument("--output",  default="training_pairs", help="Output directory")
    args = parser.parse_args()

    counts = build_pairs(split_dir=args.splits, output_dir=args.output)
    total  = sum(counts.values())
    print(f"\nTotal pairs built: {total:,}")
    print("Pair building complete.")
