# run_training.py  (UPDATED — Multi-domain version)
#
# Trains BLAIR on multiple JSONL datasets simultaneously.
# This matches the paper's approach of training across all 33 Amazon categories.
#
# NEW vs old version:
#   Old: --jsonl All_Beauty.jsonl         (one file)
#   New: --jsonl_dir datasets/            (whole folder of JSONL files)
#        --jsonl f1.jsonl f2.jsonl f3.jsonl  (explicit list)
#
# Usage examples:
#
#   # Train on a whole folder of datasets (recommended)
#   python training/run_training.py --jsonl_dir datasets/ --max_records 2000 --batch 32
#
#   # Train on specific files
#   python training/run_training.py --jsonl All_Beauty.jsonl Software.jsonl --max_records 2000
#
#   # Quick test (one file, 500 records)
#   python training/run_training.py --jsonl All_Beauty.jsonl --max_records 500 --batch 32

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from data_prep    import load_and_split_multi
from pair_builder import build_pairs
from trainer      import train


def run_pipeline(
    jsonl_paths:  list[str],
    output_root:  str   = "runs/blair_run",
    max_records:  int   = 2000,
    batch_size:   int   = 32,
    epochs:       int   = 1,
    lr:           float = 5e-5,
    model_name:   str   = "runs/blair_run/checkpoints/blair_final",
    log_every:    int   = 50,
    save_every:   int   = 500,
) -> str:
    """
    Run the complete multi-domain BLAIR training pipeline.
    """
    splits_dir  = os.path.join(output_root, "data_splits")
    pairs_dir   = os.path.join(output_root, "training_pairs")
    checkpoints = os.path.join(output_root, "checkpoints")

    print("\n" + "=" * 65)
    print("  BLAIR Multi-Domain Training Pipeline")
    print("=" * 65)
    print(f"  Datasets  : {len(jsonl_paths)} files")
    for p in jsonl_paths:
        print(f"              {os.path.basename(p)}")
    print(f"  Output    : {output_root}")
    print(f"  Batch     : {batch_size}")
    print(f"  LR        : {lr}")
    print(f"  Epochs    : {epochs}")
    print(f"  Max recs  : {max_records} per dataset")
    print(f"  Total max : ~{max_records * len(jsonl_paths):,} records")
    print("=" * 65 + "\n")

    # ── STEP 1: Multi-domain data preparation ─────────────────────────────
    print("STEP 1/5 — Loading and merging all datasets (timestamp split 8:1:1)")
    print("-" * 50)
    load_and_split_multi(
        jsonl_paths  = jsonl_paths,
        output_dir   = splits_dir,
        max_per_file = max_records,
    )

    # ── STEP 2: Pair building ─────────────────────────────────────────────
    print("\nSTEP 2/5 — Building (context, metadata) training pairs")
    print("-" * 50)
    counts = build_pairs(
        split_dir  = splits_dir,
        output_dir = pairs_dir,
    )
    total_pairs = counts.get("train", 0)
    print(f"  Training pairs available: {total_pairs:,}")

    if total_pairs == 0:
        print("\n[ERROR] No training pairs were built.")
        print("  Make sure your JSONL files have 'text' and 'title'/'asin' fields.")
        sys.exit(1)

    # ── STEP 3–5: Training ────────────────────────────────────────────────
    print("\nSTEP 3-5/5 — Training BLAIR on all domains")
    print("-" * 50)
    train(
        pairs_train = os.path.join(pairs_dir, "pairs_train.jsonl"),
        pairs_val   = os.path.join(pairs_dir, "pairs_val.jsonl"),
        output_dir  = checkpoints,
        model_name  = model_name,
        lr          = lr,
        batch_size  = batch_size,
        epochs      = epochs,
        log_every   = log_every,
        save_every  = save_every,
    )

    final_ckpt = os.path.join(checkpoints, "blair_final")

    print("\n" + "=" * 65)
    print("  Multi-Domain Training Complete!")
    print("=" * 65)
    print(f"\n  Checkpoint : {final_ckpt}/")
    print(f"  Domains    : {len(jsonl_paths)}")
    print(f"\n  Update blair_encoder.py:")
    print(f'    BLAIR_MODEL_NAME = "{final_ckpt}"')
    print("=" * 65 + "\n")

    return final_ckpt


def _collect_jsonl_paths(
    jsonl_files: list[str],
    jsonl_dir:   str | None,
) -> list[str]:
    """
    Collect all JSONL file paths from explicit list and/or directory.
    Deduplicates and validates existence.
    """
    paths = set()

    # Explicit files
    for f in (jsonl_files or []):
        p = Path(f)
        if p.exists():
            paths.add(str(p.resolve()))
        else:
            print(f"[WARNING] File not found: {f} — skipping.")

    # Directory scan
    if jsonl_dir:
        d = Path(jsonl_dir)
        if d.is_dir():
            found = list(d.glob("*.jsonl"))
            if not found:
                print(f"[WARNING] No JSONL files found in {jsonl_dir}")
            for p in sorted(found):
                paths.add(str(p.resolve()))
        else:
            print(f"[WARNING] Directory not found: {jsonl_dir}")

    result = sorted(paths)

    if not result:
        print("\n[ERROR] No JSONL files found.")
        print("  Use --jsonl <file1> <file2> or --jsonl_dir <folder>")
        sys.exit(1)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BLAIR on multiple Amazon Review datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — one of these two (or both)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--jsonl", nargs="+", metavar="FILE",
        help="One or more JSONL files. Example: --jsonl All_Beauty.jsonl Software.jsonl"
    )
    input_group.add_argument(
        "--jsonl_dir", metavar="DIR",
        help="Directory containing JSONL files. All *.jsonl files will be used."
    )

    parser.add_argument("--output",      default="runs/blair_run",
                        help="Root directory for outputs")
    parser.add_argument("--max_records", type=int, default=2000,
                        help="Max reviews per dataset file (0=all). 2000 ≈ 30min total on CPU.")
    parser.add_argument("--batch",       type=int, default=32,
                        help="Batch size. Lower if you get memory errors.")
    parser.add_argument("--epochs",      type=int, default=1,
                        help="Training epochs (paper uses 1).")
    parser.add_argument("--lr",          type=float, default=5e-5,
                        help="Learning rate (paper: 5e-5).")
    parser.add_argument("--model",       default="runs/blair_run/checkpoints/blair_final",
                        help="Base model to fine-tune (use your existing checkpoint).")
    parser.add_argument("--log",         type=int, default=50,
                        help="Log every N steps.")
    parser.add_argument("--save",        type=int, default=500,
                        help="Save checkpoint every N steps.")

    args = parser.parse_args()

    # Collect paths
    if args.jsonl:
        paths = _collect_jsonl_paths(args.jsonl, None)
    else:
        paths = _collect_jsonl_paths([], args.jsonl_dir)

    run_pipeline(
        jsonl_paths  = paths,
        output_root  = args.output,
        max_records  = args.max_records,
        batch_size   = args.batch,
        epochs       = args.epochs,
        lr           = args.lr,
        model_name   = args.model,
        log_every    = args.log,
        save_every   = args.save,
    )
