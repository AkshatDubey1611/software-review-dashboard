# download_datasets.py
#
# Downloads Amazon Reviews 2023 datasets from HuggingFace.
# All 33 categories available — this script downloads the 5-core
# (filtered) versions which are much smaller than the full versions.
#
# Usage (from project root):
#   python download_datasets.py
#
# What it does:
#   - Downloads each category as a JSONL file
#   - Saves to a datasets/ subfolder
#   - Skips files you already have
#   - Shows file size and record count after each download
#
# After running, your folder will look like:
#   datasets/
#     All_Beauty.jsonl          (you already have this)
#     Software.jsonl
#     Magazine_Subscriptions.jsonl
#     Gift_Cards.jsonl
#     Luxury_Beauty.jsonl
#     Toys_and_Games.jsonl
#     Sports_and_Outdoors.jsonl
#     Home_and_Kitchen.jsonl
#     Books.jsonl
#     Electronics.jsonl
#     Clothing_Shoes_and_Jewelry.jsonl
#     ... and more

from __future__ import annotations

import os
import json
import time
import argparse
import urllib.request
from pathlib import Path

# ── All 33 Amazon Reviews 2023 categories ────────────────────────────────────
# Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
# We use the 0-core (all reviews) versions for maximum data
# Format: (category_name, huggingface_filename, output_filename)

ALL_CATEGORIES = [
    # Small categories — fast download (< 50MB each)
    ("Gift_Cards",                  "Gift_Cards",                   "Gift_Cards.jsonl"),
    ("Magazine_Subscriptions",      "Magazine_Subscriptions",       "Magazine_Subscriptions.jsonl"),
    ("Prime_Pantry",                "Prime_Pantry",                 "Prime_Pantry.jsonl"),
    ("Luxury_Beauty",               "Luxury_Beauty",                "Luxury_Beauty.jsonl"),
    ("All_Beauty",                  "All_Beauty",                   "All_Beauty.jsonl"),
    ("Digital_Music",               "Digital_Music",                "Digital_Music.jsonl"),
    ("Musical_Instruments",         "Musical_Instruments",          "Musical_Instruments.jsonl"),
    ("Amazon_Fashion",              "Amazon_Fashion",               "Amazon_Fashion.jsonl"),
    ("Appliances",                  "Appliances",                   "Appliances.jsonl"),
    ("Arts_Crafts_and_Sewing",      "Arts_Crafts_and_Sewing",       "Arts_Crafts_and_Sewing.jsonl"),

    # Medium categories
    ("Software",                    "Software",                     "Software.jsonl"),
    ("Toys_and_Games",              "Toys_and_Games",               "Toys_and_Games.jsonl"),
    ("Office_Products",             "Office_Products",              "Office_Products.jsonl"),
    ("Handmade_Products",           "Handmade_Products",            "Handmade_Products.jsonl"),
    ("Health_and_Personal_Care",    "Health_and_Personal_Care",     "Health_and_Personal_Care.jsonl"),
    ("Patio_Lawn_and_Garden",       "Patio_Lawn_and_Garden",        "Patio_Lawn_and_Garden.jsonl"),
    ("Pet_Supplies",                "Pet_Supplies",                 "Pet_Supplies.jsonl"),
    ("Sports_and_Outdoors",         "Sports_and_Outdoors",          "Sports_and_Outdoors.jsonl"),
    ("Grocery_and_Gourmet_Food",    "Grocery_and_Gourmet_Food",     "Grocery_and_Gourmet_Food.jsonl"),
    ("Baby_Products",               "Baby_Products",                "Baby_Products.jsonl"),

    # Large categories — slower download
    ("Cell_Phones_and_Accessories", "Cell_Phones_and_Accessories",  "Cell_Phones_and_Accessories.jsonl"),
    ("Automotive",                  "Automotive",                   "Automotive.jsonl"),
    ("Home_and_Kitchen",            "Home_and_Kitchen",             "Home_and_Kitchen.jsonl"),
    ("Industrial_and_Scientific",   "Industrial_and_Scientific",    "Industrial_and_Scientific.jsonl"),
    ("Tools_and_Home_Improvement",  "Tools_and_Home_Improvement",   "Tools_and_Home_Improvement.jsonl"),
    ("Video_Games",                 "Video_Games",                  "Video_Games.jsonl"),
    ("Electronics",                 "Electronics",                  "Electronics.jsonl"),
    ("Clothing_Shoes_and_Jewelry",  "Clothing_Shoes_and_Jewelry",   "Clothing_Shoes_and_Jewelry.jsonl"),
    ("Movies_and_TV",               "Movies_and_TV",                "Movies_and_TV.jsonl"),
    ("CDs_and_Vinyl",               "CDs_and_Vinyl",                "CDs_and_Vinyl.jsonl"),
    ("Books",                       "Books",                        "Books.jsonl"),
]

# HuggingFace dataset base URL for Amazon Reviews 2023 (5-core filtered)
# These are the same files the paper used
HF_BASE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories/{category}.jsonl.gz"
)

# Alternative: direct download URLs (uncompressed, smaller subsets)
# Used as fallback if HuggingFace is slow
DIRECT_BASE = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023"
    "/raw/review_categories/{category}.jsonl.gz"
)


def _count_lines(path: str) -> int:
    """Count lines in a file efficiently."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _format_size(bytes_size: int) -> str:
    """Human-readable file size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} GB"


def _download_file(url: str, dest_path: str) -> bool:
    """
    Download a file from url to dest_path.
    Shows progress. Returns True on success.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {_format_size(downloaded)} / {_format_size(total)} ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r    {_format_size(downloaded)} downloaded", end="", flush=True)

        print()  # newline after progress
        return True

    except Exception as e:
        print(f"\n    [ERROR] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def _decompress_gz(gz_path: str, out_path: str) -> bool:
    """Decompress a .gz file to out_path."""
    import gzip
    import shutil
    try:
        print(f"    Decompressing ...", end="", flush=True)
        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(" done.")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def download_all(
    output_dir:   str  = "datasets",
    categories:   list = None,
    skip_existing: bool = True,
    max_per_file: int  = 0,    # 0 = keep all; set to trim large files
) -> list[str]:
    """
    Download all Amazon Reviews 2023 categories.

    Parameters
    ----------
    output_dir    : where to save JSONL files
    categories    : list of category names to download (None = all)
    skip_existing : skip download if file already exists
    max_per_file  : trim each file to this many lines after download (0 = keep all)

    Returns
    -------
    list of paths to successfully downloaded JSONL files
    """
    os.makedirs(output_dir, exist_ok=True)
    cats = categories if categories else ALL_CATEGORIES

    print("\n" + "=" * 65)
    print("  Amazon Reviews 2023 — Dataset Downloader")
    print("=" * 65)
    print(f"  Output dir : {output_dir}/")
    print(f"  Categories : {len(cats)}")
    print(f"  Skip existing: {skip_existing}")
    print("=" * 65 + "\n")

    downloaded_paths = []
    skipped = []
    failed  = []

    for i, (name, hf_name, out_filename) in enumerate(cats, 1):
        out_path = os.path.join(output_dir, out_filename)
        gz_path  = out_path + ".gz"

        print(f"[{i}/{len(cats)}] {name}")

        # Check if already exists
        if skip_existing and os.path.exists(out_path):
            size = os.path.getsize(out_path)
            lines = _count_lines(out_path)
            print(f"    ✔ Already exists — {_format_size(size)}, {lines:,} reviews. Skipping.")
            downloaded_paths.append(out_path)
            skipped.append(name)
            continue

        # Try HuggingFace first, then direct URL
        url = HF_BASE.format(category=hf_name)
        print(f"    Downloading from HuggingFace ...")
        print(f"    URL: {url}")

        success = _download_file(url, gz_path)

        if not success:
            # Try alternative URL
            url2 = DIRECT_BASE.format(category=hf_name)
            print(f"    Trying alternative URL ...")
            success = _download_file(url2, gz_path)

        if not success:
            print(f"    ✗ Failed to download {name}. Skipping.")
            failed.append(name)
            continue

        # Decompress
        if not _decompress_gz(gz_path, out_path):
            failed.append(name)
            continue

        # Show stats
        size  = os.path.getsize(out_path)
        lines = _count_lines(out_path)
        print(f"    ✔ {_format_size(size)} — {lines:,} reviews")

        # Optional: trim to max_per_file lines to save disk space
        if max_per_file and lines > max_per_file:
            _trim_file(out_path, max_per_file)
            print(f"    Trimmed to {max_per_file:,} lines.")

        downloaded_paths.append(out_path)
        print()

    # Summary
    print("\n" + "=" * 65)
    print("  Download Summary")
    print("=" * 65)
    print(f"  Downloaded : {len(downloaded_paths) - len(skipped)}")
    print(f"  Skipped    : {len(skipped)} (already existed)")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print(f"  Failed list: {', '.join(failed)}")
    print(f"\n  Total files ready: {len(downloaded_paths)}")
    print(f"  Location: {os.path.abspath(output_dir)}/")
    print("=" * 65)

    return downloaded_paths


def _trim_file(path: str, max_lines: int) -> None:
    """Keep only the first max_lines lines of a file."""
    tmp = path + ".tmp"
    with open(path, "r", encoding="utf-8") as fin, \
         open(tmp,  "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= max_lines:
                break
            fout.write(line)
    os.replace(tmp, path)


def list_existing(output_dir: str = "datasets") -> list[str]:
    """List all JSONL files already in the datasets directory."""
    if not os.path.exists(output_dir):
        return []
    paths = sorted(Path(output_dir).glob("*.jsonl"))
    return [str(p) for p in paths]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", default="datasets",
        help="Directory to save JSONL files"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Specific categories to download (default: all). "
             "Example: --categories All_Beauty Toys_and_Games Software"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-download even if file already exists"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Just list available categories and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable categories:")
        for i, (name, _, _) in enumerate(ALL_CATEGORIES, 1):
            print(f"  {i:2}. {name}")
        print(f"\nTotal: {len(ALL_CATEGORIES)} categories")
        exit(0)

    # Filter to requested categories if specified
    cats = None
    if args.categories:
        cats = [c for c in ALL_CATEGORIES if c[0] in args.categories]
        not_found = [c for c in args.categories if c not in [x[0] for x in ALL_CATEGORIES]]
        if not_found:
            print(f"Warning: categories not found: {not_found}")
            print("Use --list to see all available categories.")

    paths = download_all(
        output_dir    = args.output,
        categories    = cats,
        skip_existing = not args.no_skip,
    )

    if paths:
        print(f"\nNext step — train on all downloaded datasets:")
        print(f"  python training/run_training.py --jsonl_dir {args.output} --max_records 2000 --batch 32")
