# blair_dataset.py
#
# Step 3 of the BLAIR training pipeline.
#
# Paper reference (Section 3.1 — Implementation details):
#   "We use the RoBERTa tokenizer and truncate sentences with a maximum of
#    64 tokens."
#   "BLAIR_BASE is trained with per-device batch size of 384 for one epoch."
#
# This file provides:
#   BLAIRPairDataset  — PyTorch Dataset over pairs_{split}.jsonl
#   get_dataloader()  — returns a DataLoader with the correct collation
#
# Each batch contains:
#   context_input_ids,  context_attention_mask   (review text)
#   metadata_input_ids, metadata_attention_mask  (item metadata)
#   Both tokenized to max_length=64 as per the paper.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Paper §3.1: max 64 tokens
MAX_LENGTH = 64


class BLAIRPairDataset(Dataset):
    """
    PyTorch Dataset of (context, metadata) pairs for BLAIR contrastive training.

    Each item returns a dict with tokenized context and metadata tensors.
    The contrastive loss pairs context[i] with metadata[i] as the positive,
    and all other metadata[j≠i] in the same batch as negatives (in-batch negatives,
    paper Equation 2).
    """

    def __init__(
        self,
        pairs_path: str,
        tokenizer_name: str = "hyp1231/blair-roberta-base",
        max_length: int = MAX_LENGTH,
        max_samples: int = 0,       # 0 = load all
    ):
        self.max_length = max_length
        self.pairs: list[dict] = []

        # Load tokenizer (RoBERTa)
        print(f"Loading tokenizer: {tokenizer_name} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            print("  Falling back to roberta-base tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Load pairs
        path = Path(pairs_path)
        if not path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        print(f"Loading pairs from {pairs_path} ...")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("context") and rec.get("metadata"):
                        self.pairs.append(rec)
                except json.JSONDecodeError:
                    continue

                if max_samples and len(self.pairs) >= max_samples:
                    break

        print(f"  Loaded {len(self.pairs):,} pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        return {
            "context":  pair["context"],
            "metadata": pair["metadata"],
            "item_id":  pair.get("item_id", ""),
            "rating":   pair.get("rating"),
        }


class BLAIRCollator:
    """
    Collate function that tokenizes a batch of (context, metadata) string pairs
    into padded tensors ready for the model.

    Paper §3.1: "truncate sentences with a maximum of 64 tokens"
    """

    def __init__(self, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        contexts  = [item["context"]  for item in batch]
        metadatas = [item["metadata"] for item in batch]

        ctx_enc = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        meta_enc = self.tokenizer(
            metadatas,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            # Context (review text)
            "context_input_ids":       ctx_enc["input_ids"],
            "context_attention_mask":  ctx_enc["attention_mask"],
            # Metadata (item description)
            "metadata_input_ids":      meta_enc["input_ids"],
            "metadata_attention_mask": meta_enc["attention_mask"],
        }


def get_dataloader(
    pairs_path:     str,
    tokenizer_name: str = "hyp1231/blair-roberta-base",
    batch_size:     int = 128,       # paper uses 384; lower for small GPU
    shuffle:        bool = True,
    num_workers:    int = 0,
    max_samples:    int = 0,
) -> tuple["BLAIRPairDataset", DataLoader]:
    """
    Build a BLAIRPairDataset and return (dataset, DataLoader).

    Parameters
    ----------
    pairs_path     : path to pairs_{split}.jsonl
    tokenizer_name : HuggingFace model name for the tokenizer
    batch_size     : batch size (paper: 384 on A100; use 64–128 on consumer GPU)
    shuffle        : shuffle training data
    num_workers    : DataLoader worker processes
    max_samples    : cap on samples loaded (0 = all)

    Returns
    -------
    (dataset, dataloader)
    """
    dataset = BLAIRPairDataset(
        pairs_path     = pairs_path,
        tokenizer_name = tokenizer_name,
        max_samples    = max_samples,
    )

    collator = BLAIRCollator(
        tokenizer  = dataset.tokenizer,
        max_length = MAX_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collator,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    return dataset, loader


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", help="Path to pairs_train.jsonl")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    ds, dl = get_dataloader(args.pairs, batch_size=args.batch, max_samples=20)
    batch  = next(iter(dl))
    print("\nBatch keys:", list(batch.keys()))
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print("\nFirst context tokens:", batch["context_input_ids"][0])
    print("First metadata tokens:", batch["metadata_input_ids"][0])
    print("\nDataset and DataLoader OK.")
