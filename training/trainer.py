# trainer.py
#
# Step 5 of the BLAIR training pipeline.
#
# Paper reference (Section 3.1 — Implementation details):
#   "We train BLAIR_BASE on two NVIDIA A100 (80G) GPUs with a per-device
#    batch size of 384 for one epoch."
#   "We set τ = 0.05, λ = 0.1 and optimize with lr = 5×10⁻⁵  (AdamW)."
#
# This trainer works on any GPU (or CPU as fallback).
# Default batch size is 64 — safe for 8GB VRAM.  Raise to 128/256 if you
# have more memory.  The paper used 384 on an 80 GB A100.
#
# Usage:
#   python trainer.py --pairs training_pairs/pairs_train.jsonl \
#                     --val   training_pairs/pairs_val.jsonl   \
#                     --output checkpoints

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

from blair_dataset    import get_dataloader
from contrastive_loss import BLAIRLoss


# ── Hyperparameters (matching paper §3.1) ─────────────────────────────────────
DEFAULTS = dict(
    model_name  = "hyp1231/blair-roberta-base",
    lr          = 5e-5,       # paper: 5×10⁻⁵
    batch_size  = 64,         # paper: 384 on A100 — lower for consumer GPU
    epochs      = 1,          # paper: 1 epoch
    temperature = 0.05,       # τ — paper value
    lambda_pt   = 0.1,        # λ — paper value
    max_samples = 0,          # 0 = all pairs
    log_every   = 50,         # print loss every N steps
    save_every  = 500,        # save checkpoint every N steps
    warmup_frac = 0.06,       # fraction of steps used for LR warmup
    mlm_prob    = 0.15,       # MLM masking probability (standard BERT value)
)


def _get_cls_embedding(model_output, attention_mask=None) -> torch.Tensor:
    """
    Extract L2-normalised [CLS] hidden state.
    Equation 1:  s = BLAIR([[CLS]; s]),  ||s||_2 = 1
    """
    cls = model_output.hidden_states[-1][:, 0, :]   # (B, d)
    return F.normalize(cls, p=2, dim=-1)


def train(
    pairs_train: str,
    pairs_val:   str | None = None,
    output_dir:  str        = "checkpoints",
    **kwargs,
) -> None:
    """
    Main training function.  All hyperparameters can be overridden via kwargs.
    """
    cfg = {**DEFAULTS, **kwargs}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  BLAIR Training")
    print(f"{'='*60}")
    print(f"  Model      : {cfg['model_name']}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {cfg['batch_size']}")
    print(f"  LR         : {cfg['lr']}")
    print(f"  τ          : {cfg['temperature']}")
    print(f"  λ          : {cfg['lambda_pt']}")
    print(f"  Epochs     : {cfg['epochs']}")
    print(f"  Output     : {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ── Tokenizer + Model ─────────────────────────────────────────────────────
    print("Loading model ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        # AutoModelForMaskedLM gives us both the encoder AND the MLM head
        model = AutoModelForMaskedLM.from_pretrained(
            cfg["model_name"],
            output_hidden_states=True,
        )
    except Exception as exc:
        print(f"[WARNING] Could not load {cfg['model_name']}: {exc}")
        print("[WARNING] Falling back to roberta-base.")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model     = AutoModelForMaskedLM.from_pretrained(
            "roberta-base", output_hidden_states=True
        )

    model.to(device)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("Building DataLoaders ...")
    _, train_loader = get_dataloader(
        pairs_path     = pairs_train,
        tokenizer_name = cfg["model_name"],
        batch_size     = cfg["batch_size"],
        shuffle        = True,
        max_samples    = cfg["max_samples"],
    )

    val_loader = None
    if pairs_val and os.path.exists(pairs_val):
        _, val_loader = get_dataloader(
            pairs_path     = pairs_val,
            tokenizer_name = cfg["model_name"],
            batch_size     = cfg["batch_size"],
            shuffle        = False,
            max_samples    = min(cfg["max_samples"], 2000) if cfg["max_samples"] else 2000,
        )

    # ── MLM data collator ─────────────────────────────────────────────────────
    # Used to generate masked inputs for the L_PT auxiliary loss
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer  = tokenizer,
        mlm        = True,
        mlm_probability = cfg["mlm_prob"],
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])

    total_steps  = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_frac"])
    scheduler    = LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    criterion = BLAIRLoss(
        temperature = cfg["temperature"],
        lambda_pt   = cfg["lambda_pt"],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_cl   = 0.0
        epoch_pt   = 0.0
        t0         = time.time()

        print(f"\nEpoch {epoch}/{cfg['epochs']}  ({len(train_loader)} steps)")

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1

            # ── Move batch to device ──────────────────────────────────────
            ctx_ids   = batch["context_input_ids"].to(device)
            ctx_mask  = batch["context_attention_mask"].to(device)
            meta_ids  = batch["metadata_input_ids"].to(device)
            meta_mask = batch["metadata_attention_mask"].to(device)

            # ── Forward: context encoder ──────────────────────────────────
            ctx_out = model(
                input_ids      = ctx_ids,
                attention_mask = ctx_mask,
                output_hidden_states = True,
            )
            ctx_emb = _get_cls_embedding(ctx_out)   # (B, d) L2-normalised

            # ── Forward: metadata encoder ─────────────────────────────────
            meta_out = model(
                input_ids      = meta_ids,
                attention_mask = meta_mask,
                output_hidden_states = True,
            )
            meta_emb = _get_cls_embedding(meta_out)   # (B, d) L2-normalised

            # ── L_PT: MLM auxiliary loss on context ───────────────────────
            # Build masked inputs for L_PT
            mlm_inputs = mlm_collator(
                [{"input_ids": ids.tolist()} for ids in ctx_ids]
            )
            mlm_input_ids = torch.tensor(
                mlm_inputs["input_ids"], device=device
            )
            mlm_labels = torch.tensor(
                mlm_inputs["labels"], device=device
            )

            mlm_out  = model(
                input_ids      = mlm_input_ids,
                attention_mask = ctx_mask,
                labels         = mlm_labels,
                output_hidden_states = True,
            )
            mlm_loss_val = mlm_out.loss   # scalar

            # ── Combined loss L = L_CL + λ * L_PT  (Equation 3) ──────────
            losses = criterion(ctx_emb, meta_emb, mlm_loss_val=mlm_loss_val)
            loss   = losses["loss"]

            # ── Backprop ──────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step <= warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()
            epoch_cl   += losses["loss_cl"].item()
            epoch_pt   += losses["loss_pt"].item()

            # ── Logging ───────────────────────────────────────────────────
            if step % cfg["log_every"] == 0:
                avg_loss = epoch_loss / step
                avg_cl   = epoch_cl   / step
                avg_pt   = epoch_pt   / step
                elapsed  = time.time() - t0
                steps_s  = step / max(elapsed, 1e-6)
                eta_s    = (len(train_loader) - step) / max(steps_s, 1e-6)
                print(
                    f"  step {step:>5}/{len(train_loader)}  "
                    f"loss={avg_loss:.4f}  "
                    f"L_CL={avg_cl:.4f}  "
                    f"L_PT={avg_pt:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                    f"ETA={eta_s/60:.1f}min"
                )

            # ── Checkpoint ────────────────────────────────────────────────
            if global_step % cfg["save_every"] == 0:
                _save_checkpoint(model, tokenizer, output_dir,
                                 tag=f"step_{global_step}")

        # ── End of epoch ──────────────────────────────────────────────────
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} complete  |  avg loss = {avg_loss:.4f}  |  "
              f"time = {(time.time()-t0)/60:.1f} min")

        # ── Validation ────────────────────────────────────────────────────
        if val_loader:
            val_loss = _validate(model, val_loader, criterion, device, cfg)
            print(f"Validation loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, tokenizer, output_dir, tag="best")
                print(f"  ✔ New best checkpoint saved.")

        _save_checkpoint(model, tokenizer, output_dir, tag=f"epoch_{epoch}")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    _save_checkpoint(model, tokenizer, output_dir, tag="final")
    print(f"\nTraining complete.  Checkpoints in: {output_dir}/")


def _validate(model, loader, criterion, device, cfg) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in loader:
            ctx_ids   = batch["context_input_ids"].to(device)
            ctx_mask  = batch["context_attention_mask"].to(device)
            meta_ids  = batch["metadata_input_ids"].to(device)
            meta_mask = batch["metadata_attention_mask"].to(device)

            ctx_out  = model(input_ids=ctx_ids,  attention_mask=ctx_mask,
                             output_hidden_states=True)
            meta_out = model(input_ids=meta_ids, attention_mask=meta_mask,
                             output_hidden_states=True)

            ctx_emb  = _get_cls_embedding(ctx_out)
            meta_emb = _get_cls_embedding(meta_out)

            losses     = criterion(ctx_emb, meta_emb)
            total_loss += losses["loss"].item()
            n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


def _save_checkpoint(model, tokenizer, output_dir: str, tag: str) -> None:
    path = os.path.join(output_dir, f"blair_{tag}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  Checkpoint saved: {path}/")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BLAIR model.")
    parser.add_argument("--pairs",    required=True,
                        help="Path to pairs_train.jsonl")
    parser.add_argument("--val",      default=None,
                        help="Path to pairs_val.jsonl (optional)")
    parser.add_argument("--output",   default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--model",    default=DEFAULTS["model_name"],
                        help="Base model name or path")
    parser.add_argument("--lr",       type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch",    type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",   type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--max",      type=int,   default=DEFAULTS["max_samples"],
                        help="Max training pairs (0=all)")
    parser.add_argument("--log",      type=int,   default=DEFAULTS["log_every"])
    parser.add_argument("--save",     type=int,   default=DEFAULTS["save_every"])
    args = parser.parse_args()

    train(
        pairs_train = args.pairs,
        pairs_val   = args.val,
        output_dir  = args.output,
        model_name  = args.model,
        lr          = args.lr,
        batch_size  = args.batch,
        epochs      = args.epochs,
        max_samples = args.max,
        log_every   = args.log,
        save_every  = args.save,
    )
