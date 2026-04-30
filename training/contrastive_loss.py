# contrastive_loss.py
#
# Step 4 of the BLAIR training pipeline.
#
# Paper reference (Section 2.3 — Training Objective):
#
#   Equation 2 — Contrastive loss L_CL:
#     L_CL = -Σ_i  log[ exp(c_i · m_i / τ) / Σ_j exp(c_i · m_j / τ) ]
#   where:
#     c_i  = L2-normalised [CLS] embedding of review context i
#     m_j  = L2-normalised [CLS] embedding of item metadata j
#     τ    = temperature hyperparameter (paper uses τ = 0.05)
#     In-batch instances are used as negatives (j ≠ i are negatives for c_i)
#
#   Equation 3 — Total loss:
#     L = L_CL + λ * L_PT
#   where:
#     L_PT = masked language modelling (MLM) loss (auxiliary pretraining objective)
#     λ    = 0.1  (paper default)
#
# This file implements both losses as a single BLAIRLoss nn.Module.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BLAIRLoss(nn.Module):
    """
    Combined contrastive + MLM loss from Equations 2 and 3 of the BLAIR paper.

    Parameters
    ----------
    temperature : τ in Equation 2  (paper default: 0.05)
    lambda_pt   : λ in Equation 3  (paper default: 0.1)
    """

    def __init__(self, temperature: float = 0.05, lambda_pt: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_pt   = lambda_pt

    def contrastive_loss(
        self,
        context_emb:  torch.Tensor,   # (B, d)  L2-normalised
        metadata_emb: torch.Tensor,   # (B, d)  L2-normalised
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (Equation 2).

        For each context c_i, the positive is its own metadata m_i.
        All other m_j (j≠i) in the batch are negatives.
        This is InfoNCE / NT-Xent with in-batch negatives.

        context_emb  and  metadata_emb  must already be L2-normalised.
        """
        B = context_emb.size(0)

        # Similarity matrix  (B, B):  sim[i,j] = c_i · m_j / τ
        sim = torch.mm(context_emb, metadata_emb.T) / self.temperature   # (B, B)

        # Labels: diagonal is the positive pair
        labels = torch.arange(B, device=sim.device)

        # Cross-entropy over rows (each context matches its own metadata)
        loss_cl = F.cross_entropy(sim, labels)

        return loss_cl

    def mlm_loss(self, mlm_loss_value: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary MLM loss L_PT (Equation 3).
        The actual MLM forward pass is done in trainer.py using HuggingFace's
        built-in masked LM head.  This method just scales it by λ.
        """
        return self.lambda_pt * mlm_loss_value

    def forward(
        self,
        context_emb:  torch.Tensor,           # (B, d) L2-normalised context embeddings
        metadata_emb: torch.Tensor,           # (B, d) L2-normalised metadata embeddings
        mlm_loss_val: torch.Tensor | None = None,  # scalar MLM loss from HF model
    ) -> dict[str, torch.Tensor]:
        """
        Compute total loss L = L_CL + λ * L_PT  (Equation 3).

        Parameters
        ----------
        context_emb  : L2-normalised [CLS] embeddings of review texts
        metadata_emb : L2-normalised [CLS] embeddings of item metadata
        mlm_loss_val : MLM loss returned by model(..., labels=...) — optional

        Returns
        -------
        dict with keys:
          "loss"        — total loss (backpropagate this)
          "loss_cl"     — contrastive loss component
          "loss_pt"     — MLM loss component (0 if not provided)
        """
        loss_cl = self.contrastive_loss(context_emb, metadata_emb)

        if mlm_loss_val is not None:
            loss_pt    = self.mlm_loss(mlm_loss_val)
            total_loss = loss_cl + loss_pt
        else:
            loss_pt    = torch.tensor(0.0, device=loss_cl.device)
            total_loss = loss_cl

        return {
            "loss":    total_loss,
            "loss_cl": loss_cl,
            "loss_pt": loss_pt,
        }


# ── Unit test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, d = 8, 768

    # Simulate L2-normalised embeddings
    ctx  = F.normalize(torch.randn(B, d), p=2, dim=-1)
    meta = F.normalize(torch.randn(B, d), p=2, dim=-1)

    criterion = BLAIRLoss(temperature=0.05, lambda_pt=0.1)

    # Without MLM loss
    out = criterion(ctx, meta)
    print(f"L_CL  = {out['loss_cl'].item():.4f}")
    print(f"L_PT  = {out['loss_pt'].item():.4f}  (no MLM provided)")
    print(f"Total = {out['loss'].item():.4f}")

    # With dummy MLM loss
    mlm = torch.tensor(2.3)
    out = criterion(ctx, meta, mlm_loss_val=mlm)
    print(f"\nWith MLM loss = {mlm.item():.4f}:")
    print(f"L_CL  = {out['loss_cl'].item():.4f}")
    print(f"L_PT  = {out['loss_pt'].item():.4f}  (λ={criterion.lambda_pt} × {mlm.item():.4f})")
    print(f"Total = {out['loss'].item():.4f}")
    print("\nContrastive loss OK.")
