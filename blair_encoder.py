# blair_encoder.py
#
# BLAIR sentence encoder following the paper exactly:
#   "Bridging Language and Items for Retrieval and Recommendation"
#   Hou et al., 2024  (arXiv:2403.03952)
#
# Architecture (Section 2.2):
#   - Backbone: RoBERTa (encoder-only transformer)
#   - Sentence embedding = L2-normalised [CLS] hidden state (Equation 1)
#   - s = BLAIR([[CLS]; s]),  s ∈ R^d,  ||s||_2 = 1
#
# Checkpoint used: hyp1231/blair-roberta-base  (123M parameters)
# Fallback:        all-MiniLM-L6-v2  (if BLAIR cannot be downloaded)

from __future__ import annotations

import torch
import torch.nn.functional as F

BLAIR_MAX_TOKENS  = 64
BLAIR_MODEL_NAME  = "runs/blair_run/checkpoints/blair_best"
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"

_encoder = None   # singleton – loaded once, reused everywhere


def get_encoder():
    """Return the shared BLAIR encoder. Loads on first call."""
    global _encoder
    if _encoder is None:
        _encoder = _load_encoder()
    return _encoder


def _load_encoder():
    """Try BLAIR first, fall back to SentenceTransformer."""
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading BLAIR checkpoint: {BLAIR_MODEL_NAME} ...")
        tok   = AutoTokenizer.from_pretrained(BLAIR_MODEL_NAME)
        model = AutoModel.from_pretrained(BLAIR_MODEL_NAME)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"BLAIR loaded on {device}.")
        return _BLAIRWrapper(tok, model, device)

    except Exception as exc:
        print(f"[WARNING] Could not load BLAIR ({exc}).")
        print(f"[WARNING] Falling back to {FALLBACK_MODEL_NAME}.")
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(FALLBACK_MODEL_NAME)
        return _STWrapper(st)


class _BLAIRWrapper:
    """Encodes sentences via BLAIR's [CLS] hidden state, L2-normalised (Eq. 1)."""
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model     = model
        self.device    = device

    @torch.no_grad()
    def encode(self, sentences, batch_size: int = 128,
               convert_to_tensor: bool = True,
               show_progress_bar: bool = False) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embs = []
        total = len(sentences)

        for start in range(0, total, batch_size):
            batch = sentences[start: start + batch_size]
            if show_progress_bar:
                print(f"  Encoding {start}–{min(start+batch_size, total)} / {total}")

            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=BLAIR_MAX_TOKENS, return_tensors="pt"
            ).to(self.device)

            out    = self.model(**enc)
            cls    = out.last_hidden_state[:, 0, :]   # [CLS] — Equation 1
            normed = F.normalize(cls, p=2, dim=-1)
            all_embs.append(normed.cpu())

        return torch.cat(all_embs, dim=0)


class _STWrapper:
    """Thin shim so SentenceTransformer matches _BLAIRWrapper's API."""
    def __init__(self, model):
        self._model = model

    def encode(self, sentences, batch_size: int = 128,
               convert_to_tensor: bool = True,
               show_progress_bar: bool = False) -> torch.Tensor:
        embs = self._model.encode(
            sentences, batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
        )
        return F.normalize(embs.cpu(), p=2, dim=-1)
