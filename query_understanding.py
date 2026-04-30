# query_understanding.py
#
# BLAIR upgrade:
#   Now uses the shared BLAIR encoder (blair_encoder.get_encoder()) instead of
#   loading a separate all-MiniLM-L6-v2 instance.  This removes the duplicate
#   model load, saves ~350 MB of RAM, and ensures aspect detection embeddings
#   are in the same semantic space as the review/item embeddings.

from __future__ import annotations
import torch
from blair_encoder import get_encoder
from review_processing import ASPECT_DESCRIPTIONS


def detect_query_aspects(query: str, top_k: int = 2) -> list[str]:
    """
    Detect which aspects are most relevant to the user query using BLAIR
    embeddings (dot-product == cosine similarity, both sides L2-normalised).
    """
    encoder = get_encoder()

    aspect_names = list(ASPECT_DESCRIPTIONS.keys())
    aspect_texts = list(ASPECT_DESCRIPTIONS.values())

    # Encode query and aspect descriptions with BLAIR
    query_emb   = encoder.encode(query, convert_to_tensor=True)          # (d,)
    aspect_embs = encoder.encode(aspect_texts, convert_to_tensor=True)   # (A, d)

    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)   # (1, d)

    # Dot-product == cosine similarity (L2-normalised)
    similarities = (aspect_embs @ query_emb.T).squeeze(-1)               # (A,)
    top_results  = similarities.topk(k=min(top_k, len(aspect_names)))

    return [aspect_names[int(idx)] for idx in top_results.indices]
