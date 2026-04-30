# item_retrieval.py
#
# BLAIR Item-Level Retrieval
#
# KEY FIXES:
#   FIX 1 — Product titles: _best_product_title() now ONLY reads the "title"
#            field from review records (which in Amazon 2023 JSONL = product name).
#            It NEVER reads "summary" (reviewer headline like "Left my hair dull").
#            The title_counter picks the most common product title across all reviews
#            of the same product, which handles the case where a few reviews have no
#            product title but most do.
#
#   FIX 2 — ASIN validation: strict regex ^B[A-Z0-9]{9}$ or 10-digit ISBN.
#            Any ASIN that doesn't match gets amazon_url = "" so no broken links.
#
#   FIX 3 — min_reviews default = 1 so all products are indexed.
#
#   FIX 4 — FAISS optional — falls back to brute-force numpy dot-product.

from __future__ import annotations

import re
from collections import defaultdict, Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from blair_encoder import get_encoder

# ── Optional FAISS ────────────────────────────────────────────────────────────
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────

AMAZON_URL = "https://www.amazon.com/dp/{asin}"
MIN_REVIEWS_PER_ITEM = 1
PROFILE_SENTENCES    = 5

# Strict Amazon ASIN pattern
_ASIN_RE = re.compile(r'^(B[A-Z0-9]{9}|[0-9]{10})$', re.IGNORECASE)

# Fields that hold the PRODUCT title (never "summary" which is the review headline)
_PRODUCT_TITLE_FIELDS = ["title", "product_title", "name", "item_name"]

_STOPWORDS = {
    "the","a","an","this","that","and","but","or","is","are","was","were",
    "be","been","have","has","had","do","does","did","i","you","he","she",
    "it","we","they","me","him","her","us","them","my","your","its","our",
    "in","on","at","to","of","up","by","so","as","with","from","very",
    "really","just","also","not","no","more","all","any","some","one","two",
    "product","item","use","used","using","get","got","like","good","great",
    "buy","bought","would","could","will","can","should","much","many","well",
}

_SENT_RE = re.compile(r'(?<=[\.\?\!])\s+')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if len(s.split()) >= 5]


def _keywords(texts: list[str], top_n: int = 8) -> list[str]:
    counter: Counter = Counter()
    for t in texts:
        for w in re.findall(r"[a-z]{4,}", t.lower()):
            if w not in _STOPWORDS:
                counter[w] += 1
    return [w for w, _ in counter.most_common(top_n)]


def _avg_rating(reviews: list[dict]) -> Optional[float]:
    vals = []
    for r in reviews:
        try:
            v = float(r.get("rating"))
            if v is not None:
                vals.append(v)
        except (TypeError, ValueError):
            pass
    return round(sum(vals) / len(vals), 2) if vals else None


def _is_valid_asin(val: str) -> bool:
    return bool(val and _ASIN_RE.match(val.strip()))


def _best_asin(item_id: str, reviews: list[dict]) -> str:
    """Return the first real ASIN found, or '' if none."""
    candidates = []
    for r in reviews:
        for key in ("parent_asin", "asin", "item_id"):
            val = (r.get(key) or "").strip()
            if val and val not in ("N/A", "UNKNOWN"):
                candidates.append(val)
    for c in candidates:
        if _is_valid_asin(c):
            return c.upper()
    if _is_valid_asin(item_id):
        return item_id.upper()
    return ""


def _best_product_title(item_id: str, reviews: list[dict]) -> str:
    """
    Extract the PRODUCT title from review metadata.

    In Amazon Reviews 2023 JSONL the field layout is:
        "title"   → product name  (e.g. "John Frieda Detox Shampoo")
        "summary" → reviewer headline  (e.g. "Left my hair dull")   ← NEVER use this

    We count occurrences of each title value and return the most common one.
    This robustly handles the case where some reviews don't have the title populated.
    """
    title_counter: Counter = Counter()

    for r in reviews:
        # Only look at product-level fields. "summary" is explicitly excluded.
        for field in _PRODUCT_TITLE_FIELDS:
            val = (r.get(field) or "").strip()
            # Sanity checks: must be non-empty, non-generic, at least 4 chars
            if (val
                    and val not in ("N/A", "UNKNOWN", "")
                    and len(val) >= 4):
                title_counter[val] += 1
                break  # one title per review record

    if title_counter:
        return title_counter.most_common(1)[0][0]

    # Last resort: use ASIN as label
    asin = _best_asin(item_id, reviews)
    return f"Product {asin or item_id}"


# ── Item profile builder ──────────────────────────────────────────────────────

def _build_item_profile(item_id: str, reviews: list[dict]) -> dict:
    asin       = _best_asin(item_id, reviews)
    amazon_url = AMAZON_URL.format(asin=asin) if asin else ""
    title      = _best_product_title(item_id, reviews)

    texts = [r["text"] for r in reviews if isinstance(r.get("text"), str) and r["text"]]

    all_sentences: list[str] = []
    for t in texts[:50]:
        all_sentences.extend(_split_sentences(t))

    kws = set(_keywords(texts, top_n=20))

    def _score(s: str) -> int:
        return sum(1 for w in re.findall(r"[a-z]{4,}", s.lower()) if w in kws)

    top_sents  = sorted(all_sentences, key=_score, reverse=True)[:PROFILE_SENTENCES]
    aspect_kws = _keywords(texts, top_n=10)

    profile_text = (
        f"{title}. " + " ".join(top_sents) + " " + " ".join(aspect_kws)
    ).strip()

    return {
        "item_id":         asin or item_id,
        "title":           title,
        "amazon_url":      amazon_url,
        "avg_rating":      _avg_rating(reviews),
        "review_count":    len(reviews),
        "profile_text":    profile_text,
        "top_sentences":   top_sents,
        "aspect_keywords": aspect_kws,
        "sample_review":   texts[0][:300] if texts else "",
    }


# ── Item Index ────────────────────────────────────────────────────────────────

class ItemIndex:
    def __init__(self) -> None:
        self._profiles: list[dict]      = []
        self._index                     = None   # FAISS index
        self._embeddings_np: Optional[np.ndarray] = None  # brute-force fallback

    def build(self, reviews: list[dict], min_reviews: int = MIN_REVIEWS_PER_ITEM) -> int:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in reviews:
            item_id = (
                r.get("parent_asin") or r.get("item_id") or r.get("asin") or "UNKNOWN"
            )
            groups[item_id].append(r)

        filtered = {k: v for k, v in groups.items() if len(v) >= min_reviews}

        print(f"[ItemIndex] Building profiles for {len(filtered)} products ...")
        self._profiles = [
            _build_item_profile(iid, revs)
            for iid, revs in filtered.items()
        ]

        if not self._profiles:
            print("[ItemIndex] No products to index.")
            return 0

        encoder = get_encoder()
        texts   = [p["profile_text"] for p in self._profiles]

        print(f"[ItemIndex] Encoding {len(texts)} profiles with BLAIR ...")
        emb_tensor = encoder.encode(texts, batch_size=64,
                                    convert_to_tensor=True, show_progress_bar=True)
        emb_np = emb_tensor.cpu().numpy().astype(np.float32)

        if _FAISS_AVAILABLE:
            d = emb_np.shape[1]
            self._index = faiss.IndexFlatIP(d)
            self._index.add(emb_np)
            print(f"[ItemIndex] FAISS IndexFlatIP built — {self._index.ntotal} vectors.")
        else:
            self._embeddings_np = emb_np
            print(f"[ItemIndex] Brute-force index built — {len(self._profiles)} vectors.")

        return len(self._profiles)

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        if not self._profiles:
            return []

        encoder = get_encoder()
        q_emb   = encoder.encode(query_text, convert_to_tensor=True)
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        q_np = q_emb.cpu().numpy().astype(np.float32)
        k    = min(top_k, len(self._profiles))

        if _FAISS_AVAILABLE and self._index is not None:
            scores_np, indices_np = self._index.search(q_np, k)
            scores  = scores_np[0]
            indices = indices_np[0]
        else:
            scores_all = (self._embeddings_np @ q_np.T).squeeze(-1)
            top_idx    = np.argsort(scores_all)[::-1][:k]
            scores     = scores_all[top_idx]
            indices    = top_idx

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            profile     = self._profiles[int(idx)]
            explanation = _explain_match(query_text, profile)
            results.append({**profile, "score": float(score), "explanation": explanation})
        return results

    @property
    def size(self) -> int:
        return len(self._profiles)


# ── Singleton ─────────────────────────────────────────────────────────────────
_global_index: Optional[ItemIndex] = None


def get_item_index() -> ItemIndex:
    global _global_index
    if _global_index is None:
        _global_index = ItemIndex()
    return _global_index


def reset_item_index() -> None:
    global _global_index
    _global_index = ItemIndex()


# ── Match explanation ─────────────────────────────────────────────────────────
def _explain_match(query: str, profile: dict) -> str:
    query_words = set(re.findall(r"[a-z]{3,}", query.lower())) - _STOPWORDS
    matched_kws = [w for w in profile.get("aspect_keywords", []) if w in query_words]

    best_sent, best_overlap = "", 0
    for sent in profile.get("top_sentences", []):
        overlap = sum(1 for w in re.findall(r"[a-z]{3,}", sent.lower()) if w in query_words)
        if overlap > best_overlap:
            best_overlap, best_sent = overlap, sent

    parts: list[str] = []
    if matched_kws:
        parts.append(f"Keyword overlap: {', '.join(matched_kws[:5])}")
    if best_sent:
        short = best_sent[:160] + ("..." if len(best_sent) > 160 else "")
        parts.append(f'Supporting review: "{short}"')

    return " | ".join(parts) if parts else "Semantic similarity via BLAIR embedding."
