# sentence_retrieval.py
#
# BLAIR upgrades:
#   1. Uses the shared BLAIR encoder (blair_encoder.get_encoder()) — unchanged.
#   2. _build_source_line() now includes a direct Amazon product URL for the
#      review's parent_asin / asin so every result has a clickable product link.
#   3. Result dicts now carry "amazon_url" and "item_id" fields so the GUI
#      can render them as hyperlinks.

from __future__ import annotations
import re
import torch
from blair_encoder import get_encoder

_SENT_END_RE = re.compile(r'(?<=[\.\?\!])\s+')

AMAZON_URL = "https://www.amazon.com/dp/{asin}"

BOOST_KEYWORDS = {
    "price":        ["price","cost","cheap","expensive","value","worth","fair","affordable"],
    "installation": ["install","installation","setup","easy","quick","loaded","configure"],
    "performance":  ["fast","slow","lag","crash","smooth","responsive","reliable","stable"],
    "support":      ["support","help","service","assistance","staff","response","reply"],
    "interface":    ["interface","ui","layout","menu","easy","intuitive","design","navigate"],
    "quality":      ["quality","build","durable","sturdy","material","solid","broke"],
    "scent":        ["scent","smell","fragrance","aroma","odor","perfume"],
    "skin":         ["skin","face","cream","moisturizer","acne","serum","lotion"],
    "hair":         ["hair","shampoo","conditioner","scalp","frizz","curl"],
}


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_END_RE.split(text) if s.strip()]


def retrieve_relevant_sentences(query: str, reviews: list, top_k: int = 5) -> list[dict]:
    """
    Returns top-k most relevant sentences from reviews for the query.
    Uses BLAIR embeddings + keyword boost.

    Each result dict now contains:
      sentence, score, source (formatted string),
      amazon_url, item_id, title (for GUI hyperlinking)
    """
    if not reviews or not query.strip():
        return []

    encoder = get_encoder()

    # Build sentence pool
    sentence_pool: list[tuple[str, object]] = []
    for r in reviews:
        text = r["text"] if isinstance(r, dict) else str(r)
        for sent in split_into_sentences(text):
            if len(sent.split()) >= 4:
                sentence_pool.append((sent, r))

    if not sentence_pool:
        return []

    sentences_only = [s for s, _ in sentence_pool]
    print(f"Searching across {len(sentences_only)} sentences with BLAIR ...")

    # BLAIR embeddings (L2-normalised), dot-product == cosine similarity
    query_emb = encoder.encode(query, convert_to_tensor=True)
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)   # (1, d)

    all_embs = []
    for i in range(0, len(sentences_only), 128):
        all_embs.append(encoder.encode(sentences_only[i:i+128], convert_to_tensor=True))
    sent_embs = torch.cat(all_embs, dim=0)   # (N, d)

    base_scores = (sent_embs @ query_emb.T).squeeze(-1)   # (N,)

    # Keyword boost
    query_lower = query.lower()
    boost_list  = []
    for sent in sentences_only:
        sent_lower = sent.lower()
        matched = sum(
            1 for words in BOOST_KEYWORDS.values()
            if any(w in query_lower for w in words) and any(w in sent_lower for w in words)
        )
        boost = matched * 0.08 + (0.10 if matched >= 2 else 0.0)
        boost_list.append(boost)

    boost_tensor = torch.tensor(boost_list, device=base_scores.device)
    final_scores = base_scores + boost_tensor

    k = min(top_k, len(sentence_pool))
    top = final_scores.topk(k)

    results = []
    for score, idx in zip(top.values, top.indices):
        sent, review = sentence_pool[int(idx)]
        source_str, amazon_url, item_id, title = _build_source_parts(review)
        results.append({
            "sentence":   sent,
            "score":      float(score),
            "source":     source_str,
            "amazon_url": amazon_url,
            "item_id":    item_id,
            "title":      title,
        })
    return results


def _build_source_parts(r) -> tuple[str, str, str, str]:
    """
    Returns (formatted_source_string, amazon_url, item_id, title).
    amazon_url is a direct product page link built from parent_asin or asin.
    """
    if not isinstance(r, dict):
        return "", "", "", ""

    # Canonical product ID: prefer parent_asin
    asin        = r.get("asin", "")
    parent_asin = r.get("parent_asin", "") or r.get("item_id", "")
    canonical   = parent_asin if parent_asin else asin

    amazon_url = AMAZON_URL.format(asin=canonical) if canonical and canonical != "N/A" else ""
    title      = r.get("title", "")
    item_id    = canonical

    parts = []
    if canonical and canonical != "N/A":
        parts.append(f"ASIN: {canonical}")
        parts.append(f"Amazon: {amazon_url}")
    if title:
        parts.append(f"Product: {title}")
    reviewer = r.get("reviewer", "")
    if reviewer and reviewer != "Anonymous":
        parts.append(f"Reviewer: {reviewer}")
    date = r.get("date", "")
    if date:
        parts.append(f"Date: {date}")
    rating = r.get("rating")
    if rating is not None:
        stars = int(float(rating))
        parts.append(f"Rating: {'*' * stars} ({rating}/5)")
    summary = r.get("summary", "")
    if summary:
        parts.append(f'Review title: "{summary}"')

    return " | ".join(parts), amazon_url, item_id, title
