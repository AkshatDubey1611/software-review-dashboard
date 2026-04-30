# review_processing.py
#
# FIX: load_reviews() now correctly separates:
#   - "title"   = the PRODUCT title (e.g. "John Frieda Detox Shampoo")
#   - "summary" = the REVIEW headline (e.g. "Left my hair dull") — kept separately
#
# In Amazon Reviews 2023 JSONL, the field layout is:
#   "title"       -> product name (what we want to display)
#   "summary"     -> review headline written by the reviewer
#
# The old code stored whichever came first, sometimes picking review headlines.
# Now we explicitly look for product-level fields vs review-level fields.

import json
import re
import math
import heapq
from collections import Counter

from dynamic_aspects import cluster_reviews_dynamic
from config import ASPECT_KEYWORDS, STOPWORDS

# -----------------------------
# ASPECT DESCRIPTIONS (used by query_understanding.py)
# -----------------------------
ASPECT_DESCRIPTIONS = {
    "performance": "Speed, responsiveness, crashes, bugs, freezing, lag, and overall software stability.",
    "installation": "Setup process, installation ease, download speed, updates, and configuration.",
    "price": "Cost, value for money, affordability, pricing plans, and whether the software is worth it.",
    "features": "Available tools, capabilities, functions, options, and feature richness.",
    "support": "Customer service quality, responsiveness of help desk, and available assistance.",
    "compatibility": "Support for different OS versions, devices, platforms, and software versions.",
    "interface": "UI layout, menus, navigation, ease of use, and dashboard design.",
    "design": "Visual appearance, style, look and feel, and aesthetic quality of the software."
}

# Amazon Reviews 2023 JSONL field names for the PRODUCT title.
# These come from the item metadata, NOT from the reviewer.
# "summary" is EXCLUDED here — it's the review headline, not the product name.
_PRODUCT_TITLE_FIELDS = ["title", "product_title", "name", "item_name"]

# Fields that contain the actual review text
_REVIEW_TEXT_FIELDS = ["reviewText", "text", "review", "content", "comment"]


def load_reviews(path: str, max_reviews: int = 10000):
    """
    Returns a list of dicts with full review metadata.

    Field mapping (Amazon Reviews 2023 format):
      "title"       -> product name (from metadata)
      "summary"     -> review headline (written by reviewer — e.g. "Left my hair dull")
      "text"        -> full review body
      "asin"        -> item ASIN
      "parent_asin" -> parent product ASIN (groups variants)
      "rating"      -> star rating (1-5)

    KEY FIX: We now store the product title in "title" and the reviewer's
    headline in "summary" as separate fields, so item_retrieval.py can
    use the correct one for display.
    """
    reviews = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ── Review text ───────────────────────────────────────────────
            text = None
            for field in _REVIEW_TEXT_FIELDS:
                if field in data and data[field]:
                    text = str(data[field]).strip()
                    break
            if not text:
                continue

            # ── Product title (from item metadata, NOT from reviewer) ──────
            # In Amazon Reviews 2023, "title" IS the product name.
            # Example: "John Frieda Detox & Repair Shampoo, 8.45 fl oz"
            product_title = ""
            for field in _PRODUCT_TITLE_FIELDS:
                if field in data and data[field]:
                    val = str(data[field]).strip()
                    if val and val not in ("N/A", "UNKNOWN"):
                        product_title = val
                        break

            # ── Review summary/headline (written by reviewer) ─────────────
            # Example: "Left my hair dull" or "Great shampoo!"
            # This is what was mistakenly shown as the product name before.
            review_summary = str(data.get("summary", "") or "").strip()

            # ── Canonical item identifier ─────────────────────────────────
            asin        = data.get("asin", "N/A")
            parent_asin = data.get("parent_asin") or data.get("parentAsin") or ""
            item_id     = parent_asin if parent_asin else asin

            reviews.append({
                "text":        text,
                "asin":        asin,
                "parent_asin": parent_asin,
                "item_id":     item_id,
                "title":       product_title,    # ← PRODUCT name (for display)
                "summary":     review_summary,   # ← REVIEWER headline (for analysis)
                "reviewer":    data.get("reviewerName", "Anonymous"),
                "date":        data.get("reviewTime", ""),
                "rating":      data.get("overall", data.get("rating", None)),
            })

            if len(reviews) >= max_reviews:
                break

    return reviews


# -----------------------------
# KEYWORD-BASED CLUSTERING (used by retrieval.py)
# -----------------------------
def cluster_reviews(reviews):
    """
    reviews: list of dicts or plain strings.
    Returns {aspect_name: [text_string, ...]}
    """
    clusters = {aspect: [] for aspect in ASPECT_KEYWORDS}
    clusters["other"] = []

    for r in reviews:
        text = r["text"] if isinstance(r, dict) else r
        text_lower = text.lower()
        matched = False

        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                clusters[aspect].append(text)
                matched = True
                break

        if not matched:
            clusters["other"].append(text)

    return {k: v for k, v in clusters.items() if v}


# -----------------------------
# TOKENIZATION
# -----------------------------
_SENT_END_RE = re.compile(r'(?<=[\.\?\!])\s+')

def sent_tokenize(text):
    return [s.strip() for s in _SENT_END_RE.split(text) if s.strip()]

def word_tokenize(text):
    return re.findall(r"[a-zA-Z0-9']{2,}", text.lower())


# -----------------------------
# SUMMARIZATION
# -----------------------------
def score_sentences(text, top_k=3):
    sents = sent_tokenize(text)
    if not sents:
        return ""

    words = []
    for s in sents:
        words.extend([w for w in word_tokenize(s) if w not in STOPWORDS])

    if not words:
        return "No substantive content."

    freq = Counter(words)
    maxf = max(freq.values())
    for w in list(freq.keys()):
        freq[w] = freq[w] / maxf

    scores = []
    for i, s in enumerate(sents):
        wlist = [w for w in word_tokenize(s) if w not in STOPWORDS]
        score = sum(freq.get(w, 0.0) for w in wlist) / math.sqrt(len(wlist)) if wlist else 0.0
        scores.append((score, i, s))

    top = heapq.nlargest(top_k, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])

    summary = " ".join([t[2].rstrip('.') + '.' for t in top_sorted if t[0] > 0])
    if not summary:
        summary = " ".join(sents[:min(top_k, len(sents))])
    return summary


# -----------------------------
# FINAL PIPELINE
# -----------------------------
def process_reviews(reviews, num_clusters=8):
    """
    reviews: list of dicts with "text" key (from load_reviews).
    Returns (clusters, summaries):
      clusters:  {aspect -> [review_dict, ...]}
      summaries: {aspect -> str}
    """
    texts = [r["text"] if isinstance(r, dict) else r for r in reviews]
    text_clusters = cluster_reviews_dynamic(texts, num_clusters=num_clusters)

    text_to_review = {}
    for r in reviews:
        t = r["text"] if isinstance(r, dict) else r
        text_to_review[t] = r

    clusters = {}
    for aspect, cluster_texts in text_clusters.items():
        clusters[aspect] = [
            text_to_review.get(t, {
                "text": t, "asin": "N/A", "parent_asin": "",
                "item_id": "N/A", "title": "",
                "reviewer": "Anonymous", "date": "",
                "rating": None, "summary": ""
            })
            for t in cluster_texts
        ]

    summaries = {}
    for aspect, review_list in clusters.items():
        if review_list:
            combined = " ".join([r["text"] if isinstance(r, dict) else r
                                 for r in review_list[:10]])[:4000]
            summaries[aspect] = score_sentences(combined)
        else:
            summaries[aspect] = "No major discussion."

    return clusters, summaries
