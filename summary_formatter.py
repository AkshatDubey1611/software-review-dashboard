# summary_formatter.py
#
# BLAIR change:
#   Positive/negative hint words now come from dynamic_aspects.get_sentiment_hints()
#   instead of a hard-coded ASPECT_HINTS dict.  Every dynamically-discovered label
#   (Hair Care, Fragrance, Nail Care, Gaming, etc.) gets the right signal words
#   automatically, making the sentiment counts in the meta-review meaningful for
#   any dataset domain.

from __future__ import annotations
from collections import Counter
import re

from dynamic_aspects import get_sentiment_hints   # replaces hard-coded ASPECT_HINTS

# Backward-compat proxy so chart_utils.py can still import ASPECT_HINTS
class _Proxy:
    def get(self, key, default=None): return get_sentiment_hints(key)
    def __getitem__(self, key):       return get_sentiment_hints(key)
    def __contains__(self, key):      return True

ASPECT_HINTS = _Proxy()

DEFAULT_HINTS = {
    "positive":["good","great","excellent","easy","fast","helpful","nice","works","love","best","smooth","perfect"],
    "negative":["bad","poor","slow","crash","issue","problem","error","difficult","worst","hate","broken","waste"],
}

STOPWORDS = {
    "the","and","a","an","is","it","to","for","of","i","you","this","that","in","on","with",
    "was","are","be","have","has","but","not","they","we","as","or","at","by","from","so","if",
    "its","my","me","do","does","did","were","their","them","can","will","would","should","about",
    "what","which","when","how","more","no","than","also","very","really","just","all",
}


def _get_text(r) -> str:
    return r["text"] if isinstance(r, dict) else str(r)


def count_matches(reviews, keywords) -> int:
    return sum(1 for r in reviews if any(w in _get_text(r).lower() for w in keywords))


def clean_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']{3,}", text.lower())


def get_top_keywords(reviews, top_n: int = 5) -> list[str]:
    words = [w for r in reviews for w in clean_words(_get_text(r)) if w not in STOPWORDS]
    return [w for w, _ in Counter(words).most_common(top_n)] if words else []


def get_example_review(reviews, positive_words, negative_words):
    best, best_score = None, -1
    for r in reviews:
        score = sum(1 for w in positive_words + negative_words if w in _get_text(r).lower())
        if score > best_score:
            best_score, best = score, r
    if best is None and reviews:
        best = reviews[0]
    if best is None:
        return "No example available.", ""
    return shorten_text(_get_text(best)), _build_source_line(best)


def _build_source_line(r) -> str:
    if not isinstance(r, dict): return ""
    parts = []
    asin = r.get("asin","")
    if asin and asin != "N/A":
        parts += [f"ASIN: {asin}", f"Amazon: https://www.amazon.com/dp/{asin}"]
    reviewer = r.get("reviewer","")
    if reviewer and reviewer != "Anonymous": parts.append(f"Reviewer: {reviewer}")
    date = r.get("date","")
    if date: parts.append(f"Date: {date}")
    rating = r.get("rating")
    if rating is not None:
        stars = int(float(rating))
        parts.append(f"Rating: {'*'*stars} ({rating}/5)")
    summary = r.get("summary","")
    if summary: parts.append(f'Title: "{summary}"')
    return " | ".join(parts)


def shorten_text(text: str, max_len: int = 200) -> str:
    text = text.replace("\n"," ").strip()
    return (text[:max_len].rsplit(" ",1)[0] + "...") if len(text) > max_len else text


def _resolve_theme(aspect: str) -> str:
    hints = get_sentiment_hints(aspect)
    pos_kw = ", ".join(hints["positive"][:3])
    neg_kw = ", ".join(hints["negative"][:3])
    return (f"Users discuss {aspect.lower()} — "
            f"positive signals: {pos_kw}; negative signals: {neg_kw}.")


def format_aspect_summary(aspect: str, reviews: list) -> str:
    if not reviews:
        return "  No major discussion was found for this aspect."

    hints     = get_sentiment_hints(aspect)
    pos_count = count_matches(reviews, hints["positive"])
    neg_count = count_matches(reviews, hints["negative"])
    keywords  = get_top_keywords(reviews, top_n=5)
    example_text, source_line = get_example_review(reviews, hints["positive"], hints["negative"])

    ratings = []
    for r in reviews:
        try:
            v = float(r.get("rating")) if isinstance(r, dict) else None
            if v is not None: ratings.append(v)
        except (TypeError, ValueError): pass
    avg_str = f"{sum(ratings)/len(ratings):.2f} / 5" if ratings else "N/A"

    lines = [
        f"  Theme   : {_resolve_theme(aspect)}",
        f"  Avg     : {avg_str} stars",
        f"  Positive: {pos_count} reviews with favorable language.",
        f"  Negative: {neg_count} reviews with complaints or dissatisfaction.",
        f'  Example : "{example_text}"',
    ]
    if source_line: lines.append(f"  Source  : {source_line}")
    lines.append(f"  Keywords: {', '.join(keywords) if keywords else 'None found'}")
    return "\n".join(lines)


def format_all_summaries(clusters: dict) -> dict[str, str]:
    return {aspect: format_aspect_summary(aspect, reviews)
            for aspect, reviews in clusters.items()}
