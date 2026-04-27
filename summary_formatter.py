# summary_formatter.py

from collections import Counter
import re

# -----------------------------
# Aspect hint words
# -----------------------------
ASPECT_HINTS = {
    "performance": {
        "positive": ["fast", "smooth", "responsive", "quick", "stable"],
        "negative": ["slow", "lag", "crash", "freeze", "bug", "buggy"]
    },
    "installation": {
        "positive": ["easy", "simple", "quick", "straightforward", "install"],
        "negative": ["difficult", "problem", "failed", "error", "issue"]
    },
    "price": {
        "positive": ["cheap", "worth", "value", "affordable", "fair"],
        "negative": ["expensive", "overpriced", "costly", "waste"]
    },
    "features": {
        "positive": ["useful", "many", "helpful", "powerful", "feature-rich"],
        "negative": ["missing", "limited", "lacking", "confusing"]
    },
    "support": {
        "positive": ["helpful", "responsive", "support", "assistance"],
        "negative": ["poor", "useless", "slow", "unhelpful"]
    },
    "compatibility": {
        "positive": ["compatible", "works", "supported"],
        "negative": ["incompatible", "unsupported", "issue", "problem"]
    },
    "interface": {
        "positive": ["clean", "easy", "intuitive", "simple", "friendly"],
        "negative": ["confusing", "cluttered", "ugly", "hard"]
    },
    "design": {
        "positive": ["nice", "beautiful", "modern", "good-looking", "clean"],
        "negative": ["bad", "outdated", "poor", "ugly"]
    }
}

DEFAULT_HINTS = {
    "positive": ["good", "great", "excellent", "easy", "fast", "helpful",
                 "nice", "works", "love", "best", "smooth", "perfect"],
    "negative": ["bad", "poor", "slow", "crash", "issue", "problem",
                 "error", "difficult", "worst", "hate", "broken", "waste"]
}

STOPWORDS = {
    "the","and","a","an","is","it","to","for","of","i","you","this","that",
    "in","on","with","was","are","be","have","has","but","not","they","we",
    "as","or","at","by","from","so","if","its","my","me","do","does","did",
    "were","their","them","can","will","would","should","about","what","which",
    "when","how","more","no","than","also","very","really","just","all"
}


# -----------------------------
# Helpers
# -----------------------------
def _get_text(r):
    """Accept either a review dict or a plain string."""
    return r["text"] if isinstance(r, dict) else r


def count_matches(reviews, keywords):
    count = 0
    for r in reviews:
        text = _get_text(r).lower()
        if any(word in text for word in keywords):
            count += 1
    return count


def clean_words(text):
    return re.findall(r"[a-zA-Z']{3,}", text.lower())


def get_top_keywords(reviews, top_n=5):
    words = []
    for r in reviews:
        for word in clean_words(_get_text(r)):
            if word not in STOPWORDS:
                words.append(word)
    if not words:
        return []
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]


def get_example_review(reviews, positive_words, negative_words):
    """
    Pick the review with the most signal words.
    Returns (review_dict_or_str, source_line_str).
    """
    best = None
    best_score = -1

    all_signal = positive_words + negative_words
    for r in reviews:
        text_lower = _get_text(r).lower()
        score = sum(1 for w in all_signal if w in text_lower)
        if score > best_score:
            best_score = score
            best = r

    if best is None and reviews:
        best = reviews[0]

    if best is None:
        return "No example available.", ""

    text = shorten_text(_get_text(best))
    source = _build_source_line(best)
    return text, source


def _build_source_line(r):
    """Build a readable source attribution line from a review dict."""
    if not isinstance(r, dict):
        return ""

    parts = []

    asin = r.get("asin", "")
    if asin and asin != "N/A":
        parts.append(f"ASIN: {asin}")
        parts.append(f"Amazon: https://www.amazon.com/dp/{asin}")

    reviewer = r.get("reviewer", "")
    if reviewer and reviewer != "Anonymous":
        parts.append(f"Reviewer: {reviewer}")

    date = r.get("date", "")
    if date:
        parts.append(f"Date: {date}")

    rating = r.get("rating", None)
    if rating is not None:
        stars = int(rating)
        parts.append(f"Rating: {'*' * stars} ({rating}/5)")

    summary = r.get("summary", "")
    if summary:
        parts.append(f"Title: \"{summary}\"")

    return " | ".join(parts)


def shorten_text(text, max_len=200):
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _resolve_hints(aspect):
    if aspect in ASPECT_HINTS:
        return ASPECT_HINTS[aspect]
    for fixed_key, hints in ASPECT_HINTS.items():
        if fixed_key in aspect.lower():
            return hints
    return DEFAULT_HINTS


def _resolve_theme(aspect):
    theme_map = {
        "performance": "Users frequently discuss speed, responsiveness, crashes, and overall stability.",
        "installation": "Users often comment on setup experience, installation difficulty, and updates.",
        "price": "Users evaluate whether the software feels affordable, worth the cost, or overpriced.",
        "features": "Users talk about available tools, capabilities, missing functions, and usefulness.",
        "support": "Users discuss customer support quality, responsiveness, and available assistance.",
        "compatibility": "Users mention whether the software works across different devices or versions.",
        "interface": "Users comment on usability, menus, navigation, and how easy the interface feels.",
        "design": "Users describe the visual appearance, style, and overall look of the software."
    }
    if aspect in theme_map:
        return theme_map[aspect]
    for fixed_key, theme in theme_map.items():
        if fixed_key in aspect.lower():
            return theme
    return f"Users discuss topics related to: {aspect}."


# -----------------------------
# Summary builder
# -----------------------------
def format_aspect_summary(aspect, reviews):
    """
    reviews: list of review dicts (or plain strings, for backward compat).
    """
    count = len(reviews)
    if count == 0:
        return "  No major discussion was found for this aspect."

    hints = _resolve_hints(aspect)
    pos_count = count_matches(reviews, hints["positive"])
    neg_count = count_matches(reviews, hints["negative"])

    keywords = get_top_keywords(reviews, top_n=5)
    example_text, source_line = get_example_review(reviews, hints["positive"], hints["negative"])
    theme = _resolve_theme(aspect)

    lines = [
        f"  Theme   : {theme}",
        f"  Positive: {pos_count} reviews with favorable language.",
        f"  Negative: {neg_count} reviews with complaints or dissatisfaction.",
        f"  Example : \"{example_text}\"",
    ]
    if source_line:
        lines.append(f"  Source  : {source_line}")
    lines.append(f"  Keywords: {', '.join(keywords) if keywords else 'None found'}")

    return "\n".join(lines)


def format_all_summaries(clusters):
    """
    clusters: {aspect -> [review_dict, ...]}
    """
    summaries = {}
    for aspect, reviews in clusters.items():
        summaries[aspect] = format_aspect_summary(aspect, reviews)
    return summaries
