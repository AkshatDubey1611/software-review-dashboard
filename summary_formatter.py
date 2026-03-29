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

# -----------------------------
# Tiny stopword list
# -----------------------------
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
def count_matches(texts, keywords):
    count = 0
    for text in texts:
        text_lower = text.lower()
        if any(word in text_lower for word in keywords):
            count += 1
    return count


def clean_words(text):
    return re.findall(r"[a-zA-Z']{3,}", text.lower())


def get_top_keywords(texts, top_n=5):
    words = []

    for text in texts:
        for word in clean_words(text):
            if word not in STOPWORDS:
                words.append(word)

    if not words:
        return []

    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]


def get_example_review(texts, positive_words, negative_words):
    """
    Pick a review that contains useful signal words.
    Prefer balanced/meaningful reviews over random ones.
    """
    for text in texts:
        text_lower = text.lower()

        if any(w in text_lower for w in positive_words + negative_words):
            return shorten_text(text)

    if texts:
        return shorten_text(texts[0])

    return "No example available."


def shorten_text(text, max_len=180):
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


# -----------------------------
# Better summary builder
# -----------------------------
def format_aspect_summary(aspect, texts):
    count = len(texts)

    if count == 0:
        return "• No major discussion was found for this aspect."

    hints = ASPECT_HINTS.get(aspect, {"positive": [], "negative": []})

    pos_count = count_matches(texts, hints["positive"])
    neg_count = count_matches(texts, hints["negative"])

    keywords = get_top_keywords(texts, top_n=5)
    example = get_example_review(texts, hints["positive"], hints["negative"])

    # Aspect-specific theme line
    theme_map = {
        "performance": "Users frequently discuss speed, responsiveness, crashes, and overall software stability.",
        "installation": "Users often comment on setup experience, installation difficulty, and update process.",
        "price": "Users evaluate whether the software feels affordable, worth the cost, or overpriced.",
        "features": "Users talk about available tools, capabilities, missing functions, and overall usefulness.",
        "support": "Users discuss customer support quality, responsiveness, and available assistance.",
        "compatibility": "Users mention whether the software works across different devices, systems, or versions.",
        "interface": "Users comment on usability, menus, navigation, and how easy the interface feels.",
        "design": "Users describe the visual appearance, style, and overall look of the software."
    }

    summary_lines = [
        f"• Common theme: {theme_map.get(aspect, 'Users frequently discuss this aspect in their reviews.')}",
        f"• Positive signal: {pos_count} reviews contain favorable language related to this aspect.",
        f"• Negative signal: {neg_count} reviews contain complaints or dissatisfaction related to this aspect.",
        f"• Example review: \"{example}\"",
        f"• Top keywords: {', '.join(keywords) if keywords else 'No strong keywords found'}"
    ]

    return "\n".join(summary_lines)


def format_all_summaries(clusters):
    summaries = {}

    for aspect, texts in clusters.items():
        summaries[aspect] = format_aspect_summary(aspect, texts)

    return summaries