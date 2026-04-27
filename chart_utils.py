# chart_utils.py

import matplotlib.pyplot as plt
import textwrap


POSITIVE_WORDS = [
    "good", "great", "excellent", "love", "loved", "nice", "perfect",
    "amazing", "wonderful", "easy", "works", "best", "recommend",
    "happy", "smooth", "beautiful", "soft", "effective"
]

NEGATIVE_WORDS = [
    "bad", "poor", "terrible", "hate", "hated", "awful", "worst",
    "difficult", "problem", "issue", "broken", "disappointed",
    "return", "waste", "smell weird", "not good", "doesn't work"
]


def clean_aspect_label(aspect, max_words=3):
    parts = [p.strip() for p in str(aspect).split("/") if p.strip()]
    label = " / ".join(parts[:max_words])

    if len(label) > 28:
        label = textwrap.fill(label, width=18)

    return label


def get_text_and_rating(item):
    if isinstance(item, dict):
        text = item.get("text", "")
        rating = item.get("rating")
        return str(text), rating

    return str(item), None


def get_sentiment(item):
    text, rating = get_text_and_rating(item)

    try:
        rating = float(rating)
        if rating >= 4:
            return "positive"
        elif rating <= 2:
            return "negative"
        else:
            return "neutral"
    except (TypeError, ValueError):
        pass

    text_lower = text.lower()

    pos_hits = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

    if pos_hits > neg_hits:
        return "positive"
    elif neg_hits > pos_hits:
        return "negative"
    else:
        return "neutral"


def show_aspect_distribution(clusters):
    aspects = list(clusters.keys())
    counts = [len(clusters[a]) for a in aspects]
    labels = [clean_aspect_label(a) for a in aspects]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.title("Auto-Discovered Aspect Distribution")
    plt.xlabel("Aspect")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


def show_sentiment_trends(clusters):
    aspects = list(clusters.keys())
    labels = [clean_aspect_label(a) for a in aspects]

    pos_counts = []
    neu_counts = []
    neg_counts = []

    for aspect in aspects:
        items = clusters[aspect]

        pos = 0
        neu = 0
        neg = 0

        for item in items:
            sentiment = get_sentiment(item)

            if sentiment == "positive":
                pos += 1
            elif sentiment == "negative":
                neg += 1
            else:
                neu += 1

        pos_counts.append(pos)
        neu_counts.append(neu)
        neg_counts.append(neg)

    # guaranteed fallback: avoid blank chart
    if sum(pos_counts) + sum(neu_counts) + sum(neg_counts) == 0:
        neu_counts = [len(clusters[a]) for a in aspects]

    print("DEBUG sentiment counts")
    print("Positive:", pos_counts)
    print("Neutral:", neu_counts)
    print("Negative:", neg_counts)

    x = list(range(len(aspects)))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar([i - width for i in x], pos_counts, width=width, label="Positive")
    plt.bar(x, neu_counts, width=width, label="Neutral")
    plt.bar([i + width for i in x], neg_counts, width=width, label="Negative")

    plt.title("Sentiment Trends by Auto-Discovered Aspect")
    plt.xlabel("Aspect")
    plt.ylabel("Review Count")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()
