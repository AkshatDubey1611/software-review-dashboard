# chart_utils.py

import matplotlib.pyplot as plt
from summary_formatter import ASPECT_HINTS, count_matches


def show_aspect_distribution(clusters):
    """
    Display bar chart of number of reviews per aspect.
    """
    aspects = list(clusters.keys())
    counts = [len(clusters[a]) for a in aspects]

    plt.figure(figsize=(10, 5))
    plt.bar(aspects, counts)
    plt.title("Aspect Distribution")
    plt.xlabel("Aspect")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def show_sentiment_trends(clusters):
    """
    Display grouped bar chart of rough positive vs negative counts per aspect.
    """
    aspects = []
    pos_counts = []
    neg_counts = []

    for aspect, texts in clusters.items():
        hints = ASPECT_HINTS.get(aspect, {"positive": [], "negative": []})

        aspects.append(aspect)
        pos_counts.append(count_matches(texts, hints["positive"]))
        neg_counts.append(count_matches(texts, hints["negative"]))

    x = range(len(aspects))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar([i - width/2 for i in x], pos_counts, width=width, label="Positive")
    plt.bar([i + width/2 for i in x], neg_counts, width=width, label="Negative")

    plt.title("Positive vs Negative Mentions by Aspect")
    plt.xlabel("Aspect")
    plt.ylabel("Count")
    plt.xticks(list(x), aspects, rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()