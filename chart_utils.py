# chart_utils.py
#
# FIX: Charts now open every single time the button is clicked.
#
# Root cause of "works once then stops":
#   After plt.show() closes the window, matplotlib's internal figure state
#   is destroyed. The next call reuses the same figure object which is now
#   dead, so nothing appears.
#
# Solution:
#   1. Use plt.switch_backend("TkAgg") INSIDE the function, before every draw.
#   2. Call plt.close("all") at the START of each function to wipe any dead figures.
#   3. Create a BRAND NEW fig, ax with plt.subplots() every call — never reuse.
#   4. Use plt.show(block=True) so the thread blocks until the window is closed,
#      preventing the thread from dying while the window is still open.
#   5. Each button click spawns a fresh daemon=False thread, so closing one chart
#      does not kill the next one.

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # safe headless default at import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import textwrap

# Dark theme rc params — applied after every backend switch
_DARK_RC = {
    "figure.facecolor":  "#1e1e2e",
    "axes.facecolor":    "#2a2a3e",
    "axes.edgecolor":    "#585b70",
    "axes.labelcolor":   "#cdd6f4",
    "axes.titlecolor":   "#cdd6f4",
    "xtick.color":       "#a6adc8",
    "ytick.color":       "#a6adc8",
    "text.color":        "#cdd6f4",
    "grid.color":        "#313149",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#2a2a3e",
    "legend.edgecolor":  "#585b70",
    "legend.labelcolor": "#cdd6f4",
}

# Positive / negative keyword lists for sentiment fallback
_POS = ["good","great","excellent","best","love","nice","perfect","smooth",
        "easy","fast","works","clean","soft","fresh","quality","happy","recommend"]
_NEG = ["bad","poor","worst","hate","broken","waste","cheap","slow","issue",
        "problem","difficult","damaged","return","refund","crack","peel","fake"]


def _switch_backend():
    """Switch to TkAgg and re-apply dark theme. Called at the top of every chart function."""
    try:
        matplotlib.use("TkAgg")
        plt.switch_backend("TkAgg")
    except Exception:
        pass
    plt.rcParams.update(_DARK_RC)


def _get_text_and_rating(item):
    if isinstance(item, dict):
        return str(item.get("text", "")), item.get("rating")
    return str(item), None


def _sentiment(item) -> str:
    text, rating = _get_text_and_rating(item)
    try:
        r = float(rating)
        if r >= 4: return "positive"
        if r <= 2: return "negative"
        return "neutral"
    except (TypeError, ValueError):
        pass
    tl = text.lower()
    pos = sum(1 for w in _POS if w in tl)
    neg = sum(1 for w in _NEG if w in tl)
    if pos > neg:  return "positive"
    if neg > pos:  return "negative"
    return "neutral"


def _wrap(label: str, width: int = 13) -> str:
    return "\n".join(textwrap.wrap(label, width))


# Keep these as module-level proxies so chart_utils.py can still be imported
# by summary_formatter.py without errors
from dynamic_aspects import get_sentiment_hints
from summary_formatter import ASPECT_HINTS, count_matches


def show_aspect_distribution(clusters: dict) -> None:
    """
    Bar chart: review count per aspect.
    Opens a new window every time — works on repeated clicks.
    """
    # ── STEP 1: reset matplotlib completely ───────────────────────────────
    plt.close("all")
    _switch_backend()

    aspects = list(clusters.keys())
    counts  = [len(clusters[a]) for a in aspects]
    wrapped = [_wrap(a) for a in aspects]

    # ── STEP 2: create a brand-new figure ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(aspects) * 1.5), 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#2a2a3e")

    bars = ax.bar(range(len(aspects)), counts,
                  color="#7c5cbf", edgecolor="#1e1e2e", linewidth=0.8)

    peak = max(counts) if counts else 1
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + peak * 0.012,
                str(cnt), ha="center", va="bottom",
                fontsize=9, color="#cdd6f4")

    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels(wrapped, fontsize=8, ha="center")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("Auto-Discovered Aspect Distribution",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Aspect", fontsize=11)
    ax.set_ylabel("Number of Reviews", fontsize=11)
    ax.set_xlim(-0.6, len(aspects) - 0.4)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#585b70")

    plt.tight_layout()

    # ── STEP 3: block=True keeps window open; thread stays alive until closed
    plt.show(block=True)
    plt.close(fig)


def show_sentiment_trends(clusters: dict) -> None:
    """
    Grouped bar chart: positive / neutral / negative per aspect.
    Opens a new window every time — works on repeated clicks.
    """
    # ── STEP 1: reset matplotlib completely ───────────────────────────────
    plt.close("all")
    _switch_backend()

    aspects                    = list(clusters.keys())
    pos_counts, neu_counts, neg_counts = [], [], []

    for aspect in aspects:
        pos = neu = neg = 0
        for item in clusters[aspect]:
            s = _sentiment(item)
            if s == "positive":   pos += 1
            elif s == "negative": neg += 1
            else:                 neu += 1
        pos_counts.append(pos)
        neu_counts.append(neu)
        neg_counts.append(neg)

    wrapped = [_wrap(a) for a in aspects]
    x, width = list(range(len(aspects))), 0.26

    # ── STEP 2: create a brand-new figure ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(11, len(aspects) * 1.6), 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#2a2a3e")

    pb = ax.bar([i - width for i in x], pos_counts, width=width,
                label="Positive", color="#3fb950", edgecolor="#1e1e2e")
    nb = ax.bar(x,                     neu_counts, width=width,
                label="Neutral",  color="#e3a03c", edgecolor="#1e1e2e")
    rb = ax.bar([i + width for i in x], neg_counts, width=width,
                label="Negative", color="#e05c5c", edgecolor="#1e1e2e")

    peak = max(
        max(pos_counts, default=1),
        max(neu_counts, default=1),
        max(neg_counts, default=1),
        1,
    )
    for grp in (pb, nb, rb):
        for bar in grp:
            val = int(bar.get_height())
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + peak * 0.012,
                        str(val), ha="center", va="bottom",
                        fontsize=7, color="#cdd6f4")

    ax.set_xticks(x)
    ax.set_xticklabels(wrapped, fontsize=8, ha="center")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("Sentiment by Aspect  (rating-based + keyword fallback)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Aspect", fontsize=11)
    ax.set_ylabel("Review Count", fontsize=11)
    ax.set_xlim(-0.6, len(aspects) - 0.4)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#585b70")

    plt.tight_layout()

    # ── STEP 3: block=True keeps window open; thread stays alive until closed
    plt.show(block=True)
    plt.close(fig)
