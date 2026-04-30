# utils.py

import re


def clean_text(text):
    """
    Clean excessive whitespace and weird formatting.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shorten_text(text, max_chars=350):
    """
    Shorten long review text for GUI display.
    """
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    short = text[:max_chars].rsplit(" ", 1)[0]
    return short + "..."