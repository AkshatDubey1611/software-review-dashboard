# Aspect keywords used for clustering reviews by topic.

ASPECT_KEYWORDS = {
    "performance": ["fast", "slow", "speed", "lag", "crash", "crashes", "bug", "buggy", "freeze", "smooth", "responsive"],
    "installation": ["install", "installation", "setup", "download", "update", "configure"],
    "price": ["price", "cost", "expensive", "cheap", "value", "worth"],
    "features": ["feature", "option", "tool", "function", "capability"],
    "support": ["support", "help", "customer service", "assistance"],
    "compatibility": ["compatible", "windows", "mac", "linux", "version"],
    "interface": ["interface", "ui", "layout", "menu", "dashboard", "screen"],
    "design": ["design", "look", "appearance", "style"]
}

STOPWORDS = {
    "the","and","a","an","is","it","to","for","of","i","you","this","that",
    "in","on","with","was","are","be","have","has","but","not","they","we",
    "as","or","at","by","from","so","if","its","my","me","do","does","did",
    "were","their","them","can","will","would","should","about","what","which",
    "when","how","more","no","than","also"
}