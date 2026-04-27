from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import re

# Load model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Comprehensive stopword list for clean aspect label generation
STOPWORDS_EXTRA = {
    # Articles / determiners
    "the","a","an","this","that","these","those","its","their","our","your","my","his","her",
    # Conjunctions / prepositions
    "and","but","or","nor","for","yet","so","as","at","by","in","of","on","to","up","via",
    "from","into","with","about","above","after","along","among","around","before","behind",
    "below","between","beyond","during","except","inside","near","off","out","outside","over",
    "since","through","throughout","under","until","upon","within","without","than","then",
    # Pronouns
    "i","you","he","she","it","we","they","me","him","us","them","who","what","which","whose",
    # Common verbs
    "is","are","was","were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","must","shall","can","get","got","go","goes","went",
    "use","used","uses","make","made","come","came","see","saw","take","took","give","gave",
    "know","knew","think","thought","want","need","seem","feel","try","tried","keep","kept",
    "let","put","set","run","run","work","worked","works","find","found","ask","tell","told",
    "buy","bought","like","liked","love","hate","say","said","show","look","start","end",
    # Common adjectives / adverbs
    "good","bad","great","best","worst","new","old","big","small","long","short","high","low",
    "easy","hard","free","full","open","close","real","sure","able","right","wrong","true",
    "very","really","just","also","even","only","never","always","often","already","still",
    "again","well","much","many","more","most","less","least","too","quite","rather","enough",
    "first","last","next","same","other","another","both","all","any","few","some","no",
    # Generic nouns (too vague for aspect labels)
    "one","two","three","time","way","day","year","thing","things","part","place","case",
    "point","fact","lot","bit","set","kind","type","sort","side","hand","end","number","amount",
    "problem","issue","reason","result","example","information","question","answer","idea","help",
    # Product/review generic words
    "software","product","app","application","program","purchase","bought","item","order",
    "review","reviews","star","stars","rating","price","money","cost","value","paid","buy",
    "customer","service","company","brand","version","update","upgrade","release",
    # Common filler
    "however","therefore","although","because","while","when","where","how","why","whether",
    "not","now","here","there","back","just","very","well","also","else","ever","here","then",
    "yes","maybe","perhaps","actually","basically","generally","usually","simply","especially",
}

def clean_words(text):
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS_EXTRA]

def extract_keywords(texts, top_n=4):
    all_words = []
    for t in texts:
        all_words.extend(clean_words(t))
    freq = Counter(all_words)
    return [w for w, _ in freq.most_common(top_n)]

# Map top keywords to a clean human-readable aspect label
KEYWORD_TO_ASPECT = {
    # Performance
    "crash": "performance", "crashes": "performance", "freeze": "performance",
    "slow": "performance", "fast": "performance", "speed": "performance",
    "lag": "performance", "bug": "performance", "bugs": "performance",
    "stable": "performance", "smooth": "performance", "responsive": "performance",
    # Installation
    "install": "installation", "installation": "installation", "setup": "installation",
    "download": "installation", "configure": "installation", "uninstall": "installation",
    # Price / value
    "expensive": "price", "cheap": "price", "affordable": "price",
    "worth": "price", "overpriced": "price", "subscription": "price",
    # Features
    "feature": "features", "features": "features", "function": "features",
    "tool": "features", "option": "features", "capability": "features",
    "functionality": "features", "missing": "features",
    # Support
    "support": "support", "customer": "support", "refund": "support",
    "response": "support", "assistance": "support", "helpdesk": "support",
    # Compatibility
    "compatible": "compatibility", "compatibility": "compatibility",
    "windows": "compatibility", "mac": "compatibility", "linux": "compatibility",
    "driver": "compatibility", "hardware": "compatibility",
    # Interface / UX
    "interface": "interface", "menu": "interface", "navigation": "interface",
    "dashboard": "interface", "layout": "interface", "screen": "interface",
    "button": "interface", "toolbar": "interface",
    # Design
    "design": "design", "appearance": "design", "look": "design",
    "style": "design", "theme": "design", "icon": "design",
    # Games
    "game": "gaming", "games": "gaming", "gameplay": "gaming",
    "level": "gaming", "player": "gaming", "puzzle": "gaming",
    # Networking
    "router": "networking", "network": "networking", "wifi": "networking",
    "wireless": "networking", "switch": "networking", "port": "networking",
    "modem": "networking", "ethernet": "networking", "connection": "networking",
    # Security
    "virus": "security", "malware": "security", "antivirus": "security",
    "firewall": "security", "security": "security", "protection": "security",
    # Storage / backup
    "backup": "storage", "storage": "storage", "drive": "storage",
    "disk": "storage", "cloud": "storage", "sync": "storage",
}

def infer_aspect_label(keywords):
    """
    Map extracted keywords to a clean aspect label using KEYWORD_TO_ASPECT.
    Falls back to joining the keywords if no mapping is found.
    """
    votes = Counter()
    for kw in keywords:
        aspect = KEYWORD_TO_ASPECT.get(kw.lower())
        if aspect:
            votes[aspect] += 1

    if votes:
        return votes.most_common(1)[0][0]

    # Fallback: use the top keywords directly (they're already stopword-filtered)
    return " / ".join(keywords)

def cluster_reviews_dynamic(reviews, num_clusters=8):
    print("Encoding reviews for clustering...")
    embeddings = embedding_model.encode(reviews)

    print(f"Clustering into {num_clusters} aspects...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(reviews[idx])

    # Build aspect name → reviews dict with deduplicated labels
    aspect_clusters = {}
    used_labels = Counter()

    for label, texts in clusters.items():
        keywords = extract_keywords(texts, top_n=6)
        name = infer_aspect_label(keywords)

        # Deduplicate if two clusters map to the same label
        if name in aspect_clusters:
            used_labels[name] += 1
            name = f"{name} ({used_labels[name]})"

        aspect_clusters[name] = texts

    return aspect_clusters