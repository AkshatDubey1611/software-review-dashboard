# dynamic_aspects.py
#
# BLAIR-powered dynamic aspect discovery and clustering.
#
# FIX applied in this version:
#   Problem: KMeans produced 8 clusters but many got the same label
#            (e.g. "Magazine", "Magazine (1)", "Magazine (2)") because
#            keyword inference kept returning the same winner.
#
#   Solution:
#     1. After KMeans, compute cosine similarity between cluster centroids
#     2. Merge clusters whose centroids are too similar (threshold 0.85)
#     3. Use TF-IDF style scoring across ALL clusters to pick labels that
#        are DISTINCTIVE to each cluster, not just frequent in it
#     4. If two clusters still get the same label after all this,
#        merge them rather than numbering them

from __future__ import annotations
from collections import Counter
import re
import numpy as np
import torch
from sklearn.cluster import KMeans
from blair_encoder import get_encoder

# ── Stopwords ────────────────────────────────────────────────────
STOPWORDS_EXTRA = {
    "the","a","an","this","that","these","those","its","their","our","your","my","his","her",
    "and","but","or","nor","for","yet","so","as","at","by","in","of","on","to","up","via",
    "from","into","with","about","above","after","along","among","around","before","behind",
    "below","between","beyond","during","except","inside","near","off","out","outside","over",
    "since","through","throughout","under","until","upon","within","without","than","then",
    "i","you","he","she","it","we","they","me","him","us","them","who","what","which","whose",
    "is","are","was","were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","must","shall","can","get","got","go","goes","went",
    "use","used","uses","make","made","come","came","see","saw","take","took","give","gave",
    "know","knew","think","thought","want","need","seem","feel","try","tried","keep","kept",
    "let","put","set","run","work","worked","works","find","found","ask","tell","told",
    "buy","bought","like","liked","love","hate","say","said","show","look","start","end",
    "good","bad","great","best","worst","new","old","big","small","long","short","high","low",
    "easy","hard","free","full","open","close","real","sure","able","right","wrong","true",
    "very","really","just","also","even","only","never","always","often","already","still",
    "again","well","much","many","more","most","less","least","too","quite","rather","enough",
    "first","last","next","same","other","another","both","all","any","few","some","no",
    "one","two","three","time","way","day","year","thing","things","part","place","case",
    "point","fact","lot","bit","set","kind","type","sort","side","hand","end","number","amount",
    "problem","issue","reason","result","example","information","question","answer","idea","help",
    "software","product","app","application","program","purchase","bought","item","order",
    "review","reviews","star","stars","rating","price","money","cost","value","paid",
    "customer","service","company","brand","version","update","upgrade","release",
    "however","therefore","although","because","while","when","where","how","why","whether",
    "not","now","here","there","back","just","very","well","also","else","ever","then",
    "yes","maybe","perhaps","actually","basically","generally","usually","simply","especially",
    "nice","using","don","doesn","didn","isn","wasn","couldn","wouldn","shouldn",
    "pretty","bit","lot","got","also","even","just","really","quite",
    # Domain-generic words that cause false label matches
    "magazine","subscription","amazon","issue","issues","read","reading","subscriber",
    "article","articles","content","publish","published","publisher",
}

# ── Canonical aspect taxonomy ────────────────────────────────────
KEYWORD_TO_ASPECT = {
    "crash":"Performance","crashes":"Performance","freeze":"Performance",
    "slow":"Performance","fast":"Performance","speed":"Performance",
    "lag":"Performance","bug":"Performance","bugs":"Performance",
    "stable":"Performance","smooth":"Performance","responsive":"Performance",
    "glitch":"Performance","stutter":"Performance",
    "install":"Installation","installation":"Installation","setup":"Installation",
    "download":"Installation","configure":"Installation","uninstall":"Installation",
    "expensive":"Price & Value","cheap":"Price & Value","affordable":"Price & Value",
    "worth":"Price & Value","overpriced":"Price & Value","subscription":"Price & Value",
    "feature":"Features","features":"Features","function":"Features",
    "tool":"Features","option":"Features","capability":"Features",
    "functionality":"Features","missing":"Features",
    "support":"Customer Support","refund":"Customer Support",
    "response":"Customer Support","assistance":"Customer Support",
    "helpdesk":"Customer Support","staff":"Customer Support",
    "compatible":"Compatibility","compatibility":"Compatibility",
    "windows":"Compatibility","mac":"Compatibility","linux":"Compatibility",
    "driver":"Compatibility","hardware":"Compatibility","platform":"Compatibility",
    "interface":"Interface & UX","menu":"Interface & UX",
    "navigation":"Interface & UX","dashboard":"Interface & UX",
    "layout":"Interface & UX","screen":"Interface & UX",
    "button":"Interface & UX","toolbar":"Interface & UX",
    "design":"Design","appearance":"Design","look":"Design",
    "style":"Design","theme":"Design","icon":"Design",
    "hair":"Hair Care","shampoo":"Hair Care","conditioner":"Hair Care",
    "scalp":"Hair Care","frizz":"Hair Care","curl":"Hair Care",
    "skin":"Skin Care","face":"Skin Care","cream":"Skin Care",
    "moisturizer":"Skin Care","acne":"Skin Care","serum":"Skin Care",
    "scent":"Fragrance","smell":"Fragrance","fragrance":"Fragrance",
    "aroma":"Fragrance","perfume":"Fragrance","odor":"Fragrance",
    "nail":"Nail Care","nails":"Nail Care","polish":"Nail Care",
    "gel":"Nail Care","manicure":"Nail Care","lacquer":"Nail Care",
    "makeup":"Makeup","mascara":"Makeup","lipstick":"Makeup",
    "foundation":"Makeup","blush":"Makeup","eyeshadow":"Makeup",
    "color":"Makeup","colour":"Makeup","concealer":"Makeup",
    "brush":"Grooming Tools","razor":"Grooming Tools","trimmer":"Grooming Tools",
    "clipper":"Grooming Tools","bristles":"Grooming Tools","blade":"Grooming Tools",
    "wig":"Hair Styling","clip":"Hair Styling","volume":"Hair Styling",
    "thick":"Hair Styling","straight":"Hair Styling","wavy":"Hair Styling",
    "battery":"Battery & Power","charge":"Battery & Power","charging":"Battery & Power",
    "power":"Battery & Power","cable":"Battery & Power","voltage":"Battery & Power",
    "build":"Build Quality","durable":"Build Quality","sturdy":"Build Quality",
    "material":"Build Quality","plastic":"Build Quality","metal":"Build Quality",
    "solid":"Build Quality","broke":"Build Quality","flimsy":"Build Quality",
    "package":"Packaging","packaging":"Packaging","shipping":"Packaging",
    "arrived":"Packaging","delivery":"Packaging","damaged":"Packaging",
    "taste":"Taste & Flavor","flavor":"Taste & Flavor","flavour":"Taste & Flavor",
    "delicious":"Taste & Flavor","bland":"Taste & Flavor","sweet":"Taste & Flavor",
    "fresh":"Taste & Flavor","yummy":"Taste & Flavor",
    "comfort":"Comfort & Fit","comfortable":"Comfort & Fit","fit":"Comfort & Fit",
    "size":"Comfort & Fit","tight":"Comfort & Fit","soft":"Comfort & Fit",
    "game":"Gaming","gameplay":"Gaming","level":"Gaming",
    "player":"Gaming","puzzle":"Gaming","fps":"Gaming","multiplayer":"Gaming",
    "router":"Networking","network":"Networking","wifi":"Networking",
    "wireless":"Networking","ethernet":"Networking","connection":"Networking",
    "signal":"Networking","modem":"Networking",
    "virus":"Security","malware":"Security","antivirus":"Security",
    "firewall":"Security","security":"Security","protection":"Security",
    "backup":"Storage & Backup","storage":"Storage & Backup","drive":"Storage & Backup",
    "disk":"Storage & Backup","cloud":"Storage & Backup","sync":"Storage & Backup",
    # Content/media aspects
    "recipes":"Recipes & Food","cooking":"Recipes & Food","food":"Recipes & Food",
    "kosher":"Recipes & Food","baking":"Recipes & Food","chef":"Recipes & Food",
    "photos":"Photography","photography":"Photography","camera":"Photography",
    "wildlife":"Nature & Wildlife","nature":"Nature & Wildlife","animals":"Nature & Wildlife",
    "garden":"Gardening","gardening":"Gardening","plants":"Gardening","flowers":"Gardening",
    "fitness":"Health & Fitness","exercise":"Health & Fitness","workout":"Health & Fitness",
    "health":"Health & Fitness","nutrition":"Health & Fitness","diet":"Health & Fitness",
    "kids":"Children & Family","children":"Children & Family","family":"Children & Family",
    "educational":"Children & Family","learning":"Children & Family",
    "politics":"News & Politics","news":"News & Politics","current":"News & Politics",
    "science":"Science & Technology","technology":"Science & Technology","tech":"Science & Technology",
    "fashion":"Fashion & Style","style":"Fashion & Style","clothing":"Fashion & Style",
    "home":"Home & Living","decor":"Home & Living","interior":"Home & Living",
    "travel":"Travel","destination":"Travel","trip":"Travel","vacation":"Travel",
    "business":"Business & Finance","finance":"Business & Finance","investing":"Business & Finance",
    "craft":"Crafts & Hobbies","hobby":"Crafts & Hobbies","knitting":"Crafts & Hobbies",
    "music":"Music & Entertainment","entertainment":"Music & Entertainment","film":"Music & Entertainment",
    "delivery":"Delivery & Shipping","shipping":"Delivery & Shipping","arrived":"Delivery & Shipping",
    "gift":"Gift & Occasion","gifting":"Gift & Occasion","present":"Gift & Occasion",
    "digital":"Digital Access","online":"Digital Access","app":"Digital Access","access":"Digital Access",
}

# ── Sentiment hints per label ──────────────────────────────────
SENTIMENT_HINTS: dict[str, dict[str, list[str]]] = {
    "Performance":      {"positive":["fast","smooth","responsive","stable","quick"],
                         "negative":["slow","lag","crash","freeze","bug","unstable","glitch"]},
    "Installation":     {"positive":["easy","simple","quick","straightforward"],
                         "negative":["difficult","failed","error","problem","issue"]},
    "Price & Value":    {"positive":["cheap","worth","affordable","fair","value"],
                         "negative":["expensive","overpriced","costly","waste"]},
    "Features":         {"positive":["useful","powerful","helpful","rich","capable"],
                         "negative":["missing","limited","lacking","confusing"]},
    "Customer Support": {"positive":["helpful","responsive","fast","excellent"],
                         "negative":["poor","useless","slow","unhelpful","rude"]},
    "Compatibility":    {"positive":["compatible","works","supported","seamless"],
                         "negative":["incompatible","unsupported","issue","problem"]},
    "Interface & UX":   {"positive":["clean","intuitive","easy","friendly","simple"],
                         "negative":["confusing","cluttered","ugly","hard","complex"]},
    "Design":           {"positive":["nice","beautiful","modern","clean","sleek"],
                         "negative":["bad","outdated","poor","ugly","plain"]},
    "Hair Care":        {"positive":["soft","smooth","shiny","nourish","hydrate"],
                         "negative":["dry","frizzy","brittle","damage","breakage"]},
    "Skin Care":        {"positive":["soft","clear","glow","hydrate","smooth"],
                         "negative":["breakout","irritate","dry","oily","rash"]},
    "Fragrance":        {"positive":["pleasant","fresh","lasting","strong","nice"],
                         "negative":["overpowering","chemical","fake","faded","weak"]},
    "Nail Care":        {"positive":["smooth","lasting","shine","easy","quick"],
                         "negative":["chip","peel","smear","thick","streak"]},
    "Makeup":           {"positive":["pigment","smooth","lasting","blend","cover"],
                         "negative":["cakey","streak","smear","fade","patchy"]},
    "Grooming Tools":   {"positive":["sharp","smooth","easy","precise","clean"],
                         "negative":["dull","pull","irritate","broke","cheap"]},
    "Hair Styling":     {"positive":["volume","smooth","hold","shine","soft"],
                         "negative":["stiff","greasy","sticky","heavy","damage"]},
    "Battery & Power":  {"positive":["long","fast","efficient","reliable","charge"],
                         "negative":["drain","short","slow","dead","fail"]},
    "Build Quality":    {"positive":["sturdy","durable","solid","premium","strong"],
                         "negative":["cheap","flimsy","broke","crack","loose"]},
    "Packaging":        {"positive":["secure","neat","protect","intact","well"],
                         "negative":["damaged","broken","open","loose","poor"]},
    "Taste & Flavor":   {"positive":["delicious","tasty","fresh","sweet","rich"],
                         "negative":["bland","bitter","artificial","awful","stale"]},
    "Comfort & Fit":    {"positive":["comfortable","soft","perfect","fit","cozy"],
                         "negative":["tight","loose","stiff","rough","uncomfortable"]},
    "Gaming":           {"positive":["fun","smooth","immersive","great","addictive"],
                         "negative":["lag","crash","bug","glitch","boring"]},
    "Networking":       {"positive":["fast","stable","strong","reliable","range"],
                         "negative":["drop","slow","weak","disconnect","issue"]},
    "Security":         {"positive":["protect","safe","reliable","detect","block"],
                         "negative":["miss","slow","false","intrusive","fail"]},
    "Storage & Backup": {"positive":["fast","easy","reliable","large","seamless"],
                         "negative":["slow","fail","corrupt","small","error"]},
    "Recipes & Food":   {"positive":["delicious","creative","clear","helpful","varied"],
                         "negative":["bland","complicated","unclear","repetitive","few"]},
    "Health & Fitness": {"positive":["motivating","clear","helpful","effective","inspiring"],
                         "negative":["repetitive","basic","outdated","ineffective","boring"]},
    "Nature & Wildlife":{"positive":["beautiful","informative","stunning","inspiring","detailed"],
                         "negative":["repetitive","basic","dry","boring","sparse"]},
    "Children & Family":{"positive":["fun","educational","engaging","colorful","interactive"],
                         "negative":["boring","inappropriate","confusing","dull","short"]},
    "Travel":           {"positive":["inspiring","detailed","useful","accurate","beautiful"],
                         "negative":["outdated","vague","inaccurate","boring","sparse"]},
    "Fashion & Style":  {"positive":["trendy","inspiring","stylish","beautiful","diverse"],
                         "negative":["repetitive","outdated","bland","impractical","thin"]},
    "Home & Living":    {"positive":["inspiring","practical","beautiful","creative","detailed"],
                         "negative":["repetitive","basic","outdated","impractical","vague"]},
    "Digital Access":   {"positive":["easy","fast","convenient","reliable","smooth"],
                         "negative":["difficult","slow","broken","inaccessible","limited"]},
    "Gift & Occasion":  {"positive":["perfect","thoughtful","loved","great","wonderful"],
                         "negative":["disappointing","late","wrong","cheap","broken"]},
    "Delivery & Shipping":{"positive":["fast","on-time","secure","intact","reliable"],
                           "negative":["late","damaged","missing","slow","wrong"]},
    "News & Politics":  {"positive":["balanced","insightful","thorough","accurate","timely"],
                         "negative":["biased","repetitive","shallow","inaccurate","boring"]},
    "Business & Finance":{"positive":["insightful","practical","clear","useful","timely"],
                          "negative":["basic","repetitive","vague","outdated","shallow"]},
    "Crafts & Hobbies": {"positive":["inspiring","detailed","creative","clear","varied"],
                         "negative":["basic","repetitive","unclear","sparse","boring"]},
    "Music & Entertainment":{"positive":["engaging","entertaining","diverse","quality","exciting"],
                             "negative":["repetitive","boring","poor","limited","shallow"]},
    "Photography":      {"positive":["stunning","detailed","inspiring","clear","beautiful"],
                         "negative":["blurry","repetitive","sparse","poor","boring"]},
    "Science & Technology":{"positive":["informative","accurate","clear","detailed","useful"],
                            "negative":["outdated","shallow","inaccurate","boring","sparse"]},
    "Gardening":        {"positive":["helpful","detailed","inspiring","practical","clear"],
                         "negative":["vague","basic","repetitive","outdated","sparse"]},
}

_DEFAULT_HINTS = {
    "positive":["good","great","excellent","easy","fast","helpful","nice","love","best","smooth"],
    "negative":["bad","poor","slow","crash","issue","problem","error","difficult","worst","hate"],
}


def get_sentiment_hints(label: str) -> dict[str, list[str]]:
    clean = re.sub(r"\s*\(\d+\)$", "", label).strip()
    if clean in SENTIMENT_HINTS:
        return SENTIMENT_HINTS[clean]
    for key, hints in SENTIMENT_HINTS.items():
        if key.lower() in clean.lower() or clean.lower() in key.lower():
            return hints
    return _DEFAULT_HINTS


# ── Text helpers ─────────────────────────────────────────────────

def clean_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z]{3,}", text.lower())
            if w not in STOPWORDS_EXTRA]


def extract_keywords(texts: list[str], top_n: int = 8) -> list[str]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(clean_words(t))
    return [w for w, _ in counter.most_common(top_n)]


def extract_tfidf_keywords(
    cluster_texts: list[str],
    all_clusters_texts: list[list[str]],
    top_n: int = 8,
) -> list[str]:
    """
    Extract keywords that are DISTINCTIVE to this cluster vs all others.
    Uses TF-IDF logic: high frequency in THIS cluster, low in ALL others.
    This prevents generic words like "magazine" from dominating every label.
    """
    # Term frequency in this cluster
    tf: Counter = Counter()
    for t in cluster_texts:
        tf.update(clean_words(t))

    if not tf:
        return []

    # Document frequency across ALL clusters (how many clusters contain each word)
    all_words = set(tf.keys())
    df: Counter = Counter()
    for other_texts in all_clusters_texts:
        other_words = set()
        for t in other_texts:
            other_words.update(clean_words(t))
        for w in all_words:
            if w in other_words:
                df[w] += 1

    n_clusters = len(all_clusters_texts)

    # TF-IDF score: high tf, low df = distinctive keyword
    scores: dict[str, float] = {}
    for w, freq in tf.items():
        # Normalize tf by cluster size
        tf_norm = freq / max(len(cluster_texts), 1)
        # IDF: penalize words that appear in many clusters
        idf = n_clusters / max(df[w], 1)
        scores[w] = tf_norm * idf

    top_words = sorted(scores, key=lambda w: -scores[w])[:top_n]
    return top_words


def infer_aspect_label(keywords: list[str]) -> str:
    votes: Counter = Counter()
    for kw in keywords:
        label = KEYWORD_TO_ASPECT.get(kw.lower())
        if label:
            votes[label] += 1
    if votes:
        return votes.most_common(1)[0][0]
    # Fallback: capitalize the most distinctive keyword
    return keywords[0].capitalize() if keywords else "General"


# ── Cluster merging ───────────────────────────────────────────────

def _merge_similar_clusters(
    raw_clusters: dict[int, list[str]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    similarity_threshold: float = 0.92,
) -> dict[int, list[str]]:
    """
    Merge clusters whose centroids are too similar.
    This prevents near-duplicate clusters getting numbered suffixes.

    similarity_threshold: cosine similarity above which two clusters are merged.
    0.92 is aggressive merging — lower to 0.85 if you want more clusters.
    """
    cluster_ids = list(raw_clusters.keys())
    if len(cluster_ids) <= 1:
        return raw_clusters

    # Compute centroids
    centroids: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        idxs = np.where(labels == cid)[0]
        if len(idxs) > 0:
            centroid = embeddings[idxs].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[cid] = centroid / norm if norm > 0 else centroid

    # Union-Find for merging
    parent = {cid: cid for cid in cluster_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # Check all pairs
    for i, cid1 in enumerate(cluster_ids):
        for cid2 in cluster_ids[i+1:]:
            if cid1 not in centroids or cid2 not in centroids:
                continue
            sim = float(np.dot(centroids[cid1], centroids[cid2]))
            if sim >= similarity_threshold:
                union(cid1, cid2)
                print(f"  [Merge] Clusters {cid1} and {cid2} merged (similarity={sim:.3f})")

    # Build merged clusters
    merged: dict[int, list[str]] = {}
    for cid in cluster_ids:
        root = find(cid)
        if root not in merged:
            merged[root] = []
        merged[root].extend(raw_clusters[cid])

    if len(merged) < len(raw_clusters):
        print(f"  [Merge] Reduced from {len(raw_clusters)} to {len(merged)} clusters.")

    return merged


# ── BLAIR-powered clustering ─────────────────────────────────────

def cluster_reviews_dynamic(reviews: list[str], num_clusters: int = 8) -> dict[str, list[str]]:
    encoder = get_encoder()

    print("Encoding reviews with BLAIR ...")
    emb_tensor = encoder.encode(reviews, batch_size=128,
                                convert_to_tensor=True, show_progress_bar=True)
    embeddings = emb_tensor.numpy()

    # Clamp num_clusters to avoid more clusters than reviews
    n_clusters = min(num_clusters, len(reviews) // 5, 8)
    n_clusters = max(n_clusters, 2)

    print(f"Clustering into {n_clusters} aspects ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Build raw clusters
    raw_clusters: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        raw_clusters.setdefault(int(label), []).append(reviews[idx])

    # ── STEP 1: Merge similar clusters ────────────────────────────
    merged_clusters = _merge_similar_clusters(raw_clusters, embeddings, labels,
                                              similarity_threshold=0.92)

    # ── STEP 2: Label each cluster using TF-IDF distinctive keywords ──
    all_texts_per_cluster = list(merged_clusters.values())
    used_labels: dict[str, int] = {}   # label -> cluster_id that owns it
    aspect_clusters: dict[str, list[str]] = {}

    for cluster_id, texts in sorted(merged_clusters.items(),
                                    key=lambda x: -len(x[1])):  # largest first

        # Get distinctive keywords for THIS cluster vs all others
        distinctive_kws = extract_tfidf_keywords(texts, all_texts_per_cluster, top_n=10)

        # Fall back to plain frequency if TF-IDF gives nothing
        if not distinctive_kws:
            distinctive_kws = extract_keywords(texts, top_n=8)

        name = infer_aspect_label(distinctive_kws)

        # ── STEP 3: Handle label collision ────────────────────────
        if name in used_labels:
            # Same label already used — merge into existing cluster
            # instead of creating "Magazine (1)", "Magazine (2)" etc.
            print(f"  [Label] '{name}' already used — merging cluster {cluster_id} into it.")
            existing_name = name
            aspect_clusters[existing_name].extend(texts)
        else:
            used_labels[name] = cluster_id
            aspect_clusters[name] = texts

    print(f"  Final aspects: {list(aspect_clusters.keys())}")
    return aspect_clusters
