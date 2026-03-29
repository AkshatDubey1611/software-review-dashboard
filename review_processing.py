import json
import re
import math
import heapq
from collections import Counter
import torch
from sentence_transformers import SentenceTransformer, util


from config import STOPWORDS

# -----------------------------
# SEMANTIC ASPECT DESCRIPTIONS
# -----------------------------
ASPECT_DESCRIPTIONS = {
    "performance": "software speed, lag, crashes, responsiveness",
    "installation": "installation process, setup, downloading, updates",
    "price": "cost, pricing, value for money, expensive or cheap",
    "features": "software features, functions, tools, capabilities",
    "support": "customer support, help, assistance, service",
    "compatibility": "compatibility with windows mac linux versions",
    "interface": "user interface, UI, layout, menus, dashboard usability",
    "design": "visual design, look, appearance, style"
}

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute aspect embeddings
aspect_names = list(ASPECT_DESCRIPTIONS.keys())
aspect_texts = list(ASPECT_DESCRIPTIONS.values())
aspect_embeddings = embedding_model.encode(aspect_texts, convert_to_tensor=True)

# -----------------------------
# LOAD REVIEWS
# -----------------------------
def load_reviews(path: str, max_reviews: int = 10000):
    reviews = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "reviewText" in data:
                reviews.append(data["reviewText"])

            if len(reviews) >= max_reviews:
                break

    return reviews

# -----------------------------
# SEMANTIC CLUSTERING (FIXED)
# -----------------------------
def cluster_reviews(reviews):
    clusters = {a: [] for a in ASPECT_DESCRIPTIONS}

    print("Encoding reviews in batches (this may take a little time)...")

    batch_size = 256
    all_embeddings = []

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(batch_embeddings)

    review_embeddings = torch.cat(all_embeddings, dim=0)

    similarity_matrix = util.cos_sim(review_embeddings, aspect_embeddings)

    for i, review in enumerate(reviews):
        best_idx = similarity_matrix[i].argmax().item()
        best_aspect = aspect_names[best_idx]
        clusters[best_aspect].append(review)

    return clusters

# -----------------------------
# TOKENIZATION
# -----------------------------
_SENT_END_RE = re.compile(r'(?<=[\.\?\!])\s+')

def sent_tokenize(text: str):
    sents = [s.strip() for s in _SENT_END_RE.split(text) if s.strip()]
    if not sents:
        sents = [s.strip() for s in text.splitlines() if s.strip()]
    return sents

def word_tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9']{2,}", text.lower())

# -----------------------------
# SUMMARIZATION
# -----------------------------
def score_sentences(text: str, top_k: int = 3):
    sents = sent_tokenize(text)
    if not sents:
        return ""

    words = []
    for s in sents:
        words.extend([w for w in word_tokenize(s) if w not in STOPWORDS])

    if not words:
        return "No substantive content."

    freq = Counter(words)
    maxf = max(freq.values())

    for w in list(freq.keys()):
        freq[w] = freq[w] / maxf

    scores = []
    for i, s in enumerate(sents):
        wlist = [w for w in word_tokenize(s) if w not in STOPWORDS]

        if not wlist:
            score = 0.0
        else:
            score = sum(freq.get(w, 0.0) for w in wlist) / math.sqrt(len(wlist))

        scores.append((score, i, s))

    top = heapq.nlargest(top_k, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])

    summary = " ".join([t[2].rstrip('.') + '.' for t in top_sorted if t[0] > 0])

    if not summary:
        summary = " ".join(sents[:min(top_k, len(sents))])

    return summary

# -----------------------------
# FINAL SUMMARIES
# -----------------------------
def generate_aspect_summaries(clusters):
    aspect_summaries = {}

    for aspect, texts in clusters.items():
        if texts:
            combined = " ".join(texts[:10])
            combined = combined.replace("\n", " ")[:4000]
            summary = score_sentences(combined, top_k=3)
            aspect_summaries[aspect] = summary
        else:
            aspect_summaries[aspect] = "No major discussion."

    return aspect_summaries