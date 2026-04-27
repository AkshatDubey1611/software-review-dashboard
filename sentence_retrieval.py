# sentence_retrieval.py

import re
import torch
from sentence_transformers import SentenceTransformer, util

embedding_model = None
_SENT_END_RE = re.compile(r'(?<=[\.\?\!])\s+')


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading sentence retrieval model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


def split_into_sentences(text):
    return [s.strip() for s in _SENT_END_RE.split(text) if s.strip()]


def retrieve_relevant_sentences(query, reviews, top_k=5):
    """
    reviews: list of review dicts (with "text" key) or plain strings.
    Returns list of {sentence, score, source} dicts.
    """
    if not reviews or not query.strip():
        return []

    model = get_embedding_model()

    # Build sentence pool, keeping track of which review each sentence came from
    sentence_pool = []   # list of (sentence_str, review_dict_or_str)

    for r in reviews:
        text = r["text"] if isinstance(r, dict) else r
        for sent in split_into_sentences(text):
            if len(sent.split()) >= 4:
                sentence_pool.append((sent, r))

    if not sentence_pool:
        return []

    sentences_only = [s for s, _ in sentence_pool]
    print(f"Searching across {len(sentences_only)} sentences...")

    query_embedding = model.encode(query, convert_to_tensor=True)

    batch_size = 256
    all_embeddings = []
    for i in range(0, len(sentences_only), batch_size):
        batch = sentences_only[i:i + batch_size]
        all_embeddings.append(model.encode(batch, convert_to_tensor=True))

    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

    # Keyword boosting
    BOOST_KEYWORDS = {
        "price": ["price", "cost", "cheap", "expensive", "value", "worth", "fair"],
        "installation": ["install", "installation", "setup", "easy", "quick", "loaded"],
    }
    query_lower = query.lower()
    boost_scores = []
    for sentence in sentences_only:
        sentence_lower = sentence.lower()
        boost = 0.0
        for aspect, words in BOOST_KEYWORDS.items():
            for word in words:
                if word in query_lower and word in sentence_lower:
                    boost += 0.05
        boost_scores.append(boost)

    boost_tensor = torch.tensor(boost_scores, device=similarities.device)
    final_scores = similarities + boost_tensor

    top_results = final_scores.topk(k=min(top_k, len(sentence_pool)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        sent, review = sentence_pool[idx.item()]
        source = _build_source_line(review)
        results.append({
            "sentence": sent,
            "score":    float(score.item()),
            "source":   source
        })

    return results


def _build_source_line(r):
    if not isinstance(r, dict):
        return ""
    parts = []
    asin = r.get("asin", "")
    if asin and asin != "N/A":
        parts.append(f"ASIN: {asin} | https://www.amazon.com/dp/{asin}")
    reviewer = r.get("reviewer", "")
    if reviewer and reviewer != "Anonymous":
        parts.append(f"Reviewer: {reviewer}")
    date = r.get("date", "")
    if date:
        parts.append(f"Date: {date}")
    rating = r.get("rating")
    if rating is not None:
        parts.append(f"Rating: {'*' * int(rating)} ({rating}/5)")
    summary = r.get("summary", "")
    if summary:
        parts.append(f"Title: \"{summary}\"")
    return " | ".join(parts)
