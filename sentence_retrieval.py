# sentence_retrieval.py

import re
import torch
from sentence_transformers import SentenceTransformer, util

# Lazy-load model
embedding_model = None

_SENT_END_RE = re.compile(r'(?<=[\.\?\!])\s+')


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading sentence retrieval model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return embedding_model


def split_into_sentences(text):
    sentences = [s.strip() for s in _SENT_END_RE.split(text) if s.strip()]
    return sentences


def retrieve_relevant_sentences(query, reviews, top_k=5):
    """
    Retrieve top-k most relevant sentences from reviews for a user query.
    """
    if not reviews or not query.strip():
        return []

    model = get_embedding_model()

    # Step 1: build sentence pool
    sentence_pool = []

    for review in reviews:
        sentences = split_into_sentences(review)
        for sent in sentences:
            if len(sent.split()) >= 4:  # ignore tiny useless fragments
                sentence_pool.append(sent)

    if not sentence_pool:
        return []

    print(f"Searching across {len(sentence_pool)} sentences...")

    # Step 2: encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Step 3: encode sentences in batches
    batch_size = 256
    all_embeddings = []

    for i in range(0, len(sentence_pool), batch_size):
        batch = sentence_pool[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(batch_embeddings)

    sentence_embeddings = torch.cat(all_embeddings, dim=0)

    # Step 4: semantic similarity
    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

    # Step 5: keyword boosting
    BOOST_KEYWORDS = {
        "price": ["price", "cost", "cheap", "expensive", "value", "worth", "fair"],
        "installation": ["install", "installation", "setup", "easy", "quick", "loaded"],
    }

    query_lower = query.lower()
    boost_scores = []

    for sentence in sentence_pool:
        sentence_lower = sentence.lower()
        boost = 0.0

        for aspect, words in BOOST_KEYWORDS.items():
            for word in words:
                if word in query_lower and word in sentence_lower:
                    boost += 0.05  # small boost

        boost_scores.append(boost)

    boost_tensor = torch.tensor(boost_scores, device=similarities.device)

    # Step 6: combine semantic + boost
    final_scores = similarities + boost_tensor

    # Step 7: top-k
    top_results = final_scores.topk(k=min(top_k, len(sentence_pool)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "sentence": sentence_pool[idx.item()],
            "score": float(score.item())
        })

    return results