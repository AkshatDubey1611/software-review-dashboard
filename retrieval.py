# retrieval.py

import torch
from sentence_transformers import SentenceTransformer, util
from query_understanding import detect_query_aspects
from review_processing import cluster_reviews   # now properly defined

# Lazy-load model
embedding_model = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading retrieval model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return embedding_model


def retrieve_relevant_reviews(query, reviews, top_k=5):
    """
    Retrieve top-k most relevant reviews for a user query,
    after filtering by detected aspects.
    """
    if not reviews or not query.strip():
        return []

    model = get_embedding_model()

    # Step 1: detect query aspects
    detected_aspects = detect_query_aspects(query, top_k=2)
    print("Detected query aspects:", detected_aspects)

    # Step 2: cluster reviews by aspect
    clusters = cluster_reviews(reviews)

    # Step 3: collect only reviews from detected aspects
    filtered_reviews = []
    for aspect in detected_aspects:
        filtered_reviews.extend(clusters.get(aspect, []))

    # fallback if filtering becomes too narrow
    if not filtered_reviews:
        filtered_reviews = reviews

    print(f"Searching within {len(filtered_reviews)} filtered reviews...")

    # Step 4: encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Step 5: encode filtered reviews
    batch_size = 256
    all_embeddings = []

    for i in range(0, len(filtered_reviews), batch_size):
        batch = filtered_reviews[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(batch_embeddings)

    review_embeddings = torch.cat(all_embeddings, dim=0)

    # Step 6: similarity
    similarities = util.cos_sim(query_embedding, review_embeddings)[0]

    # Step 7: top-k
    top_results = similarities.topk(k=min(top_k, len(filtered_reviews)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "review": filtered_reviews[idx.item()],
            "score": float(score.item())
        })

    return results
