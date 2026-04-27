# query_understanding.py

from sentence_transformers import SentenceTransformer, util
from review_processing import ASPECT_DESCRIPTIONS   # now properly defined

# Lazy-load model
embedding_model = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading query understanding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return embedding_model


def detect_query_aspects(query, top_k=2):
    """
    Detect which aspects are most relevant to the user query.
    """
    model = get_embedding_model()

    aspect_names = list(ASPECT_DESCRIPTIONS.keys())
    aspect_texts = list(ASPECT_DESCRIPTIONS.values())

    query_embedding = model.encode(query, convert_to_tensor=True)
    aspect_embeddings = model.encode(aspect_texts, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, aspect_embeddings)[0]
    top_results = similarities.topk(k=min(top_k, len(aspect_names)))

    detected_aspects = []
    for idx in top_results.indices:
        detected_aspects.append(aspect_names[idx.item()])

    return detected_aspects
