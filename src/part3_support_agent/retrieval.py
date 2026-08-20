"""
Part 3 Task 2 -- Embed and index.
Embed every chunk with a free, local sentence-transformer model and build a
vector index over them with FAISS. Both are free, local, no account/API key.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.part3_support_agent.knowledge_base import build_all_chunks

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model_singleton = None
_index_singleton = None
_chunks_singleton = None


def get_embedder():
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_singleton


def build_index():
    """Embed all chunks and build a FAISS index. Returns (index, chunks, embeddings)."""
    global _index_singleton, _chunks_singleton
    chunks = build_all_chunks()
    model = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    _index_singleton = index
    _chunks_singleton = chunks
    return index, chunks, embeddings


def get_index():
    global _index_singleton, _chunks_singleton
    if _index_singleton is None:
        build_index()
    return _index_singleton, _chunks_singleton


def retrieve(query: str, k: int = 3):
    """Retrieve top-k chunks for a query, with similarity scores."""
    index, chunks = get_index()
    model = get_embedder()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "similarity": float(score)})
    return results


if __name__ == "__main__":
    index, chunks, embeddings = build_index()
    print(f"Indexed {len(chunks)} chunks, embedding dim={embeddings.shape[1]}")

    test_query = "How long do I have to return a t-shirt I bought?"
    results = retrieve(test_query, k=3)
    print(f"\nQuery: {test_query}")
    for r in results:
        print(f"  [{r['similarity']:.4f}] ({r['doc_id']}) {r['text']}")
