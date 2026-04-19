"""
RAG Module: Vector Store Builder & Retriever
Uses FAISS with sentence-transformers for local, free-tier embedding.
"""

import os
import numpy as np

# We use a simple TF-IDF + cosine similarity approach as fallback
# if sentence-transformers is not available, ensuring Streamlit Cloud compatibility.
from retention_knowledge import RETENTION_DOCS


def _tfidf_embed(texts):
    """Simple TF-IDF bag-of-words vectorizer (no external deps)."""
    from collections import Counter
    import math

    # Build vocabulary
    vocab = {}
    tokenized = []
    for text in texts:
        tokens = text.lower().split()
        tokenized.append(tokens)
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    V = len(vocab)
    N = len(texts)

    # Compute TF-IDF
    df_counts = Counter()
    for tokens in tokenized:
        for t in set(tokens):
            df_counts[t] += 1

    matrix = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        for t, count in tf.items():
            if t in vocab:
                tf_val = count / len(tokens)
                idf_val = math.log((N + 1) / (df_counts[t] + 1)) + 1
                matrix[i, vocab[t]] = tf_val * idf_val

    # L2 normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix = matrix / norms
    return matrix, vocab


def _cosine_sim(query_vec, matrix):
    return matrix @ query_vec


class RetentionRAG:
    """
    Retrieval-Augmented Generation module for retention strategies.
    Indexes the knowledge base and retrieves top-k relevant documents
    given a customer profile query.
    """

    def __init__(self):
        self._built = False
        self.matrix = None
        self.vocab = None

    def build_index(self):
        """Build the TF-IDF index from the knowledge base."""
        texts = [doc["title"] + " " + doc["content"] for doc in RETENTION_DOCS]
        self.matrix, self.vocab = _tfidf_embed(texts)
        self._built = True

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve top_k most relevant retention documents for a given query.
        Returns list of dicts with title, content, and similarity score.
        """
        if not self._built:
            self.build_index()

        # Embed query
        import math
        from collections import Counter

        tokens = query.lower().split()
        tf = Counter(tokens)
        V = len(self.vocab)
        query_vec = np.zeros(V, dtype=np.float32)
        for t, count in tf.items():
            if t in self.vocab:
                query_vec[self.vocab[t]] = count / len(tokens)
        norm = np.linalg.norm(query_vec) + 1e-10
        query_vec = query_vec / norm

        scores = _cosine_sim(query_vec, self.matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = RETENTION_DOCS[idx]
            results.append({
                "title": doc["title"],
                "content": doc["content"],
                "tags": doc["tags"],
                "score": float(scores[idx]),
            })
        return results


# Singleton instance
_rag_instance = None


def get_rag() -> RetentionRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RetentionRAG()
        _rag_instance.build_index()
    return _rag_instance
