import numpy as np


class VectorIndex:
    """
    B-21 fix: was using raw dot product which gives meaningless similarity
    scores for unnormalized embeddings. Now uses cosine similarity (dot product
    of L2-normalized vectors), so scores are always in [-1, 1] regardless of
    embedding magnitude.
    """

    def __init__(self, dim: int = 384):
        self.dim      = dim
        self.vectors  = []   # normalized unit vectors
        self.payloads = []

    def add(self, embedding: np.ndarray, payload: dict) -> None:
        unit = self._normalize(embedding)
        self.vectors.append(unit)
        self.payloads.append(payload)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list:
        if not self.vectors:
            return []

        q = self._normalize(query_embedding)

        # Cosine similarity = dot product of unit vectors
        scores = [(float(np.dot(q, vec)), i) for i, vec in enumerate(self.vectors)]
        scores.sort(reverse=True)

        return [self.payloads[i] for _, i in scores[:k]]

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm
