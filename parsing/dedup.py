from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Deduplicator:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def deduplicate(self, paragraphs: List[Dict]) -> List[Dict]:
        if not paragraphs:
            return []
        texts = [p["text"] for p in paragraphs]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        sim = cosine_similarity(X)
        keep = []
        kept_idx = []
        for i in range(len(paragraphs)):
            if not any(sim[i, j] > self.threshold for j in kept_idx):
                keep.append(paragraphs[i])
                kept_idx.append(i)
        return keep
