import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmailRetriever:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.emails: list[dict] = self._load_dataset()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.corpus_matrix = self.vectorizer.fit_transform(
            [email["customer_email"] for email in self.emails]
        )

    def _load_dataset(self) -> list[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def retrieve_similar(self, query_email: str, top_k: int = 3) -> list[dict]:
        query_vector = self.vectorizer.transform([query_email])
        similarity_scores = cosine_similarity(query_vector, self.corpus_matrix).flatten()
        ranked_indices = similarity_scores.argsort()[::-1][:top_k]

        results = []
        for index in ranked_indices:
            match = self.emails[index].copy()
            match["similarity_score"] = round(float(similarity_scores[index]), 4)
            results.append(match)
        return results