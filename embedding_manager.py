# embeddings/embedding_manager.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # or use a better one if needed
        self.dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            embeddings = np.array([[0.0] * self.dimension for _ in texts])
        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            return np.array([[0.0] * self.dimension])

    def get_dimension(self) -> int:
        return self.dimension
