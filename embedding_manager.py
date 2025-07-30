# embeddings/embedding_manager.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and effective
        self.dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"❌ Error generating local embeddings: {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

    def generate_query_embedding(self, query: str) -> np.ndarray:
        try:
            return np.array([self.model.encode(query)])
        except Exception as e:
            print(f"❌ Error generating local query embedding: {e}")
            return np.array([[0.0] * self.dimension])

    def get_dimension(self) -> int:
        return self.dimension
