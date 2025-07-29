# embeddings/embedding_manager.py
import google.generativeai as genai
import numpy as np
from typing import List, Union
from settings import settings


class EmbeddingManager:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = "models/embedding-001"  # Gemini embedding model
        self.dimension = 768  # Gemini embedding dimension

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Google Gemini API"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Error generating embedding for text: {str(e)}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.dimension)

        return np.array(embeddings)

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query text"""
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            return np.array([result['embedding']])
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            return np.array([[0.0] * self.dimension])

    def get_dimension(self) -> int:
        return self.dimension