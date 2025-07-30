# database/vectordb.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from settings import settings


class FAISSManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = f"{self.index_path}.metadata"
        self.index = None
        self.metadata = []
        self.load_or_create_index()

    def load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
                self.index = faiss.read_index(self.index_path)
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                self._create_new_index()
        except Exception as e:
            print(f"Error loading index: {e}")
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.metadata = []
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        print(f"Created new FAISS index with dimension {self.dimension}")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None) -> List[int]:
        """Add vectors to index and return their IDs"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        start_id = self.index.ntotal
        self.index.add(vectors)

        # Store metadata
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * vectors.shape[0])

        self.save_index()
        vector_ids = list(range(start_id, self.index.ntotal))
        print(f"Added {vectors.shape[0]} vectors, total vectors: {self.index.ntotal}")
        return vector_ids

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return [], []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure query vector is float32
        query_vector = query_vector.astype(np.float32)

        # Normalize query vector
        faiss.normalize_L2(query_vector)

        # Limit k to the number of vectors in index
        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, k)

        # Filter out invalid indices (-1)
        valid_pairs = [(score, idx) for score, idx in zip(scores[0], indices[0]) if idx != -1]

        if not valid_pairs:
            return [], []

        scores_list, indices_list = zip(*valid_pairs)
        return list(scores_list), list(indices_list)

    def save_index(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving index: {e}")

    def get_metadata(self, vector_id: int) -> Dict[str, Any]:
        """Get metadata for a vector ID"""
        if 0 <= vector_id < len(self.metadata):
            return self.metadata[vector_id]
        return {}

    def get_total_vectors(self) -> int:
        """Get total number of vectors in the index"""
        return self.index.ntotal if self.index else 0
        
    def clear_index(self):
        """
        Deletes the on-disk index and re-initializes an empty index in memory.
        DANGEROUS: This deletes all learned vectors.
        """
        print("WARNING: CLEARING ALL DATA FROM FAISS VECTOR INDEX.")
        self.index = None
        
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                print(f"Removed FAISS index file: {self.index_path}")
            except Exception as e:
                print(f"Error removing FAISS index file: {e}")

        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
                print(f"Removed FAISS metadata file: {self.metadata_path}")
            except Exception as e:
                print(f"Error removing FAISS metadata file: {e}")
                
        self._create_new_index()
