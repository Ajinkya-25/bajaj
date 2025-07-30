# database/enhanced_vectordb.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from settings import settings
import json
from collections import defaultdict


class FAISSManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = f"{self.index_path}.metadata"
        self.category_mapping_path = f"{self.index_path}.categories"

        self.index = None
        self.metadata = []
        self.category_mapping = defaultdict(list)  # category -> [vector_ids]
        self.vector_to_category = {}  # vector_id -> category

        # Increase index capacity for large documents
        self.max_vectors = 1000000  # 1M vectors capacity

        self.load_or_create_index()

    def load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
                self.index = faiss.read_index(self.index_path)

                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)

                # Load category mappings
                if os.path.exists(self.category_mapping_path):
                    with open(self.category_mapping_path, 'r') as f:
                        data = json.load(f)
                        self.category_mapping = defaultdict(list, data.get('category_mapping', {}))
                        self.vector_to_category = data.get('vector_to_category', {})
                        # Convert string keys back to int for vector_to_category
                        self.vector_to_category = {int(k): v for k, v in self.vector_to_category.items()}

                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
                print(f"Categories: {list(self.category_mapping.keys())}")
            else:
                self._create_new_index()
        except Exception as e:
            print(f"Error loading index: {e}")
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index - use simple FlatIP for small datasets"""
        # Start with a simple flat index that doesn't require training
        self.index = faiss.IndexFlatIP(self.dimension)

        self.metadata = []
        self.category_mapping = defaultdict(list)
        self.vector_to_category = {}

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        print(f"Created new Flat FAISS index with dimension {self.dimension}")

    def _create_ivf_index_if_needed(self, num_vectors: int):
        """Convert to IVF index if we have enough vectors"""
        if num_vectors < 1000:  # Keep flat index for small datasets
            return

        if isinstance(self.index, faiss.IndexFlatIP):
            print(f"Converting to IVF index for {num_vectors} vectors...")

            # Calculate appropriate number of clusters
            n_clusters = min(100, max(4, num_vectors // 50))

            # Create IVF index
            quantizer = faiss.IndexFlatIP(self.dimension)
            new_index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
            new_index.nprobe = min(10, n_clusters)

            # Get all vectors from current index
            if self.index.ntotal > 0:
                vectors = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
                for i in range(self.index.ntotal):
                    vectors[i] = self.index.reconstruct(i)

                # Train and add to new index
                new_index.train(vectors)
                new_index.add(vectors)

                # Replace index
                self.index = new_index
                print(f"Successfully converted to IVF index with {n_clusters} clusters")

    def clear_database(self):
        """Clear all vectors and metadata - used when adding new document"""
        print("Clearing existing database for new document...")
        self._create_new_index()
        self.save_index()

    def add_vectors_with_categories(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None) -> List[int]:
        """Add vectors to index with category information"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        # Check if we should convert to IVF index
        total_vectors_after = self.index.ntotal + vectors.shape[0]
        self._create_ivf_index_if_needed(total_vectors_after)

        # Train index if it's IVF and not trained yet
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Training IVF index...")
            # Combine existing vectors with new ones for training
            if self.index.ntotal > 0:
                existing_vectors = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
                for i in range(self.index.ntotal):
                    existing_vectors[i] = self.index.reconstruct(i)
                training_vectors = np.vstack([existing_vectors, vectors])
            else:
                training_vectors = vectors

            self.index.train(training_vectors)

        start_id = self.index.ntotal
        self.index.add(vectors)

        # Store metadata and category mappings
        vector_ids = list(range(start_id, self.index.ntotal))

        if metadata:
            self.metadata.extend(metadata)

            # Update category mappings
            for i, meta in enumerate(metadata):
                vector_id = vector_ids[i]
                category = meta.get('category', 'general')
                importance = meta.get('importance_score', 0.5)

                self.category_mapping[category].append({
                    'vector_id': vector_id,
                    'importance': importance
                })
                self.vector_to_category[vector_id] = category
        else:
            self.metadata.extend([{}] * vectors.shape[0])

        self.save_index()
        print(f"Added {vectors.shape[0]} vectors, total vectors: {self.index.ntotal}")
        return vector_ids

    def search_by_category(self, query_vector: np.ndarray, categories: List[str] = None,
                           k: int = 3, importance_threshold: float = 0.0) -> Tuple[List[float], List[int], List[str]]:
        """Search for similar vectors within specific categories"""
        if self.index.ntotal == 0:
            return [], [], []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure query vector is float32
        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Search all categories by default
        all_results = self._search_all_categories(query_vector, k * 3)  # Get more results initially

        # Filter by importance threshold and limit results
        filtered_results = [
            (score, idx, cat) for score, idx, cat in all_results
            if idx < len(self.metadata) and self.metadata[idx].get('importance_score', 0) >= importance_threshold
        ]

        # Sort by combined score (similarity + importance)
        filtered_results.sort(
            key=lambda x: x[0] + self.metadata[x[1]].get('importance_score', 0) * 0.2,
            reverse=True
        )

        # Limit to k results
        filtered_results = filtered_results[:k]

        if filtered_results:
            scores, indices, cats = zip(*filtered_results)
            return list(scores), list(indices), list(cats)
        else:
            return [], [], []

    def search(self, query_vector: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """Basic search method for compatibility"""
        if self.index.ntotal == 0:
            return [], []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure query vector is float32
        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        # Filter out invalid indices
        valid_scores = []
        valid_indices = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                valid_scores.append(float(score))
                valid_indices.append(int(idx))

        return valid_scores, valid_indices

    def _search_all_categories(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, int, str]]:
        """Search across all categories and return results with category info"""
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                category = self.vector_to_category.get(idx, 'general')
                results.append((float(score), int(idx), category))

        return results

    def get_category_statistics(self) -> Dict[str, Any]:
        """Get statistics about categories in the index"""
        stats = {}
        for category, vectors in self.category_mapping.items():
            stats[category] = {
                'count': len(vectors),
                'avg_importance': sum(v['importance'] for v in vectors) / len(vectors) if vectors else 0,
                'max_importance': max(v['importance'] for v in vectors) if vectors else 0
            }
        return stats

    def save_index(self):
        """Save index, metadata, and category mappings to disk"""
        try:
            faiss.write_index(self.index, self.index_path)

            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

            # Save category mappings
            category_data = {
                'category_mapping': dict(self.category_mapping),
                'vector_to_category': {str(k): v for k, v in self.vector_to_category.items()}
            }
            with open(self.category_mapping_path, 'w') as f:
                json.dump(category_data, f)

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

    def get_vectors_by_category(self, category: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get all vectors from a specific category"""
        if category not in self.category_mapping:
            return []

        vectors = self.category_mapping[category]

        # Sort by importance
        vectors.sort(key=lambda x: x['importance'], reverse=True)

        if limit:
            vectors = vectors[:limit]

        # Get full metadata
        result = []
        for vector_info in vectors:
            vector_id = vector_info['vector_id']
            if vector_id < len(self.metadata):
                metadata = self.metadata[vector_id].copy()
                metadata['vector_id'] = vector_id
                metadata['category'] = category
                metadata['importance_score'] = vector_info['importance']
                result.append(metadata)

        return result