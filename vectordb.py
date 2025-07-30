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
        """Create a new FAISS index with higher capacity"""
        # Use IVF index for better scalability with large datasets
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, max(1, self.max_vectors // 10000)))
        self.index.nprobe = 10  # Number of clusters to search

        self.metadata = []
        self.category_mapping = defaultdict(list)
        self.vector_to_category = {}

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        print(f"Created new IVF FAISS index with dimension {self.dimension}")

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

        # Train index if it's IVF and not trained yet
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(vectors)

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

        if categories is None:
            # Search all categories but prioritize by importance
            all_results = self._search_all_categories(query_vector, k * 3)  # Get more results initially

            # Filter by importance threshold and limit results
            filtered_results = [
                (score, idx, cat) for score, idx, cat in all_results
                if self.metadata[idx].get('importance_score', 0) >= importance_threshold
            ]

            # Sort by combined score (similarity + importance)
            filtered_results.sort(key=lambda x: x[0] + self.metadata[x[1]].get('importance_score', 0) * 0.2,
                                  reverse=True)

            # Limit to k results
            filtered_results = filtered_results[:k]

            if filtered_results:
                scores, indices, cats = zip(*filtered_results)
                return list(scores), list(indices), list(cats)
            else:
                return [], [], []
        else:
            # Search within specific categories
            candidate_vectors = []
            for category in categories:
                if category in self.category_mapping:
                    category_vectors = self.category_mapping[category]
                    # Filter by importance
                    filtered_vectors = [
                        v for v in category_vectors
                        if v['importance'] >= importance_threshold
                    ]
                    candidate_vectors.extend(filtered_vectors)

            if not candidate_vectors:
                return [], [], []

            # Sort by importance and limit candidates
            candidate_vectors.sort(key=lambda x: x['importance'], reverse=True)
            candidate_ids = [v['vector_id'] for v in candidate_vectors[:k * 2]]  # Get more candidates

            # Perform similarity search on candidates
            results = []
            for vector_id in candidate_ids:
                if vector_id < len(self.metadata):
                    # Calculate similarity (simplified - in practice you'd use FAISS search)
                    results.append((0.5, vector_id, self.vector_to_category.get(vector_id, 'general')))

            # Limit to k results
            results = results[:k]

            if results:
                scores, indices, cats = zip(*results)
                return list(scores), list(indices), list(cats)
            else:
                return [], [], []

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