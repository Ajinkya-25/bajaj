# database/vectordb.py
# Improved version with better search capabilities

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
                print(f"✅ Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                self._create_new_index()
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index with better configuration"""
        # Use IndexFlatIP for exact cosine similarity search
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        print(f"✅ Created new FAISS index with dimension {self.dimension}")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None) -> List[int]:
        """Add vectors to index with improved preprocessing"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)
        
        # Check for and handle any NaN or inf values
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            print("⚠️  Warning: Found NaN or inf values in vectors, replacing with zeros")
            vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize vectors for cosine similarity (L2 normalization)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        vectors = vectors / norms

        start_id = self.index.ntotal
        self.index.add(vectors)

        # Store metadata with enhanced information
        if metadata:
            for i, meta in enumerate(metadata):
                enhanced_meta = {
                    **meta,
                    'vector_norm': float(norms[i][0]),
                    'added_at': start_id + i
                }
                self.metadata.append(enhanced_meta)
        else:
            self.metadata.extend([{'added_at': start_id + i} for i in range(vectors.shape[0])])

        self.save_index()
        vector_ids = list(range(start_id, self.index.ntotal))
        print(f"✅ Added {vectors.shape[0]} vectors, total: {self.index.ntotal}")
        return vector_ids

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
        """Enhanced search with better preprocessing and error handling"""
        if self.index.ntotal == 0:
            print("⚠️  No vectors in index")
            return [], []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure query vector is float32
        query_vector = query_vector.astype(np.float32)
        
        # Handle NaN or inf values
        if np.any(np.isnan(query_vector)) or np.any(np.isinf(query_vector)):
            print("⚠️  Warning: Query vector contains NaN or inf values")
            query_vector = np.nan_to_num(query_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize query vector for cosine similarity
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        else:
            print("⚠️  Warning: Query vector has zero norm")
            return [], []

        # Limit k to available vectors
        k = min(k, self.index.ntotal)
        
        try:
            # Perform search
            scores, indices = self.index.search(query_vector, k)
            
            # Convert to lists and filter valid results
            scores_list = scores[0].tolist()
            indices_list = indices[0].tolist()
            
            # Filter out invalid indices and low scores
            valid_results = []
            for score, idx in zip(scores_list, indices_list):
                if idx != -1 and not np.isnan(score) and not np.isinf(score):
                    valid_results.append((float(score), int(idx)))
            
            if not valid_results:
                print("⚠️  No valid search results found")
                return [], []
            
            # Sort by score (descending for cosine similarity)
            valid_results.sort(key=lambda x: x[0], reverse=True)
            
            final_scores, final_indices = zip(*valid_results)
            
            print(f"🔍 Search completed: {len(final_indices)} results, top score: {final_scores[0]:.4f}")
            
            return list(final_scores), list(final_indices)
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return [], []

    def save_index(self):
        """Save index and metadata with error handling"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            print(f"💾 Saved index and metadata ({len(self.metadata)} entries)")
        except Exception as e:
            print(f"❌ Error saving index: {e}")

    def get_metadata(self, vector_id: int) -> Dict[str, Any]:
        """Get metadata for a vector ID with bounds checking"""
        if 0 <= vector_id < len(self.metadata):
            return self.metadata[vector_id]
        print(f"⚠️  Invalid vector_id: {vector_id} (valid range: 0-{len(self.metadata)-1})")
        return {}

    def get_total_vectors(self) -> int:
        """Get total number of vectors in the index"""
        return self.index.ntotal if self.index else 0
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the index for debugging"""
        if not self.index:
            return {"status": "no_index"}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "metadata_count": len(self.metadata),
            "index_type": type(self.index).__name__,
            "is_trained": getattr(self.index, 'is_trained', True)
        }
        
    def clear_index(self):
        """
        Deletes the on-disk index and re-initializes an empty index in memory.
        DANGEROUS: This deletes all learned vectors.
        """
        print("⚠️  WARNING: CLEARING ALL DATA FROM FAISS VECTOR INDEX.")
        self.index = None
        
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                print(f"🗑️  Removed FAISS index file: {self.index_path}")
            except Exception as e:
                print(f"❌ Error removing FAISS index file: {e}")

        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
                print(f"🗑️  Removed FAISS metadata file: {self.metadata_path}")
            except Exception as e:
                print(f"❌ Error removing FAISS metadata file: {e}")
                
        self._create_new_index()

    def debug_search(self, query_vector: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """Debug version of search that returns detailed information"""
        stats = self.get_search_stats()
        
        if self.index.ntotal == 0:
            return {"error": "No vectors in index", "stats": stats}
        
        scores, indices = self.search(query_vector, k)
        
        debug_info = {
            "stats": stats,
            "query_norm": float(np.linalg.norm(query_vector)),
            "results_count": len(scores),
            "top_scores": scores[:5] if scores else [],
            "top_indices": indices[:5] if indices else [],
            "score_range": {
                "min": float(min(scores)) if scores else None,
                "max": float(max(scores)) if scores else None,
                "mean": float(np.mean(scores)) if scores else None
            }
        }
        
        return debug_info