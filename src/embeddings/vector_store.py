import logging
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from dataclasses import dataclass

from ..core.config import Config
from ..chunking.semantic_chunker import Chunk

@dataclass
class VectorIndex:
    embeddings: np.ndarray
    chunk_ids: List[str]
    metadata: List[Dict[str, Any]]
    index_timestamp: float

class VectorStore:
    """Vector storage and similarity search with caching"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Config.CACHE_DIR / "vector_store"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index: Optional[VectorIndex] = None
        self.chunks_map: Dict[str, Chunk] = {}
    
    def build_index(self, chunks: List[Chunk], force_rebuild: bool = False) -> None:
        """Build vector index from chunks"""
        start_time = time.time()
        
        # Check for cached index
        index_file = self.cache_dir / "vector_index.pkl"
        if index_file.exists() and not force_rebuild:
            try:
                self.index = self._load_index(index_file)
                self.logger.info("Loaded vector index from cache")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load cached index: {str(e)}")
        
        # Filter chunks with embeddings
        embedded_chunks = [c for c in chunks if c.embedding is not None]
        
        if not embedded_chunks:
            raise ValueError("No chunks with embeddings found")
        
        # Create index
        embeddings = np.array([c.embedding for c in embedded_chunks])
        chunk_ids = [f"{c.document_id}_{c.chunk_index}" for c in embedded_chunks]
        metadata = []
        
        for chunk in embedded_chunks:
            meta = {
                'document_id': chunk.document_id,
                'chunk_index': chunk.chunk_index,
                'page_number': chunk.page_number,
                'section_title': chunk.section_title,
                'word_count': chunk.word_count,
                'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            metadata.append(meta)
        
        self.index = VectorIndex(
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            metadata=metadata,
            index_timestamp=time.time()
        )
        
        # Build chunks map for quick access
        self.chunks_map = {f"{c.document_id}_{c.chunk_index}": c for c in embedded_chunks}
        
        # Cache the index
        try:
            self._save_index(index_file)
        except Exception as e:
            self.logger.warning(f"Failed to cache index: {str(e)}")
        
        build_time = time.time() - start_time
        self.logger.info(f"Built vector index with {len(embedded_chunks)} vectors in {build_time:.2f}s")
    
    def _save_index(self, index_file: Path) -> None:
        """Save index to cache"""
        with open(index_file, 'wb') as f:
            pickle.dump(self.index, f)
    
    def _load_index(self, index_file: Path) -> VectorIndex:
        """Load index from cache"""
        with open(index_file, 'rb') as f:
            return pickle.load(f)
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 10, 
                         threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.index is None:
            raise RuntimeError("Vector index not built")
        
        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding, self.index.embeddings)
        
        # Get indices of top k similar vectors above threshold
        valid_indices = np.where(similarities >= threshold)[0]
        valid_similarities = similarities[valid_indices]
        
        # Sort by similarity
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]
        top_indices = sorted_indices[:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            result = {
                'chunk_id': self.index.chunk_ids[idx],
                'similarity': float(similarities[idx]),
                'metadata': self.index.metadata[idx],
                'chunk': self.chunks_map.get(self.index.chunk_ids[idx])
            }
            results.append(result)
        
        return results
    
    def _calculate_similarities(self, query_embedding: np.ndarray, 
                              index_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities efficiently"""
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(index_embeddings))
        
        normalized_query = query_embedding / query_norm
        
        # Normalize index embeddings
        index_norms = np.linalg.norm(index_embeddings, axis=1)
        valid_mask = index_norms > 0
        
        similarities = np.zeros(len(index_embeddings))
        
        if np.any(valid_mask):
            normalized_index = index_embeddings[valid_mask] / index_norms[valid_mask, np.newaxis]
            
            # Calculate cosine similarities
            cos_similarities = np.dot(normalized_index, normalized_query)
            
            # Convert to [0, 1] range
            similarities[valid_mask] = (cos_similarities + 1) / 2
        
        return similarities
    
    def multi_query_search(self, query_embeddings: List[np.ndarray], 
                          top_k: int = 10, threshold: float = 0.3) -> List[List[Dict[str, Any]]]:
        """Search for multiple queries"""
        results = []
        
        for query_embedding in query_embeddings:
            query_results = self.similarity_search(query_embedding, top_k, threshold)
            results.append(query_results)
        
        return results
    
    def filtered_search(self, query_embedding: np.ndarray, filters: Dict[str, Any], 
                       top_k: int = 10, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search with metadata filters"""
        if self.index is None:
            raise RuntimeError("Vector index not built")
        
        # Find indices that match filters
        valid_indices = []
        for i, metadata in enumerate(self.index.metadata):
            if self._matches_filters(metadata, filters):
                valid_indices.append(i)
        
        if not valid_indices:
            return []
        
        # Calculate similarities only for valid indices
        valid_embeddings = self.index.embeddings[valid_indices]
        similarities = self._calculate_similarities(query_embedding, valid_embeddings)
        
        # Filter by threshold
        threshold_mask = similarities >= threshold
        filtered_indices = np.array(valid_indices)[threshold_mask]
        filtered_similarities = similarities[threshold_mask]
        
        # Sort and get top k
        sorted_order = np.argsort(filtered_similarities)[::-1]
        top_indices = filtered_indices[sorted_order[:top_k]]
        top_similarities = filtered_similarities[sorted_order[:top_k]]
        
        # Prepare results
        results = []
        for idx, similarity in zip(top_indices, top_similarities):
            result = {
                'chunk_id': self.index.chunk_ids[idx],
                'similarity': float(similarity),
                'metadata': self.index.metadata[idx],
                'chunk': self.chunks_map.get(self.index.chunk_ids[idx])
            }
            results.append(result)
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.chunks_map.get(chunk_id)
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        if self.index is None:
            return {'status': 'not_built'}
        
        embedding_dim = self.index.embeddings.shape[1] if self.index.embeddings.size > 0 else 0
        
        # Calculate average similarity between all vectors (sample if too many)
        n_vectors = len(self.index.embeddings)
        if n_vectors > 1000:
            # Sample for efficiency
            sample_indices = np.random.choice(n_vectors, 1000, replace=False)
            sample_embeddings = self.index.embeddings[sample_indices]
        else:
            sample_embeddings = self.index.embeddings
        
        # Calculate pairwise similarities for sample
        avg_similarity = 0.0
        if len(sample_embeddings) > 1:
            similarities = []
            for i in range(min(100, len(sample_embeddings))):  # Limit to 100 for efficiency
                for j in range(i+1, min(i+11, len(sample_embeddings))):  # Compare with next 10
                    sim = self._calculate_similarities(sample_embeddings[i], sample_embeddings[j:j+1])[0]
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
        
        # Document distribution
        doc_counts = {}
        for meta in self.index.metadata:
            doc_id = meta['document_id']
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        return {
            'status': 'built',
            'total_vectors': n_vectors,
            'embedding_dimension': embedding_dim,
            'index_age_seconds': time.time() - self.index.index_timestamp,
            'average_similarity': float(avg_similarity),
            'documents_count': len(doc_counts),
            'chunks_per_document': doc_counts,
            'memory_usage_mb': self.index.embeddings.nbytes / (1024 * 1024) if self.index.embeddings.size > 0 else 0
        }
    
    def update_chunk(self, chunk: Chunk) -> None:
        """Update a chunk in the index"""
        chunk_id = f"{chunk.document_id}_{chunk.chunk_index}"
        
        if chunk_id in self.chunks_map:
            self.chunks_map[chunk_id] = chunk
            
            # Update in index if it exists
            if self.index is not None:
                try:
                    idx = self.index.chunk_ids.index(chunk_id)
                    if chunk.embedding is not None:
                        self.index.embeddings[idx] = chunk.embedding
                    
                    # Update metadata
                    self.index.metadata[idx] = {
                        'document_id': chunk.document_id,
                        'chunk_index': chunk.chunk_index,
                        'page_number': chunk.page_number,
                        'section_title': chunk.section_title,
                        'word_count': chunk.word_count,
                        'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    }
                    
                except ValueError:
                    self.logger.warning(f"Chunk {chunk_id} not found in index for update")
    
    def remove_document(self, document_id: str) -> None:
        """Remove all chunks from a document"""
        if self.index is None:
            return
        
        # Find indices to remove
        indices_to_remove = []
        for i, chunk_id in enumerate(self.index.chunk_ids):
            if chunk_id.startswith(f"{document_id}_"):
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return
        
        # Create mask for keeping indices
        keep_mask = np.ones(len(self.index.chunk_ids), dtype=bool)
        keep_mask[indices_to_remove] = False
        
        # Update index
        self.index.embeddings = self.index.embeddings[keep_mask]
        self.index.chunk_ids = [cid for i, cid in enumerate(self.index.chunk_ids) if keep_mask[i]]
        self.index.metadata = [meta for i, meta in enumerate(self.index.metadata) if keep_mask[i]]
        
        # Update chunks map
        for chunk_id in list(self.chunks_map.keys()):
            if chunk_id.startswith(f"{document_id}_"):
                del self.chunks_map[chunk_id]
        
        self.logger.info(f"Removed {len(indices_to_remove)} chunks for document {document_id}")
