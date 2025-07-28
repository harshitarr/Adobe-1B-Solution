import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import hashlib
from sentence_transformers import SentenceTransformer

from ..core.config import Config, ModelConfig
from ..chunking.semantic_chunker import Chunk

class EmbeddingEngine:
    """Embedding generation and model management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = ModelConfig.MODELS['embeddings']['name']
        self.model_path = Config.PROJECT_ROOT / ModelConfig.MODELS['embeddings']['path']
        self.cache_dir = Config.CACHE_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            start_time = time.time()
            
            # Check if local model exists
            if self.model_path.exists():
                self.logger.info(f"Loading local model from {self.model_path}")
                self.model = SentenceTransformer(str(self.model_path))
            else:
                self.logger.info(f"Loading model {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Save model locally for future use
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.model_path))
                self.logger.info(f"Model saved to {self.model_path}")
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, chunks: List[Chunk], use_cache: bool = True) -> List[Chunk]:
        """Generate embeddings for chunks with caching"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        start_time = time.time()
        
        # Separate cached and non-cached chunks
        cached_chunks = []
        new_chunks = []
        
        if use_cache:
            for chunk in chunks:
                cache_key = self._generate_cache_key(chunk.content)
                cache_file = self.cache_dir / f"{cache_key}.npy"
                
                if cache_file.exists():
                    try:
                        embedding = np.load(cache_file)
                        chunk.embedding = embedding
                        cached_chunks.append(chunk)
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached embedding: {str(e)}")
                
                new_chunks.append(chunk)
        else:
            new_chunks = chunks
        
        self.logger.info(f"Found {len(cached_chunks)} cached embeddings, generating {len(new_chunks)} new ones")
        
        # Generate embeddings for new chunks
        if new_chunks:
            texts = [chunk.content for chunk in new_chunks]
            
            try:
                # Generate in batches for memory efficiency
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_chunks = new_chunks[i:i + batch_size]
                    
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    # Assign embeddings to chunks and cache them
                    for chunk, embedding in zip(batch_chunks, batch_embeddings):
                        chunk.embedding = embedding
                        
                        # Cache the embedding
                        if use_cache:
                            cache_key = self._generate_cache_key(chunk.content)
                            cache_file = self.cache_dir / f"{cache_key}.npy"
                            try:
                                np.save(cache_file, embedding)
                            except Exception as e:
                                self.logger.warning(f"Failed to cache embedding: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings: {str(e)}")
                raise
        
        # Combine all chunks
        all_chunks = cached_chunks + new_chunks
        
        processing_time = time.time() - start_time
        self.logger.info(f"Generated embeddings for {len(chunks)} chunks in {processing_time:.2f}s")
        
        return all_chunks
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Check cache first
        cache_key = self._generate_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        # Generate new embedding
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # Cache it
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {str(e)}")
        
        return embedding
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of text content for cache key
        hasher = hashlib.md5()
        hasher.update(text.encode('utf-8'))
        hasher.update(self.model_name.encode('utf-8'))
        return hasher.hexdigest()
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure similarity is in [0, 1] range
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def find_similar_chunks(self, query_embedding: np.ndarray, chunks: List[Chunk], 
                          top_k: int = 10, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Find most similar chunks to query embedding"""
        similarities = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                similarity = self.calculate_similarity(query_embedding, chunk.embedding)
                if similarity >= threshold:
                    similarities.append({
                        'chunk': chunk,
                        'similarity': similarity
                    })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def get_embedding_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        embedded_chunks = [c for c in chunks if c.embedding is not None]
        
        if not embedded_chunks:
            return {'total_chunks': len(chunks), 'embedded_chunks': 0}
        
        embeddings = np.array([c.embedding for c in embedded_chunks])
        
        return {
            'total_chunks': len(chunks),
            'embedded_chunks': len(embedded_chunks),
            'embedding_coverage': len(embedded_chunks) / len(chunks),
            'embedding_dimension': embeddings.shape[1] if embeddings.size > 0 else 0,
            'avg_embedding_norm': np.mean([np.linalg.norm(emb) for emb in embeddings]),
            'embedding_similarity_matrix_shape': (len(embedded_chunks), len(embedded_chunks))
        }
    
    def batch_similarity_search(self, query_embeddings: List[np.ndarray], 
                               chunks: List[Chunk], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Perform batch similarity search for multiple queries"""
        results = []
        
        for query_embedding in query_embeddings:
            similar_chunks = self.find_similar_chunks(query_embedding, chunks, top_k)
            results.append(similar_chunks)
        
        return results
    
    def clear_cache(self):
        """Clear embedding cache"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Embedding cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
