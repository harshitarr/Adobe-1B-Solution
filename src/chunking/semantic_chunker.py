import logging
from typing import List, Dict, Any
import re
from datetime import datetime
import numpy as np

from ..core.schemas import Chunk, ChunkMetadata
from ..shared.text_processor import TextProcessor
from ..core.config import Config

class SemanticChunker:
    """Semantic-aware document chunking with structure preservation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_processor = TextProcessor()
        
        # Chunking parameters
        self.chunk_size = getattr(Config, 'CHUNK_SIZE', 512) 
        self.chunk_overlap = getattr(Config, 'CHUNK_OVERLAP', 128)
        self.min_chunk_size = 50  # Minimum characters
        self.max_chunk_size = 1000  # Maximum characters
        
        # Sentence boundary detection patterns
        self.sentence_patterns = [
            r'\.(?=\s+[A-Z])',  # Period followed by space and capital letter
            r'[.!?]+(?=\s+[A-Z])',  # Multiple punctuation followed by capital
            r'\.(?=\n)',  # Period at end of line
        ]
        
    def chunk_documents(self, processed_docs: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk multiple documents while preserving semantic structure"""
        all_chunks = []
        
        for doc in processed_docs:
            try:
                doc_chunks = self._chunk_document(doc)
                all_chunks.extend(doc_chunks)
                self.logger.info(f"Created {len(doc_chunks)} chunks for document {doc.get('document_id', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Failed to chunk document {doc.get('document_id', 'unknown')}: {str(e)}")
                continue
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single document"""
        chunks = []
        
        # Handle both ProcessedDocument objects and dicts
        if hasattr(doc, 'document_id'):
            document_id = doc.document_id
            full_text = doc.full_text
            pages = doc.pages
        else:
            document_id = doc.get('document_id', f"doc_{id(doc)}")
            full_text = doc.get('full_text', '')
            pages = doc.get('pages', [])
        
        if not full_text.strip():
            self.logger.warning(f"Document {document_id} has no text content")
            return chunks
        
        # Method 1: Structure-aware chunking (preferred)
        if pages:
            chunks = self._chunk_by_structure(document_id, pages)
        
        # Method 2: Fallback to text-only chunking
        if not chunks:
            chunks = self._chunk_by_text(document_id, full_text)
        
        return chunks
    
    def _chunk_by_structure(self, document_id: str, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk document using structural information"""
        chunks = []
        chunk_index = 0
        
        for page_num, page in enumerate(pages, 1):
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # Try to identify sections/paragraphs
            sections = self._extract_sections(page_text)
            
            for section in sections:
                if len(section.strip()) < self.min_chunk_size:
                    continue
                
                # Create chunks from this section
                section_chunks = self._split_text_into_chunks(section)
                
                for chunk_text in section_chunks:
                    if len(chunk_text.strip()) >= self.min_chunk_size:
                        chunk_metadata = ChunkMetadata(
                            chunk_id=f"{document_id}_chunk_{chunk_index}",
                            document_id=document_id,
                            chunk_index=chunk_index,
                            start_page=page_num,
                            end_page=page_num,
                            chunk_type="content",
                            word_count=len(chunk_text.split()),
                            character_count=len(chunk_text)
                        )
                        
                        chunk = Chunk(
                            chunk_id=f"{document_id}_chunk_{chunk_index}",
                            document_id=document_id,
                            chunk_index=chunk_index,
                            content=chunk_text.strip(),
                            start_page=page_num,
                            end_page=page_num,
                            chunk_type="content",
                            metadata=chunk_metadata
                        )
                        
                        chunks.append(chunk)
                        chunk_index += 1
        
        return chunks
    
    def _chunk_by_text(self, document_id: str, full_text: str) -> List[Chunk]:
        """Fallback method to chunk by text only"""
        chunks = []
        chunk_index = 0
        
        # Clean the text
        clean_text = self.text_processor.clean_text(full_text)
        
        # Split into chunks
        text_chunks = self._split_text_into_chunks(clean_text)
        
        for chunk_text in text_chunks:
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_page=1,
                    end_page=1,
                    chunk_type="content",
                    word_count=len(chunk_text.split()),
                    character_count=len(chunk_text)
                )
                
                chunk = Chunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_text.strip(),
                    start_page=1,
                    end_page=1,
                    chunk_type="content",
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract logical sections from text"""
        # Try to split by common section markers
        section_patterns = [
            r'\n\n+',  # Double newlines
            r'\n[A-Z][A-Za-z\s]+:',  # Section headers
            r'\n\d+\.',  # Numbered sections
            r'\n[•\-\*]\s',  # Bullet points
        ]
        
        sections = [text]  # Start with full text
        
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([part.strip() for part in parts if part.strip()])
            sections = new_sections
        
        # Filter out very short sections
        sections = [s for s in sections if len(s) >= self.min_chunk_size]
        
        return sections if sections else [text]
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of appropriate size"""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        sentences = self.text_processor.extract_sentences(text)
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, save current chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If chunks are still too large, split more aggressively
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                # Split by words as last resort
                words = chunk.split()
                current_word_chunk = ""
                
                for word in words:
                    if len(current_word_chunk) + len(word) > self.max_chunk_size and current_word_chunk:
                        final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word
                    else:
                        current_word_chunk += " " + word if current_word_chunk else word
                
                if current_word_chunk.strip():
                    final_chunks.append(current_word_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using pattern matching"""
        sentences = []
        current_pos = 0
        
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                for match in matches:
                    sentence = text[current_pos:match.end()].strip()
                    if sentence:
                        sentences.append(sentence)
                    current_pos = match.end()
        
        # Add remaining text
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sentences.append(remaining)
        
        # Fallback: if no sentences found, split by length
        if not sentences:
            words = text.split()
            for i in range(0, len(words), self.chunk_size):
                sentence = " ".join(words[i:i + self.chunk_size])
                sentences.append(sentence)
        
        return sentences
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add overlap between consecutive chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i-1]
            
            # Get overlap words from previous chunk
            prev_words = previous_chunk.content.split()
            overlap_words = prev_words[-self.chunk_overlap:] if len(prev_words) > self.chunk_overlap else prev_words
            
            # Add overlap to current chunk
            overlapped_content = " ".join(overlap_words) + " " + current_chunk.content
            
            # Create new metadata for overlapped chunk
            new_metadata = ChunkMetadata(
                chunk_id=current_chunk.chunk_id,
                document_id=current_chunk.document_id,
                chunk_index=current_chunk.chunk_index,
                start_page=current_chunk.start_page,
                end_page=current_chunk.end_page,
                chunk_type=current_chunk.chunk_type,
                word_count=len(overlapped_content.split()),
                character_count=len(overlapped_content)
            )
            
            overlapped_chunk = Chunk(
                chunk_id=current_chunk.chunk_id,
                document_id=current_chunk.document_id,
                chunk_index=current_chunk.chunk_index,
                content=overlapped_content,
                start_page=current_chunk.start_page,
                end_page=current_chunk.end_page,
                chunk_type=current_chunk.chunk_type,
                metadata=new_metadata
            )
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about the generated chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'average_chunk_size': 0,
                'documents_covered': 0
            }
        
        chunk_sizes = []
        word_counts = []
        documents = set()
        
        for chunk in chunks:
            chunk_sizes.append(len(chunk.content))
            word_counts.append(len(chunk.content.split()))
            documents.add(chunk.document_id)
        
        return {
            'total_chunks': len(chunks),
            'average_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0,
            'average_word_count': np.mean(word_counts) if word_counts else 0,
            'documents_covered': len(documents),
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'chunk_size_distribution': {
                'small': len([s for s in chunk_sizes if s < 200]),
                'medium': len([s for s in chunk_sizes if 200 <= s < 500]),
                'large': len([s for s in chunk_sizes if s >= 500])
            }
        }
    
    def create_chunk_metadata_list(self, chunks: List[Chunk]) -> List[ChunkMetadata]:
        """Convert chunks to metadata objects for serialization"""
        metadata_list = []
        
        for chunk in chunks:
            # Use the metadata from the chunk if it exists, otherwise create new
            if hasattr(chunk, 'metadata') and chunk.metadata:
                metadata_list.append(chunk.metadata)
            else:
                metadata = ChunkMetadata(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    chunk_type=chunk.chunk_type,
                    word_count=len(chunk.content.split()),
                    character_count=len(chunk.content)
                )
                metadata_list.append(metadata)
        
        return metadata_list
