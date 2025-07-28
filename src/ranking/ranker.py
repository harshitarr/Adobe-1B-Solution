import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import heapq
from collections import defaultdict

from ..chunking.semantic_chunker import Chunk
from ..ranking.scorer import SectionScore
from ..core.config import Config
from ..core.schemas import ExtractedSection

class DocumentRanker:
    """Cross-document ranking with MMR diversity and top-k selection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.diversity_weight = 0.3  # Weight for diversity in MMR
        self.max_sections_per_doc = 5  # Limit sections per document
    
    def rank_sections(self, scored_sections: List[SectionScore], chunks: List[Chunk],
                     top_k: int = None) -> List[ExtractedSection]:
        """Rank sections across all documents with diversity"""
        if not scored_sections:
            return []
        
        top_k = top_k or Config.TOP_K_SECTIONS
        
        # Create chunk lookup
        chunk_map = {f"{c.document_id}_{c.chunk_index}": c for c in chunks}
        
        # Apply MMR (Maximal Marginal Relevance) for diversity
        selected_sections = self._apply_mmr_selection(scored_sections, chunk_map, top_k)
        
        # Convert to ExtractedSection objects
        extracted_sections = []
        for i, (score, chunk) in enumerate(selected_sections):
            extracted_section = ExtractedSection(
                document_reference=chunk.document_id,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                content=chunk.content,
                importance_rank=i + 1,  # 1-based ranking
                confidence_score=score.confidence
            )
            extracted_sections.append(extracted_section)
        
        self.logger.info(f"Ranked and selected {len(extracted_sections)} sections")
        return extracted_sections
    
    def _apply_mmr_selection(self, scored_sections: List[SectionScore], 
                           chunk_map: Dict[str, Chunk], top_k: int) -> List[Tuple[SectionScore, Chunk]]:
        """Apply Maximal Marginal Relevance for diverse selection"""
        if not scored_sections:
            return []
        
        # Sort by combined score
        scored_sections.sort(key=lambda x: x.combined_score, reverse=True)
        
        selected = []
        remaining = [(score, chunk_map.get(score.section_id)) for score in scored_sections 
                    if chunk_map.get(score.section_id) is not None]
        
        # Document tracking for diversity
        doc_counts = defaultdict(int)
        section_type_counts = defaultdict(int)
        
        while len(selected) < top_k and remaining:
            if not selected:
                # Select the highest scoring section first
                best_item = remaining.pop(0)
                selected.append(best_item)
                doc_counts[best_item[1].document_id] += 1
                section_type_counts[self._get_section_type(best_item[1].section_title)] += 1
                continue
            
            # Calculate MMR scores for remaining items
            best_mmr_score = -1
            best_index = -1
            
            for i, (score, chunk) in enumerate(remaining):
                # Skip if document already has too many sections
                if doc_counts[chunk.document_id] >= self.max_sections_per_doc:
                    continue
                
                # Calculate diversity penalty
                diversity_penalty = self._calculate_diversity_penalty(
                    chunk, selected, doc_counts, section_type_counts
                )
                
                # MMR score: relevance - diversity_penalty
                mmr_score = score.combined_score - self.diversity_weight * diversity_penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_index = i
            
            if best_index >= 0:
                best_item = remaining.pop(best_index)
                selected.append(best_item)
                doc_counts[best_item[1].document_id] += 1
                section_type_counts[self._get_section_type(best_item[1].section_title)] += 1
            else:
                # No valid items remaining
                break
        
        return selected
    
    def _calculate_diversity_penalty(self, chunk: Chunk, selected: List[Tuple[SectionScore, Chunk]],
                                   doc_counts: Dict[str, int], 
                                   section_type_counts: Dict[str, int]) -> float:
        """Calculate penalty for lack of diversity"""
        penalty = 0.0
        
        # Document diversity penalty
        doc_penalty = doc_counts[chunk.document_id] / len(selected) if selected else 0
        penalty += doc_penalty * 0.5
        
        # Section type diversity penalty
        section_type = self._get_section_type(chunk.section_title)
        type_penalty = section_type_counts[section_type] / len(selected) if selected else 0
        penalty += type_penalty * 0.3
        
        # Content similarity penalty (simplified)
        content_penalty = self._calculate_content_similarity_penalty(chunk, selected)
        penalty += content_penalty * 0.2
        
        return min(1.0, penalty)
    
    def _get_section_type(self, section_title: str) -> str:
        """Classify section type"""
        title_lower = section_title.lower()
        
        type_mapping = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'intro', 'background'],
            'methodology': ['method', 'approach', 'procedure'],
            'results': ['result', 'finding', 'outcome'],
            'discussion': ['discussion', 'analysis', 'interpretation'],
            'conclusion': ['conclusion', 'summary', 'final']
        }
        
        for section_type, keywords in type_mapping.items():
            if any(keyword in title_lower for keyword in keywords):
                return section_type
        
        return 'other'
    
    def _calculate_content_similarity_penalty(self, chunk: Chunk, 
                                            selected: List[Tuple[SectionScore, Chunk]]) -> float:
        """Calculate penalty based on content similarity to already selected"""
        if not selected:
            return 0.0
        
        max_similarity = 0.0
        chunk_words = set(chunk.content.lower().split())
        
        for _, selected_chunk in selected:
            selected_words = set(selected_chunk.content.lower().split())
            
            if chunk_words and selected_words:
                overlap = len(chunk_words.intersection(selected_words))
                union = len(chunk_words.union(selected_words))
                similarity = overlap / union if union > 0 else 0.0
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def rank_by_document_coverage(self, scored_sections: List[SectionScore], 
                                 chunks: List[Chunk], target_docs: int = None) -> List[ExtractedSection]:
        """Rank sections ensuring coverage across all documents"""
        if not scored_sections:
            return []
        
        chunk_map = {f"{c.document_id}_{c.chunk_index}": c for c in chunks}
        
        # Group sections by document
        doc_sections = defaultdict(list)
        for score in scored_sections:
            chunk = chunk_map.get(score.section_id)
            if chunk:
                doc_sections[chunk.document_id].append((score, chunk))
        
        # Sort sections within each document
        for doc_id in doc_sections:
            doc_sections[doc_id].sort(key=lambda x: x[0].combined_score, reverse=True)
        
        # Select sections round-robin from documents
        selected_sections = []
        total_docs = len(doc_sections)
        sections_per_doc = Config.TOP_K_SECTIONS // total_docs if total_docs > 0 else Config.TOP_K_SECTIONS
        extra_sections = Config.TOP_K_SECTIONS % total_docs
        
        # First, take equal number from each document
        for doc_id in sorted(doc_sections.keys()):
            doc_limit = sections_per_doc + (1 if extra_sections > 0 else 0)
            if extra_sections > 0:
                extra_sections -= 1
            
            selected_sections.extend(doc_sections[doc_id][:doc_limit])
        
        # Sort final selection by combined score
        selected_sections.sort(key=lambda x: x[0].combined_score, reverse=True)
        
        # Convert to ExtractedSection objects
        extracted_sections = []
        for i, (score, chunk) in enumerate(selected_sections[:Config.TOP_K_SECTIONS]):
            extracted_section = ExtractedSection(
                document_reference=chunk.document_id,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                content=chunk.content,
                importance_rank=i + 1,
                confidence_score=score.confidence
            )
            extracted_sections.append(extracted_section)
        
        return extracted_sections
    
    def apply_final_ranking_adjustments(self, extracted_sections: List[ExtractedSection]) -> List[ExtractedSection]:
        """Apply final adjustments to ranking"""
        if not extracted_sections:
            return extracted_sections
        
        # Adjust rankings based on section types
        type_boosts = {
            'abstract': 0.1,
            'executive summary': 0.1,
            'conclusion': 0.05,
            'key findings': 0.08,
            'summary': 0.05
        }
        
        # Calculate adjusted scores
        adjusted_sections = []
        for section in extracted_sections:
            boost = 0.0
            section_title = section.section_title.lower()
            
            for section_type, boost_value in type_boosts.items():
                if section_type in section_title:
                    boost = boost_value
                    break
            
            # Create new section with adjusted confidence (used for re-ranking)
            adjusted_section = ExtractedSection(
                document_reference=section.document_reference,
                page_number=section.page_number,
                section_title=section.section_title,
                content=section.content,
                importance_rank=section.importance_rank,
                confidence_score=min(1.0, section.confidence_score + boost)
            )
            adjusted_sections.append(adjusted_section)
        
        # Re-rank based on adjusted confidence scores
        adjusted_sections.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Update importance ranks
        for i, section in enumerate(adjusted_sections):
            section.importance_rank = i + 1
        
        return adjusted_sections
    
    def get_ranking_statistics(self, extracted_sections: List[ExtractedSection]) -> Dict[str, Any]:
        """Get statistics about the final ranking"""
        if not extracted_sections:
            return {}
        
        # Document distribution
        doc_counts = defaultdict(int)
        section_types = defaultdict(int)
        confidence_scores = []
        
        for section in extracted_sections:
            doc_counts[section.document_reference] += 1
            section_types[self._get_section_type(section.section_title)] += 1
            confidence_scores.append(section.confidence_score)
        
        return {
            'total_sections': len(extracted_sections),
            'documents_covered': len(doc_counts),
            'document_distribution': dict(doc_counts),
            'section_type_distribution': dict(section_types),
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'min_confidence': min(confidence_scores) if confidence_scores else 0,
            'max_confidence': max(confidence_scores) if confidence_scores else 0,
            'ranking_quality_score': self._calculate_ranking_quality(extracted_sections)
        }
    
    def _calculate_ranking_quality(self, extracted_sections: List[ExtractedSection]) -> float:
        """Calculate overall ranking quality score"""
        if not extracted_sections:
            return 0.0
        
        # Quality factors
        confidence_quality = np.mean([s.confidence_score for s in extracted_sections])
        
        # Diversity quality (document and section type diversity)
        doc_diversity = len(set(s.document_reference for s in extracted_sections)) / len(extracted_sections)
        type_diversity = len(set(self._get_section_type(s.section_title) for s in extracted_sections)) / len(extracted_sections)
        
        # Ranking consistency (higher confidence should have better ranks)
        consistency_score = 1.0
        for i in range(1, len(extracted_sections)):
            if extracted_sections[i].confidence_score > extracted_sections[i-1].confidence_score:
                consistency_score -= 0.1
        
        consistency_score = max(0.0, consistency_score)
        
        # Combined quality score
        quality_score = (
            confidence_quality * 0.4 +
            doc_diversity * 0.3 +
            type_diversity * 0.2 +
            consistency_score * 0.1
        )
        
        return min(1.0, quality_score)
