import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import math
from dataclasses import dataclass

from ..chunking.semantic_chunker import Chunk
from ..persona.processor import PersonaQuery, PersonaProcessor
from ..core.schemas import Persona, JobToBeDone

@dataclass
class SectionScore:
    section_id: str
    relevance_score: float
    importance_score: float
    combined_score: float
    confidence: float
    justification: str

class RelevanceScorer:
    """Calculate relevance and importance scores for document sections"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.persona_processor = PersonaProcessor()
        
        # Scoring weights
        self.weights = {
            'semantic_similarity': 0.3,
            'keyword_overlap': 0.25,
            'persona_relevance': 0.25,
            'section_importance': 0.2
        }
        
        # Section type importance mapping
        self.section_importance_map = {
            'abstract': 0.95,
            'executive summary': 0.9,
            'introduction': 0.8,
            'conclusion': 0.85,
            'summary': 0.85,
            'key findings': 0.9,
            'results': 0.8,
            'discussion': 0.75,
            'methodology': 0.7,
            'methods': 0.7,
            'literature review': 0.6,
            'background': 0.6,
            'references': 0.3,
            'appendix': 0.4,
            'acknowledgments': 0.2
        }
    
    def score_sections(self, chunks: List[Chunk], persona_query: PersonaQuery,
                      persona: Persona, jtbd: JobToBeDone) -> List[SectionScore]:
        """Score all sections for relevance and importance"""
        section_scores = []
        
        # Get persona insights
        persona_insights = self.persona_processor.extract_persona_insights(persona, jtbd)
        
        for chunk in chunks:
            score = self._score_single_section(chunk, persona_query, persona_insights)
            section_scores.append(score)
        
        self.logger.info(f"Scored {len(section_scores)} sections")
        return section_scores
    
    def _score_single_section(self, chunk: Chunk, persona_query: PersonaQuery,
                            persona_insights: Dict[str, Any]) -> SectionScore:
        """Score a single section"""
        section_id = f"{chunk.document_id}_{chunk.chunk_index}"
        
        # Calculate individual score components
        semantic_score = self._calculate_semantic_relevance(chunk, persona_query)
        keyword_score = self._calculate_keyword_overlap(chunk, persona_query)
        persona_score = self._calculate_persona_relevance(chunk, persona_query, persona_insights)
        importance_score = self._calculate_section_importance(chunk, persona_insights)
        
        # Combine scores
        relevance_score = (
            self.weights['semantic_similarity'] * semantic_score +
            self.weights['keyword_overlap'] * keyword_score +
            self.weights['persona_relevance'] * persona_score
        )
        
        combined_score = (
            relevance_score * 0.7 +
            importance_score * 0.3
        )
        
        # Calculate confidence based on score consistency
        scores = [semantic_score, keyword_score, persona_score, importance_score]
        confidence = self._calculate_confidence(scores)
        
        # Generate justification
        justification = self._generate_justification(
            chunk, semantic_score, keyword_score, persona_score, importance_score
        )
        
        return SectionScore(
            section_id=section_id,
            relevance_score=relevance_score,
            importance_score=importance_score,
            combined_score=combined_score,
            confidence=confidence,
            justification=justification
        )
    
    def _calculate_semantic_relevance(self, chunk: Chunk, persona_query: PersonaQuery) -> float:
        """Calculate semantic relevance using embeddings"""
        if chunk.embedding is None:
            return 0.0
        
        # Generate query embeddings for comparison
        search_queries = self.persona_processor.generate_search_queries(persona_query, 3)
        query_similarities = []
        
        # This would require the embedding engine, simplified for now
        # In practice, you'd generate embeddings for queries and compare
        
        # Fallback to text-based similarity
        max_similarity = 0.0
        for query in search_queries:
            # Simplified semantic similarity calculation
            query_words = set(query.lower().split())
            chunk_words = set(chunk.content.lower().split())
            
            if query_words and chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                union = len(query_words.union(chunk_words))
                similarity = overlap / union if union > 0 else 0.0
                max_similarity = max(max_similarity, similarity)
        
        return min(1.0, max_similarity * 2)  # Scale up for better distribution
    
    def _calculate_keyword_overlap(self, chunk: Chunk, persona_query: PersonaQuery) -> float:
        """Calculate keyword overlap score"""
        chunk_text = chunk.content.lower()
        total_score = 0.0
        total_weight = 0.0
        
        # Check primary keywords
        for keyword in persona_query.primary_keywords:
            if keyword.lower() in chunk_text:
                weight = persona_query.priority_weights.get(keyword.lower(), 0.5)
                total_score += weight
            total_weight += persona_query.priority_weights.get(keyword.lower(), 0.5)
        
        # Check domain terms
        domain_score = 0.0
        for term in persona_query.domain_terms:
            if term.lower() in chunk_text:
                domain_score += 0.3
        
        # Combine scores
        keyword_score = total_score / total_weight if total_weight > 0 else 0.0
        domain_factor = min(1.0, domain_score)
        
        return min(1.0, keyword_score * 0.7 + domain_factor * 0.3)
    
    def _calculate_persona_relevance(self, chunk: Chunk, persona_query: PersonaQuery,
                                   persona_insights: Dict[str, Any]) -> float:
        """Calculate persona-specific relevance"""
        return self.persona_processor.calculate_persona_relevance(chunk.content, persona_query)
    
    def _calculate_section_importance(self, chunk: Chunk, persona_insights: Dict[str, Any]) -> float:
        """Calculate inherent section importance"""
        section_title = chunk.section_title.lower()
        
        # Base importance from section type
        base_importance = 0.5
        for section_type, importance in self.section_importance_map.items():
            if section_type in section_title:
                base_importance = importance
                break
        
        # Adjust based on persona insights
        section_priorities = persona_insights.get('section_priorities', {})
        for section_type, priority in section_priorities.items():
            if section_type in section_title:
                base_importance = min(1.0, base_importance + priority * 0.3)
                break
        
        # Adjust based on content length (longer sections might be more important)
        length_factor = min(1.0, len(chunk.content.split()) / 200)  # Normalize around 200 words
        
        # Adjust based on section level (higher level = more important)
        level_factor = max(0.5, 1.0 - (chunk.metadata.get('section_level', 1) - 1) * 0.1) if chunk.metadata else 1.0
        
        final_importance = base_importance * 0.6 + length_factor * 0.2 + level_factor * 0.2
        
        return min(1.0, final_importance)
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on score consistency"""
        if not scores:
            return 0.0
        
        # Calculate standard deviation
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Lower standard deviation = higher confidence
        # Also consider mean score (higher mean = higher confidence)
        confidence = mean_score * (1 - min(0.5, std_score))
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_justification(self, chunk: Chunk, semantic_score: float,
                               keyword_score: float, persona_score: float,
                               importance_score: float) -> str:
        """Generate human-readable justification for the score"""
        justifications = []
        
        if semantic_score > 0.7:
            justifications.append("high semantic relevance to query")
        elif semantic_score > 0.4:
            justifications.append("moderate semantic relevance")
        
        if keyword_score > 0.6:
            justifications.append("strong keyword overlap")
        elif keyword_score > 0.3:
            justifications.append("some keyword matches")
        
        if persona_score > 0.7:
            justifications.append("highly relevant to persona needs")
        elif persona_score > 0.4:
            justifications.append("moderately relevant to persona")
        
        if importance_score > 0.8:
            justifications.append("high intrinsic section importance")
        elif importance_score > 0.6:
            justifications.append("moderate section importance")
        
        # Section-specific justifications
        section_title = chunk.section_title.lower()
        if any(key in section_title for key in ['abstract', 'summary', 'conclusion']):
            justifications.append("summary-type section")
        elif any(key in section_title for key in ['method', 'approach']):
            justifications.append("methodology section")
        elif any(key in section_title for key in ['result', 'finding']):
            justifications.append("results section")
        
        if not justifications:
            justifications.append("basic relevance criteria met")
        
        return "; ".join(justifications)
    
    def calculate_section_diversity(self, scored_sections: List[SectionScore],
                                  chunks: List[Chunk]) -> Dict[str, float]:
        """Calculate diversity metrics for section selection"""
        if not scored_sections:
            return {}
        
        # Create chunk lookup
        chunk_map = {f"{c.document_id}_{c.chunk_index}": c for c in chunks}
        
        # Document diversity
        documents = set()
        section_types = {}
        
        for score in scored_sections:
            chunk = chunk_map.get(score.section_id)
            if chunk:
                documents.add(chunk.document_id)
                
                # Categorize section type
                section_title = chunk.section_title.lower()
                section_type = 'other'
                for key_type in ['abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion']:
                    if key_type in section_title:
                        section_type = key_type
                        break
                
                section_types[section_type] = section_types.get(section_type, 0) + 1
        
        # Calculate diversity scores
        doc_diversity = len(documents) / len(set(c.document_id for c in chunks)) if chunks else 0
        type_diversity = len(section_types) / max(1, len(section_types))
        
        return {
            'document_diversity': doc_diversity,
            'section_type_diversity': type_diversity,
            'documents_covered': len(documents),
            'section_types_covered': len(section_types),
            'section_type_distribution': section_types
        }
    
    def apply_score_normalization(self, scored_sections: List[SectionScore]) -> List[SectionScore]:
        """Normalize scores for better distribution"""
        if not scored_sections:
            return scored_sections
        
        # Get score statistics
        combined_scores = [s.combined_score for s in scored_sections]
        relevance_scores = [s.relevance_score for s in scored_sections]
        importance_scores = [s.importance_score for s in scored_sections]
        
        # Calculate normalization parameters
        def normalize_scores(scores):
            if not scores:
                return scores
            
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range == 0:
                return [0.5] * len(scores)  # All scores are the same
            
            # Min-max normalization to [0, 1]
            return [(score - min_score) / score_range for score in scores]
        
        normalized_combined = normalize_scores(combined_scores)
        normalized_relevance = normalize_scores(relevance_scores)
        normalized_importance = normalize_scores(importance_scores)
        
        # Update section scores
        normalized_sections = []
        for i, section in enumerate(scored_sections):
            normalized_section = SectionScore(
                section_id=section.section_id,
                relevance_score=normalized_relevance[i],
                importance_score=normalized_importance[i],
                combined_score=normalized_combined[i],
                confidence=section.confidence,
                justification=section.justification
            )
            normalized_sections.append(normalized_section)
        
        return normalized_sections
