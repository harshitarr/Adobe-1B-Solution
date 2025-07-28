import logging
import re
from typing import List, Dict, Any, Tuple
import numpy as np

from ..core.schemas import ExtractedSection, ExtractedSubsection
from ..core.config import Config
from ..shared.text_processor import TextProcessor
from ..nlp.processor import NLPProcessor
from ..persona.processor import PersonaQuery

class SubsectionExtractor:
    """Granular subsection extraction and text refinement with ranking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_processor = TextProcessor()
        self.nlp_processor = NLPProcessor()
        
        # Subsection extraction parameters
        self.min_subsection_length = 50  # Minimum characters
        self.max_subsection_length = 800  # Maximum characters
        self.min_sentences = 2  # Minimum sentences per subsection
        
        # Refinement parameters
        self.key_sentence_indicators = [
            'in conclusion', 'therefore', 'as a result', 'importantly',
            'significantly', 'notably', 'it should be noted', 'key finding',
            'main result', 'primary outcome', 'critical factor'
        ]
        
        # Content quality indicators
        self.quality_indicators = {
            'statistical': ['p <', 'significant', 'correlation', 'regression', 'n ='],
            'methodological': ['method', 'approach', 'procedure', 'technique', 'analysis'],
            'findings': ['found', 'discovered', 'revealed', 'showed', 'demonstrated'],
            'quantitative': ['%', 'percent', 'ratio', 'rate', 'measure', 'score']
        }
    
    def extract_subsections(self, extracted_sections: List[ExtractedSection],
                           persona_query: PersonaQuery) -> List[ExtractedSubsection]:
        """Extract refined subsections from main sections"""
        all_subsections = []
        
        for section in extracted_sections:
            subsections = self._extract_from_section(section, persona_query)
            all_subsections.extend(subsections)
        
        # Rank subsections
        ranked_subsections = self._rank_subsections(all_subsections, persona_query)
        
        # Limit to top subsections
        top_subsections = ranked_subsections[:Config.TOP_K_SUBSECTIONS]
        
        self.logger.info(f"Extracted {len(top_subsections)} top subsections from {len(extracted_sections)} sections")
        
        return top_subsections
    
    def _extract_from_section(self, section: ExtractedSection,
                            persona_query: PersonaQuery) -> List[ExtractedSubsection]:
        """Extract subsections from a single main section"""
        subsections = []
        content = section.content
        
        if not content or len(content.strip()) < self.min_subsection_length:
            return subsections
        
        # Method 1: Split by paragraphs
        paragraph_subsections = self._extract_by_paragraphs(section, persona_query)
        subsections.extend(paragraph_subsections)
        
        # Method 2: Split by key sentences
        sentence_subsections = self._extract_by_key_sentences(section, persona_query)
        subsections.extend(sentence_subsections)
        
        # Method 3: Extract by content patterns
        pattern_subsections = self._extract_by_patterns(section, persona_query)
        subsections.extend(pattern_subsections)
        
        # Remove duplicates and very similar subsections
        unique_subsections = self._remove_duplicates(subsections)
        
        return unique_subsections
    
    def _extract_by_paragraphs(self, section: ExtractedSection,
                             persona_query: PersonaQuery) -> List[ExtractedSubsection]:
        """Extract subsections by splitting paragraphs"""
        subsections = []
        paragraphs = [p.strip() for p in section.content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if (self.min_subsection_length <= len(paragraph) <= self.max_subsection_length
                and len(self.text_processor.extract_sentences(paragraph)) >= self.min_sentences):
                
                # Refine the paragraph content
                refined_text = self._refine_text(paragraph, persona_query)
                
                if refined_text:
                    relevance_score = self._calculate_subsection_relevance(refined_text, persona_query)
                    
                    subsection = ExtractedSubsection(
                        document_reference=section.document_reference,
                        page_number=section.page_number,
                        refined_text=refined_text,
                        parent_section=section.section_title,
                        relevance_score=relevance_score
                    )
                    subsections.append(subsection)
        
        return subsections
    
    def _extract_by_key_sentences(self, section: ExtractedSection,
                                 persona_query: PersonaQuery) -> List[ExtractedSubsection]:
        """Extract subsections around key sentences"""
        subsections = []
        sentences = self.text_processor.extract_sentences(section.content)
        
        # Identify key sentences
        key_sentence_indices = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in self.key_sentence_indicators):
                key_sentence_indices.append(i)
        
        # Extract context around key sentences
        for key_idx in key_sentence_indices:
            # Take sentence and surrounding context
            start_idx = max(0, key_idx - 2)
            end_idx = min(len(sentences), key_idx + 3)
            
            context_sentences = sentences[start_idx:end_idx]
            context_text = ' '.join(context_sentences)
            
            if self.min_subsection_length <= len(context_text) <= self.max_subsection_length:
                refined_text = self._refine_text(context_text, persona_query)
                
                if refined_text:
                    relevance_score = self._calculate_subsection_relevance(refined_text, persona_query)
                    
                    subsection = ExtractedSubsection(
                        document_reference=section.document_reference,
                        page_number=section.page_number,
                        refined_text=refined_text,
                        parent_section=section.section_title,
                        relevance_score=relevance_score
                    )
                    subsections.append(subsection)
        
        return subsections
    
    def _extract_by_patterns(self, section: ExtractedSection,
                           persona_query: PersonaQuery) -> List[ExtractedSubsection]:
        """Extract subsections based on content patterns"""
        subsections = []
        content = section.content
        
        # Pattern 1: Lists (numbered or bulleted)
        list_patterns = [
            r'(?:^|\n)(?:\d+\.|\•|\*|-)\s+([^\n]+(?:\n(?!\d+\.|\•|\*|-)[^\n]*)*)',
            r'(?:^|\n)(?:[a-zA-Z]\.|\([a-zA-Z]\))\s+([^\n]+(?:\n(?![a-zA-Z]\.|\([a-zA-Z]\))[^\n]*)*)'
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if self.min_subsection_length <= len(match) <= self.max_subsection_length:
                    refined_text = self._refine_text(match, persona_query)
                    
                    if refined_text:
                        relevance_score = self._calculate_subsection_relevance(refined_text, persona_query)
                        
                        subsection = ExtractedSubsection(
                            document_reference=section.document_reference,
                            page_number=section.page_number,
                            refined_text=refined_text,
                            parent_section=section.section_title,
                            relevance_score=relevance_score
                        )
                        subsections.append(subsection)
        
        # Pattern 2: Statistical results
        stat_pattern = r'([^.!?]*(?:p\s*[<>=]\s*[\d.]+|significant|correlation|regression)[^.!?]*[.!?])'
        stat_matches = re.findall(stat_pattern, content, re.IGNORECASE)
        
        for match in stat_matches:
            if self.min_subsection_length <= len(match) <= self.max_subsection_length:
                refined_text = self._refine_text(match, persona_query)
                
                if refined_text:
                    relevance_score = self._calculate_subsection_relevance(refined_text, persona_query)
                    
                    subsection = ExtractedSubsection(
                        document_reference=section.document_reference,
                        page_number=section.page_number,
                        refined_text=refined_text,
                        parent_section=section.section_title,
                        relevance_score=relevance_score + 0.1  # Boost for statistical content
                    )
                    subsections.append(subsection)
        
        return subsections
    
    def _refine_text(self, text: str, persona_query: PersonaQuery) -> str:
        """Refine and clean extracted text"""
        if not text.strip():
            return ""
        
        # Clean whitespace and formatting
        refined = re.sub(r'\s+', ' ', text.strip())
        
        # Remove incomplete sentences at the beginning and end
        sentences = self.text_processor.extract_sentences(refined)
        if sentences:
            # Keep only complete sentences
            complete_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if (sentence.endswith(('.', '!', '?', ':')) and 
                    len(sentence.split()) >= 3):  # Minimum 3 words
                    complete_sentences.append(sentence)
            
            if complete_sentences:
                refined = ' '.join(complete_sentences)
            else:
                return ""  # No complete sentences found
        
        # Ensure minimum quality
        if len(refined.split()) < 5:  # At least 5 words
            return ""
        
        # Remove excessive technical jargon for beginners
        if any(level in ['beginner', 'student'] for level in [persona_query.primary_keywords]):
            refined = self._simplify_technical_language(refined)
        
        return refined
    
    def _simplify_technical_language(self, text: str) -> str:
        """Simplify technical language for beginners"""
        # This is a simplified approach - in practice, you might use more sophisticated NLP
        simplifications = {
            'methodology': 'method',
            'utilized': 'used',
            'demonstrate': 'show',
            'facilitate': 'help',
            'subsequently': 'then',
            'approximately': 'about',
            'significant': 'important'
        }
        
        for technical, simple in simplifications.items():
            text = re.sub(r'\b' + technical + r'\b', simple, text, flags=re.IGNORECASE)
        
        return text
    
    def _calculate_subsection_relevance(self, text: str, persona_query: PersonaQuery) -> float:
        """Calculate relevance score for a subsection"""
        base_score = 0.0
        
        # Keyword matching
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in persona_query.primary_keywords 
                             if keyword.lower() in text_lower)
        keyword_score = keyword_matches / max(1, len(persona_query.primary_keywords))
        base_score += keyword_score * 0.4
        
        # Domain term matching
        domain_matches = sum(1 for term in persona_query.domain_terms 
                           if term.lower() in text_lower)
        domain_score = domain_matches / max(1, len(persona_query.domain_terms))
        base_score += domain_score * 0.2
        
        # Content quality indicators
        quality_score = 0.0
        for category, indicators in self.quality_indicators.items():
            matches = sum(1 for indicator in indicators if indicator.lower() in text_lower)
            if matches > 0:
                quality_score += 0.1
        
        base_score += min(0.3, quality_score)
        
        # Sentence structure and readability
        complexity = self.nlp_processor.analyze_text_complexity(text)
        if complexity:
            # Adjust based on complexity preference
            complexity_level = complexity.get('complexity_level', 'moderate')
            if complexity_level == 'moderate':
                base_score += 0.1  # Prefer moderate complexity
        
        return min(1.0, base_score)
    
    def _remove_duplicates(self, subsections: List[ExtractedSubsection]) -> List[ExtractedSubsection]:
        """Remove duplicate and very similar subsections"""
        if not subsections:
            return subsections
        
        unique_subsections = []
        seen_texts = set()
        
        for subsection in subsections:
            # Normalize text for comparison
            normalized = re.sub(r'\s+', ' ', subsection.refined_text.lower().strip())
            
            # Check for exact duplicates
            if normalized in seen_texts:
                continue
            
            # Check for high similarity with existing subsections
            is_duplicate = False
            for existing in unique_subsections:
                existing_normalized = re.sub(r'\s+', ' ', existing.refined_text.lower().strip())
                
                # Calculate simple similarity
                similarity = self._calculate_text_similarity(normalized, existing_normalized)
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_subsections.append(subsection)
                seen_texts.add(normalized)
        
        return unique_subsections
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _rank_subsections(self, subsections: List[ExtractedSubsection],
                         persona_query: PersonaQuery) -> List[ExtractedSubsection]:
        """Rank subsections by relevance and quality"""
        if not subsections:
            return subsections
        
        # Calculate additional ranking factors
        for subsection in subsections:
            # Add diversity bonus for different parent sections
            parent_sections = set(s.parent_section for s in subsections)
            if len(parent_sections) > 1:
                subsection.relevance_score += 0.05
            
            # Add length bonus for optimal length
            text_length = len(subsection.refined_text)
            if 200 <= text_length <= 500:  # Optimal length range
                subsection.relevance_score += 0.05
            
            # Add statistical content bonus
            if any(indicator in subsection.refined_text.lower() 
                   for indicator in self.quality_indicators['statistical']):
                subsection.relevance_score += 0.1
        
        # Sort by relevance score
        ranked_subsections = sorted(subsections, 
                                  key=lambda x: x.relevance_score, 
                                  reverse=True)
        
        return ranked_subsections
    
    def get_subsection_statistics(self, subsections: List[ExtractedSubsection]) -> Dict[str, Any]:
        """Get statistics about extracted subsections"""
        if not subsections:
            return {}
        
        # Basic statistics
        total_subsections = len(subsections)
        avg_relevance = np.mean([s.relevance_score for s in subsections])
        
        # Parent section distribution
        parent_distribution = {}
        for subsection in subsections:
            parent_distribution[subsection.parent_section] = parent_distribution.get(
                subsection.parent_section, 0) + 1
        
        # Document distribution
        doc_distribution = {}
        for subsection in subsections:
            doc_distribution[subsection.document_reference] = doc_distribution.get(
                subsection.document_reference, 0) + 1
        
        # Text length statistics
        lengths = [len(s.refined_text) for s in subsections]
        
        return {
            'total_subsections': total_subsections,
            'average_relevance_score': avg_relevance,
            'parent_section_distribution': parent_distribution,
            'document_distribution': doc_distribution,
            'text_length_stats': {
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'avg': np.mean(lengths) if lengths else 0,
                'median': np.median(lengths) if lengths else 0
            },
            'quality_indicators_found': self._count_quality_indicators(subsections)
        }
    
    def _count_quality_indicators(self, subsections: List[ExtractedSubsection]) -> Dict[str, int]:
        """Count quality indicators across all subsections"""
        indicator_counts = {category: 0 for category in self.quality_indicators}
        
        for subsection in subsections:
            text_lower = subsection.refined_text.lower()
            for category, indicators in self.quality_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    indicator_counts[category] += 1
        
        return indicator_counts
