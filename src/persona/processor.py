import logging
import re
from typing import Dict, List, Any, Tuple
import json
from dataclasses import dataclass

from ..core.schemas import Persona, JobToBeDone
from ..shared.text_processor import TextProcessor
from ..embeddings.engine import EmbeddingEngine

@dataclass
class PersonaQuery:
    primary_keywords: List[str]
    domain_terms: List[str]
    task_objectives: List[str]
    priority_weights: Dict[str, float]
    experience_modifiers: Dict[str, float]

class PersonaProcessor:
    """Parse and analyze persona and job-to-be-done for intelligent extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_processor = TextProcessor()
        self.embedding_engine = EmbeddingEngine()
        
        # Domain-specific term mappings
        self.domain_mappings = {
            'academic': ['research', 'methodology', 'analysis', 'findings', 'literature', 'hypothesis'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'growth', 'competitive'],
            'technical': ['implementation', 'architecture', 'performance', 'optimization', 'design'],
            'medical': ['diagnosis', 'treatment', 'symptoms', 'patient', 'clinical', 'therapy'],
            'legal': ['regulation', 'compliance', 'policy', 'rights', 'contract', 'liability']
        }
        
        # Experience level modifiers
        self.experience_weights = {
            'beginner': {'overview': 0.8, 'introduction': 0.9, 'basic': 0.8, 'fundamental': 0.9},
            'intermediate': {'method': 0.8, 'approach': 0.8, 'application': 0.9, 'example': 0.7},
            'advanced': {'advanced': 0.9, 'complex': 0.8, 'detailed': 0.9, 'in-depth': 0.9},
            'expert': {'novel': 0.9, 'cutting-edge': 0.9, 'innovative': 0.8, 'theoretical': 0.8}
        }
    
    def process_persona(self, persona_data: Dict[str, Any]) -> Persona:
        """Parse and validate persona data"""
        try:
            # Handle both string and dict formats
            if isinstance(persona_data, str):
                persona_data = self._parse_persona_string(persona_data)
            
            # Validate required fields
            required_fields = ['role', 'expertise_areas', 'experience_level', 'domain']
            for field in required_fields:
                if field not in persona_data:
                    raise ValueError(f"Missing required persona field: {field}")
            
            # Ensure expertise_areas and priorities are lists
            if isinstance(persona_data.get('expertise_areas'), str):
                persona_data['expertise_areas'] = [persona_data['expertise_areas']]
            
            if 'priorities' not in persona_data:
                persona_data['priorities'] = persona_data.get('expertise_areas', [])
            elif isinstance(persona_data.get('priorities'), str):
                persona_data['priorities'] = [persona_data['priorities']]
            
            persona = Persona(**persona_data)
            self.logger.info(f"Processed persona: {persona.role} ({persona.experience_level})")
            
            return persona
            
        except Exception as e:
            self.logger.error(f"Error processing persona: {str(e)}")
            raise
    
    def _parse_persona_string(self, persona_str: str) -> Dict[str, Any]:
        """Parse persona from string format"""
        # Try to parse as JSON first
        try:
            return json.loads(persona_str)
        except json.JSONDecodeError:
            pass
        
        # Parse natural language persona description
        persona_data = {
            'role': 'Unknown',
            'expertise_areas': [],
            'experience_level': 'intermediate',
            'domain': 'general',
            'priorities': []
        }
        
        # Extract role
        role_patterns = [
            r'(?:I am a|I\'m a|As a|Role:\s*)([A-Za-z\s]+?)(?:\s*(?:with|in|specializing|focusing))',
            r'([A-Za-z\s]+?)(?:\s*(?:with|specializing in))'
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, persona_str, re.IGNORECASE)
            if match:
                persona_data['role'] = match.group(1).strip()
                break
        
        # Extract experience level
        if any(word in persona_str.lower() for word in ['phd', 'doctoral', 'professor', 'senior']):
            persona_data['experience_level'] = 'expert'
        elif any(word in persona_str.lower() for word in ['graduate', 'master', 'experienced']):
            persona_data['experience_level'] = 'advanced'
        elif any(word in persona_str.lower() for word in ['student', 'undergraduate', 'beginner']):
            persona_data['experience_level'] = 'beginner'
        
        # Extract domain and expertise areas
        for domain, terms in self.domain_mappings.items():
            if any(term in persona_str.lower() for term in terms):
                persona_data['domain'] = domain
                persona_data['expertise_areas'] = [term for term in terms if term in persona_str.lower()]
                break
        
        # Extract priorities from keywords
        keywords = self.text_processor.extract_keywords(persona_str, top_k=10)
        if keywords.get('frequency_keywords'):
            persona_data['priorities'] = [kw['word'] for kw in keywords['frequency_keywords'][:5]]
        
        return persona_data
    
    def process_job_to_be_done(self, jtbd_data: Dict[str, Any]) -> JobToBeDone:
        """Parse and validate job-to-be-done data"""
        try:
            # Handle string format
            if isinstance(jtbd_data, str):
                jtbd_data = self._parse_jtbd_string(jtbd_data)
            
            # Validate required fields
            if 'task_description' not in jtbd_data:
                raise ValueError("Missing required field: task_description")
            
            # Set defaults
            jtbd_data.setdefault('expected_outcomes', [])
            jtbd_data.setdefault('priority_keywords', [])
            jtbd_data.setdefault('context', '')
            jtbd_data.setdefault('urgency', 'medium')
            
            # Extract keywords from task description if not provided
            if not jtbd_data['priority_keywords']:
                keywords = self.text_processor.extract_keywords(jtbd_data['task_description'])
                if keywords.get('frequency_keywords'):
                    jtbd_data['priority_keywords'] = [kw['word'] for kw in keywords['frequency_keywords'][:10]]
            
            jtbd = JobToBeDone(**jtbd_data)
            self.logger.info(f"Processed JTBD: {jtbd.task_description[:50]}...")
            
            return jtbd
            
        except Exception as e:
            self.logger.error(f"Error processing JTBD: {str(e)}")
            raise
    
    def _parse_jtbd_string(self, jtbd_str: str) -> Dict[str, Any]:
        """Parse JTBD from string format"""
        # Try JSON first
        try:
            return json.loads(jtbd_str)
        except json.JSONDecodeError:
            pass
        
        # Parse natural language
        jtbd_data = {
            'task_description': jtbd_str,
            'expected_outcomes': [],
            'priority_keywords': [],
            'context': '',
            'urgency': 'medium'
        }
        
        # Extract expected outcomes
        outcome_patterns = [
            r'(?:expected outcomes?|results?|goals?):\s*([^.]+)',
            r'(?:to|in order to|so that|for)\s+([^.]+)'
        ]
        
        for pattern in outcome_patterns:
            matches = re.findall(pattern, jtbd_str, re.IGNORECASE)
            jtbd_data['expected_outcomes'].extend([match.strip() for match in matches])
        
        # Extract urgency
        if any(word in jtbd_str.lower() for word in ['urgent', 'immediately', 'asap', 'critical']):
            jtbd_data['urgency'] = 'high'
        elif any(word in jtbd_str.lower() for word in ['later', 'eventually', 'when possible']):
            jtbd_data['urgency'] = 'low'
        
        return jtbd_data
    
    def build_persona_query(self, persona: Persona, jtbd: JobToBeDone) -> PersonaQuery:
        """Build query representation for persona and JTBD"""
        # Extract primary keywords from both persona and JTBD
        primary_keywords = []
        
        # From persona
        primary_keywords.extend(persona.expertise_areas)
        primary_keywords.extend(persona.priorities)
        primary_keywords.append(persona.role.lower())
        
        # From JTBD
        primary_keywords.extend(jtbd.priority_keywords)
        task_keywords = self.text_processor.extract_keywords(jtbd.task_description)
        if task_keywords.get('frequency_keywords'):
            primary_keywords.extend([kw['word'] for kw in task_keywords['frequency_keywords'][:5]])
        
        # Get domain-specific terms
        domain_terms = self.domain_mappings.get(persona.domain, [])
        
        # Extract task objectives
        task_objectives = jtbd.expected_outcomes.copy()
        if not task_objectives:
            # Extract potential objectives from task description
            objective_patterns = [
                r'(?:analyze|identify|determine|find|extract|compare|evaluate)\s+([^.]+)',
                r'(?:prepare|create|develop|build|generate)\s+([^.]+)'
            ]
            
            for pattern in objective_patterns:
                matches = re.findall(pattern, jtbd.task_description, re.IGNORECASE)
                task_objectives.extend([match.strip() for match in matches])
        
        # Calculate priority weights
        priority_weights = self._calculate_priority_weights(persona, jtbd, primary_keywords)
        
        # Get experience modifiers
        experience_modifiers = self.experience_weights.get(persona.experience_level, {})
        
        query = PersonaQuery(
            primary_keywords=list(set(primary_keywords)),  # Remove duplicates
            domain_terms=domain_terms,
            task_objectives=task_objectives,
            priority_weights=priority_weights,
            experience_modifiers=experience_modifiers
        )
        
        self.logger.info(f"Built persona query with {len(query.primary_keywords)} keywords")
        return query
    
    def _calculate_priority_weights(self, persona: Persona, jtbd: JobToBeDone, 
                                  keywords: List[str]) -> Dict[str, float]:
        """Calculate priority weights for different aspects"""
        weights = {}
        
        # Base weights
        for keyword in keywords:
            weights[keyword] = 0.5
        
        # Boost persona expertise areas
        for expertise in persona.expertise_areas:
            weights[expertise.lower()] = weights.get(expertise.lower(), 0.5) + 0.3
        
        # Boost JTBD priority keywords
        for priority in jtbd.priority_keywords:
            weights[priority.lower()] = weights.get(priority.lower(), 0.5) + 0.2
        
        # Boost based on urgency
        urgency_multipliers = {'high': 1.3, 'medium': 1.0, 'low': 0.8}
        multiplier = urgency_multipliers.get(jtbd.urgency, 1.0)
        
        for key in weights:
            weights[key] *= multiplier
        
        # Normalize weights to [0, 1] range
        if weights:
            max_weight = max(weights.values())
            if max_weight > 0:
                for key in weights:
                    weights[key] = min(1.0, weights[key] / max_weight)
        
        return weights
    
    def generate_search_queries(self, persona_query: PersonaQuery, num_queries: int = 5) -> List[str]:
        """Generate multiple search queries for comprehensive retrieval"""
        queries = []
        
        # Primary query with top keywords
        primary_query = " ".join(persona_query.primary_keywords[:5])
        queries.append(primary_query)
        
        # Domain-specific query
        if persona_query.domain_terms:
            domain_query = " ".join(persona_query.domain_terms[:3] + persona_query.primary_keywords[:2])
            queries.append(domain_query)
        
        # Task-objective queries
        for objective in persona_query.task_objectives[:2]:
            if len(objective.split()) <= 10:  # Keep queries reasonable
                queries.append(objective)
        
        # Weighted keyword query
        top_weighted = sorted(persona_query.priority_weights.items(), 
                            key=lambda x: x[1], reverse=True)[:4]
        weighted_query = " ".join([kw for kw, _ in top_weighted])
        queries.append(weighted_query)
        
        # Experience-level specific query
        experience_terms = list(persona_query.experience_modifiers.keys())[:3]
        if experience_terms:
            exp_query = " ".join(experience_terms + persona_query.primary_keywords[:2])
            queries.append(exp_query)
        
        # Remove duplicates and empty queries
        unique_queries = []
        for query in queries:
            if query.strip() and query not in unique_queries:
                unique_queries.append(query.strip())
        
        return unique_queries[:num_queries]
    
    def calculate_persona_relevance(self, text: str, persona_query: PersonaQuery) -> float:
        """Calculate how relevant text is to the persona and JTBD"""
        if not text.strip():
            return 0.0
        
        text_lower = text.lower()
        relevance_score = 0.0
        total_weight = 0.0
        
        # Check primary keywords
        for keyword in persona_query.primary_keywords:
            if keyword.lower() in text_lower:
                weight = persona_query.priority_weights.get(keyword.lower(), 0.5)
                relevance_score += weight
                total_weight += 1.0
        
        # Check domain terms
        for term in persona_query.domain_terms:
            if term.lower() in text_lower:
                relevance_score += 0.3
                total_weight += 0.5
        
        # Check experience modifiers
        for modifier, weight in persona_query.experience_modifiers.items():
            if modifier.lower() in text_lower:
                relevance_score += weight * 0.4
                total_weight += 0.3
        
        # Check task objectives (semantic similarity)
        for objective in persona_query.task_objectives:
            similarity = self.text_processor.semantic_similarity(text[:500], objective)
            relevance_score += similarity * 0.6
            total_weight += 0.4
        
        # Normalize by total possible weight
        if total_weight > 0:
            relevance_score = min(1.0, relevance_score / total_weight)
        
        return relevance_score
    
    def extract_persona_insights(self, persona: Persona, jtbd: JobToBeDone) -> Dict[str, Any]:
        """Extract insights about persona preferences and needs"""
        insights = {
            'complexity_preference': self._get_complexity_preference(persona.experience_level),
            'content_focus': self._get_content_focus(persona, jtbd),
            'information_depth': self._get_information_depth(persona.experience_level),
            'section_priorities': self._get_section_priorities(persona, jtbd),
            'keyword_emphasis': self._get_keyword_emphasis(persona, jtbd)
        }
        
        return insights
    
    def _get_complexity_preference(self, experience_level: str) -> str:
        """Get complexity preference based on experience"""
        mapping = {
            'beginner': 'simple_clear',
            'intermediate': 'moderate_detail',
            'advanced': 'detailed_comprehensive',
            'expert': 'highly_technical'
        }
        return mapping.get(experience_level, 'moderate_detail')
    
    def _get_content_focus(self, persona: Persona, jtbd: JobToBeDone) -> List[str]:
        """Determine content focus areas"""
        focus_areas = []
        
        # From persona domain
        domain_focus = {
            'academic': ['methodology', 'results', 'discussion', 'literature_review'],
            'business': ['executive_summary', 'financial_data', 'market_analysis', 'strategy'],
            'technical': ['implementation', 'architecture', 'performance', 'specifications'],
            'medical': ['clinical_findings', 'treatment', 'diagnosis', 'patient_outcomes'],
            'legal': ['regulations', 'compliance', 'policy', 'case_studies']
        }
        
        focus_areas.extend(domain_focus.get(persona.domain, ['overview', 'key_findings']))
        
        # From JTBD
        if any(word in jtbd.task_description.lower() for word in ['review', 'analysis', 'summary']):
            focus_areas.extend(['abstract', 'conclusion', 'key_points'])
        
        if any(word in jtbd.task_description.lower() for word in ['compare', 'benchmark']):
            focus_areas.extend(['comparison', 'evaluation', 'metrics'])
        
        return list(set(focus_areas))
    
    def _get_information_depth(self, experience_level: str) -> str:
        """Get preferred information depth"""
        mapping = {
            'beginner': 'surface_level',
            'intermediate': 'moderate_depth',
            'advanced': 'detailed',
            'expert': 'comprehensive'
        }
        return mapping.get(experience_level, 'moderate_depth')
    
    def _get_section_priorities(self, persona: Persona, jtbd: JobToBeDone) -> Dict[str, float]:
        """Get section type priorities"""
        priorities = {
            'introduction': 0.6,
            'methodology': 0.5,
            'results': 0.7,
            'discussion': 0.6,
            'conclusion': 0.8,
            'abstract': 0.9,
            'references': 0.3
        }
        
        # Adjust based on experience level
        experience_adjustments = {
            'beginner': {'introduction': 0.3, 'abstract': 0.3, 'overview': 0.4},
            'expert': {'methodology': 0.2, 'detailed_analysis': 0.3, 'technical_details': 0.3}
        }
        
        adjustments = experience_adjustments.get(persona.experience_level, {})
        for section, adjustment in adjustments.items():
            priorities[section] = priorities.get(section, 0.5) + adjustment
        
        # Adjust based on JTBD
        if 'method' in jtbd.task_description.lower():
            priorities['methodology'] += 0.2
        
        if 'result' in jtbd.task_description.lower():
            priorities['results'] += 0.3
        
        # Normalize
        max_priority = max(priorities.values()) if priorities else 1.0
        if max_priority > 1.0:
            for key in priorities:
                priorities[key] = priorities[key] / max_priority
        
        return priorities
    
    def _get_keyword_emphasis(self, persona: Persona, jtbd: JobToBeDone) -> Dict[str, float]:
        """Get keyword emphasis weights"""
        emphasis = {}
        
        # High emphasis on expertise areas
        for expertise in persona.expertise_areas:
            emphasis[expertise.lower()] = 0.9
        
        # Medium emphasis on priorities
        for priority in persona.priorities:
            emphasis[priority.lower()] = emphasis.get(priority.lower(), 0) + 0.6
        
        # High emphasis on JTBD keywords
        for keyword in jtbd.priority_keywords:
            emphasis[keyword.lower()] = emphasis.get(keyword.lower(), 0) + 0.8
        
        # Normalize
        if emphasis:
            max_emphasis = max(emphasis.values())
            if max_emphasis > 1.0:
                for key in emphasis:
                    emphasis[key] = min(1.0, emphasis[key] / max_emphasis)
        
        return emphasis
