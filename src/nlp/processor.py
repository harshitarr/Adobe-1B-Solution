import logging
import spacy
import nltk
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from collections import Counter, defaultdict
import re

from ..core.config import Config

class NLPProcessor:
    """Advanced NLP processing with spaCy, NLTK, and domain classification"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Disable unnecessary components for speed
            self.nlp.disable_pipes(["parser", "ner"])
        except OSError:
            self.logger.error("spaCy model not found")
            self.nlp = None
        
        # Initialize NLTK
        self._ensure_nltk_data()
        
        # Domain classification patterns
        self.domain_patterns = {
            'academic': {
                'keywords': ['research', 'study', 'analysis', 'methodology', 'hypothesis', 'findings'],
                'patterns': [r'\b(?:study|research|analysis|methodology)\b', r'\bcitation\b', r'\bp\s*<\s*0\.05\b']
            },
            'business': {
                'keywords': ['revenue', 'profit', 'market', 'strategy', 'growth', 'financial'],
                'patterns': [r'\$[\d,]+', r'\b\d+%\b', r'\bROI\b', r'\bKPI\b']
            },
            'technical': {
                'keywords': ['algorithm', 'implementation', 'system', 'architecture', 'performance'],
                'patterns': [r'\bAPI\b', r'\bSQL\b', r'\bJSON\b', r'\bHTTP\b']
            },
            'medical': {
                'keywords': ['patient', 'treatment', 'diagnosis', 'clinical', 'medical', 'therapy'],
                'patterns': [r'\bmg\b', r'\bml\b', r'\bdose\b', r'\bsymptom\b']
            },
            'legal': {
                'keywords': ['law', 'regulation', 'policy', 'compliance', 'legal', 'contract'],
                'patterns': [r'\bsection\s+\d+\b', r'\barticle\s+\d+\b', r'\bclause\b']
            }
        }
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available"""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{data_name}')
                    except LookupError:
                        try:
                            nltk.download(data_name, quiet=True)
                        except Exception as e:
                            self.logger.warning(f"Could not download NLTK data {data_name}: {e}")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        entities = []
        
        if self.nlp:
            # Re-enable NER for this operation
            if "ner" not in self.nlp.pipe_names:
                self.nlp.enable_pipe("ner")
            
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_) or ent.label_
                })
            
            # Disable NER again for performance
            self.nlp.disable_pipe("ner")
        
        return entities
    
    def extract_key_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[Dict[str, Any]]:
        """Extract key phrases using linguistic patterns"""
        phrases = []
        
        if not self.nlp:
            return phrases
        
        doc = self.nlp(text)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            phrase_text = chunk.text.strip()
            if min_length <= len(phrase_text.split()) <= max_length:
                phrases.append({
                    'text': phrase_text,
                    'type': 'noun_phrase',
                    'root': chunk.root.text,
                    'pos_tags': [token.pos_ for token in chunk]
                })
        
        # Extract verb phrases (simplified)
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                # Get children of the verb
                verb_phrase = [token.text]
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj', 'acomp']:
                        verb_phrase.append(child.text)
                
                if min_length <= len(verb_phrase) <= max_length:
                    phrases.append({
                        'text': ' '.join(verb_phrase),
                        'type': 'verb_phrase',
                        'root': token.text,
                        'pos_tags': [t.pos_ for t in [token] + list(token.children)]
                    })
        
        return phrases
    
    def classify_domain(self, text: str) -> Dict[str, Any]:
        """Classify the domain of the text"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, config in self.domain_patterns.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            keyword_score = keyword_matches / len(config['keywords'])
            
            # Check patterns
            pattern_matches = 0
            for pattern in config['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
            pattern_score = pattern_matches / len(config['patterns'])
            
            # Combined score
            domain_scores[domain] = (keyword_score * 0.6 + pattern_score * 0.4)
        
        # Determine primary domain
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        confidence = domain_scores.get(primary_domain, 0.0)
        
        # If confidence is too low, classify as general
        if confidence < 0.2:
            primary_domain = 'general'
            confidence = 0.5
        
        return {
            'primary_domain': primary_domain,
            'confidence': confidence,
            'domain_scores': domain_scores,
            'is_mixed_domain': len([s for s in domain_scores.values() if s > 0.3]) > 1
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using VADER"""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
            
            # Determine overall sentiment
            if scores['compound'] >= 0.05:
                overall = 'positive'
            elif scores['compound'] <= -0.05:
                overall = 'negative'
            else:
                overall = 'neutral'
            
            return {
                'overall': overall,
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
            
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return {
                'overall': 'neutral',
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
    
    def extract_statistical_info(self, text: str) -> Dict[str, List[str]]:
        """Extract statistical information from text"""
        stats_info = {
            'numbers': [],
            'percentages': [],
            'statistical_terms': [],
            'measurements': []
        }
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        stats_info['numbers'] = re.findall(number_pattern, text)
        
        # Extract percentages
        percentage_pattern = r'\b\d+(?:\.\d+)?%\b'
        stats_info['percentages'] = re.findall(percentage_pattern, text)
        
        # Extract statistical terms
        stat_terms = ['mean', 'average', 'median', 'standard deviation', 'correlation', 
                     'significant', 'p-value', 'confidence interval', 'regression', 
                     'chi-square', 't-test', 'ANOVA', 'variance']
        
        for term in stat_terms:
            if term.lower() in text.lower():
                stats_info['statistical_terms'].append(term)
        
        # Extract measurements
        measurement_pattern = r'\b\d+(?:\.\d+)?\s*(?:mg|ml|kg|cm|mm|inches?|feet|meters?|km|miles?|seconds?|minutes?|hours?|days?)\b'
        stats_info['measurements'] = re.findall(measurement_pattern, text, re.IGNORECASE)
        
        return stats_info
    
    def semantic_matching(self, query_terms: List[str], text: str, threshold: float = 0.7) -> Dict[str, Any]:
        """Perform semantic matching between query terms and text"""
        if not self.nlp:
            # Fallback to simple string matching
            matches = []
            text_lower = text.lower()
            for term in query_terms:
                if term.lower() in text_lower:
                    matches.append({
                        'term': term,
                        'similarity': 1.0,
                        'match_type': 'exact'
                    })
            
            return {
                'matches': matches,
                'total_matches': len(matches),
                'average_similarity': 1.0 if matches else 0.0
            }
        
        # Process query terms and text
        query_docs = [self.nlp(term) for term in query_terms]
        text_doc = self.nlp(text)
        
        matches = []
        similarities = []
        
        for i, query_doc in enumerate(query_docs):
            similarity = query_doc.similarity(text_doc)
            similarities.append(similarity)
            
            if similarity >= threshold:
                match_type = 'semantic'
                if query_terms[i].lower() in text.lower():
                    match_type = 'exact'
                elif any(token.lemma_ in text.lower() for token in query_doc):
                    match_type = 'lemma'
                
                matches.append({
                    'term': query_terms[i],
                    'similarity': similarity,
                    'match_type': match_type
                })
        
        return {
            'matches': matches,
            'total_matches': len(matches),
            'average_similarity': np.mean(similarities) if similarities else 0.0,
            'max_similarity': max(similarities) if similarities else 0.0
        }
    
    def extract_definitions(self, text: str) -> List[Dict[str, str]]:
        """Extract definitions from text using linguistic patterns"""
        definitions = []
        
        # Definition patterns
        patterns = [
            r'([A-Z][a-zA-Z\s]+?)\s+(?:is|are|refers to|means?|defined as)\s+([^.!?]+[.!?])',
            r'([A-Z][a-zA-Z\s]+?):\s*([^.!?\n]+[.!?]?)',
            r'(?:Definition|Define):\s*([A-Z][a-zA-Z\s]+?)\s*[-–—]\s*([^.!?\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                term = match[0].strip()
                definition = match[1].strip()
                
                if len(term.split()) <= 5 and len(definition) > 10:  # Basic quality filter
                    definitions.append({
                        'term': term,
                        'definition': definition
                    })
        
        return definitions
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity using multiple metrics"""
        if not text.strip():
            return {}
        
        # Basic metrics
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Filter out punctuation for word-based metrics
        words_only = [word for word in words if word.isalnum()]
        
        if not sentences or not words_only:
            return {'complexity_level': 'unknown'}
        
        # Calculate metrics
        avg_sentence_length = len(words_only) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words_only])
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = set(word.lower() for word in words_only)
        ttr = len(unique_words) / len(words_only) if words_only else 0
        
        # POS tag complexity
        pos_complexity = 0
        if self.nlp:
            doc = self.nlp(text)
            pos_tags = [token.pos_ for token in doc]
            unique_pos = len(set(pos_tags))
            pos_complexity = unique_pos / len(pos_tags) if pos_tags else 0
        
        # Readability score (simplified Flesch)
        syllable_count = sum(self._count_syllables(word) for word in words_only)
        avg_syllables = syllable_count / len(words_only) if words_only else 0
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        flesch_score = max(0, min(100, flesch_score))
        
        # Determine complexity level
        if flesch_score >= 80:
            complexity_level = 'simple'
        elif flesch_score >= 60:
            complexity_level = 'moderate'
        elif flesch_score >= 30:
            complexity_level = 'complex'
        else:
            complexity_level = 'very_complex'
        
        return {
            'complexity_level': complexity_level,
            'flesch_score': flesch_score,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'vocabulary_diversity': ttr,
            'pos_complexity': pos_complexity,
            'total_sentences': len(sentences),
            'total_words': len(words_only),
            'unique_words': len(unique_words)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified algorithm)"""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract academic citations from text"""
        citations = []
        
        # Citation patterns
        patterns = [
            # Author (Year) format
            r'([A-Z][a-zA-Z\s,&]+?)\s*\((\d{4})\)',
            # [Number] format
            r'\[(\d+)\]',
            # (Author, Year) format
            r'\(([A-Z][a-zA-Z\s,&]+?),\s*(\d{4})\)'
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text)
            for match in matches:
                if i == 0:  # Author (Year)
                    citations.append({
                        'type': 'author_year',
                        'author': match[0].strip(),
                        'year': match[1],
                        'full_text': f"{match[0].strip()} ({match[1]})"
                    })
                elif i == 1:  # [Number]
                    citations.append({
                        'type': 'numbered',
                        'number': match,
                        'full_text': f"[{match}]"
                    })
                elif i == 2:  # (Author, Year)
                    citations.append({
                        'type': 'parenthetical',
                        'author': match[0].strip(),
                        'year': match[1],
                        'full_text': f"({match[0].strip()}, {match[1]})"
                    })
        
        return citations
    
    def analyze_writing_style(self, text: str) -> Dict[str, Any]:
        """Analyze writing style characteristics"""
        if not text.strip():
            return {}
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Sentence type analysis
        question_count = sum(1 for s in sentences if s.strip().endswith('?'))
        exclamation_count = sum(1 for s in sentences if s.strip().endswith('!'))
        
        # Passive voice detection (simplified)
        passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are', 'am']
        passive_count = sum(1 for word in words if word.lower() in passive_indicators)
        
        # First person usage
        first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        first_person_count = sum(1 for word in words if word.lower() in first_person)
        
        # Academic language indicators
        academic_words = ['therefore', 'however', 'moreover', 'furthermore', 'consequently',
                         'nevertheless', 'additionally', 'specifically', 'particularly']
        academic_count = sum(1 for word in words if word.lower() in academic_words)
        
        total_words = len(words)
        
        return {
            'formality_score': (academic_count / max(1, total_words)) * 10,
            'personal_tone': first_person_count / max(1, total_words),
            'passive_voice_ratio': passive_count / max(1, total_words),
            'question_ratio': question_count / max(1, len(sentences)),
            'exclamation_ratio': exclamation_count / max(1, len(sentences)),
            'writing_style': self._classify_writing_style(
                academic_count / max(1, total_words),
                first_person_count / max(1, total_words)
            )
        }
    
    def _classify_writing_style(self, formality_score: float, personal_score: float) -> str:
        """Classify writing style based on metrics"""
        if formality_score > 0.02:
            return 'academic'
        elif personal_score > 0.03:
            return 'personal'
        elif formality_score > 0.01:
            return 'professional'
        else:
            return 'general'
