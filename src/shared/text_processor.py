import re
import string
from typing import List, Dict, Set
import logging
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from pathlib import Path
import os

class TextProcessor:
    """Advanced text processing for document analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Ensure NLTK can find our custom data location
        self._setup_nltk_paths()
        
        # Initialize NLTK components
        self._ensure_nltk_data()
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Some features may be limited.")
            self.nlp = None
    
    def _setup_nltk_paths(self):
        """Setup NLTK data paths to find our custom location"""
        # Priority order: custom location first, then standard locations
        custom_nltk_paths = [
            "/app/models/nlp/nltk_data",
            "/app/models/nlp/nltk_data",
            "./models/nlp/nltk_data"
        ]
        
        for custom_path in custom_nltk_paths:
            if Path(custom_path).exists():
                if custom_path not in nltk.data.path:
                    nltk.data.path.insert(0, custom_path)
                    self.logger.info(f"Added NLTK data path: {custom_path}")
                break
        
        # Also check environment variable
        if 'NLTK_DATA' in os.environ:
            env_path = os.environ['NLTK_DATA']
            if env_path not in nltk.data.path:
                nltk.data.path.insert(0, env_path)
                self.logger.info(f"Added NLTK data path from env: {env_path}")
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available"""
        required_data = ['punkt', 'stopwords', 'wordnet']
        
        for data_name in required_data:
            try:
                # Try different possible locations for the data
                for data_type in ['tokenizers', 'corpora', 'taggers']:
                    try:
                        nltk.data.find(f'{data_type}/{data_name}')
                        self.logger.debug(f"Found NLTK data: {data_name}")
                        break
                    except LookupError:
                        continue
                else:
                    # If we get here, data was not found in any location
                    self.logger.warning(f"NLTK data {data_name} not found in any location")
                    # Don't try to download since we're offline
                    
            except Exception as e:
                self.logger.warning(f"Could not verify NLTK data {data_name}: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\-\,\:\;\(\)]', ' ', text)
        
        # Fix common OCR errors
        text = self._fix_ocr_errors(text)
        
        return text.strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors"""
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase l to uppercase I
            r'\b0\b': 'O',  # zero to letter O in context
            r'(?<=\w)1(?=\w)': 'l',  # 1 to l in middle of words
            r'\bm\b': 'in',  # common m/in confusion
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            return word_tokenize(text.lower())
        except LookupError:
            # Fallback if NLTK punkt is not available
            self.logger.warning("NLTK punkt tokenizer not available, using simple split")
            return text.lower().split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        try:
            return [token for token in tokens if token not in self.stop_words]
        except LookupError:
            # Fallback if stopwords not available
            self.logger.warning("NLTK stopwords not available, using basic filtering")
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            return [token for token in tokens if token not in basic_stopwords]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            # Fallback if wordnet not available
            self.logger.warning("NLTK wordnet not available, using stemming instead")
            return self.stem_tokens(tokens)
    
    def extract_keywords(self, text: str, top_k: int = 20) -> List[Dict]:
        """Extract keywords using multiple methods"""
        # Clean and tokenize
        clean_text = self.clean_text(text)
        tokens = self.tokenize(clean_text)
        
        # Remove stopwords and punctuation
        filtered_tokens = [
            token for token in tokens 
            if token not in string.punctuation and len(token) > 2
        ]
        
        # Apply stopword removal (with fallback)
        filtered_tokens = self.remove_stopwords(filtered_tokens)
        
        # Frequency-based keywords
        freq_keywords = Counter(filtered_tokens).most_common(top_k)
        
        # Stemmed keywords
        stemmed_tokens = self.stem_tokens(filtered_tokens)
        stemmed_keywords = Counter(stemmed_tokens).most_common(top_k)
        
        # spaCy-based entity extraction if available
        entities = []
        if self.nlp:
            try:
                doc = self.nlp(clean_text)
                entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
            except Exception as e:
                self.logger.warning(f"spaCy entity extraction failed: {e}")
        
        return {
            'frequency_keywords': [{'word': word, 'count': count} for word, count in freq_keywords],
            'stemmed_keywords': [{'word': word, 'count': count} for word, count in stemmed_keywords],
            'entities': entities
        }
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            return sent_tokenize(text)
        except LookupError:
            # Fallback sentence splitting
            self.logger.warning("NLTK sentence tokenizer not available, using regex fallback")
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def calculate_readability(self, text: str) -> Dict:
        """Calculate readability metrics"""
        sentences = self.extract_sentences(text)
        words = self.tokenize(text)
        
        # Basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(word) for word in words)
        
        if num_sentences == 0 or num_words == 0:
            return {'flesch_score': 0, 'avg_words_per_sentence': 0, 'avg_syllables_per_word': 0}
        
        # Flesch Reading Ease Score
        avg_sentence_length = num_words / num_sentences
        avg_syllables_per_word = num_syllables / num_words
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),
            'avg_words_per_sentence': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word,
            'complexity_level': self._get_complexity_level(flesch_score)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word"""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_char_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    def _get_complexity_level(self, flesch_score: float) -> str:
        """Get complexity level based on Flesch score"""
        if flesch_score >= 90:
            return 'very_easy'
        elif flesch_score >= 80:
            return 'easy'
        elif flesch_score >= 70:
            return 'fairly_easy'
        elif flesch_score >= 60:
            return 'standard'
        elif flesch_score >= 50:
            return 'fairly_difficult'
        elif flesch_score >= 30:
            return 'difficult'
        else:
            return 'very_difficult'
    
    def extract_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[Dict]:
        """Extract meaningful phrases from text"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            phrases = []
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if min_length <= len(chunk.text.split()) <= max_length:
                    phrases.append({
                        'text': chunk.text,
                        'type': 'noun_phrase',
                        'root': chunk.root.text
                    })
            
            # Extract key verb phrases
            for token in doc:
                if token.pos_ == 'VERB' and not token.is_stop:
                    # Get verb with its immediate dependencies
                    verb_phrase = [token.text]
                    for child in token.children:
                        if child.dep_ in ['dobj', 'pobj', 'acomp']:
                            verb_phrase.append(child.text)
                    
                    if len(verb_phrase) >= min_length:
                        phrases.append({
                            'text': ' '.join(verb_phrase),
                            'type': 'verb_phrase',
                            'root': token.text
                        })
            
            return phrases
        except Exception as e:
            self.logger.warning(f"Phrase extraction failed: {e}")
            return []
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts using spaCy"""
        if not self.nlp:
            # Fallback to simple token overlap
            tokens1 = set(self.tokenize(text1))
            tokens2 = set(self.tokenize(text2))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            overlap = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return overlap / union if union > 0 else 0.0
        
        try:
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            return doc1.similarity(doc2)
        except Exception as e:
            self.logger.warning(f"spaCy similarity calculation failed: {e}")
            # Fallback to token overlap
            tokens1 = set(self.tokenize(text1))
            tokens2 = set(self.tokenize(text2))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            overlap = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return overlap / union if union > 0 else 0.0
