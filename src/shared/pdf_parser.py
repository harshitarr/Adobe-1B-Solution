import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import PyPDF2
import fitz  # PyMuPDF
import re
from datetime import datetime

class PDFParser:
    """PDF parsing with multiple extraction methods and error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF and extract content with metadata"""
        start_time = time.time()
        
        try:
            # First, try PyMuPDF (fitz) - generally more reliable
            result = self._parse_with_pymupdf(file_path)
            
            # If PyMuPDF fails or returns empty content, try PyPDF2
            if not result.get('full_text', '').strip():
                self.logger.warning(f"PyMuPDF returned empty content for {file_path.name}, trying PyPDF2")
                result = self._parse_with_pypdf2(file_path)
            
            # Add processing metadata
            result['processing_time'] = time.time() - start_time
            result['file_size'] = file_path.stat().st_size
            result['filename'] = file_path.name
            
            self.logger.info(f"Successfully parsed {file_path.name} ({len(result['pages'])} pages)")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path.name}: {str(e)}")
            # Return minimal structure to prevent downstream failures
            return {
                'pages': [],
                'full_text': '',
                'title': file_path.stem,
                'author': None,
                'structure': {},
                'processing_time': time.time() - start_time,
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'filename': file_path.name,
                'error': str(e)
            }
    
    def _parse_with_pymupdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using PyMuPDF (fitz)"""
        try:
            doc = fitz.open(str(file_path))
            pages = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Clean the text
                cleaned_text = self._clean_text(page_text)
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()) if cleaned_text else 0
                }
                
                pages.append(page_data)
                full_text += cleaned_text + "\n\n"
            
            # Extract metadata
            metadata = doc.metadata
            doc.close()
            
            return {
                'pages': pages,
                'full_text': full_text.strip(),
                'title': metadata.get('title') or file_path.stem,
                'author': metadata.get('author'),
                'structure': self._extract_structure(pages),
                'page_count': len(pages),
                'extraction_method': 'pymupdf'
            }
            
        except Exception as e:
            self.logger.warning(f"PyMuPDF parsing failed for {file_path.name}: {str(e)}")
            raise
    
    def _parse_with_pypdf2(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using PyPDF2 as fallback"""
        try:
            pages = []
            full_text = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        cleaned_text = self._clean_text(page_text)
                        
                        page_data = {
                            'page_number': page_num + 1,
                            'text': cleaned_text,
                            'char_count': len(cleaned_text),
                            'word_count': len(cleaned_text.split()) if cleaned_text else 0
                        }
                        
                        pages.append(page_data)
                        full_text += cleaned_text + "\n\n"
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {page_num + 1} from {file_path.name}: {str(e)}")
                        # Add empty page to maintain page numbering
                        pages.append({
                            'page_number': page_num + 1,
                            'text': '',
                            'char_count': 0,
                            'word_count': 0,
                            'error': str(e)
                        })
                
                # Extract metadata
                info = pdf_reader.metadata if hasattr(pdf_reader, 'metadata') else {}
                
                return {
                    'pages': pages,
                    'full_text': full_text.strip(),
                    'title': info.get('/Title') or file_path.stem,
                    'author': info.get('/Author'),
                    'structure': self._extract_structure(pages),
                    'page_count': len(pages),
                    'extraction_method': 'pypdf2'
                }
                
        except Exception as e:
            self.logger.warning(f"PyPDF2 parsing failed for {file_path.name}: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null characters
        text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Control characters
        
        # Fix common encoding issues
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u2018', "'")  # Left single quotation mark
        text = text.replace('\u201C', '"')  # Left double quotation mark
        text = text.replace('\u201D', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '--') # Em dash
        
        return text.strip()
    
    def _extract_structure(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract basic document structure"""
        total_words = sum(page.get('word_count', 0) for page in pages)
        total_chars = sum(page.get('char_count', 0) for page in pages)
        non_empty_pages = len([p for p in pages if p.get('word_count', 0) > 0])
        
        # Simple structure analysis
        structure = {
            'total_pages': len(pages),
            'non_empty_pages': non_empty_pages,
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_page': total_words / len(pages) if pages else 0,
            'pages_with_content': non_empty_pages / len(pages) if pages else 0
        }
        
        # Try to identify potential sections based on text patterns
        sections = self._identify_sections(pages)
        if sections:
            structure['sections'] = sections
        
        return structure
    
    def _identify_sections(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic section identification"""
        sections = []
        
        for page in pages:
            text = page.get('text', '')
            if not text:
                continue
            
            # Look for potential headings (lines that are short and start with capital letters)
            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                line = line.strip()
                if (len(line) < 100 and  # Short lines
                    len(line) > 5 and     # But not too short
                    line[0].isupper() and # Starts with capital
                    not line.endswith('.') and  # Doesn't end with period
                    ':' not in line):    # Doesn't contain colon
                    
                    sections.append({
                        'page_number': page['page_number'],
                        'line_number': line_num + 1,
                        'potential_heading': line,
                        'confidence': 0.5  # Basic confidence score
                    })
        
        return sections[:20]  # Limit to first 20 potential sections
    
    def validate_extraction(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate that extraction was successful"""
        if not parsed_data:
            return False
        
        # Check basic structure
        required_keys = ['pages', 'full_text', 'page_count']
        if not all(key in parsed_data for key in required_keys):
            return False
        
        # Check that we have actual content
        if not parsed_data['pages'] or parsed_data['page_count'] == 0:
            return False
        
        # Check that we extracted some text
        full_text = parsed_data.get('full_text', '')
        if not full_text or len(full_text.strip()) < 10:
            return False
        
        return True
    
    def get_extraction_statistics(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the extraction"""
        if not parsed_data:
            return {}
        
        pages = parsed_data.get('pages', [])
        full_text = parsed_data.get('full_text', '')
        
        stats = {
            'total_pages': len(pages),
            'total_characters': len(full_text),
            'total_words': len(full_text.split()) if full_text else 0,
            'extraction_method': parsed_data.get('extraction_method', 'unknown'),
            'processing_time': parsed_data.get('processing_time', 0),
            'file_size_bytes': parsed_data.get('file_size', 0),
            'characters_per_page': len(full_text) / len(pages) if pages else 0,
            'words_per_page': len(full_text.split()) / len(pages) if pages and full_text else 0
        }
        
        # Page-level statistics
        if pages:
            page_word_counts = [p.get('word_count', 0) for p in pages]
            stats.update({
                'pages_with_content': len([c for c in page_word_counts if c > 0]),
                'average_words_per_page': sum(page_word_counts) / len(page_word_counts),
                'max_words_per_page': max(page_word_counts),
                'min_words_per_page': min(page_word_counts)
            })
        
        return stats
