import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..shared.pdf_parser import PDFParser
from ..core.schemas import ProcessedDocument, DocumentMetadata

class BatchProcessor:
    """Batch processing of multiple documents with parallelization"""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.pdf_parser = PDFParser()
        self.max_workers = max_workers
        self._lock = threading.Lock()
        
    def process_documents(self, document_paths: List[Path]) -> List[ProcessedDocument]:
        """Process multiple documents in parallel"""
        start_time = time.time()
        processed_docs = []
        
        self.logger.info(f"Starting batch processing of {len(document_paths)} documents")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_path = {
                executor.submit(self._process_single_document, path): path 
                for path in document_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        with self._lock:
                            processed_docs.append(result)
                        self.logger.info(f"Successfully processed: {path.name}")
                    else:
                        self.logger.warning(f"Processing returned None for: {path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to process {path.name}: {str(e)}")
        
        processing_time = time.time() - start_time
        self.logger.info(f"Batch processing completed in {processing_time:.2f}s")
        
        return processed_docs
    
    def _process_single_document(self, document_path: Path) -> ProcessedDocument:
        """Process a single document"""
        start_time = time.time()
        
        try:
            # Extract content from PDF using the correct method name
            pdf_data = self.pdf_parser.parse_pdf(document_path)
            
            # Validate the extraction
            if not self.pdf_parser.validate_extraction(pdf_data):
                self.logger.error(f"Invalid extraction for {document_path.name}")
                return None
            
            # Create metadata with ALL required fields
            metadata = DocumentMetadata(
                filename=document_path.name,
                file_size=document_path.stat().st_size,
                page_count=len(pdf_data.get('pages', [])),
                processing_time=time.time() - start_time,
                document_type="pdf",  # Explicitly set the required field
                title=pdf_data.get('title'),
                author=pdf_data.get('author'),
                language="en"  # Default language
            )
            
            # Create processed document
            processed_doc = ProcessedDocument(
                document_id=self._generate_document_id(document_path),
                metadata=metadata,
                pages=pdf_data.get('pages', []),
                full_text=pdf_data.get('full_text', ''),
                structure=pdf_data.get('structure', {})
            )
            
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Error processing {document_path.name}: {str(e)}")
            # Don't re-raise, just return None to continue with other documents
            return None
    
    def _generate_document_id(self, document_path: Path) -> str:
        """Generate a unique document ID"""
        # Use filename without extension as base ID
        base_name = document_path.stem
        # Clean the name to make it a valid ID
        clean_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
        return clean_name.lower()
    
    def get_processing_statistics(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """Get statistics about the processed documents"""
        if not processed_docs:
            return {
                'total_documents': 0,
                'total_pages': 0,
                'total_processing_time': 0,
                'average_pages_per_doc': 0,
                'average_processing_time': 0
            }
        
        total_pages = sum(doc.metadata.page_count for doc in processed_docs)
        total_processing_time = sum(doc.metadata.processing_time for doc in processed_docs)
        
        return {
            'total_documents': len(processed_docs),
            'total_pages': total_pages,
            'total_processing_time': total_processing_time,
            'average_pages_per_doc': total_pages / len(processed_docs),
            'average_processing_time': total_processing_time / len(processed_docs),
            'document_sizes': [doc.metadata.file_size for doc in processed_docs],
            'processing_times': [doc.metadata.processing_time for doc in processed_docs]
        }
    
    def validate_documents(self, document_paths: List[Path]) -> List[Path]:
        """Validate document paths and filter out invalid ones"""
        valid_paths = []
        
        for path in document_paths:
            if not path.exists():
                self.logger.warning(f"Document not found: {path}")
                continue
                
            if not path.is_file():
                self.logger.warning(f"Path is not a file: {path}")
                continue
                
            if path.suffix.lower() != '.pdf':
                self.logger.warning(f"Not a PDF file: {path}")
                continue
                
            if path.stat().st_size == 0:
                self.logger.warning(f"Empty file: {path}")
                continue
                
            valid_paths.append(path)
        
        self.logger.info(f"Validated {len(valid_paths)} of {len(document_paths)} documents")
        return valid_paths
    
    def process_with_fallback(self, document_paths: List[Path]) -> List[ProcessedDocument]:
        """Process documents with fallback error handling"""
        valid_paths = self.validate_documents(document_paths)
        
        if not valid_paths:
            self.logger.error("No valid documents to process")
            return []
        
        processed_docs = []
        
        # Try parallel processing first
        try:
            processed_docs = self.process_documents(valid_paths)
        except Exception as e:
            self.logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
            
            # Fallback to sequential processing
            for path in valid_paths:
                try:
                    doc = self._process_single_document(path)
                    if doc:
                        processed_docs.append(doc)
                except Exception as e:
                    self.logger.error(f"Failed to process {path.name}: {e}")
                    continue
        
        if not processed_docs:
            self.logger.error("No documents could be processed successfully")
            return []
        
        self.logger.info(f"Successfully processed {len(processed_docs)} of {len(valid_paths)} documents")
        return processed_docs
    
    def extract_document_sections(self, processed_docs: List[ProcessedDocument]) -> List[Dict[str, Any]]:
        """Extract sections from processed documents for further analysis"""
        all_sections = []
        
        for doc in processed_docs:
            sections = self._extract_sections_from_doc(doc)
            all_sections.extend(sections)
        
        return all_sections
    
    def _extract_sections_from_doc(self, doc: ProcessedDocument) -> List[Dict[str, Any]]:
        """Extract logical sections from a processed document"""
        sections = []
        
        # Use the structure information if available
        if doc.structure and 'sections' in doc.structure:
            # Use pre-identified sections
            for section_info in doc.structure['sections']:
                sections.append({
                    'document_id': doc.document_id,
                    'document_title': doc.metadata.title or doc.metadata.filename,
                    'page_number': section_info.get('page_number', 1),
                    'section_title': section_info.get('potential_heading', 'Untitled Section'),
                    'content': self._extract_section_content(doc, section_info),
                    'confidence': section_info.get('confidence', 0.5),
                    'section_type': 'identified'
                })
        else:
            # Fallback: create sections from pages
            for page in doc.pages:
                if page.get('word_count', 0) > 20:  # Only include pages with substantial content
                    sections.append({
                        'document_id': doc.document_id,
                        'document_title': doc.metadata.title or doc.metadata.filename,
                        'page_number': page['page_number'],
                        'section_title': f"Page {page['page_number']}",
                        'content': page['text'],
                        'confidence': 1.0,
                        'section_type': 'page_based'
                    })
        
        return sections
    
    def _extract_section_content(self, doc: ProcessedDocument, section_info: Dict[str, Any]) -> str:
        """Extract content for a specific section"""
        page_num = section_info.get('page_number', 1)
        
        # Find the page containing this section
        target_page = None
        for page in doc.pages:
            if page['page_number'] == page_num:
                target_page = page
                break
        
        if not target_page:
            return ""
        
        # For now, return the full page content
        # In a more sophisticated implementation, you could extract content between headings
        return target_page.get('text', '')
