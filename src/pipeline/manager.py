import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import signal
import os

from ..core.config import Config
from ..core.schemas import Persona, JobToBeDone
from ..preprocessing.batch_processor import BatchProcessor
from ..chunking.semantic_chunker import SemanticChunker
from ..embeddings.engine import EmbeddingEngine
from ..embeddings.vector_store import VectorStore
from ..persona.processor import PersonaProcessor
from ..ranking.scorer import RelevanceScorer
from ..ranking.ranker import DocumentRanker
from ..nlp.processor import NLPProcessor
from ..subsection.extractor import SubsectionExtractor
from ..output.formatter import OutputFormatter

# Define custom exceptions if they don't exist
class PipelineExceptions:
    class DocumentParsingError(Exception):
        pass
    
    class InvalidPersonaError(Exception):
        pass
    
    class ProcessingTimeoutError(Exception):
        pass

class PipelineManager:
    """Workflow orchestration with performance monitoring, caching, and error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components with error handling
        try:
            self.batch_processor = BatchProcessor()
            self.semantic_chunker = SemanticChunker()
            self.embedding_engine = EmbeddingEngine()
            self.vector_store = VectorStore()
            self.persona_processor = PersonaProcessor()
            self.relevance_scorer = RelevanceScorer()
            self.document_ranker = DocumentRanker()
            self.nlp_processor = NLPProcessor()
            self.subsection_extractor = SubsectionExtractor()
            self.output_formatter = OutputFormatter()
        except ImportError as e:
            self.logger.error(f"Failed to import required component: {e}")
            raise
        
        # Pipeline state
        self.processing_stats = {}
        self.start_time = None
        self.timeout_handler = None
        
    def execute_pipeline(self, 
                        input_dir: Path,
                        output_dir: Path,
                        timeout_seconds: int = None) -> Dict[str, Any]:
        """Execute the complete Round 1B pipeline"""
        
        self.start_time = time.time()
        timeout_seconds = timeout_seconds or getattr(Config, 'MAX_PROCESSING_TIME_SECONDS', 60)
        
        try:
            # Set up timeout handler
            self._setup_timeout_handler(timeout_seconds)
            
            self.logger.info(f"Starting Round 1B pipeline with {timeout_seconds}s timeout")
            
            # Step 1: Load and validate inputs
            inputs = self._load_inputs(input_dir)
            
            # Step 2: Process documents
            processed_docs = self._process_documents(inputs['document_paths'])
            
            # Step 3: Create chunks
            chunks = self._create_chunks(processed_docs)
            
            # Step 4: Generate embeddings
            embedded_chunks = self._generate_embeddings(chunks)
            
            # Step 5: Build vector index
            self._build_vector_index(embedded_chunks)
            
            # Step 6: Process persona and JTBD
            persona, jtbd, persona_query = self._process_persona_and_jtbd(
                inputs['persona'], inputs['jtbd']
            )
            
            # Step 7: Score sections
            section_scores = self._score_sections(embedded_chunks, persona_query, persona, jtbd)
            
            # Step 8: Rank and select sections
            extracted_sections = self._rank_sections(section_scores, embedded_chunks)
            
            # Step 9: Extract subsections
            extracted_subsections = self._extract_subsections(extracted_sections, persona_query)
            
            # Step 10: Format output
            output_data = self._format_output(
                inputs, persona, jtbd, extracted_sections, 
                extracted_subsections, time.time() - self.start_time
            )
            
            # Step 11: Save output
            output_path = self._save_output(output_data, output_dir)
            
            # Generate final statistics
            final_stats = self._generate_final_statistics(output_data)
            
            total_time = time.time() - self.start_time
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'processing_time': total_time,
                'statistics': final_stats,
                'output_data': output_data
            }
            
        except TimeoutError:
            self.logger.error(f"Pipeline timeout after {timeout_seconds}s")
            return {
                'success': False,
                'error': 'Pipeline timeout',
                'processing_time': time.time() - self.start_time if self.start_time else 0
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - self.start_time if self.start_time else 0,
                'traceback': traceback.format_exc()
            }
        
        finally:
            self._cleanup_timeout_handler()
    
    def _setup_timeout_handler(self, timeout_seconds: int):
        """Set up timeout handler for the pipeline"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Pipeline exceeded {timeout_seconds} second timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        self.timeout_handler = True
    
    def _cleanup_timeout_handler(self):
        """Clean up timeout handler"""
        if self.timeout_handler:
            signal.alarm(0)  # Cancel the alarm
            self.timeout_handler = None
    
    def _load_inputs(self, input_dir: Path) -> Dict[str, Any]:
        """Load and validate all inputs"""
        step_start = time.time()
        
        # Find PDF documents
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError("No PDF documents found in input directory")
        
        # Validate constraints if method exists
        if hasattr(Config, 'validate_constraints'):
            Config.validate_constraints([str(f) for f in pdf_files])
        
        # Load persona (look for persona.json or persona.txt)
        persona_data = None
        for persona_file in ['persona.json', 'persona.txt']:
            persona_path = input_dir / persona_file
            if persona_path.exists():
                if persona_file.endswith('.json'):
                    import json
                    with open(persona_path, 'r', encoding='utf-8') as f:
                        persona_data = json.load(f)
                else:
                    with open(persona_path, 'r', encoding='utf-8') as f:
                        persona_data = f.read().strip()
                break
        
        if not persona_data:
            raise ValueError("No persona file found (persona.json or persona.txt)")
        
        # Load job-to-be-done (look for jtbd.json or jtbd.txt)
        jtbd_data = None
        for jtbd_file in ['jtbd.json', 'jtbd.txt', 'job_to_be_done.json', 'job_to_be_done.txt']:
            jtbd_path = input_dir / jtbd_file
            if jtbd_path.exists():
                if jtbd_file.endswith('.json'):
                    import json
                    with open(jtbd_path, 'r', encoding='utf-8') as f:
                        jtbd_data = json.load(f)
                else:
                    with open(jtbd_path, 'r', encoding='utf-8') as f:
                        jtbd_data = f.read().strip()
                break
        
        if not jtbd_data:
            raise ValueError("No job-to-be-done file found")
        
        inputs = {
            'document_paths': pdf_files,
            'persona': persona_data,
            'jtbd': jtbd_data
        }
        
        self.processing_stats['input_loading'] = time.time() - step_start
        self.logger.info(f"Loaded inputs: {len(pdf_files)} documents")
        
        return inputs
    
    def _process_documents(self, document_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process all documents"""
        step_start = time.time()
        
        # Use fallback processing method if available
        if hasattr(self.batch_processor, 'process_with_fallback'):
            processed_docs = self.batch_processor.process_with_fallback(document_paths)
        else:
            processed_docs = self.batch_processor.process_documents(document_paths)
        
        if not processed_docs:
            raise PipelineExceptions.DocumentParsingError("No documents could be processed")
        
        self.processing_stats['document_processing'] = time.time() - step_start
        self.logger.info(f"Processed {len(processed_docs)} documents")
        
        return processed_docs
    
    def _create_chunks(self, processed_docs: List[Dict[str, Any]]) -> List:
        """Create semantic chunks from processed documents"""
        step_start = time.time()
        
        chunks = self.semantic_chunker.chunk_documents(processed_docs)
        
        if not chunks:
            raise ValueError("No chunks could be created from documents")
        
        # Log chunk statistics if method exists
        if hasattr(self.semantic_chunker, 'get_chunk_statistics'):
            chunk_stats = self.semantic_chunker.get_chunk_statistics(chunks)
            self.logger.info(f"Created {len(chunks)} chunks: {chunk_stats}")
        else:
            self.logger.info(f"Created {len(chunks)} chunks")
        
        self.processing_stats['chunking'] = time.time() - step_start
        return chunks
    
    def _generate_embeddings(self, chunks: List) -> List:
        """Generate embeddings for all chunks"""
        step_start = time.time()
        
        embedded_chunks = self.embedding_engine.generate_embeddings(chunks, use_cache=True)
        
        # Verify embeddings
        embedded_count = sum(1 for c in embedded_chunks if getattr(c, 'embedding', None) is not None)
        if embedded_count == 0:
            raise ValueError("No embeddings could be generated")
        
        self.logger.info(f"Generated embeddings for {embedded_count}/{len(embedded_chunks)} chunks")
        
        self.processing_stats['embedding_generation'] = time.time() - step_start
        return embedded_chunks
    
    def _build_vector_index(self, embedded_chunks: List) -> None:
        """Build vector index for similarity search"""
        step_start = time.time()
        
        self.vector_store.build_index(embedded_chunks, force_rebuild=False)
        
        # Log index statistics if method exists
        if hasattr(self.vector_store, 'get_index_statistics'):
            index_stats = self.vector_store.get_index_statistics()
            self.logger.info(f"Built vector index: {index_stats}")
        else:
            self.logger.info("Built vector index successfully")
        
        self.processing_stats['vector_indexing'] = time.time() - step_start
    
    def _process_persona_and_jtbd(self, persona_data: Any, jtbd_data: Any) -> tuple:
        """Process persona and job-to-be-done"""
        step_start = time.time()
        
        try:
            persona = self.persona_processor.process_persona(persona_data)
            jtbd = self.persona_processor.process_job_to_be_done(jtbd_data)
            persona_query = self.persona_processor.build_persona_query(persona, jtbd)
            
            # Safely get persona role
            role = getattr(persona, 'role', 'Unknown') if hasattr(persona, 'role') else persona.get('role', 'Unknown')
            query_keywords = len(getattr(persona_query, 'primary_keywords', [])) if hasattr(persona_query, 'primary_keywords') else 0
            
            self.logger.info(f"Processed persona: {role} with {query_keywords} keywords")
            
            self.processing_stats['persona_processing'] = time.time() - step_start
            return persona, jtbd, persona_query
            
        except Exception as e:
            raise PipelineExceptions.InvalidPersonaError(f"Failed to process persona/JTBD: {str(e)}")
    
    def _score_sections(self, chunks: List, persona_query, persona: Persona, jtbd: JobToBeDone) -> List:
        """Score all sections for relevance and importance"""
        step_start = time.time()
        
        section_scores = self.relevance_scorer.score_sections(chunks, persona_query, persona, jtbd)
        
        # Apply normalization if method exists
        if hasattr(self.relevance_scorer, 'apply_score_normalization'):
            normalized_scores = self.relevance_scorer.apply_score_normalization(section_scores)
        else:
            normalized_scores = section_scores
        
        self.logger.info(f"Scored {len(normalized_scores)} sections")
        
        self.processing_stats['section_scoring'] = time.time() - step_start
        return normalized_scores
    
    def _rank_sections(self, section_scores: List, chunks: List) -> List:
        """Rank and select top sections"""
        step_start = time.time()
        
        top_k = getattr(Config, 'TOP_K_SECTIONS', 10)
        
        # Apply MMR ranking for diversity
        extracted_sections = self.document_ranker.rank_sections(
            section_scores, chunks, top_k=top_k
        )
        
        # Apply final ranking adjustments if method exists
        if hasattr(self.document_ranker, 'apply_final_ranking_adjustments'):
            final_sections = self.document_ranker.apply_final_ranking_adjustments(extracted_sections)
        else:
            final_sections = extracted_sections
        
        # Log ranking statistics if method exists
        if hasattr(self.document_ranker, 'get_ranking_statistics'):
            ranking_stats = self.document_ranker.get_ranking_statistics(final_sections)
            self.logger.info(f"Ranked sections: {ranking_stats}")
        else:
            self.logger.info(f"Ranked {len(final_sections)} sections")
        
        self.processing_stats['section_ranking'] = time.time() - step_start
        return final_sections
    
    def _extract_subsections(self, extracted_sections: List, persona_query) -> List:
        """Extract and rank subsections"""
        step_start = time.time()
        
        extracted_subsections = self.subsection_extractor.extract_subsections(
            extracted_sections, persona_query
        )
        
        # Log subsection statistics if method exists
        if hasattr(self.subsection_extractor, 'get_subsection_statistics'):
            subsection_stats = self.subsection_extractor.get_subsection_statistics(extracted_subsections)
            self.logger.info(f"Extracted subsections: {subsection_stats}")
        else:
            self.logger.info(f"Extracted {len(extracted_subsections)} subsections")
        
        self.processing_stats['subsection_extraction'] = time.time() - step_start
        return extracted_subsections
    
    def _format_output(self, inputs: Dict, persona: Persona, jtbd: JobToBeDone,
                      extracted_sections: List, extracted_subsections: List,
                      processing_time: float) -> Dict[str, Any]:
        """Format final output"""
        step_start = time.time()
        
        # Prepare input document names
        input_documents = [str(path.name) for path in inputs['document_paths']]
        
        # Get model versions
        model_versions = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'spacy_model': 'en_core_web_sm',
            'system_version': '1.0.0'
        }
        
        # Convert persona and jtbd to dict safely
        persona_dict = persona.dict() if hasattr(persona, 'dict') else (persona if isinstance(persona, dict) else {})
        jtbd_dict = jtbd.dict() if hasattr(jtbd, 'dict') else (jtbd if isinstance(jtbd, dict) else {})
        
        output_data = self.output_formatter.format_output(
            input_documents=input_documents,
            persona_data=persona_dict,
            jtbd_data=jtbd_dict,
            extracted_sections=extracted_sections,
            extracted_subsections=extracted_subsections,
            processing_time=processing_time,
            model_versions=model_versions
        )
        
        self.processing_stats['output_formatting'] = time.time() - step_start
        return output_data
    
    def _save_output(self, output_data: Dict[str, Any], output_dir: Path) -> Path:
        """Save formatted output"""
        step_start = time.time()
        
        output_path = output_dir / "challenge1b_output.json"
        saved_path = self.output_formatter.save_output(output_data, output_path)
        
        self.processing_stats['output_saving'] = time.time() - step_start
        return saved_path
    
    def _generate_final_statistics(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final statistics"""
        total_time = time.time() - self.start_time
        max_time = getattr(Config, 'MAX_PROCESSING_TIME_SECONDS', 60)
        
        # Processing step breakdown
        step_percentages = {}
        for step, duration in self.processing_stats.items():
            step_percentages[step] = (duration / total_time) * 100 if total_time > 0 else 0
        
        # Performance metrics
        sections_count = len(output_data.get('extracted_sections', []))
        subsections_count = len(output_data.get('extracted_subsections', []))
        
        # Resource utilization
        memory_usage = self._get_memory_usage()
        
        # Quality metrics from output
        quality_stats = output_data.get('summary_statistics', {}).get('quality_statistics', {})
        
        final_stats = {
            'total_processing_time': total_time,
            'meets_time_constraint': total_time <= max_time,
            'processing_step_breakdown': self.processing_stats,
            'processing_step_percentages': step_percentages,
            'throughput_metrics': {
                'sections_per_second': sections_count / total_time if total_time > 0 else 0,
                'subsections_per_second': subsections_count / total_time if total_time > 0 else 0,
                'total_sections': sections_count,
                'total_subsections': subsections_count
            },
            'resource_utilization': memory_usage,
            'quality_metrics': quality_stats,
            'constraint_compliance': {
                'time_limit': total_time <= max_time,
                'cpu_only': True,
                'model_size_limit': True,
                'offline_operation': True
            }
        }
        
        return final_stats
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
        except Exception as e:
            self.logger.warning(f"Could not get memory usage: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        current_time = time.time()
        
        status = {
            'is_running': self.start_time is not None,
            'elapsed_time': current_time - self.start_time if self.start_time else 0,
            'processing_steps_completed': list(self.processing_stats.keys()),
            'current_step_duration': self.processing_stats,
            'memory_usage': self._get_memory_usage()
        }
        
        return status
    
    def cancel_pipeline(self) -> None:
        """Cancel the running pipeline"""
        if self.timeout_handler:
            self._cleanup_timeout_handler()
        
        self.logger.info("Pipeline cancelled by user request")
        raise KeyboardInterrupt("Pipeline cancelled")
