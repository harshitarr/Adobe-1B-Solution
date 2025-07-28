import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from ..core.schemas import Challenge1BOutput, ProcessingMetadata, ExtractedSection, ExtractedSubsection
from ..core.config import Config

class OutputFormatter:
    """JSON formatting with schema validation, metadata, and timestamps"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_output(self, 
                     input_documents: List[str],
                     persona_data: Dict[str, Any],
                     jtbd_data: Dict[str, Any],
                     extracted_sections: List[ExtractedSection],
                     extracted_subsections: List[ExtractedSubsection],
                     processing_time: float,
                     model_versions: Dict[str, str] = None) -> Dict[str, Any]:
        """Format complete output according to Round 1B schema"""
        
        # Create processing metadata
        metadata = ProcessingMetadata(
            input_documents=input_documents,
            persona=persona_data,
            job_to_be_done=jtbd_data,
            processing_timestamp=datetime.now(),
            total_processing_time=processing_time,
            model_versions=model_versions or self._get_default_model_versions()
        )
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(
            extracted_sections, extracted_subsections, processing_time
        )
        
        # Create main output object
        output = Challenge1BOutput(
            metadata=metadata,
            extracted_sections=extracted_sections,
            extracted_subsections=extracted_subsections,
            summary_statistics=summary_stats
        )
        
        # Convert to dictionary for JSON serialization
        output_dict = self._to_serializable_dict(output)
        
        # Validate output
        self._validate_output(output_dict)
        
        self.logger.info(f"Formatted output with {len(extracted_sections)} sections and {len(extracted_subsections)} subsections")
        
        return output_dict
    
    def _get_default_model_versions(self) -> Dict[str, str]:
        """Get default model versions"""
        return {
            'embedding_model': 'all-MiniLM-L6-v2',
            'spacy_model': 'en_core_web_sm',
            'system_version': '1.0.0'
        }
    
    def _calculate_summary_statistics(self, 
                                    extracted_sections: List[ExtractedSection],
                                    extracted_subsections: List[ExtractedSubsection],
                                    processing_time: float) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        
        # Document coverage
        documents_in_sections = set(s.document_reference for s in extracted_sections)
        documents_in_subsections = set(s.document_reference for s in extracted_subsections)
        
        # Section statistics
        section_stats = {
            'total_sections': len(extracted_sections),
            'documents_covered': len(documents_in_sections),
            'avg_confidence_score': sum(s.confidence_score for s in extracted_sections) / len(extracted_sections) if extracted_sections else 0,
            'confidence_distribution': self._calculate_confidence_distribution(extracted_sections),
            'section_length_stats': self._calculate_length_statistics([s.content for s in extracted_sections])
        }
        
        # Subsection statistics
        subsection_stats = {
            'total_subsections': len(extracted_subsections),
            'documents_covered': len(documents_in_subsections),
            'avg_relevance_score': sum(s.relevance_score for s in extracted_subsections) / len(extracted_subsections) if extracted_subsections else 0,
            'relevance_distribution': self._calculate_relevance_distribution(extracted_subsections),
            'subsection_length_stats': self._calculate_length_statistics([s.refined_text for s in extracted_subsections])
        }
        
        # Page coverage
        pages_covered = set()
        for section in extracted_sections:
            pages_covered.add(f"{section.document_reference}_page_{section.page_number}")
        
        # Parent section diversity
        parent_sections = set(s.parent_section for s in extracted_subsections)
        
        # Performance metrics
        performance_stats = {
            'processing_time_seconds': processing_time,
            'sections_per_second': len(extracted_sections) / processing_time if processing_time > 0 else 0,
            'subsections_per_second': len(extracted_subsections) / processing_time if processing_time > 0 else 0,
            'meets_time_constraint': processing_time <= Config.MAX_PROCESSING_TIME_SECONDS
        }
        
        # Quality metrics
        quality_stats = {
            'section_ranking_quality': self._calculate_ranking_quality(extracted_sections),
            'content_diversity_score': len(parent_sections) / len(extracted_subsections) if extracted_subsections else 0,
            'document_coverage_ratio': len(documents_in_sections) / len(set(documents_in_sections | documents_in_subsections)) if documents_in_sections or documents_in_subsections else 0
        }
        
        return {
            'section_statistics': section_stats,
            'subsection_statistics': subsection_stats,
            'coverage_statistics': {
                'total_pages_covered': len(pages_covered),
                'parent_sections_diversity': len(parent_sections)
            },
            'performance_statistics': performance_stats,
            'quality_statistics': quality_stats,
            'extraction_efficiency': {
                'sections_per_document': len(extracted_sections) / len(documents_in_sections) if documents_in_sections else 0,
                'subsections_per_section': len(extracted_subsections) / len(extracted_sections) if extracted_sections else 0
            }
        }
    
    def _calculate_confidence_distribution(self, sections: List[ExtractedSection]) -> Dict[str, int]:
        """Calculate confidence score distribution"""
        if not sections:
            return {}
        
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for section in sections:
            if section.confidence_score >= 0.7:
                distribution['high'] += 1
            elif section.confidence_score >= 0.4:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _calculate_relevance_distribution(self, subsections: List[ExtractedSubsection]) -> Dict[str, int]:
        """Calculate relevance score distribution"""
        if not subsections:
            return {}
        
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for subsection in subsections:
            if subsection.relevance_score >= 0.7:
                distribution['high'] += 1
            elif subsection.relevance_score >= 0.4:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _calculate_length_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate text length statistics"""
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        return {
            'avg_characters': sum(lengths) / len(lengths),
            'avg_words': sum(word_counts) / len(word_counts),
            'min_characters': min(lengths),
            'max_characters': max(lengths),
            'min_words': min(word_counts),
            'max_words': max(word_counts)
        }
    
    def _calculate_ranking_quality(self, sections: List[ExtractedSection]) -> float:
        """Calculate ranking quality based on confidence score ordering"""
        if len(sections) < 2:
            return 1.0
        
        correct_order = 0
        total_pairs = 0
        
        for i in range(len(sections) - 1):
            for j in range(i + 1, len(sections)):
                total_pairs += 1
                # Check if higher rank (lower number) has higher confidence
                if (sections[i].importance_rank < sections[j].importance_rank and 
                    sections[i].confidence_score >= sections[j].confidence_score):
                    correct_order += 1
                elif (sections[i].importance_rank > sections[j].importance_rank and 
                      sections[i].confidence_score <= sections[j].confidence_score):
                    correct_order += 1
        
        return correct_order / total_pairs if total_pairs > 0 else 1.0
    
    def _to_serializable_dict(self, output: Challenge1BOutput) -> Dict[str, Any]:
        """Convert Pydantic model to JSON-serializable dictionary"""
        
        def convert_value(value):
            """Recursively convert values to JSON-serializable format"""
            if isinstance(value, datetime):
                return value.isoformat()
            elif hasattr(value, 'dict'):  # Pydantic model
                return value.dict()
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {key: convert_value(val) for key, val in value.items()}
            else:
                return value
        
        return convert_value(output.dict())
    
    def _validate_output(self, output_dict: Dict[str, Any]) -> None:
        """Validate output against schema requirements"""
        errors = []
        
        # Check required top-level fields
        required_fields = ['metadata', 'extracted_sections', 'extracted_subsections', 'summary_statistics']
        for field in required_fields:
            if field not in output_dict:
                errors.append(f"Missing required field: {field}")
        
        # Validate metadata
        if 'metadata' in output_dict:
            metadata = output_dict['metadata']
            metadata_required = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
            for field in metadata_required:
                if field not in metadata:
                    errors.append(f"Missing metadata field: {field}")
        
        # Validate sections
        if 'extracted_sections' in output_dict:
            sections = output_dict['extracted_sections']
            if not isinstance(sections, list):
                errors.append("extracted_sections must be a list")
            elif len(sections) == 0:
                errors.append("At least one section must be extracted")
            else:
                for i, section in enumerate(sections):
                    required_section_fields = ['document_reference', 'page_number', 'section_title', 'content', 'importance_rank']
                    for field in required_section_fields:
                        if field not in section:
                            errors.append(f"Missing field '{field}' in section {i}")
                    
                    # Validate importance_rank
                    if 'importance_rank' in section and section['importance_rank'] < 1:
                        errors.append(f"importance_rank must be >= 1 in section {i}")
        
        # Validate subsections
        if 'extracted_subsections' in output_dict:
            subsections = output_dict['extracted_subsections']
            if not isinstance(subsections, list):
                errors.append("extracted_subsections must be a list")
            else:
                for i, subsection in enumerate(subsections):
                    required_subsection_fields = ['document_reference', 'page_number', 'refined_text', 'parent_section']
                    for field in required_subsection_fields:
                        if field not in subsection:
                            errors.append(f"Missing field '{field}' in subsection {i}")
        
        if errors:
            error_msg = f"Output validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Output validation passed")
    
    def save_output(self, output_dict: Dict[str, Any], output_path: Path = None) -> Path:
        """Save formatted output to JSON file"""
        if output_path is None:
            output_path = Config.OUTPUT_DIR / "challenge1b_output.json"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Output saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save output: {str(e)}")
            raise
    
    def load_output(self, output_path: Path) -> Dict[str, Any]:
        """Load and validate output from JSON file"""
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                output_dict = json.load(f)
            
            # Validate the loaded output
            self._validate_output(output_dict)
            
            self.logger.info(f"Output loaded from {output_path}")
            return output_dict
            
        except Exception as e:
            self.logger.error(f"Failed to load output: {str(e)}")
            raise
    
    def generate_summary_report(self, output_dict: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        try:
            sections = output_dict.get('extracted_sections', [])
            subsections = output_dict.get('extracted_subsections', [])
            stats = output_dict.get('summary_statistics', {})
            metadata = output_dict.get('metadata', {})
            
            report_lines = [
                "# Round 1B Processing Summary Report",
                f"**Generated:** {metadata.get('processing_timestamp', 'Unknown')}",
                f"**Processing Time:** {stats.get('performance_statistics', {}).get('processing_time_seconds', 0):.2f} seconds",
                "",
                "## Input Information",
                f"**Documents Processed:** {len(metadata.get('input_documents', []))}",
                f"**Persona:** {metadata.get('persona', {}).get('role', 'Unknown')}",
                f"**Experience Level:** {metadata.get('persona', {}).get('experience_level', 'Unknown')}",
                f"**Task:** {metadata.get('job_to_be_done', {}).get('task_description', 'Not specified')[:100]}...",
                "",
                "## Extraction Results",
                f"**Sections Extracted:** {len(sections)}",
                f"**Subsections Extracted:** {len(subsections)}",
                f"**Documents Covered:** {stats.get('section_statistics', {}).get('documents_covered', 0)}",
                f"**Pages Covered:** {stats.get('coverage_statistics', {}).get('total_pages_covered', 0)}",
                "",
                "## Quality Metrics",
                f"**Average Section Confidence:** {stats.get('section_statistics', {}).get('avg_confidence_score', 0):.3f}",
                f"**Average Subsection Relevance:** {stats.get('subsection_statistics', {}).get('avg_relevance_score', 0):.3f}",
                f"**Content Diversity Score:** {stats.get('quality_statistics', {}).get('content_diversity_score', 0):.3f}",
                f"**Ranking Quality:** {stats.get('quality_statistics', {}).get('section_ranking_quality', 0):.3f}",
                "",
                "## Performance",
                f"**Time Constraint Met:** {'✓' if stats.get('performance_statistics', {}).get('meets_time_constraint', False) else '✗'}",
                f"**Sections/Second:** {stats.get('performance_statistics', {}).get('sections_per_second', 0):.2f}",
                f"**Subsections/Second:** {stats.get('performance_statistics', {}).get('subsections_per_second', 0):.2f}",
                "",
                "## Top Extracted Sections",
            ]
            
            # Add top 5 sections
            for i, section in enumerate(sections[:5]):
                report_lines.extend([
                    f"### {i+1}. {section.get('section_title', 'Untitled')}",
                    f"**Document:** {section.get('document_reference', 'Unknown')}",
                    f"**Page:** {section.get('page_number', 'Unknown')}",
                    f"**Confidence:** {section.get('confidence_score', 0):.3f}",
                    f"**Content Preview:** {section.get('content', '')[:200]}...",
                    ""
                ])
            
            return '\n'.join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {str(e)}")
            return f"Error generating report: {str(e)}"
