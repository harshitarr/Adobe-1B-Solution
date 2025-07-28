from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path

class DocumentMetadata(BaseModel):
    """Metadata for a processed document"""
    filename: str
    file_size: int
    page_count: int
    processing_time: float
    document_type: str = "pdf"  # Add default value
    creation_date: Optional[datetime] = None
    title: Optional[str] = None
    author: Optional[str] = None
    language: str = "en"

class ChunkMetadata(BaseModel):
    """Metadata for a document chunk"""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_page: int
    end_page: int
    chunk_type: str = "content"
    word_count: int = 0
    character_count: int = 0
    creation_timestamp: datetime = Field(default_factory=datetime.now)

class ProcessedDocument(BaseModel):
    """Represents a processed document with extracted content"""
    document_id: str
    metadata: DocumentMetadata
    pages: List[Dict[str, Any]]
    full_text: str
    structure: Dict[str, Any]
    
class Chunk(BaseModel):
    """Represents a semantic chunk of document content"""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    start_page: int
    end_page: int
    chunk_type: str = "content"
    metadata: ChunkMetadata  # Changed from Dict to ChunkMetadata
    embedding: Optional[List[float]] = None
    
class Persona(BaseModel):
    """User persona for personalized document intelligence"""
    role: str
    expertise_areas: List[str]
    experience_level: str
    domain: str
    priorities: List[str] = Field(default_factory=list)
    
class JobToBeDone(BaseModel):
    """Job-to-be-done specification"""
    task_description: str
    expected_outcomes: List[str] = Field(default_factory=list)
    priority_keywords: List[str] = Field(default_factory=list)
    context: str = ""
    urgency: str = "medium"
    
class ExtractedSection(BaseModel):
    """Extracted document section with importance ranking"""
    document_reference: str
    page_number: int
    section_title: str
    content: str
    importance_rank: int = Field(ge=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    section_type: str = "content"
    
class ExtractedSubsection(BaseModel):
    """Refined subsection with relevance scoring"""
    document_reference: str
    page_number: int
    refined_text: str
    parent_section: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    
class ProcessingMetadata(BaseModel):
    """Processing metadata for the pipeline"""
    input_documents: List[str]
    persona: Dict[str, Any]
    job_to_be_done: Dict[str, Any]
    processing_timestamp: datetime
    total_processing_time: float
    model_versions: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        # Fix the Pydantic warning about model_versions
        protected_namespaces = ()
    
class Challenge1BOutput(BaseModel):
    """Complete Round 1B output schema"""
    metadata: ProcessingMetadata
    extracted_sections: List[ExtractedSection]
    extracted_subsections: List[ExtractedSubsection]
    summary_statistics: Dict[str, Any]
    
    class Config:
        protected_namespaces = ()
