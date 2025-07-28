from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    CACHE_DIR = DATA_DIR / "cache"
    OUTPUT_DIR = DATA_DIR / "output"
    
    # Processing parameters
    MAX_PROCESSING_TIME_SECONDS = 60
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    TOP_K_SECTIONS = 10
    TOP_K_SUBSECTIONS = 15
    SIMILARITY_THRESHOLD = 0.3
    MAX_SECTIONS_PER_DOC = 5
    
    # Model configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"
    
    @staticmethod
    def validate_constraints(file_paths: list) -> None:
        """Validate Round 1B constraints"""
        if len(file_paths) < 3:
            raise ValueError(f"At least 3 PDF files required, got {len(file_paths)}")
        
        if len(file_paths) > 10:
            raise ValueError(f"Maximum 10 PDF files allowed, got {len(file_paths)}")
        
        # Check file sizes
        total_size_mb = 0
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                raise ValueError(f"File not found: {file_path}")
            
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            
            if size_mb > 50:  # 50MB per file limit
                raise ValueError(f"File too large: {file_path} ({size_mb:.1f}MB)")
        
        if total_size_mb > 200:  # 200MB total limit
            raise ValueError(f"Total files too large: {total_size_mb:.1f}MB")

class ModelConfig:
    """Model-specific configuration"""
    MODELS = {
        "embeddings": {
            "all-MiniLM-L6-v2": {
                "path": "models/embeddings/sentence-transformer/all-MiniLM-L6-v2",
                "size_mb": 23,
                "dimensions": 384
            }
        },
        "spacy": {
            "en_core_web_sm": {
                "path": "models/nlp/en_core_web_sm",
                "size_mb": 15
            }
        },
        "llm": {
            "name": "llama-3.2-1b-q4.gguf",
            "path": "models/llm/llama-3.2-1b-q4.gguf",
            "optional": True
        }
    }
