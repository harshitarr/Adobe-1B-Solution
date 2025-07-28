#!/usr/bin/env python3
"""
Setup script for Round 1B - Download models and validate environment
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
import urllib.request
import tarfile
import zipfile
import yaml

# Add project root to Python path (not src directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import from src module
from src.core.config import Config, ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Download and setup required models"""
    
    def __init__(self):
        self.models_dir = Config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_sentence_transformer(self):
        """Download sentence transformer model"""
        logger.info("Setting up sentence transformer model...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = "all-MiniLM-L6-v2"
            model_path = self.models_dir / "embeddings" / "sentence-transformer" / model_name
            
            # Check if model already exists with proper validation
            if model_path.exists() and (model_path / "config.json").exists():
                logger.info(f"Model {model_name} already exists and is valid")
                try:
                    # Verify the model can actually be loaded
                    test_model = SentenceTransformer(str(model_path))
                    logger.info(f"Model {model_name} verified and working")
                    return True
                except Exception as e:
                    logger.warning(f"Existing model is corrupted, re-downloading: {e}")
                    # Continue to download fresh model
            
            logger.info(f"Downloading {model_name}...")
            
            # Download the model from HuggingFace
            model = SentenceTransformer(model_name)
            
            # Ensure directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model to our custom location
            model.save(str(model_path))
            
            # Verify the model was saved correctly with all required files
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            missing_files = []
            
            for req_file in required_files:
                if not (model_path / req_file).exists():
                    missing_files.append(req_file)
            
            if missing_files:
                logger.error(f"Model saved but missing files: {missing_files}")
                return False
            
            # Final verification: try to load the saved model
            try:
                verification_model = SentenceTransformer(str(model_path))
                logger.info(f"Model saved and verified successfully at {model_path}")
                return True
            except Exception as e:
                logger.error(f"Model saved but failed verification load: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to download sentence transformer: {e}")
            return False
    
    def download_spacy_model(self):
        """Download spaCy model"""
        logger.info("Setting up spaCy model...")
        
        try:
            model_name = "en_core_web_sm"
            
            # Try to import first
            try:
                import spacy
                nlp = spacy.load(model_name)
                logger.info(f"spaCy model {model_name} already available")
                return True
            except OSError:
                pass
            
            # Download model
            logger.info(f"Downloading spaCy model {model_name}...")
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", model_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("spaCy model downloaded successfully")
                
                # Verify the model can be loaded
                try:
                    import spacy
                    nlp = spacy.load(model_name)
                    logger.info("spaCy model verified and working")
                    return True
                except Exception as e:
                    logger.error(f"spaCy model downloaded but cannot be loaded: {e}")
                    return False
            else:
                logger.error(f"Failed to download spaCy model: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup spaCy model: {e}")
            return False
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        logger.info("Setting up NLTK data...")
        
        try:
            import nltk
            
            # Set download directory
            nltk_data_dir = self.models_dir / "nlp" / "nltk_data"
            nltk_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Add the custom path BEFORE downloading (insert at beginning for priority)
            nltk.data.path.insert(0, str(nltk_data_dir))
            
            required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']
            
            for data_name in required_data:
                try:
                    nltk.data.find(f'tokenizers/{data_name}')
                    logger.info(f"NLTK data {data_name} already available")
                except LookupError:
                    try:
                        nltk.data.find(f'corpora/{data_name}')
                        logger.info(f"NLTK data {data_name} already available")
                    except LookupError:
                        try:
                            nltk.data.find(f'taggers/{data_name}')
                            logger.info(f"NLTK data {data_name} already available")
                        except LookupError:
                            logger.info(f"Downloading NLTK data: {data_name}")
                            nltk.download(data_name, download_dir=str(nltk_data_dir))
            
            # Create a symlink to standard location as backup
            import os
            standard_nltk_dir = Path("/root/nltk_data")
            if not standard_nltk_dir.exists():
                try:
                    os.symlink(str(nltk_data_dir), str(standard_nltk_dir))
                    logger.info(f"Created symlink from {standard_nltk_dir} to {nltk_data_dir}")
                except Exception as e:
                    logger.warning(f"Could not create symlink: {e}")
            
            # Also create environment variable file for runtime
            env_file = Path("/app/.env")
            try:
                with open(env_file, "w") as f:
                    f.write(f"NLTK_DATA={nltk_data_dir}\n")
                logger.info(f"Created environment file at {env_file}")
            except Exception as e:
                logger.warning(f"Could not create env file: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup NLTK data: {e}")
            return False
    
    def setup_optional_llm(self):
        """Setup optional LLM (if space allows)"""
        logger.info("Checking optional LLM setup...")
        
        llm_config = ModelConfig.MODELS.get('llm', {})
        if not llm_config.get('optional', True):
            return True
        
        model_path = self.models_dir / llm_config.get('path', 'models/llm/llama-3.2-1b-q4.gguf')
        
        if model_path.exists():
            logger.info("Optional LLM already exists")
            return True
        
        # Calculate available space
        try:
            import shutil
            free_space_gb = shutil.disk_usage(self.models_dir).free / (1024**3)
            
            if free_space_gb < 1.0:  # Need at least 1GB for LLM
                logger.info("Insufficient space for optional LLM, skipping...")
                return True
            
            logger.info("Optional LLM setup skipped (would require manual download)")
            return True
            
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True

def validate_python_version():
    """Validate Python version"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install Python requirements"""
    logger.info("Installing Python requirements...")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.warning("requirements.txt not found, skipping...")
        return True
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Requirements installed successfully")
            return True
        else:
            logger.error(f"Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directory structure...")
    
    directories = [
        Config.MODELS_DIR,
        Config.DATA_DIR,
        Config.CACHE_DIR,
        Config.OUTPUT_DIR,
        Config.DATA_DIR / "input",
        Config.DATA_DIR / "temp",
        Path("logs")
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    
    return True

def validate_installation():
    """Validate the installation"""
    logger.info("Validating installation...")
    
    try:
        # Test imports
        import sentence_transformers
        import spacy
        import nltk
        import torch
        import numpy
        import pandas
        import sklearn
        import pydantic
        
        logger.info("All required packages imported successfully")
        
        # Test model loading with proper paths
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try loading from our custom location first
            custom_model_path = Config.MODELS_DIR / "embeddings" / "sentence-transformer" / "all-MiniLM-L6-v2"
            if custom_model_path.exists():
                model = SentenceTransformer(str(custom_model_path))
                logger.info("Sentence transformer model loaded successfully from custom location")
            else:
                # Fallback to downloading fresh
                model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence transformer model loaded successfully from HuggingFace")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
        
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        return False
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("=== Round 1B Setup Script ===")
    
    # Validate Python version
    if not validate_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install requirements (only if requirements.txt exists)
    if not install_requirements():
        logger.warning("Requirements installation failed, continuing...")
    
    # Setup models
    downloader = ModelDownloader()
    
    success = True
    
    if not downloader.download_sentence_transformer():
        logger.error("Failed to setup sentence transformer")
        success = False
    
    if not downloader.download_spacy_model():
        logger.error("Failed to setup spaCy model")
        success = False
    
    if not downloader.download_nltk_data():
        logger.error("Failed to setup NLTK data")
        success = False
    
    # Optional LLM setup
    downloader.setup_optional_llm()
    
    # Validate installation
    if not validate_installation():
        logger.error("Installation validation failed")
        success = False
    
    if success:
        logger.info("=== Setup completed successfully ===")
        return 0
    else:
        logger.error("=== Setup completed with errors ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())
