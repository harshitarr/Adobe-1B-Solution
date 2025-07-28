#!/usr/bin/env python3
"""
Round 1B Persona-Driven Document Intelligence
Entry point for the complete system
"""

import sys
import logging
import argparse
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.manager import PipelineManager
from src.core.config import Config

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "processing.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Round 1B Persona-Driven Document Intelligence"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/app/input"),
        help="Input directory containing PDFs and persona/JTBD files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path("/app/output"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=Config.MAX_PROCESSING_TIME_SECONDS,
        help="Processing timeout in seconds"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Custom configuration file"
    )
    
    return parser.parse_args()

def validate_environment():
    """Validate the runtime environment"""
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append("Python 3.8+ is required")
    
    # Check essential directories
    essential_dirs = [
        Config.MODELS_DIR,
        Config.DATA_DIR,
        Config.CACHE_DIR
    ]
    
    for dir_path in essential_dirs:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")
    
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return False
    
    return True

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Round 1B Persona-Driven Document Intelligence")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Timeout: {args.timeout} seconds")
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            return 1
        
        # Validate inputs
        if not args.input_dir.exists():
            logger.error(f"Input directory does not exist: {args.input_dir}")
            return 1
        
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        pipeline = PipelineManager()
        
        # Execute pipeline
        result = pipeline.execute_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout
        )
        
        if result['success']:
            logger.info("Pipeline completed successfully!")
            logger.info(f"Output saved to: {result['output_path']}")
            logger.info(f"Processing time: {result['processing_time']:.2f}s")
            
            # Print summary statistics
            stats = result.get('statistics', {})
            if stats:
                logger.info("Performance Summary:")
                logger.info(f"  - Sections extracted: {stats.get('throughput_metrics', {}).get('total_sections', 0)}")
                logger.info(f"  - Subsections extracted: {stats.get('throughput_metrics', {}).get('total_subsections', 0)}")
                logger.info(f"  - Time constraint met: {stats.get('constraint_compliance', {}).get('time_limit', False)}")
            
            return 0
        else:
            logger.error(f"Pipeline failed: {result['error']}")
            if 'traceback' in result:
                logger.debug(f"Traceback: {result['traceback']}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
