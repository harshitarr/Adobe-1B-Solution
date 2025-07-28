#!/usr/bin/env python3
"""
Round 1B Persona-Driven Document Intelligence - Main Entry Point
"""

import logging
import sys
import argparse
from pathlib import Path
import time

# Absolute imports since main.py is at project root
from src.pipeline.manager import PipelineManager
from src.core.config import Config

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_pipeline(input_dir: str, output_dir: str = None, timeout_seconds: int = 60):
    """Run the Round 1B pipeline programmatically"""
    
    logger = logging.getLogger(__name__)
    
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else Path("./data/output")
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Round 1B Persona-Driven Document Intelligence")
        logger.info(f"Input directory: {input_path}")
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Timeout: {timeout_seconds} seconds")
        
        # Initialize and run pipeline
        pipeline = PipelineManager()
        result = pipeline.execute_pipeline(input_path, output_path, timeout_seconds=timeout_seconds)
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Round 1B Persona-Driven Document Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input-dir data/input --output-dir data/output
  python main.py --input-dir /path/to/pdfs --timeout 45
        """
    )
    
    parser.add_argument(
        '--input-dir', 
        type=str, 
        required=True,
        help='Directory containing PDF documents, persona.json, and jtbd.json'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./data/output',
        help='Output directory for results (default: ./data/output)'
    )
    
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=60,
        help='Processing timeout in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Run the pipeline
    start_time = time.time()
    result = run_pipeline(args.input_dir, args.output_dir, args.timeout)
    total_time = time.time() - start_time
    
    if result['success']:
        print(f"✅ Success! Processing completed in {total_time:.2f}s")
        print(f"📄 Output saved to: {result.get('output_path', args.output_dir)}")
        sys.exit(0)
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        sys.exit(1)

if __name__ == "__main__":
    main()
