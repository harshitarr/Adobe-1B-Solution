#!/bin/bash

# Round 1B Docker Entrypoint Script

set -e

echo "=== Round 1B Persona-Driven Document Intelligence ==="
echo "Starting pipeline execution..."

# Check if input directory exists and has content
if [ ! -d "/app/input" ]; then
    echo "ERROR: Input directory /app/input not found"
    exit 1
fi

# Count PDF files
PDF_COUNT=$(find /app/input -name "*.pdf" | wc -l)
echo "Found $PDF_COUNT PDF documents"

if [ $PDF_COUNT -eq 0 ]; then
    echo "ERROR: No PDF documents found in /app/input"
    exit 1
fi

# Check for persona and JTBD files
PERSONA_FILE=""
JTBD_FILE=""

for file in persona.json persona.txt; do
    if [ -f "/app/input/$file" ]; then
        PERSONA_FILE="$file"
        break
    fi
done

for file in jtbd.json jtbd.txt job_to_be_done.json job_to_be_done.txt; do
    if [ -f "/app/input/$file" ]; then
        JTBD_FILE="$file"
        break
    fi
done

if [ -z "$PERSONA_FILE" ]; then
    echo "ERROR: No persona file found (persona.json or persona.txt)"
    exit 1
fi

if [ -z "$JTBD_FILE" ]; then
    echo "ERROR: No job-to-be-done file found"
    exit 1
fi

echo "Using persona file: $PERSONA_FILE"
echo "Using JTBD file: $JTBD_FILE"

# Ensure output directory exists
mkdir -p /app/output

# Set default timeout if not provided
TIMEOUT=${TIMEOUT:-60}

echo "Processing with ${TIMEOUT}s timeout..."

# Run the pipeline
python main.py \
    --input-dir /app/input \
    --output-dir /app/output \
    --timeout $TIMEOUT \
    --log-level INFO

# Check if output was generated
if [ -f "/app/output/challenge1b_output.json" ]; then
    echo "SUCCESS: Output generated at /app/output/challenge1b_output.json"
    
    # Show output summary
    echo "=== OUTPUT SUMMARY ==="
    python -c "
import json
import sys
try:
    with open('/app/output/challenge1b_output.json', 'r') as f:
        data = json.load(f)
    
    sections = len(data.get('extracted_sections', []))
    subsections = len(data.get('extracted_subsections', []))
    processing_time = data.get('metadata', {}).get('total_processing_time', 0)
    
    print(f'Sections extracted: {sections}')
    print(f'Subsections extracted: {subsections}')
    print(f'Processing time: {processing_time:.2f}s')
    
except Exception as e:
    print(f'Could not read output summary: {e}')
    "
else
    echo "ERROR: No output file generated"
    exit 1
fi

echo "=== Pipeline completed successfully ==="
