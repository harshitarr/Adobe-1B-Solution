"""
Very lightweight sanity-tests that run in < 2 s on CPU-only boxes.
They guarantee that all key modules can be imported and basic
end-to-end processing works on the smallest possible input.
"""

import json
from pathlib import Path

import pytest

from src.pipeline.manager import PipelineManager
from src.core.config import Config

ROOT = Path(__file__).resolve().parents[2]      # project root


def test_imports():
    """
    Make sure every top-level package is importable. If anything fails
    here, the codebase is structurally broken.
    """
    import src.shared.pdf_parser
    import src.preprocessing.batch_processor
    import src.chunking.semantic_chunker
    import src.embeddings.engine
    import src.persona.processor
    import src.ranking.scorer
    import src.subsection.extractor
    import src.output.formatter


@pytest.mark.timeout(20)
def test_end_to_end_minimal(tmp_path: Path, monkeypatch):
    """
    End-to-end with a **single one-page dummy PDF** (generated on the fly)
    plus minimal persona/JTBD.
    """
    # --- 1. create dummy input set -------------------------------------- #
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # 1-page PDF with PyPDF2
    from PyPDF2 import PdfWriter

    pdf_path = input_dir / "dummy.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as fp:
        writer.write(fp)

    # persona + jtbd
    (input_dir / "persona.json").write_text(
        json.dumps(
            {
                "role": "Tester",
                "expertise_areas": ["qa"],
                "experience_level": "beginner",
                "domain": "general",
                "priorities": ["correctness"],
            }
        )
    )
    (input_dir / "jtbd.json").write_text(
        json.dumps(
            {
                "task_description": "Just make sure the pipeline runs.",
                "expected_outcomes": [],
                "priority_keywords": [],
                "context": "unit test",
                "urgency": "low",
            }
        )
    )

    # --- 2. run pipeline ------------------------------------------------ #
    output_dir = tmp_path / "output"
    pipeline = PipelineManager()
    result = pipeline.execute_pipeline(input_dir, output_dir, timeout_seconds=30)

    assert result["success"], f"pipeline failed: {result.get('error')}"

    output_path = Path(result["output_path"])
    assert output_path.exists(), "no output JSON produced"

    data = json.loads(output_path.read_text())
    # Minimal schema checks
    assert "metadata" in data
    assert "extracted_sections" in data
    assert "extracted_subsections" in data
