"""Tests for the hybrid RAG/QA train loader's pure helpers."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from script.load_hybrid_rag_qa_train import (  # noqa: E402
    SPECS,
    SourceSpec,
    is_clean_pair,
    make_row,
    normalize_text,
    stable_id,
    validate_selection,
)


def test_normalization_and_quality_filter() -> None:
    assert normalize_text("  Câu\n hỏi\t ") == "Câu hỏi"
    assert is_clean_pair("Câu hỏi hợp lệ?", "Nội dung trả lời hợp lệ. " * 10)
    assert not is_clean_pair("Câu hỏi hợp lệ?", "ngắn")
    assert not is_clean_pair("Câu hỏi hợp lệ?", "Nội dung [[ lỗi ]] " * 10)


def test_make_row_has_train_provenance() -> None:
    spec = SourceSpec("general", "UIT ViQuAD2", "rag_qa_gold", False, 100, 1)
    row = make_row(
        spec,
        {
            "data_id": "data_001",
            "title": "Tiêu đề",
            "anchor": "Câu hỏi hợp lệ?",
            "positive": "Nội dung trả lời hợp lệ. " * 10,
        },
    )

    assert row["split"] == "train"
    assert row["task_type"] == "rag_qa_gold"
    assert not row["is_synthetic"]
    assert row["data_id"] == stable_id(spec, "data_001")


def test_validate_selection_rejects_benchmark_source() -> None:
    row = {
        "data_id": "one",
        "split": "train",
        "task_type": "rag_qa_gold",
        "is_synthetic": False,
        "quality_score": 100,
        "source_table": "general",
        "source": "GreenNode/nq-vn",
        "source_data_id": "one",
        "title": None,
        "anchor": "Câu hỏi hợp lệ?",
        "positive": "Nội dung trả lời hợp lệ. " * 10,
        "hard_negative": None,
    }
    try:
        validate_selection([row] * sum(spec.quota for spec in SPECS))
    except RuntimeError as error:
        assert "duplicate" in str(error) or "VN-MTEB" in str(error)
    else:
        raise AssertionError("Benchmark rows must be rejected.")
