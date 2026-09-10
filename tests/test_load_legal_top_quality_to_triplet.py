"""Unit tests for legal top-quality triplet selection helpers."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from script.load_legal_top_quality_to_triplet import (  # noqa: E402
    LegalCandidate,
    infer_topic,
    normalize_text,
    to_triplet_row,
)


def test_normalize_text_and_infer_topic() -> None:
    assert normalize_text("  Điều\n\n  3\t Luật  ") == "Điều 3 Luật"
    assert infer_topic("Điều 3 Nghị định 12/2024/NĐ-CP") == "Nghị định"
    assert infer_topic("Hướng dẫn áp dụng") is None


def test_to_triplet_row_uses_namespaced_id_and_legal_domain() -> None:
    row = to_triplet_row(
        LegalCandidate(
            data_id="legal_002_abc",
            source="Law Reading Comprehension QA",
            title="Điều 3 Luật mẫu",
            anchor="Câu hỏi pháp luật đủ độ dài.",
            positive="Nội dung trả lời pháp luật đủ độ dài để kiểm thử.",
        )
    )

    assert row["data_id"] == "legal_top100k_v1_legal_002_abc"
    assert row["domain"] == "legal"
    assert row["topic"] == "Luật"
    assert row["hard_negative"] is None
