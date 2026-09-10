"""Tests for English QA/RAG cleaning helpers."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from script.clean_english_rag_qa_triplets import (  # noqa: E402
    clean_text,
    invalid_reason,
    lexical_overlap,
    make_row,
)


def test_clean_text_repairs_common_mojibake() -> None:
    assert clean_text("Chelsea is 5 ft 9Â½ inches.") == "Chelsea is 5 ft 9½ inches."


def test_invalid_reason_rejects_short_and_degenerate_passages() -> None:
    assert invalid_reason("valid query", "too short", "A valid hard negative passage." * 3) == "positive_length"
    assert invalid_reason(
        "valid query",
        "A valid positive passage." * 3,
        "A valid positive passage." * 3,
    ) == "degenerate_triplet"


def test_make_row_uses_source_local_clean_pair_identity() -> None:
    original = {
        "data_id": "raw-1",
        "source": "example/source",
        "source_revision": "revision",
        "source_split": "train",
        "source_row_index": 1,
        "domain": "web_passage_qa",
        "task_type": "rag_qa_retrieval_triplet",
        "anchor": "What is the capital of France?",
        "positive": "Paris is the capital and most populous city of France." * 2,
        "hard_negative": "Berlin is the capital and largest city of Germany." * 2,
    }
    row, reason, changed = make_row(original)

    assert reason is None
    assert not changed
    assert row is not None
    assert row["data_id"].startswith("en_clean_")
    assert row["hard_negative_score"] == lexical_overlap(original["anchor"], original["hard_negative"])
