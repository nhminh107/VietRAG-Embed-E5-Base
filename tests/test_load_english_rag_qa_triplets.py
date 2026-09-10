"""Unit tests for the English QA/RAG triplet loader's pure helpers."""

import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from script.load_english_rag_qa_triplets import (  # noqa: E402
    SOURCES,
    extract_triplet,
    is_valid_triplet,
    make_row,
)


def test_extracts_named_retrieval_triplet_with_provenance() -> None:
    source = SOURCES[0]
    row = make_row(
        source,
        7,
        {
            "query": "What regulates potassium secretion?",
            "positive": "Insulin and catecholamines regulate potassium distribution.",
            "negative": "Bile secretion is regulated by cholecystokinin and secretin.",
        },
    )

    assert row["source"] == "bclavie/msmarco-500k-triplets"
    assert row["source_split"] == "train"
    assert row["task_type"] == "rag_qa_retrieval_triplet"
    assert row["hard_negative"].startswith("Bile secretion")


def test_extracts_list_backed_triplet_and_rejects_pair() -> None:
    source = SOURCES[1]
    assert extract_triplet(
        source,
        {
            "texts": [
                "What did the article discuss?",
                "The article discussed musicians' health problems in detail.",
                "The city council approved an unrelated construction project.",
            ]
        },
    )[0] == "What did the article discuss?"

    with pytest.raises(ValueError, match="three-text triplet"):
        extract_triplet(source, {"texts": ["question", "positive"]})


def test_rejects_degenerate_retrieval_supervision() -> None:
    assert not is_valid_triplet("Q", "A valid passage long enough.", "Another valid passage long enough.")
    assert not is_valid_triplet(
        "Valid query",
        "A valid passage long enough.",
        "A valid passage long enough.",
    )
