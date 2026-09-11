#!/usr/bin/env python3
"""Stream Vietnamese mMARCO reranker scores into top-hard-negative Parquet.

The source dataset is read with Hugging Face streaming mode.  Raw source
shards are not downloaded to the workspace.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


DATASET_ID = "hotchpotch/mmarco-hard-negatives-reranker-score"
REVISION = "6e7d9ae9eea4ed4de2fcb7c9885fd4addf348b5b"
CONFIG = "vietnamese_bge-reranker-v2-m3"
DEFAULT_OUTPUT = Path(
    "data/mmarco-vietnamese-reranker/"
    "mmarco_vi_top_hard_negative_score_gt_0_5.parquet"
)
BATCH_SIZE = 50_000

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.int64()),
        pa.field("positive_doc_id", pa.int64()),
        pa.field("positive_score", pa.float32()),
        pa.field("hard_negative_doc_id", pa.int64()),
        pa.field("hard_negative_score", pa.float32()),
    ],
    metadata={
        b"source_dataset": DATASET_ID.encode(),
        b"source_revision": REVISION.encode(),
        b"source_config": CONFIG.encode(),
        b"selection": (
            b"One highest-scoring negative per qid after excluding every "
            b"positive document ID; hard_negative_score strictly greater than 0.5."
        ),
    },
)


def choose_top_document(
    document_ids: list[Any], scores: list[Any], excluded_ids: set[int]
) -> tuple[int, float] | None:
    """Return the deterministic best document-score pair not in exclusions."""
    candidates: list[tuple[int, float]] = []
    for document_id, score in zip(document_ids, scores, strict=True):
        if not isinstance(document_id, int) or not isinstance(score, (int, float)):
            continue
        score_value = float(score)
        if document_id in excluded_ids or not math.isfinite(score_value):
            continue
        candidates.append((document_id, score_value))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1], -item[0]))


def iter_source_rows(source_files: list[Path] | None) -> Iterable[dict[str, Any]]:
    """Read local shards when supplied, otherwise use Hugging Face streaming."""
    required_columns = {"qid", "pos", "neg", "pos.score", "neg.score"}
    if source_files:
        for source_file in source_files:
            parquet_file = pq.ParquetFile(source_file)
            if set(parquet_file.schema_arrow.names) != required_columns:
                raise ValueError(
                    f"Unexpected source columns in {source_file}: "
                    f"{parquet_file.schema_arrow.names}"
                )
            for batch in parquet_file.iter_batches(batch_size=5_000):
                yield from batch.to_pylist()
        return

    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        split="train",
        streaming=True,
        revision=REVISION,
    )
    if set(dataset.column_names) != required_columns:
        raise ValueError(f"Unexpected source columns: {dataset.column_names}")
    yield from dataset


def iter_selected_rows(
    threshold: float,
    stats: Counter[str],
    source_files: list[Path] | None,
) -> Iterator[dict[str, int | float]]:
    """Yield one usable positive/top-hard-negative row per source query."""
    for raw_row in iter_source_rows(source_files):
        stats["examined"] += 1
        positives = raw_row["pos"]
        positive_scores = raw_row["pos.score"]
        negatives = raw_row["neg"]
        negative_scores = raw_row["neg.score"]
        qid = raw_row["qid"]
        if not isinstance(qid, int) or not all(
            isinstance(value, list)
            for value in (positives, positive_scores, negatives, negative_scores)
        ):
            stats["rejected_invalid_schema"] += 1
            continue
        if len(positives) != len(positive_scores) or len(negatives) != len(negative_scores):
            stats["rejected_unaligned_scores"] += 1
            continue

        positive = choose_top_document(positives, positive_scores, set())
        if positive is None:
            stats["rejected_no_positive"] += 1
            continue
        positive_ids = {document_id for document_id in positives if isinstance(document_id, int)}
        hard_negative = choose_top_document(negatives, negative_scores, positive_ids)
        if hard_negative is None:
            stats["rejected_no_negative"] += 1
            continue
        if hard_negative[1] <= threshold:
            stats["rejected_below_threshold"] += 1
            continue

        stats["accepted"] += 1
        yield {
            "qid": qid,
            "positive_doc_id": positive[0],
            "positive_score": positive[1],
            "hard_negative_doc_id": hard_negative[0],
            "hard_negative_score": hard_negative[1],
        }


def write_parquet(
    output_path: Path, threshold: float, source_files: list[Path] | None
) -> Counter[str]:
    """Write selected rows in bounded batches to a single Parquet artifact."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    batch: list[dict[str, int | float]] = []
    with pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="zstd") as writer:
        for row in iter_selected_rows(threshold, stats, source_files):
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA))
    return stats


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--source-file",
        type=Path,
        action="append",
        help="Local Vietnamese score shard; may be repeated. Defaults to online streaming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    stats = write_parquet(args.output, args.threshold, args.source_file)
    print(f"Output: {args.output}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
