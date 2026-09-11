#!/usr/bin/env python3
"""Attach mMARCO query/collection row indices to mined hard negatives."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path(
    "data/mmarco-vietnamese-reranker/"
    "mmarco_vi_top_hard_negative_score_gt_0_5.parquet"
)
DEFAULT_MAPPING = Path(
    "data/mmarco-vietnamese-reranker/mapping/mappings/vietnamese_joblib.pkl.gz"
)
DEFAULT_OUTPUT = Path(
    "data/mmarco-vietnamese-reranker/"
    "mmarco_vi_top_hard_negative_score_gt_0_5_with_indices.parquet"
)

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.int64()),
        pa.field("query_row_index", pa.int64()),
        pa.field("positive_doc_id", pa.int64()),
        pa.field("positive_row_index", pa.int64()),
        pa.field("positive_score", pa.float32()),
        pa.field("hard_negative_doc_id", pa.int64()),
        pa.field("hard_negative_row_index", pa.int64()),
        pa.field("hard_negative_score", pa.float32()),
    ],
    metadata={
        b"mapping_source": (
            b"hotchpotch/mmarco-hard-negatives-reranker-score/"
            b"mappings/vietnamese_joblib.pkl.gz"
        ),
        b"mapping_revision": b"6e7d9ae9eea4ed4de2fcb7c9885fd4addf348b5b",
        b"index_targets": b"unicamp-dl/mmarco queries-vietnamese and collection-vietnamese",
    },
)


def load_mapping(mapping_file: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Load and validate the official mMARCO ID-to-row-index mapping."""
    mapping: dict[str, Any] = joblib.load(mapping_file)
    query_indices = mapping.get("query_id_dict")
    collection_indices = mapping.get("collection_id_dict")
    if not isinstance(query_indices, dict) or not isinstance(collection_indices, dict):
        raise ValueError("Mapping does not contain the required query and collection dictionaries.")
    return query_indices, collection_indices


def attach_indices(
    input_file: Path, mapping_file: Path, output_file: Path
) -> Counter[str]:
    """Stream selected ID rows and write a fully mapped Parquet artifact."""
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_file}")
    query_indices, collection_indices = load_mapping(mapping_file)
    input_parquet = pq.ParquetFile(input_file)
    expected_columns = {
        "qid",
        "positive_doc_id",
        "positive_score",
        "hard_negative_doc_id",
        "hard_negative_score",
    }
    if set(input_parquet.schema_arrow.names) != expected_columns:
        raise ValueError(f"Unexpected input columns: {input_parquet.schema_arrow.names}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    with pq.ParquetWriter(output_file, OUTPUT_SCHEMA, compression="zstd") as writer:
        for batch in input_parquet.iter_batches(batch_size=20_000):
            output_rows: list[dict[str, int | float]] = []
            for row in batch.to_pylist():
                stats["examined"] += 1
                try:
                    query_row_index = query_indices[row["qid"]]
                    positive_row_index = collection_indices[row["positive_doc_id"]]
                    hard_negative_row_index = collection_indices[row["hard_negative_doc_id"]]
                except KeyError:
                    stats["unmapped"] += 1
                    continue
                output_rows.append(
                    {
                        "qid": row["qid"],
                        "query_row_index": query_row_index,
                        "positive_doc_id": row["positive_doc_id"],
                        "positive_row_index": positive_row_index,
                        "positive_score": row["positive_score"],
                        "hard_negative_doc_id": row["hard_negative_doc_id"],
                        "hard_negative_row_index": hard_negative_row_index,
                        "hard_negative_score": row["hard_negative_score"],
                    }
                )
            if output_rows:
                writer.write_table(pa.Table.from_pylist(output_rows, schema=OUTPUT_SCHEMA))
                stats["mapped"] += len(output_rows)
    return stats


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    stats = attach_indices(args.input, args.mapping, args.output)
    print(f"Output: {args.output}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")
    if stats["unmapped"]:
        raise RuntimeError(f"{stats['unmapped']:,} selected rows could not be resolved by mapping.")


if __name__ == "__main__":
    main()
