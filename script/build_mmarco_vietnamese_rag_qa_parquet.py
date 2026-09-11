"""Build a filtered mMARCO Vietnamese RAG/QA triplet Parquet dataset.

The input shards are processed one Parquet row group at a time.  This keeps
memory bounded and avoids downloading the full 39.8-million-row dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("query", pa.string()),
        pa.field("positive", pa.string()),
        pa.field("negative", pa.string()),
    ],
    metadata={
        b"source_dataset": b"minhnguyent546/mmarco-vietnamese-split",
        b"subset": b"triples/train",
        b"selection": (
            b"Non-empty, distinct query/positive/negative; query 8-384 "
            b"characters; passages 80-2,000 characters."
        ),
    },
)


def quality_mask(table: pa.Table) -> pa.Array:
    """Vectorized minimum-quality checks for RAG/QA training triplets."""
    query_length = pc.utf8_length(table["query"])
    positive_length = pc.utf8_length(table["positive"])
    negative_length = pc.utf8_length(table["negative"])
    return pc.and_kleene(
        pc.and_kleene(
            pc.and_kleene(
                pc.and_kleene(
                    pc.greater_equal(query_length, 8),
                    pc.less_equal(query_length, 384),
                ),
                pc.and_kleene(
                    pc.greater_equal(positive_length, 80),
                    pc.less_equal(positive_length, 2_000),
                ),
            ),
            pc.and_kleene(
                pc.greater_equal(negative_length, 80),
                pc.less_equal(negative_length, 2_000),
            ),
        ),
        pc.and_kleene(
            pc.not_equal(table["positive"], table["negative"]),
            pc.not_equal(table["query"], table["positive"]),
        ),
    )


def write_dataset(
    shard_paths: list[Path],
    output_path: Path,
    rows_per_shard: int,
    batch_size: int,
) -> Counter[str]:
    """Write the selected triplets to one compressed Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    stats: Counter[str] = Counter()
    with pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="zstd") as writer:
        for shard_path in shard_paths:
            accepted_from_shard = 0
            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["query", "positive", "negative"],
            ):
                table = pa.Table.from_batches([batch])
                stats["examined"] += table.num_rows
                selected = table.filter(quality_mask(table))
                remaining = rows_per_shard - accepted_from_shard
                if selected.num_rows > remaining:
                    selected = selected.slice(0, remaining)
                if selected.num_rows:
                    writer.write_table(selected.cast(OUTPUT_SCHEMA))
                    accepted_from_shard += selected.num_rows
                    stats["accepted"] += selected.num_rows
                if accepted_from_shard >= rows_per_shard:
                    break
            stats[f"accepted:{shard_path.name}"] = accepted_from_shard
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/mmarco-vietnamese-rag-qa/source/triples"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mmarco-vietnamese-rag-qa/mmarco_vi_rag_qa_600k.parquet"),
    )
    parser.add_argument("--rows-per-shard", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.source_dir.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {args.source_dir}")
    if args.rows_per_shard <= 0 or args.batch_size <= 0:
        raise ValueError("--rows-per-shard and --batch-size must be positive")

    stats = write_dataset(
        shard_paths=shard_paths,
        output_path=args.output,
        rows_per_shard=args.rows_per_shard,
        batch_size=args.batch_size,
    )
    print(f"Output: {args.output}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
