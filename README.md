# VietRAG-Embed-E5-Base

A data preparation, fine-tuning, and evaluation pipeline for a Vietnamese RAG embedding model based on `multilingual-e5-base`. The project focuses on traceable retrieval training data in pair or triplet form: `anchor`, `positive`, and `hard_negative`.

```mermaid
flowchart LR
    A[Collect data] --- B[Clean and normalize]
    B --- C[PostgreSQL]
    C --- D[Build pairs and hard negatives]
    D --- E[Quality checks]
    E --- F[SQLite or Parquet for training]
    F --- G[Fine-tune and benchmark]
```

## Benchmark highlights

The final model was evaluated on an NVIDIA Tesla T4 with E5-compatible `query:` and `passage:` prefixes.

### mMARCO-VI held-out retrieval

The in-domain test set contains 4,599 unseen queries and 7,949 candidate passages from mMARCO-VI shards 04–05.

| Metric | Score |
| --- | ---: |
| Recall@1 | 0.8419 |
| Recall@5 | 0.9533 |
| Recall@10 | 0.9698 |
| MRR@10 | 0.8907 |
| nDCG@10 | 0.9102 |

### Selected VN-MTEB retrieval results

| Task | Score |
| --- | ---: |
| SciFact-VN | 0.6130 |
| TRECCOVID-VN | 0.6068 |
| Quora-VN | 0.5652 |

The mMARCO-VI result is an in-domain diagnostic and is reported separately from VN-MTEB. Detailed outputs and run notes are available in [`benchmark/kaggle_v6_results`](benchmark/kaggle_v6_results/).

## Project structure

| Directory | Purpose |
| --- | --- |
| `preprocess/` | Normalize, merge, and reconstruct general and scientific data. |
| `extractor/` | Mine hard negatives from the `general` table with FAISS IVF-PQ. |
| `script/` | Ingest data, build mMARCO datasets, clean records, and export training artifacts. |
| `database/` | Define SQLAlchemy schemas and manage PostgreSQL connections. |
| `train/` | Fine-tune the embedding model from pair and triplet records. |
| `benchmark/` | Evaluate the model on VN-MTEB and the held-out mMARCO-VI set. |
| `tests/` | Test cleaning, ingestion, and data-selection utilities. |

## Environment setup

Python 3 and PostgreSQL are required. Create a `.env` file in the repository root:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
```

Install the locked dependencies:

```bash
uv sync
```

Or use the project's Conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
uv sync
```

## Typical workflow

### 1. Build and validate the mMARCO dataset

```bash
python script/build_mmarco_vietnamese_rag_qa_parquet.py \
  --source-dir data/mmarco-vietnamese-rag-qa/source/triples \
  --output data/mmarco-vietnamese-rag-qa/mmarco_vi_rag_qa_600k.parquet

python script/load_mmarco_vietnamese_triplets_to_postgres.py --dry-run
```

`--dry-run` validates the Parquet schema and data quality without writing to PostgreSQL. Remove the flag to insert records into the `triplet` table.

### 2. Mine hard negatives

The extractor uses the local checkpoint at `models/checkpoint-40236`, selects CUDA when available, and writes results to `general_triplet`.

```bash
python extractor/extract_general_data.py
```

Rebuild the FAISS index after changing the source data or embedding model:

```bash
python extractor/extract_general_data.py --rebuild-index
```

### 3. Load curated data

```bash
python script/load_curated_triplet_to_postgres.py
python script/load_legal_top_quality_to_triplet.py --dry-run
python script/load_hybrid_rag_qa_train.py
```

The hybrid loader uses training splits only and explicitly excludes VN-MTEB benchmark data.

### 4. Export the training data

```bash
python script/export_clean_triplet_to_sqlite.py \
  --output database/triplet_clean.db

python script/export_general_triplet_to_parquet.py \
  --output data/general_triplet.parquet
```

The export scripts validate schema and integrity and do not overwrite existing artifacts by default. Use `--overwrite` only when replacement is intentional.

### 5. Fine-tune and evaluate

Use [`train/final_train.ipynb`](train/final_train.ipynb) to fine-tune on both pair and triplet records, then run [`benchmark/VietEmbed-RAG-V1-VN-MTEB-Benchmark.ipynb`](benchmark/VietEmbed-RAG-V1-VN-MTEB-Benchmark.ipynb) on Kaggle to reproduce the evaluation workflow.

## Core data schema

| Field | Description |
| --- | --- |
| `data_id` | Stable record identifier. |
| `source`, `title`, `topic`, `domain` | Provenance metadata for filtering and auditing. |
| `anchor` | Query or anchor text. |
| `positive` | Passage relevant to the anchor. |
| `hard_negative` | Difficult but irrelevant passage; optional for pair-only sources. |

E5 inputs must use the corresponding `query:` and `passage:` prefixes. The same convention is applied during training, hard-negative mining, and evaluation.

## Tests

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
pytest -q
```

## Data safety

- Do not commit `.env` files, checkpoints, FAISS indexes, databases, or large generated datasets.
- PostgreSQL import and export commands require `DATABASE_URL`; review command options before running scripts that write data or create artifacts.
