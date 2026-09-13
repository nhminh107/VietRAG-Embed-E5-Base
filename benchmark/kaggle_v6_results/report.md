# VietEmbed RAG V1 — Kaggle benchmark v6

Nguồn: Kaggle run v6 hoàn tất thành công trên Tesla T4, model
`VietRAG/final`. Runtime đo riêng các task là 28,6 phút; thời gian toàn bộ
run trên Kaggle là 31 phút 6 giây.

## VN-MTEB RAG Core

| Task | Main retrieval score |
| --- | ---: |
| SciFact-VN | 0.61302 |
| TRECCOVID-VN | 0.60684 |
| Quora-VN | 0.56517 |
| CQADupstackAndroid-VN | 0.34816 |
| NFCorpus-VN | 0.28164 |
| FiQA2018-VN | 0.26928 |
| SCIDOCS-VN | 0.13370 |
| Mean | 0.40254 |

The original per-task CSV was written to `/content` by the notebook and is
not included in Kaggle Output. `vn_mteb_rag_core_recovered.csv` is a faithful
transcription from the completed Kaggle notebook's rendered result table, not
the original MTEB export.

## mMARCO-VI ShardHoldout 04-05

- Queries: 4,599; corpus documents: 7,949; qrels: 4,599.
- Recall@1: 0.841922; Recall@5: 0.953251; Recall@10: 0.969776.
- MRR@10: 0.890667; nDCG@10: 0.910250.
- Mean rank: 4.509024; median rank: 1.

This is a query-held-out, in-domain diagnostic. It must be reported separately
from the VN-MTEB RAG Core mean.
