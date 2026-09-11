# VietRAG-Embed-E5-Base

Pipeline chuẩn bị dữ liệu huấn luyện cho mô hình embedding/RAG tiếng Việt dựa trên `multilingual-e5-base`. Project tập trung vào tạo triplet truy hồi có thể truy vết nguồn: `anchor`, `positive`, `hard_negative`.

```mermaid
flowchart LR
    A[Thu thập dữ liệu] --- B[Làm sạch và chuẩn hoá]
    B --- C[PostgreSQL]
    C --- D[Tạo triplet và hard negative]
    D --- E[Kiểm tra chất lượng]
    E --- F[SQLite hoặc Parquet để huấn luyện]
```

## Thành phần chính

| Thư mục | Vai trò |
| --- | --- |
| `preprocess/` | Chuẩn hoá, gộp và tái tạo dữ liệu tổng quát/khoa học. |
| `extractor/` | Tạo hard negative từ bảng `general` bằng FAISS IVF-PQ. |
| `script/` | Nạp dữ liệu, xây dựng tập mMARCO, làm sạch và xuất artifact huấn luyện. |
| `database/` | Khai báo schema SQLAlchemy và quản lý kết nối PostgreSQL. |
| `train/` | Notebook fine-tuning mô hình embedding. |
| `benchmark/` | Notebook đánh giá VN-MTEB. |
| `tests/` | Kiểm thử các hàm làm sạch, nạp và chọn dữ liệu. |

## Chuẩn bị môi trường

Yêu cầu Python 3 và PostgreSQL. Tạo file `.env` tại thư mục gốc:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
```

Cài dependencies từ lockfile:

```bash
uv sync
```

Hoặc dùng Conda theo môi trường của dự án:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
uv sync
```

## Luồng sử dụng tiêu biểu

### 1. Xây dựng và kiểm tra dữ liệu mMARCO

```bash
python script/build_mmarco_vietnamese_rag_qa_parquet.py \
  --source-dir data/mmarco-vietnamese-rag-qa/source/triples \
  --output data/mmarco-vietnamese-rag-qa/mmarco_vi_rag_qa_600k.parquet

python script/load_mmarco_vietnamese_triplets_to_postgres.py --dry-run
```

`--dry-run` chỉ xác thực schema và chất lượng Parquet, không ghi PostgreSQL. Bỏ cờ này để nạp vào bảng `triplet`.

### 2. Tạo hard negative cho dữ liệu tổng quát

Script dưới đây dùng checkpoint cục bộ tại `models/checkpoint-40236`, tự chọn CUDA khi PyTorch hỗ trợ, và ghi kết quả vào bảng `general_triplet`.

```bash
python extractor/extract_general_data.py
```

Khi cần tạo lại chỉ mục FAISS do thay đổi dữ liệu/model:

```bash
python extractor/extract_general_data.py --rebuild-index
```

### 3. Nạp hoặc chọn dữ liệu đã tuyển lọc

```bash
python script/load_curated_triplet_to_postgres.py
python script/load_legal_top_quality_to_triplet.py --dry-run
python script/load_hybrid_rag_qa_train.py
```

Tập hybrid chỉ lấy nguồn huấn luyện và chủ động loại dữ liệu benchmark VN-MTEB.

### 4. Xuất dữ liệu để huấn luyện

```bash
python script/export_clean_triplet_to_sqlite.py \
  --output database/triplet_clean.db

python script/export_general_triplet_to_parquet.py \
  --output data/general_triplet.parquet
```

Các script xuất kiểm tra schema, tính toàn vẹn và tránh ghi đè mặc định. Dùng `--overwrite` chỉ khi bạn thực sự muốn thay artifact đầu ra.

## Schema dữ liệu cốt lõi

| Trường | Ý nghĩa |
| --- | --- |
| `data_id` | Định danh ổn định của bản ghi. |
| `source`, `title`, `topic`, `domain` | Thông tin nguồn để truy vết và lọc. |
| `anchor` | Câu truy vấn hoặc văn bản neo. |
| `positive` | Đoạn phù hợp với `anchor`. |
| `hard_negative` | Đoạn khó nhưng không phù hợp; có thể rỗng ở nguồn chưa khai thác negative. |

Với E5, mã hoá truy vấn và đoạn văn theo tiền tố tương ứng `query:` và `passage:`; logic này nằm trong `extractor/extract_general_data.py`.

## Kiểm thử

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
pytest -q
```

## Lưu ý dữ liệu

- Không commit `.env`, checkpoint, chỉ mục FAISS, cơ sở dữ liệu hoặc artifact dữ liệu lớn.
- Các thao tác nạp/xuất cần `DATABASE_URL`; đọc kỹ tham số trước khi chạy vì một số script ghi vào PostgreSQL hoặc tạo file đầu ra.
