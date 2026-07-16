# 🔬 KG-TABI: Knowledge Graph — Toulmin-Abductive Bucketed Inference

> **Hệ thống tự động phát hiện khoảng trống nghiên cứu (Research Gap) bằng phân tích cấu trúc Đồ thị Tri thức kết hợp Suy luận Logic Toulmin.**

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Cấu hình API Key](#-cấu-hình-api-key)
- [Hướng dẫn chạy](#-hướng-dẫn-chạy)
- [Giải thích từng bước](#-giải-thích-từng-bước-pipeline)
- [Dashboard kiểm duyệt](#-dashboard-kiểm-duyệt-chuyên-gia-hcai)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Tùy chỉnh nâng cao](#-tùy-chỉnh-nâng-cao)

---

## 🎯 Tổng quan

**KG-TABI** là một framework end-to-end giải quyết bài toán: *"Làm thế nào để phát hiện các khoảng trống nghiên cứu ẩn (implicit research gaps) mà không bài báo đơn lẻ nào tự khai báo?"*

Thay vì chỉ đọc văn bản như các phương pháp truyền thống (RAG, LLM prompting), KG-TABI:

1. **Xây dựng Đồ thị Tri thức (Knowledge Graph)** từ các bài báo khoa học.
2. **Phân tích cấu trúc topo** (topology) của đồ thị để tìm ra các vùng tri thức bị cô lập hoặc đình trệ.
3. **Suy luận logic** theo khung lập luận Toulmin để sinh ra các phát biểu khoảng trống có cơ sở.
4. **So sánh đối chứng** với 6 phương pháp baseline (bao gồm GraphRAG, LightRAG, HippoRAG) và đánh giá bằng NLI (Natural Language Inference).

---

## 🏗️ Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────────┐
│                        KG-TABI Pipeline                              │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐               │
│  │  Stage 1    │──▶│  Stage 2    │──▶│  Stage 3     │               │
│  │  Literature │   │  KG Build   │   │  Topology    │               │
│  │  Search     │   │  + Entity   │   │  Analysis    │               │
│  │  & Screen   │   │  Resolution │   │  (Louvain +  │               │
│  │             │   │             │   │   Decay)     │               │
│  └─────────────┘   └─────────────┘   └──────┬───────┘               │
│                                              │                       │
│                                              ▼                       │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐               │
│  │  Stage 6    │◀──│  Stage 5    │◀──│  Stage 4     │               │
│  │  Expert     │   │  Baselines  │   │  TABI        │               │
│  │  Dashboard  │   │  & Evaluate │   │  Inference   │               │
│  │  (Streamlit)│   │  (B1-B6)    │   │  (Toulmin)   │               │
│  └─────────────┘   └─────────────┘   └──────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Cấu trúc thư mục

```
research-paper-gap/
├── .env                          # Cấu hình API Key (không push lên git)
├── .env.example                  # Mẫu cấu hình API Key
├── requirements.txt              # Danh sách thư viện Python
├── README.md                     # File hướng dẫn này
│
├── src/                          # Mã nguồn chính
│   ├── config.py                 # Cấu hình và ngưỡng toán học
│   ├── llm_client.py             # Factory gọi LLM (OpenAI/Groq/Gemini) + retry logic
│   ├── fetch_papers.py           # Stage 1: Tìm kiếm và sàng lọc bài báo
│   ├── extract_triples.py        # Stage 2a: Trích xuất bộ ba (Subject, Relation, Object)
│   ├── entity_resolution.py      # Stage 2b: Khử trùng lặp thực thể (Fuzzy + S-BERT)
│   ├── graph_analysis.py         # Stage 3: Phân tích cấu trúc topo đồ thị
│   ├── tabi_inference.py         # Stage 4: Suy luận khoảng trống theo khung Toulmin
│   ├── evaluate.py               # Stage 5: Đánh giá đối chứng (Jaccard, NLI, Uniqueness)
│   ├── app.py                    # Stage 6: Dashboard Streamlit cho chuyên gia kiểm duyệt
│   └── main.py                   # Bộ điều phối pipeline chính (orchestrator)
│
├── baselines/                    # Các phương pháp baseline để so sánh
│   ├── mulla_rag.py              # B1: Mulla et al. RAG (Retrieve + LLM)
│   ├── simple_llm.py             # B2: Simple LLM (LLM trực tiếp, không KG)
│   ├── gapmap_text.py            # B3: GAPMAP Text-only (TABI trên text thô)
│   ├── graphrag.py               # B4: GraphRAG (subgraph retrieval + LLM)
│   ├── lightrag.py               # B5: LightRAG (entity co-occurrence + LLM)
│   └── hipporag.py               # B6: HippoRAG (KG-based re-ranking + LLM)
│
├── data/                         # Dữ liệu đầu vào và đầu ra
│   ├── raw_papers/               # Bài báo thô, chunks, sample data
│   │   ├── sample_papers.json    # Dữ liệu mẫu 5 bài báo (chạy nhanh)
│   │   ├── screened_papers.json  # Bài báo đã sàng lọc
│   │   └── chunks.json           # Văn bản đã cắt thành mảnh
│   ├── triples/                  # Bộ ba trích xuất và mapping thực thể
│   │   ├── raw_triples.json      # Bộ ba thô từ LLM
│   │   ├── resolved_triples.json # Bộ ba sau khử trùng lặp
│   │   └── entity_mapping.json   # Bảng ánh xạ thực thể đồng nghĩa
│   ├── graph/                    # Đồ thị tri thức và kết quả phân tích topo
│   │   ├── knowledge_graph.gml   # Đồ thị tri thức (GML format)
│   │   ├── orphan_clusters.json  # Danh sách cụm mồ côi (Louvain)
│   │   └── temporal_decay.json   # Khái niệm bị đình trệ theo thời gian
│   ├── gaps/                     # Kết quả khoảng trống nghiên cứu
│   │   ├── kgtabi_gaps.json      # ⭐ Khoảng trống từ KG-TABI (phương pháp đề xuất)
│   │   ├── baseline_mulla_rag.json
│   │   ├── baseline_simple_llm.json
│   │   ├── baseline_gapmap.json
│   │   ├── baseline_graphrag.json
│   │   ├── baseline_lightrag.json
│   │   └── baseline_hipporag.json
│   └── evaluation_results.md     # 📊 Bảng so sánh thực nghiệm cuối cùng
│
├── paper.tex                     # Bài báo khoa học LaTeX
└── references.bib                # Tài liệu tham khảo
```

---

## ⚙️ Cài đặt

### 1. Yêu cầu hệ thống

- **Python** ≥ 3.10
- **pip** (trình quản lý gói Python)

### 2. Cài đặt thư viện

```powershell
# Dùng pip thông thường
pip install -r requirements.txt

# Hoặc nếu pip không nhận dạng được trong PATH
python -m pip install -r requirements.txt
```

**Danh sách thư viện chính:**

| Thư viện | Vai trò |
|---|---|
| `openai` | Gọi API LLM (OpenAI, Groq, Gemini qua OpenAI-compatible endpoint) |
| `requests` | Gọi API Semantic Scholar |
| `networkx` | Xây dựng và phân tích đồ thị tri thức |
| `python-louvain` | Thuật toán phát hiện cộng đồng Louvain |
| `rapidfuzz` | So khớp mờ (fuzzy matching) cho khử trùng lặp thực thể |
| `sentence-transformers` | Sentence-BERT cho embedding ngữ nghĩa |
| `nltk` | Tách câu cho chunking văn bản |
| `numpy` | Tính toán ma trận Cosine Similarity |
| `streamlit` | Dashboard kiểm duyệt chuyên gia (HCAI) |
| `pyvis` | Trực quan hóa đồ thị tương tác trên trình duyệt |
| `python-dotenv` | Đọc biến môi trường từ file `.env` |

---

## 🔑 Cấu hình API Key

### Bước 1: Tạo file `.env`

Sao chép file mẫu:

```powershell
copy .env.example .env
```

### Bước 2: Điền API Key

Mở file `.env` bằng editor và điền thông tin:

#### Dùng Google Gemini (Google AI Studio) — Miễn phí

```env
LLM_PROVIDER=openai
LLM_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-3.5-flash-low
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

> 💡 Lấy API Key tại: https://aistudio.google.com/apikey

#### Dùng OpenAI — Trả phí

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-your_openai_key_here
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=
```

#### Dùng Groq — Miễn phí (giới hạn)

```env
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_groq_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=
```

---

## 🚀 Hướng dẫn chạy

### Chạy toàn bộ pipeline (end-to-end) với dữ liệu mẫu

```powershell
python -m src.main --sample
```

Lệnh này chạy tất cả 5 stage liên tiếp trên tập dữ liệu mẫu 5 bài báo (`sample_papers.json`).

### Chạy toàn bộ pipeline với dữ liệu thật từ Semantic Scholar

```powershell
python -m src.main --topic "Microservices security" --limit 10
```

| Tham số | Ý nghĩa | Giá trị mặc định |
|---|---|---|
| `--topic` | Chủ đề nghiên cứu cần tìm kiếm | `"Microservices security"` |
| `--limit` | Số bài báo tải về trên mỗi từ khóa mở rộng | `5` |
| `--sample` | Dùng dữ liệu mẫu thay vì gọi Semantic Scholar | `False` |
| `--stage` | Chạy một stage cụ thể (1-5), `0` = toàn bộ | `0` |

### Chạy từng stage riêng lẻ

Nếu bạn muốn tiết kiệm API quota hoặc debug từng bước:

```powershell
# Stage 1: Tìm kiếm và sàng lọc bài báo
python -m src.main --stage 1 --sample

# Stage 2: Xây dựng đồ thị tri thức
python -m src.main --stage 2

# Stage 3: Phân tích cấu trúc topo (Louvain + Temporal Decay)
python -m src.main --stage 3

# Stage 4: Suy luận TABI để sinh khoảng trống nghiên cứu
python -m src.main --stage 4

# Stage 5: Chạy baselines và đánh giá so sánh
python -m src.main --stage 5
```

> ⚠️ **Lưu ý:** Mỗi stage phụ thuộc vào đầu ra của stage trước. Hãy chạy theo thứ tự 1 → 2 → 3 → 4 → 5.

### Mở Dashboard kiểm duyệt chuyên gia

```powershell
python -m streamlit run src/app.py
```

Trình duyệt sẽ tự mở tại: **http://localhost:8501**

---

## 📖 Giải thích từng bước Pipeline

### Stage 1 — Thu thập & Sàng lọc Bài báo (`fetch_papers.py`)

```
Input: Chủ đề nghiên cứu (VD: "Microservices security")
Output: data/raw_papers/chunks.json
```

1. **Mở rộng truy vấn (Query Expansion):** LLM sinh ra 5 từ khóa ngách bổ sung để tăng độ phủ tìm kiếm.
2. **Tìm kiếm bài báo:** Gọi API Semantic Scholar tải tiêu đề, tóm tắt, năm xuất bản.
3. **Sàng lọc độ tương quan:** LLM chấm điểm mức độ liên quan (0.0 → 1.0). Chỉ giữ bài có điểm ≥ 0.7.
4. **Cắt văn bản (Chunking):** Tách văn bản thành các mảnh ≤ 1000 từ, ngắt tại dấu kết thúc câu.

### Stage 2 — Xây dựng Đồ thị Tri thức (`extract_triples.py` + `entity_resolution.py`)

```
Input: data/raw_papers/chunks.json
Output: data/triples/resolved_triples.json
```

1. **Trích xuất bộ ba (Triple Extraction):** LLM đọc từng chunk và trích xuất thông tin theo định dạng:
   ```
   ⟨Subject, Relation, Object⟩
   VD: ⟨Envoy Proxy, USES, mTLS⟩
   ```
   Các kiểu thực thể: `METHOD`, `DATASET`, `METRIC`, `CONCEPT`, `FINDING`, `TOOL`.
   Các kiểu quan hệ: `USES`, `IMPROVES`, `ADDRESSES`, `EVALUATES_ON`, `PRODUCES`, `CONTRADICTS`, `EXTENDS`, `LACKS`, `COMBINES`, `APPLIED_TO`.

2. **Khử trùng lặp thực thể (Entity Resolution):**
   - *Vòng 1 (Fuzzy Lexical):* So khớp ký tự bằng Levenshtein Distance (ngưỡng ≥ 85%).
   - *Vòng 2 (Semantic Embedding):* Sentence-BERT chuyển thực thể thành vector, tính Cosine Similarity (ngưỡng ≥ 0.85).
   - Gộp các thực thể đồng nghĩa (VD: `"mTLS"` ↔ `"mutual TLS"`).

### Stage 3 — Phân tích Cấu trúc Topo Đồ thị (`graph_analysis.py`)

```
Input: data/triples/resolved_triples.json
Output: data/graph/orphan_clusters.json + temporal_decay.json
```

1. **Xây dựng đồ thị NetworkX:** Tạo Directed Graph với các node (thực thể) và edge (quan hệ) kèm thuộc tính năm.
2. **Phát hiện cụm mồ côi (Orphan Cluster Detection):**
   - Thuật toán **Louvain** phân chia đồ thị thành các cộng đồng (communities).
   - Nếu một cộng đồng chiếm ≥ 5% tổng node nhưng tỉ lệ kết nối ra ngoài < 10%, nó được đánh dấu là **cụm mồ côi** (structural hole).
3. **Phân tích suy tàn khái niệm (Temporal Decay):**
   - Theo dõi số lượng cạnh mới kết nối tới mỗi node qua các năm.
   - Nếu giảm ≥ 30% trong 3 năm gần nhất → đánh dấu là **khái niệm đình trệ**.

### Stage 4 — Suy luận TABI (`tabi_inference.py`)

```
Input: orphan_clusters.json + temporal_decay.json
Output: data/gaps/kgtabi_gaps.json ⭐
```

LLM đóng vai Nhà nghiên cứu cao cấp, nhận bằng chứng topo cụ thể và suy luận theo **khung Toulmin**:

| Trường | Ý nghĩa |
|---|---|
| **Grounds** | Bằng chứng cấu trúc đồ thị (VD: "Community A và B không có liên kết nào") |
| **Claim** | Phát biểu khoảng trống nghiên cứu (VD: "Cần tích hợp Autoencoder vào Envoy Proxy") |
| **Warrant** | Giải thích chuyên môn tại sao kết nối này quan trọng |
| **Bucket** | `more_probable` (khả thi ngay) hoặc `least_probable` (dài hạn/suy đoán) |

Hệ thống sử dụng **3-shot prompting** với 3 ví dụ lập luận mẫu để đảm bảo chất lượng đầu ra.

### Stage 5 — Chạy Baselines & Đánh giá (`evaluate.py`)

```
Input: kgtabi_gaps.json + gaps từ các baselines
Output: data/evaluation_results.md 📊
```

**6 phương pháp baseline để so sánh:**

| Baseline | Mô tả | File |
|---|---|---|
| **B1 (Mulla RAG)** | Lấy 3 bài báo tương tự nhất (S-BERT Cosine), truyền vào LLM | `baselines/mulla_rag.py` |
| **B2 (Simple LLM)** | Gửi trực tiếp abstract cho LLM, không dùng KG | `baselines/simple_llm.py` |
| **B3 (GAPMAP Text)** | Chạy TABI trên text thô, không qua đồ thị | `baselines/gapmap_text.py` |
| **B4 (GraphRAG)** | Xây subgraph quanh entity, truy xuất multi-hop neighbor | `baselines/graphrag.py` |
| **B5 (LightRAG)** | Xây đồ thị co-occurrence, truy xuất entity liên kết | `baselines/lightrag.py` |
| **B6 (HippoRAG)** | Re-rank text chunks dùng KG dựa trên entity density | `baselines/hipporag.py` |

**5 chỉ số đánh giá tự động:**

| Chỉ số | Cách tính |
|---|---|
| **Total Gaps** | Tổng số lượng gaps phát hiện |
| **Unique Gaps** | Gộp các gap trùng lặp bằng S-BERT (ngưỡng ≥ 0.85) |
| **Avg Words/Claim** | Độ dài trung bình của phát biểu Claim |
| **NLI Entailment Rate** | LLM đánh giá: "Grounds + Warrant ⟹ Claim" có hợp logic không? |
| **Jaccard Overlap** | Tỉ lệ từ vựng trùng nhau giữa KG-TABI và baseline |

### Stage 6 — Dashboard Kiểm duyệt Chuyên gia (`app.py`)

```
Lệnh khởi chạy: python -m streamlit run src/app.py
Truy cập: http://localhost:8501
```

Xem chi tiết ở phần [Dashboard kiểm duyệt](#-dashboard-kiểm-duyệt-chuyên-gia-hcai).

---

## 🖥️ Dashboard kiểm duyệt Chuyên gia (HCAI)

Dashboard Streamlit cho phép chuyên gia nghiên cứu kiểm duyệt (human-in-the-loop) từng khoảng trống:

- **Sidebar:** Thống kê tổng quan (số gaps, số nodes/edges đồ thị) + nhập Reviewer ID.
- **Bộ chọn Gap:** Dropdown chọn bất kỳ gap nào trong danh sách.
- **Cột trái:** Hiển thị cấu trúc lập luận TABI (Grounds → Claim → Warrant → Bucket).
- **Cột phải:** Bản đồ đồ thị con tương tác (pyvis) — kéo thả, phóng to, xem kiểu thực thể.
- **Form kiểm duyệt:** Chỉnh sửa Claim/Warrant và bấm **Accept** / **Modify** / **Reject**.
- **Lịch sử kiểm duyệt:** Bảng log ghi lại toàn bộ thao tác kiểm định.

Kết quả kiểm duyệt được lưu tự động vào `data/expert_reviews.json`.

---

## 📊 Kết quả thực nghiệm

Kết quả trên tập dữ liệu **100 bài báo thực tế** về **Research Gap Identification** (LLM: `gemini-3.5-flash-low`):

| Phương pháp | Total Gaps | Unique Gaps | Avg Words | NLI Entailment | Jaccard Overlap |
|:---|:---:|:---:|:---:|:---:|:---:|
| **KG-TABI (Ours)** | **46** | **45** | **24.4** | **76.1%** | N/A |
| B1 (Mulla RAG) | 100 | 100 | 61.3 | 50.0% | 0.166 |
| B2 (Simple LLM) | 300 | 296 | 22.9 | 65.7% | 0.147 |
| B3 (GAPMAP Text) | 101 | 100 | 26.1 | 63.4% | 0.212 |
| B4 (GraphRAG) | 97 | 97 | 28.0 | 72.2% | 0.213 |
| B5 (LightRAG) | 100 | 98 | 30.6 | 83.0% | 0.186 |
| B6 (HippoRAG) | 92 | 88 | 31.6 | 79.3% | 0.204 |

**Nhận xét chính:**
- KG-TABI đạt **76.1% NLI Entailment**, chứng minh phần lớn Claim được suy luận logic chặt chẽ từ bằng chứng topo.
- **Jaccard Overlap thấp** (<0.22) cho thấy KG-TABI phát hiện các gaps **hoàn toàn mới** mà các phương pháp text-only hoặc RAG không tìm ra.
- KG-TABI tạo ra ít gaps hơn (46) nhưng **97.8% là độc bản** (45/46), so với B2 tạo 300 gaps nhưng chất lượng NLI chỉ 65.7%.
- Đồ thị tri thức gồm **550 nodes**, **425 edges**, phát hiện **4 cụm mồ côi** và **40 khái niệm đình trệ**.

---

## 🔧 Tùy chỉnh nâng cao

### Thay đổi chủ đề nghiên cứu

```powershell
python -m src.main --topic "Federated Learning privacy" --limit 15
```

### Điều chỉnh ngưỡng thuật toán

Mở file `src/config.py` và chỉnh các tham số:

```python
RELEVANCE_THRESHOLD = 0.7          # Ngưỡng sàng lọc bài báo (0.0 → 1.0)
TRIPLE_CONFIDENCE_THRESHOLD = 0.3  # Ngưỡng tối thiểu cho bộ ba
FUZZY_MATCH_THRESHOLD = 85         # Ngưỡng khử trùng lặp ký tự (%)
COSINE_SIMILARITY_THRESHOLD = 0.85 # Ngưỡng khử trùng lặp ngữ nghĩa
LOUVAIN_MIN_SIZE_RATIO = 0.05      # Kích thước tối thiểu cụm mồ côi (%)
LOUVAIN_MAX_BRIDGE_RATIO = 0.10    # Tỉ lệ kết nối tối đa để coi là cô lập
TEMPORAL_DECAY_THRESHOLD = 0.30    # Ngưỡng suy tàn khái niệm (%)
```

### Xử lý giới hạn tốc độ API (Rate Limit)

Hệ thống đã tích hợp cơ chế **exponential backoff** tự động trong `src/llm_client.py`:
- Mỗi lần gọi LLM đều có khoảng nghỉ 2 giây.
- Khi gặp lỗi 429 (quota exceeded), hệ thống tự động đọc thời gian chờ từ API và sleep tương ứng.
- Tối đa 7 lần retry trước khi báo lỗi.

Nếu dùng **Gemini Free Tier** (15 RPM), toàn bộ pipeline mất khoảng **10–15 phút**.
Nếu dùng **API trả phí**, pipeline hoàn thành trong **1–2 phút**.

---

## 📄 Trích dẫn

Nếu bạn sử dụng framework này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@inproceedings{kgtabi2026,
  title     = {KG-TABI: Automating Research Gap Detection via Dynamic
               Knowledge Graphs and Toulmin-Abductive Inference},
  author    = {Le, Anh Hoa and Nguyen, Duc Hoang and Phan, Ly Van Khoa
               and Nguyen, Dinh Thanh and Ho, Dinh Anh and Truong, Long},
  booktitle = {Proceedings of the International Conference on Software Engineering},
  year      = {2026}
}
```

---

## 📝 License

Dự án này được phát triển phục vụ nghiên cứu học thuật.
