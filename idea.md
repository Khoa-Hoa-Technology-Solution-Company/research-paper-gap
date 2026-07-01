# HƯỚNG DẪN TRIỂN KHAI: HỆ THỐNG PHÁT HIỆN KHOẢNG TRỐNG NGHIÊN CỨU (RESEARCH GAP) NGÀNH KỸ THUẬT PHẦN MỀM

**Mô tả:** Dự án này xây dựng một hệ thống tự động hóa quá trình Tổng quan Tài liệu Hệ thống (SLR). Nó kết hợp sức mạnh phân tích cấu trúc minh bạch của **Đồ thị Tri thức Động (Dynamic Knowledge Graph)** và khả năng suy luận ngôn ngữ tự nhiên của **Khung TABI (Toulmin-Abductive Bucketed Inference)** để phát hiện các khoảng trống nghiên cứu tiềm ẩn (implicit gaps).

---

## 1. CÔNG CỤ VÀ THƯ VIỆN CẦN CHUẨN BỊ
*   **LLMs:** `Llama-3.3-70B` (cho tác vụ suy luận phức tạp) và `Llama-3.1-8B` hoặc `GPT-4o mini` (cho trích xuất thực thể tốc độ cao).
*   **API Dữ liệu:** `Semantic Scholar API` (để tải metadata và bài báo).
*   **Thư viện Python:** 
    *   Xử lý văn bản: `Stanza` (cho chunking), `Sentence-BERT` (cho embedding).
    *   Đồ thị tri thức: `NetworkX` (để chạy thuật toán Louvain), `PyKEEN` (để chạy TransE).
    *   Giao diện: `Streamlit` (để xây dựng dashboard review cho chuyên gia).

---

## 2. QUY TRÌNH TRIỂN KHAI 5 GIAI ĐOẠN CHI TIẾT

### Giai đoạn 1: Thu thập và Sàng lọc (Literature Search & Screening)
Giai đoạn này tự động hóa việc tìm bài báo và chuẩn bị dữ liệu đầu vào.
1.  **Truy vấn tự động:** Dùng LLM sinh ra các câu truy vấn từ khóa liên quan đến một ngách trong Kỹ thuật phần mềm (VD: *Microservices architecture, Cloud-native security*). Gọi API Semantic Scholar để tải bài báo.
2.  **Sàng lọc (Relevance):** Đưa Title và Abstract của từng bài cho `Llama-3.1-8B` chấm điểm (0-1). Giữ lại các bài có điểm > 0.7.
3.  **Phân mảnh (Chunking):** Cắt nội dung bài báo thành các đoạn tối đa **1000 từ**, ngắt tại ranh giới kết thúc câu để bảo toàn ngữ nghĩa.

### Giai đoạn 2: Xây dựng Đồ thị Tri thức Động (Dynamic KG Construction)
Trích xuất thông tin để xây dựng bản đồ công nghệ phần mềm.
1.  **Trích xuất Bộ ba (Triple Extraction):** 
    *   Yêu cầu LLM trích xuất các bộ ba `<Chủ thể, Quan hệ, Đối tượng>` kèm theo Điểm Tự tin (Confidence > 0.3).
    *   *Giới hạn loại Thực thể (Entities):* `METHOD`, `DATASET`, `METRIC`, `CONCEPT`, `FINDING`, `TOOL`.
    *   *Giới hạn loại Quan hệ (Relations):* `USES`, `IMPROVES`, `ADDRESSES`, `EVALUATES_ON`, `PRODUCES`, `CONTRADICTS`, `EXTENDS`, `LACKS`, `COMBINES`, `APPLIED_TO`.
2.  **Khử trùng lặp (Deduplication):** 
    *   Dùng Fuzzy String Matching (ngưỡng 85) và Sentence-BERT Cosine Similarity (ngưỡng 0.85) để gộp các node đồng nghĩa thành 1 node duy nhất trên đồ thị.
3.  **Gắn nhãn thời gian (Temporal Tagging):** Gắn năm xuất bản của bài báo vào từng cạnh (edge) để phục vụ phân tích xu hướng.

### Giai đoạn 3: Phát hiện Điểm đứt gãy Cấu trúc (Topological Gap Detection)
Phát hiện lỗ hổng tự động bằng thuật toán toán học.
1.  **Tìm Cụm mồ côi (Orphan Clusters) bằng thuật toán Louvain:** 
    *   Chạy phân cụm Louvain trên đồ thị. 
    *   Xác định lỗ hổng: Một cụm được tính là "khoảng trống" nếu nó chứa > 5% tổng số node nhưng có tỷ lệ cạnh liên kết (bridges) với các cụm khác < 10%.
2.  **Phân tích Suy tàn Công nghệ (Temporal Decay):** 
    *   Tính toán tỷ lệ cạnh mới sinh ra so với lịch sử. Nếu một khái niệm suy giảm > 30% và bị đình trệ, đánh dấu nó là một hướng nghiên cứu bị bỏ ngỏ.

### Giai đoạn 4: Suy luận Khoảng trống bằng Khung TABI (Information Synthesis)
Biến các con số từ Giai đoạn 3 thành ngôn ngữ tự nhiên giải thích được thông qua RAG và LLM.
1.  **Trích xuất Bằng chứng (Subgraph Evidence):** Lấy danh sách các node thuộc 2 cụm bị đứt gãy (hoặc node bị suy tàn).
2.  **Prompting theo chuẩn TABI (In-context 3-shot):** Cung cấp 3 ví dụ mẫu cho LLM. Yêu cầu LLM output theo đúng JSON định dạng 4 phần:
    *   **Grounds:** Bằng chứng mạng con từ đồ thị (VD: "Cụm A không có liên kết với cụm B").
    *   **Claim:** Khoảng trống nghiên cứu được suy luận.
    *   **Warrant:** Giải thích logic chuyên môn tại sao sự kết nối này lại quan trọng.
    *   **Bucket:** Mức độ tự tin của LLM (more_probable / least_probable).

### Giai đoạn 5: Đánh giá bởi Chuyên gia (HCAI Review)
Tuân thủ tiêu chuẩn Trí tuệ Nhân tạo lấy Con người làm trung tâm (HCAI).
1.  Xây dựng giao diện bằng `Streamlit` hiển thị các Khoảng trống (Gaps) đã được tính điểm xếp hạng (Ranking).
2.  Hiển thị trực quan Subgraph làm bằng chứng bên cạnh mỗi Gap.
3.  Chuyên gia/Nghiên cứu sinh thực hiện thao tác: `Accept`, `Modify` (chỉnh sửa text), hoặc `Reject` (loại bỏ). Lưu lại log file `expert_reviews.json`.

---

## 3. CÁC MẪU PROMPT LÕI (CORE PROMPTS)

### Prompt 1: Trích xuất Bộ ba (Dùng cho Giai đoạn 2)
```text
You are an expert in Software Engineering. Read the following text chunk and extract triples in the format <Subject, Relation, Object>.
CONSTRAINTS:
- Entities MUST belong to: [METHOD, DATASET, METRIC, CONCEPT, FINDING, TOOL].
- Relations MUST belong to: [USES, IMPROVES, ADDRESSES, EVALUATES_ON, PRODUCES, CONTRADICTS, EXTENDS, LACKS, COMBINES, APPLIED_TO].
- Assign a Confidence Score (0.0 to 1.0) for each triple. 
- Provide a short "evidence_quote" from the text.

Text Chunk: {chunk_text}
Output format: JSON array of objects.
```

### Prompt 2: Suy luận TABI (Dùng cho Giai đoạn 4)
```text
Based on the topological analysis of the Knowledge Graph, the algorithm found a structural hole: 
There are ZERO bridge edges between Community A (Nodes: {nodes_A}) and Community B (Nodes: {nodes_B}).

Act as a Senior Software Engineering Researcher. Deduce an implicit research gap addressing this missing bridge using the TABI framework.
Output strictly as JSON:
{
  "Grounds": "State the graph evidence clearly (the disconnection between the specific nodes).",
  "Claim": "Formulate a clear research gap statement combining concepts from both communities.",
  "Warrant": "Explain the technical rationale: Why is bridging these two communities important for the Software Engineering field?",
  "Bucket": "Classify as 'more_probable' or 'least_probable' feasibility."
}
```