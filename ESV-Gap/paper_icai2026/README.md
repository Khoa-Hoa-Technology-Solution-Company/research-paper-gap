# Hồ sơ sửa bài ESV-Gap cho ICAI 2026

## Đọc theo thứ tự

1. `REVISION_PLAN_VI.md`: chẩn đoán feedback ban đầu, ma trận sửa đổi, hướng cải tiến và hai lộ trình theo thời gian.
2. `EVIDENCE_AUDIT.md`: số liệu theo từng run, sai khác bản thảo/ảnh feedback, hạn chế nhãn và kết quả replay.
3. `main.pdf` và `main.tex`: bản thảo tiếng Anh 5 trang, đã sửa theo feedback mới; source độc lập dùng IEEEtran và bibliography nhúng. Bản PDF bàn giao giống hệt được lưu tại `../../output/pdf/ESV-Gap_ICAI2026_Draft.pdf`.
4. `EXPERIMENT_PROTOCOL.md`: thiết kế nghiên cứu tiếp theo để kiểm nghiệm đóng góp discovery, không phải kết quả đã đạt.
5. `SUBMISSION_CHECKLIST_VI.md`: yêu cầu hội nghị và những việc tác giả phải hoàn thành trước nộp.
6. `MANUSCRIPT_NOTES.md`: phạm vi tuyên bố, trạng thái nguồn và lưu ý biên tập.
7. `expanded_run/multidomain_summary.json`, `expanded_run/run_multidomain_backtest.py`, `expanded_run/candidate_source_ablation.json`, `expanded_run/analyze_candidate_sources.py`, `expanded_run/bm25_lacks_baseline.json`, `expanded_run/run_bm25_lacks_baseline.py`, `expanded_run/bm25_query_sensitivity.json`, `expanded_run/run_bm25_query_sensitivity.py` và `expanded_run/BM25_MATCH_INSPECTION.md`: kết quả, lệnh chạy lại, baseline BM25 trên cùng pool LACKS, độ nhạy với ba query hậu kiểm, phân tách nguồn ứng viên và kiểm tra năm trường hợp khớp.
8. `expanded_audit/FEASIBILITY.md`: khả năng thu thập mới trên 300 bài và các điều kiện còn thiếu.

## Những gì đã thực hiện

- GPT-6 Sol được giao kiểm toán dữ liệu và viết lại bản thảo; GPT-6 Luna được giao kế hoạch cải tiến/protocol và phản biện thí nghiệm đa miền; agent chính chạy lại, kiểm tra chéo, biên dịch và bổ sung prototype.
- Giữ nguyên các bản thảo và raw runs cũ. Đầu ra nghiên cứu mới nằm trong thư mục này; bản PDF bàn giao nằm trong `output/pdf/`, ảnh kiểm tra PDF nằm trong `tmp/` của workspace.
- Prototype tại `prototype/` có 10 kiểm thử tổng hợp cho logic closure cùng phạm vi/cùng thí nghiệm. Không dùng kết quả test này để tuyên bố hiệu quả nghiên cứu.
- Lệnh audit tại `audit/` kiểm tra số đếm/hashes và replay temporal trên dữ liệu đã lưu. Thí nghiệm mới ở `expanded_run/` chạy lại mã hiện hành cho 378 bài không trùng trên bốn miền, 10 lát cắt thời gian, cùng một ngân sách top-20 với baseline. Phân biệt **recount artifact cũ** với **chạy lại mã hiện hành**; khác biệt silver labels giữa hai lần không phải cải thiện chất lượng.
- Baseline BM25 mới chỉ xếp hạng lại cùng pool limitation triples, không phải hệ truy hồi abstract độc lập. Macro proxy Recall@5/10/20 của Q1 là 0.0556/0.0694/0.0933, so với 0/0/0.0139 của ESV-Gap. Hai query hậu kiểm Q2/Q3 đạt Recall@20 = 0.0853, cho thấy kết quả nhạy một phần với wording. Năm heuristic matches Q1 được kiểm tra ở mức abstract và đã thêm citation/DOI trong bản thảo; không trường hợp nào được xác nhận độc lập là giải quyết cùng phạm vi.

## Trạng thái bàn giao

Đây là bản thảo đầy đủ để tác giả duyệt, **không phải chứng nhận sẵn sàng nộp hoặc bảo đảm accept**. Đã có thí nghiệm ngoại tuyến bốn miền, nhưng chưa có đánh giá chuyên gia độc lập, thu thập mới theo cùng protocol, kiểm chứng chứng nhận research gap hoặc thực nghiệm chứng minh phương pháp residual-evidence nâng chất lượng discovery. Đã điền tên tác giả duy nhất Lê Anh Hòa, FPT University, Vietnam và email `leanhhoa3002004@gmail.com` theo thông tin người dùng cung cấp. Phân tách nguồn cho thấy 39/39 ứng viên là explicit limitation; nhánh typed evidence-map cho 0 ứng viên. Không có thao tác nộp bài, gửi email hoặc công bố artifact.

Định vị bản thảo là một nghiên cứu kiểm toán hồi cứu đa miền và phân tích thất bại, không coi prototype chưa đánh giá là đóng góp thực nghiệm. Hướng discovery mạnh hơn cần nghiên cứu tiếp theo với nhãn chuyên gia và baseline truy hồi độc lập.

Trình biên dịch LaTeX tích hợp đã được thử nhưng báo lỗi thư mục nền tảng. PDF được tạo bằng MiKTeX có sẵn, tắt tự động cài gói; không cài thêm TeX/plugin. Giữ source trong editor để tiếp tục chỉnh sửa. Bản hiện tại dài 5 trang A4 do thêm năm citation truy vết; vẫn dưới giới hạn hội nghị.
