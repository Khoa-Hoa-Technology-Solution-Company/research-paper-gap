# Báo cáo công việc đêm 07-08/10/2026

Bài: *Absence of Evidence Is Not a Research Gap: Certified Gap Claims under Corpus Incompleteness* (`main.tex`, `main.pdf`).
Mọi bảng, hình và con số trong bài được sinh từ kết quả bằng `make_tables.py`, không chép tay.

## Đã sửa theo review (7.2/10)

| Điểm review | Đã làm |
|---|---|
| #1 Định lý 2 thiếu giả định | Thêm exchangeability vào phát biểu và chứng minh, thêm đoạn nói rõ bảo đảm phụ thuộc giả định này |
| #2 Benchmark chưa phải "research gap" thật | Gọi rõ là *controlled surrogate*; chuẩn bị file chấm tay (xem dưới) |
| #3 "No local signal" quá mạnh | Giảm mức khẳng định + thêm Proposition non-identifiability (có lập luận) |
| #4 Dùng lại tập test / gộp dữ liệu | Thêm giao thức train→test (hiệu chỉnh trên train, đánh giá trên test chưa dùng cho thiết kế chứng nhận) |
| #5 Giả định MCAR | Thêm thí nghiệm xóa có chủ đích (bằng chứng mạnh bị mất nhiều hơn) |
| #6 Thiếu 74 câu OpenAlex | Đã tải bù lúc 8h50 (chuỗi tự động bị treo do máy ngủ), đủ 1.109/1.109 câu, đã chạy lại toàn bộ |
| #7 Baseline yếu | Thêm BGE-large dense và BGE reranker |
| Số liệu không thống nhất | Abstract, kết luận dùng cùng nguồn (Bảng I) |

## Phát hiện mới (đã đưa vào bài, viết trung thực)

1. **BGE reranker có F1 cao nhất (0.713) khi chưa chứng nhận**, hơn ESV-Learned (0.687). Bài đã sửa: ESV chỉ dẫn đầu trong nhóm có trích câu bằng chứng (58% câu trùng nhãn vàng).
2. **Sau khi xóa bằng chứng, BGE cũng chỉ còn AUC 0.496 và 0.502**. Mô hình hiện đại nhất cũng không phân biệt được "đã có lời giải nhưng corpus thiếu" với "thật sự mở". Đây là bằng chứng mạnh cho luận điểm chính.
3. **Xóa có chủ đích (non-MCAR)**: bảo đảm gần như không đổi (trung bình lớn nhất khoảng 0.104).
4. **Train→test** (dữ liệu đầy đủ): giá trị lớn nhất đo được là 0.126 (ESV-Scope transfer 0.104). Tập test cố định có 188 câu CLOSED nên sai số chuẩn khoảng 0.022; 0.126 chỉ cao hơn mục tiêu 1,2 lần sai số chuẩn, và đó lại là giá trị lớn nhất trong 70 ước lượng. Bài viết đúng như vậy: không khẳng định có lệch phân phối, chỉ khuyến cáo lấy câu hỏi hiệu chỉnh cùng lĩnh vực.
5. **Mô hình chính xác nhất lại nguy hiểm nhất**: BGE reranker với ngưỡng tối ưu F1 có tỷ lệ gap giả 0.731 khi mất hết bằng chứng (cao nhất). Khi có chứng nhận, ranker mạnh giúp công nhận được nhiều gap thật hơn (44% so với 28% của ESV khi corpus đủ), nhưng lợi thế mất dần khi corpus thiếu. BGE reranker + OpenAlex có power cao nhất ở mức thiếu 30 đến 50%.

## Việc cần nhóm làm (máy không làm thay được)

1. **Kiểm tra hạn nộp trên CMT**: trang web hội nghị ghi hai hạn khác nhau (30/9 và 31/10).
2. *(Không bắt buộc, nhóm đã chọn bỏ qua)* **Chấm tay** theo `../gapclose/annotation/HUONG_DAN_CHAM.md` (77 câu bằng chứng + 60 câu NEI có bằng chứng ngoài; khoảng 1,5 đến 2 giờ cho 2 người). Chạy `audit_stats.py` để có số đưa vào Discussion. Đây là cách trả lời mạnh nhất cho điểm review #2.
3. **Đối chiếu 25 tài liệu tham khảo** (tôi viết từ trí nhớ, cần kiểm tra số trang, hội nghị).
4. **Link repo** cho câu "We release the benchmark, deletion protocol and code", hoặc đổi thành "will be released".
5. Email tác giả đầu: bài dùng `leanhhoa30012004@gmail.com`; Paper 63 cũ ghi `leanhhoa3002004`.
