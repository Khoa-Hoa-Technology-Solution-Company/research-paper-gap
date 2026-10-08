# ICAI 2026: hồ sơ sửa bài và điều kiện trước khi nộp

Kiểm tra website ngày 29/09/2026. Đây là ICAI của CMC University, không phải các hội nghị khác trùng tên.

## Quy định đã xác minh

- Hạn toàn văn: **30/09/2026**. Trang công khai không nêu giờ đóng hoặc múi giờ; kiểm tra CMT/Ban tổ chức trước khi chờ đến cuối ngày.
- Thông báo kết quả: 20/10/2026; camera-ready: 30/10/2026; hội nghị: 03/12/2026, Hà Nội.
- Dùng template chính thức, tối đa **8 trang tính cả tài liệu tham khảo**; không tự thu nhỏ lề/cỡ chữ để lách giới hạn.
- Trang Authors dẫn đến IEEE Conference Template trên Overleaf. Bản mới dùng IEEEtran conference; cần kiểm tra PDF và thông tin tác giả trước khi nộp.
- Trang Submission yêu cầu họ tên, đơn vị, email và quốc gia khớp giữa bản thảo và CMT. Không suy diễn rằng hội nghị yêu cầu ẩn danh.
- Kỷ yếu có ISBN; không đồng nhất với cam kết IEEE Xplore, Scopus hoặc xuất bản tạp chí.
- Nhánh phù hợp nhất để cân nhắc: Information Technology and Communications; AI for Education chỉ khi framing và bằng chứng thực sự phục vụ nghiên cứu/đào tạo.

Nguồn chính thức:

- https://icai.cmcu.edu.vn/icai-2026/authors
- https://icai.cmcu.edu.vn/icai-2026/dates
- https://icai.cmcu.edu.vn/icai-2026/join
- https://icai.cmcu.edu.vn/icai-2026
- Template được hội nghị dẫn: https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn

## Trước khi tác giả bấm Submit

- [ ] Xác nhận PDF nào thực sự được hội nghị trước đánh giá. PDF trong thư mục và con số trong ảnh feedback không khớp hoàn toàn; không tự quy lỗi cho phản biện.
- [ ] Tên tác giả duy nhất đã được cung cấp: Lê Anh Hòa, FPT University, Vietnam, `leanhhoa3002004@gmail.com`. Bảo đảm thông tin và đơn vị/khoa (nếu CMT yêu cầu) khớp CMT. AI hỗ trợ không được đưa vào danh sách tác giả.
- [ ] Tác giả đọc và chịu trách nhiệm cho từng nhận định, bảng số liệu, trích dẫn, giới hạn và mô tả hỗ trợ AI.
- [ ] Không gọi dữ liệu đã trích xuất sẵn là bằng chứng kiểm định độc lập; không gọi test tổng hợp là thí nghiệm khám phá khoa học.
- [ ] Giữ rõ ranh giới giữa kết quả được chạy lại, kết quả lưu từ trước và phương pháp mới chưa được kiểm nghiệm trên corpus.
- [ ] Đối chiếu nhãn corpus/run, mẫu số, đơn vị, mốc thời gian và hash với EVIDENCE_AUDIT.md.
- [ ] Kiểm tra artifact không chứa API keys, thông tin riêng tư hoặc toàn văn không được phép chia sẻ. Lần làm việc này không công bố/upload artifact.
- [ ] Nếu muốn tăng bằng chứng thực nghiệm trước khi nộp, bổ sung nhãn phạm vi do tác giả/chuyên gia đọc độc lập và một baseline BM25 trên raw abstract; BM25 hiện tại chỉ rerank triples LACKS, không khẳng định so sánh độc lập text-vs-graph.
- [ ] Đối chiếu DOI, tác giả và venue của năm bài trong Table IV với trang xuất bản gốc; hai metadata IoT 2026 và MongoDB 2024 hiện chưa được xác minh trực tiếp ở nguồn gốc.
- [ ] Xác nhận bài cũ đã kết thúc quy trình đánh giá; không nộp song song trái chính sách và không trình bày bản đã xuất bản như bài mới.
- [ ] Kiểm tra tính phù hợp template, số trang thực tế, font nhúng, bảng không tràn, công thức và link hoạt động trong PDF cuối cùng.
- [ ] Kiểm tra CMT và chính sách AI/originality mới nhất; không có cam kết accept.

## Đánh giá sẵn sàng về khoa học

Bản mới là một bản thảo đầy đủ theo hướng **kiểm toán bằng chứng và phân tích thất bại**, không phải chứng minh thành công một hệ thống phát hiện gap. Chuyển hướng này làm tuyên bố trung thực hơn nhưng không tự giải quyết yêu cầu discovery của reviewer. Muốn theo đuổi đóng góp discovery mạnh cần hoàn thành nghiên cứu trong EXPERIMENT_PROTOCOL.md, đặc biệt là gold labels độc lập, baselines công bằng, đối chứng topology và kiểm tra nhiều miền.

Nếu hội nghị không phù hợp bài nghiên cứu hệ thống/negative-results, hoặc không hoàn thành các bước bắt buộc trước hạn, nên hỏi Ban tổ chức về gia hạn hoặc chọn vòng nộp khác. Không viết kết quả kỳ vọng thành kết quả đã có.
