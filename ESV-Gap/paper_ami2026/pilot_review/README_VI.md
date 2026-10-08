# Pilot review cố định: 12 câu hỏi, 53 cặp, 40 paper IDs

Đây là **instrument chưa có nhãn**, không phải kết quả đánh giá controller. Không dùng bộ này để báo accuracy hoặc Cohen's kappa khi chưa có hai người đọc độc lập.

## Cách chọn

Loại toàn miền MongoDB. Gộp các hàng cùng normalized subject/capability trong từng miền và giữ cutoff sớm nhất. Chọn tối đa bốn câu hỏi mỗi miền bằng seed 20261002, giữ toàn bộ evidence pairs đã chuẩn bị cho các câu hỏi đó. Tổng 4 IoT, 4 microservices, 4 handwritten math; 53 cặp. Không đọc future controls hoặc dùng labels/system judgments để chọn.

`PILOT_PROTOCOL.json` ghi source hashes và frozen manifest hash. Selection chỉ là một pilot từ legacy candidates; không đại diện literature, không đủ chứng minh statistical power, không thay fair proposal pooling.

## Quy trình cho hai người có chuyên môn

1. Giữ bản gốc CSV/JSON chưa annotate. Mỗi reviewer dùng một bản sao tên `reviewer_A_completed.csv` hoặc `reviewer_B_completed.csv`. Không xem organizer_manifest (rank/score) khi gán nhãn.
2. Trước khi mở evidence pairs, hai người độc lập đọc nguồn candidate trước cutoff để xác định task, conditions và outcome requirements hợp lý. Phải giải thích vì sao chọn thresholds. Consensus/adjudication tạo một contract chung, cố định version/hash. Nếu candidate không đủ thông tin, giữ underspecified; không sáng tác scope cho dễ tìm closure.
3. Candidate inputs trong packet chỉ cung cấp origin span. Nếu nguồn gốc không đủ để định scope, cần truy xuất nguồn đầy đủ và ghi provenance trước cutoff; không đọc future outcomes để chọn threshold.
4. Dùng contract chung để hai người annotate độc lập sáu labels: SAME_SCOPE_CLOSURE, PARTIAL_EVIDENCE, SUPPORTING_LIMITATION, TOPICAL_ONLY, INSUFFICIENT_EVIDENCE, UNDERSPECIFIED_CANDIDATE. Có source quote, experiment ID và reviewer ID riêng. Annotate cả cặp khó; không xóa hàng bất lợi hoặc hàng chưa rõ.
5. CSV/scorer hiện tại kiểm tra quotes có trong abstract được cung cấp. Nếu closure cần full text, ghi INSUFFICIENT_EVIDENCE ở abstract screening và tạo evidence JSON/source riêng cho vòng full-text. Không sửa abstract để chèn quote; không coi abstract không có metric là proof metric không tồn tại trong paper.
6. Adjudicate disagreements bằng người thứ ba hoặc consensus ghi lại. Kappa chỉ đo agreement, không phải độ chính xác scientific decision. Reviewer identities là khai báo, không phải xác thực độc lập của phần mềm.

## Chấm agreement cho đúng pilot

Chạy từ thư mục ESV-Gap sau khi cả 53 cặp đã được gán nhãn:

```powershell
python -B paper_ami2026/score_scope_review.py --manifest paper_ami2026/pilot_review/organizer_manifest.json --reviewer-a paper_ami2026/pilot_review/reviewer_A_completed.csv --reviewer-b paper_ami2026/pilot_review/reviewer_B_completed.csv --output paper_ami2026/pilot_review/human_agreement.json
```

Scorer từ chối thiếu nhãn, thiếu reviewer identity, cùng reviewer IDs giữa hai bộ, sai item set và closure không có experiment ID. Không chạy để tạo score từ template trống.

## Đánh giá controller sau agreement

- Hoàn thiện contracts/evidence theo schema `src/scoped_verification.py`; lưu source text, metadata/experiment review và facet quotes. Full-text evidence phải được đánh giá riêng, có ground truth độc lập cho closure.
- Cùng evidence pool và budget cho scoped rule, co-mention/action-rule baselines và ablations. Không dùng adjudicated gold judgments để tạo system predictions rồi báo prediction accuracy.
- Freeze predictions trước khi chấm với ground truth. Khi human facet annotations là đầu vào controller, phải nói rõ đây là đánh giá quyết định có annotations cung cấp, không phải tự động hiểu evidence.
- Báo adjudicated closure precision, decision coverage, abstention/review workload, errors và chi phí. Gold chưa đủ thì giữ unresolved; không biến chưa có witness thành negative truth. Precision undefined khi không có closure.

Không gửi packet cho bên ngoài hoặc liên hệ reviewer trong phiên này. Tác giả đã xác nhận hiện chưa có người hoặc nhãn độc lập.
