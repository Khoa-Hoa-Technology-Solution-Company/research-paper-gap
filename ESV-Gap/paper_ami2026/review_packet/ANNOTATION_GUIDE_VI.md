# Hướng dẫn đánh giá độc lập

## Phạm vi

Đây là 188 cặp candidate-evidence được chuẩn bị để đánh giá pilot, không phải gold dataset. Hai người nhận `reviewer_A.csv` và `reviewer_B.csv` riêng. Không trao đổi nhãn trước khi hoàn tất; không mở `organizer_manifest.json` vì file này chứa rank/score và mapping. Packet có nhiều cặp dùng chung candidate/paper, nên không coi 188 hàng là 188 mẫu độc lập.

## Quy trình

1. Đọc candidate subject/capability và origin span. Nếu nhiệm vụ, điều kiện hoặc outcome chưa cụ thể, chọn `UNDERSPECIFIED_CANDIDATE`; không suy đoán ngưỡng thành công.
2. Hai chuyên gia cần xác lập scope contract từ nguồn trước cutoff, trước khi đọc các future-control outcomes. `scope_contracts_unannotated.json` là template chưa đủ điều kiện chạy certification; các threshold phải có lý do chuyên môn được lưu lại.
3. Đọc title/abstract bằng chứng. Chỉ chọn closure nếu một thí nghiệm đáp ứng **toàn bộ** hợp đồng đã freeze. Nếu abstract không đủ để xác định, dùng `INSUFFICIENT_EVIDENCE`, hoặc thu thập full text vào bộ đánh giá được quản lý phiên bản riêng.
4. Điền nhãn, quoted source span, experiment ID, reviewer ID và notes. Một span phải là substring thật của abstract trong packet; scorer hiện tại chấm abstract-only. Đối với full-text study, cần nâng schema/source snapshot tương ứng, không dán quote full text vào trường abstract.
5. Đóng băng hai bản CSV, chấm agreement, rồi adjudicate disagreement. Kappa cao không tự chứng minh nhãn đúng. Người đánh giá phải thực sự có chuyên môn; script chỉ kiểm tra identity strings, không xác minh được độc lập ngoài đời.

## Nhãn

| evidence_label | Khi dùng |
|---|---|
| `SAME_SCOPE_CLOSURE` | Có một thí nghiệm đáp ứng tất cả task/domain/conditions/outcomes đã xác định; `scope_sufficient=yes`, cần span và experiment ID |
| `PARTIAL_EVIDENCE` | Có thí nghiệm đúng hướng nhưng thiếu/không đạt một số outcome hoặc conditions |
| `SUPPORTING_LIMITATION` | Nguồn nói vấn đề vẫn là limitation, không báo cáo closure trong scope |
| `TOPICAL_ONLY` | Có chung từ khóa nhưng không chứng minh quan hệ thực nghiệm cần kiểm tra |
| `INSUFFICIENT_EVIDENCE` | Abstract/metadata không đủ để đưa ra phán quyết |
| `UNDERSPECIFIED_CANDIDATE` | Câu hỏi chưa đủ rõ để biết thế nào là được giải quyết |

`scope_sufficient` dùng `yes`/`no`. `reviewer_id` nên là mã người đánh giá; không cần thông tin cá nhân trong bản chia sẻ. `source_span` bắt buộc cho closure/partial/limitation. `notes` ghi residual obligations, missing data, full-text need hoặc metadata concern. Không biến “không tìm được witness” thành nhãn research gap mới.

## Giới hạn bộ chấm

`score_scope_review.py` chấm agreement trên labels của **toàn bộ packet**; packet chưa được điền sẽ bị từ chối. Script chưa đo system precision/recall, chưa adjudicate hoặc kiểm tra entailment. Để đo controller, cần chuyển các annotations đã adjudicate sang records, freeze predictions, xây dựng gold labels riêng, và báo cáo coverage cùng closure precision. Không dùng output synthetic tests như ground truth của những hàng này.
