# Hướng dẫn chấm tay (khoảng 1,5 đến 2 giờ cho 2 người)

Mục đích: trả lời hai câu hỏi của reviewer mà máy không tự phán được. Nên có 2 người chấm độc lập rồi đối chiếu, để báo cáo được độ đồng thuận.

## 1. `witness_audit.csv` (77 dòng)

Mỗi dòng là một câu hỏi mà ESV-Scope kết luận CLOSED, kèm câu bằng chứng nó trích ra.

- Cột `VALID_EVIDENCE`: điền `yes` nếu câu trích ra thật sự trả lời câu hỏi (ủng hộ hoặc bác bỏ), `partial` nếu chỉ liên quan một phần hoặc khác điều kiện/quần thể, `no` nếu không trả lời.
- Cột `DIRECTION`: `support` hoặc `refute` (bỏ trống nếu `no`).
- Không nhìn cột `matches_gold_rationale` khi chấm (có thể ẩn cột này trong Excel).

## 2. `nei_external_audit.csv` (60 dòng)

Mỗi dòng là một câu hỏi mà SciFact gán NEI (tức "chưa có bằng chứng" trong corpus 5.183 bài), kèm câu mạnh nhất tìm được từ OpenAlex.

- Cột `ANSWERS_QUESTION`: `support` hoặc `refute` nếu câu này trả lời câu hỏi, `no` nếu không.
- Cột `SAME_SCOPE`: `yes` nếu cùng đối tượng, can thiệp, kết cục; `no` nếu khác phạm vi (ví dụ chuột so với người).

## Sau khi chấm

Đổi tên file đã chấm thành `witness_audit_A.csv`, `witness_audit_B.csv` (người A, B), tương tự cho file NEI, rồi chạy:

```
cd ESV-Gap/gapclose/src
../.venv/Scripts/python audit_stats.py
```

Script in ra: tỷ lệ bằng chứng hợp lệ (tách theo trùng/không trùng nhãn vàng), tỷ lệ câu NEI thực ra đã có lời giải, và Cohen's kappa giữa hai người. Các con số này đưa vào mục Discussion của bài.
