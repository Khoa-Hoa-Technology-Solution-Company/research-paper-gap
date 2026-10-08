# Xử lý phản biện bạn gửi — 01/10/2026

Phản biện tham chiếu PDF `ESV-Gap_AMI2026_Scoped_Draft` cũ 11 trang. Source hiện tại đã có Context24; không dùng nhận xét về độ dài hoặc “chỉ synthetic tests” của PDF cũ để mô tả toàn bộ source mới. Nhận xét về controller chưa có nhãn thực vẫn đúng.

| Yêu cầu trong phản biện | Đã xử lý | Giới hạn còn lại |
|---|---|---|
| Pilot human evaluation | Freeze 12 câu hỏi/53 cặp/40 papers, hai packet chưa annotate; scorer nhận frozen manifest | Tác giả xác nhận chưa có người/nhãn. Chưa có human agreement hoặc real-literature closure metric |
| Positioning | Title/abstract/RQs mở bằng suppression và abstention theo scope; legacy R20 ra khỏi abstract | Không tuyên bố autonomous semantic review hoặc reduction of false closure đã đo được |
| MongoDB provenance | Loại cả miền khỏi primary diagnostic analysis và regenerate tám folds từ original inputs, giữ secondary accounting | Post hoc domain exclusion; không chữa DOI/year hoặc chứng nhận ba miền sạch |
| System figure | Thêm figure ngay trong main.tex, dùng LaTeX picture, không thêm package/assets | Native compiler lỗi môi trường, chưa xác nhận rendered layout |
| Technical depth | Conditional witness integrity; noncomposition; incomplete-search abstention; restricted insertion persistence; contract strengthening; repeatability | Rule guarantees có điều kiện. Không coi correctness-by-construction là theorem semantic truth hoặc một ML algorithm mới |
| Monotonicity/idempotence | Witness predicate ghi rõ dependence on evidence multiset; duplicate key có counterexample. Disposition/action permutation invariant; trace/hash có thể thay đổi | Không phát biểu unrestricted monotonicity hoặc idempotent autonomous loop |
| Full/short paper | Nội dung thêm là phương pháp, thí nghiệm, thuộc tính và sơ đồ; rút gọn historical audit, bỏ forced page break trước references | Chưa xác nhận số trang, không nhận source mới là short/full trước khi compile thành công |

## Kết quả mới của vòng này

**Primary diagnostic replay** (`primary_audit/primary_audit_report.json`): 325 source records, 2,545 triples, ba miền; tám completed folds, bảy scored; 61 non-independent control rows/29 distinct within-domain control IDs. 38 candidate rows, typed branch zero. ESV proxy recall R20 **0.0179**, BM25 **0.0485**. Regenerated candidates, controls và metrics bằng archive ở từng fold. Tên “primary” mô tả ranh giới kết quả trong bài sửa, không có nghĩa preregistered primary endpoint.

**Conditional property verification** (`controller_properties_report.json`): 8,192 constructed inputs pass. Mỗi record có sáu binary states; hai records distinct keys, hai ledger states. 180 state pairs có combined facets đủ nhưng không individual witness vẫn nonclosed. 64 unique-key insertions, 64 permutations, 64 repeats, 144 threshold-tightening combinations pass. Duplicate-key nonmonotonicity checked. Các kiểm tra này là kiểm chứng software trên inputs dựng, không phải thực nghiệm closure trên papers.

**Pilot preparation** (`pilot_review/PILOT_PROTOCOL.json`): 4 câu hỏi mỗi miền non-MongoDB, chọn earliest cutoff theo exact normalized subject/capability rồi seeded sample; giữ đủ evidence pairs. Không dùng future controls. Labels trống; kiểm tra scorer từ chối missing labels và không sinh agreement report. Bộ này ưu tiên tính khả thi cho annotators, không đại diện corpus hoặc bảo đảm statistical power.

## Đánh giá lại

Nhận định hiện tại vẫn **borderline/weak reject tùy kỳ vọng đóng góp**. Positioning, ảnh hưởng của disputed domain và technical specification đã được xử lý tốt hơn; phần khó nhất của phản biện — real-literature decision effectiveness — chưa có dữ liệu để giải quyết. Context24 cung cấp bằng chứng riêng cho within-paper grounding retrieval, không thay thế closure ground truth.

Không dùng lời “reviewer-proof”, “competitive short paper” hoặc một phần trăm accept như bằng chứng. [AMI guidelines](https://ami.ctu.edu.vn/2026/submission) công bố full 12–15/short 6–11 trang, không xác nhận một “poster paper” track riêng trên trang này. Tiêu chí và formatting là điều kiện cần; không bảo đảm accept.

Built-in compiler của main.tex vẫn báo `Unable to find standard directories for platform`. Current source chưa được chứng nhận compile/layout/page count; PDF cũ là stale. Không biên dịch PDF riêng, mở tab khác hoặc nộp bài trong vòng sửa này.
