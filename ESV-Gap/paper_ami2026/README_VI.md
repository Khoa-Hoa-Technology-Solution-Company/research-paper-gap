# Bản sửa ESV-Gap cho AMI 2026

Cập nhật: 01/10/2026, giờ Việt Nam. Đây là bản sửa để tác giả duyệt và một triển khai nghiên cứu có thể chạy. **Chưa có căn cứ để khẳng định bài đạt mức accept cao.** Review mới nhất nằm ở `SELF_REVIEW_VI.md`.

## Cập nhật mới nhất: benchmark expert-labelled và tự review

Sau phản biện bạn gửi, đã tiếp tục sửa title/abstract theo scope-control contribution, thêm system figure và conditional properties; chạy verifier trên 8,192 constructed inputs; và **regenerate primary audit ba miền** (325 records, không MongoDB) thay cho aggregate omission: ESV proxy R20 0.0179, BM25 0.0485. Historical bốn miền chuyển xuống secondary accounting; không gọi đây là corrected provenance.

Đã freeze pilot nhỏ `pilot_review/`: 12 câu hỏi/53 cặp/40 paper IDs. Hướng dẫn trong `pilot_review/README_VI.md`; `score_scope_review.py` có tùy chọn `--manifest` để dùng chính tập này. Bạn xác nhận chưa có người/nhãn; packet vẫn trống, không có agreement/closure score. Xem `REVIEW_RESPONSE_VI.md` để đối chiếu từng yêu cầu với thay đổi.

Hai kiểm tra mới có thể chạy từ ESV-Gap:

```powershell
python -B paper_ami2026/replay_primary_audit.py
python -B paper_ami2026/verify_controller_properties.py
```

Replay dùng Python với dependencies pipeline sẵn có (networkx, PyYAML, numpy, python-louvain); outputs nằm trong `paper_ami2026/primary_audit`, không thay historical runs.

Đã sửa trực tiếp `main.tex` đang mở, thêm phương pháp lấy methodological context từ full text và thí nghiệm Context24: 42 claims/31 papers, leave-one-paper-out. Claim-macro ROUGE-L F1 đạt **0.2122**, BM25 **0.1805**, TF-IDF **0.1857**, cùng ngân sách 512 words. Report có per-claim scores, source offsets, input/code hashes và paper-bootstrap. Đây là retrieval ngữ cảnh trong paper đã biết; chưa phải xác minh research gap hoặc numerical witness.

Prior đóng góp phần lớn gain. Paired-paper intervals khi so với prior-only hoặc hybrid không diversification đều chứa 0. Cross-lab transfer không ổn định: MegaCog transfer **0.1504**, thấp hơn BM25 **0.1714**. Không điều chỉnh weights sau khi thấy kết quả; giữ cả kết quả bất lợi trong bài. Không chấm official test thiếu gold, không nhận đây là leaderboard score.

- Code: `src/context_retrieval.py`.
- Protocol: `public_benchmark/CONTEXT24_PROTOCOL.md`.
- Results: `public_benchmark/results/context24_report.json`.
- Verification/budgets: `public_benchmark/results/verification_and_sensitivity.json`.
- Ablations/transfer: `public_benchmark/results/transfer_and_ablations.json`.
- Tổng 31 unit tests pass: 25 controller, 6 retrieval; 252 benchmark predictions pass source/budget checks.

Compiler tích hợp vẫn trả `Unable to find standard directories for platform`. **Số trang và layout source mới chưa xác nhận.** Không xuất PDF riêng trong vòng sửa này. PDF cũ 11 trang và các nhận định layout trước đây chỉ áp dụng cho phiên bản trước; không dùng PDF cũ làm bản nộp của source hiện tại.

Chạy từ `ESV-Gap`, sau khi có các data files đúng hashes trong download manifest:

```powershell
python -B -m unittest discover -s tests -p test_context_retrieval.py -v
python -B paper_ami2026/public_benchmark/run_context24.py
python -B paper_ami2026/public_benchmark/check_context24.py
python -B paper_ami2026/public_benchmark/check_transfer.py
```

Phần bên dưới giữ thông tin audit và workflow annotation. Các mục mô tả bản 11 trang là lịch sử của vòng sửa trước.

## 1. Đánh giá bản PDF bạn gửi

Bản `C:/Users/leanh/Downloads/ESV-gap submit AMI.pdf` có 15 trang, SHA-256 `4770fb65597dfa28b0d7c368ac053ad9ed1eb7efa5145e6e287c4e392a9f862e`, trùng bản Springer cũ trong workspace. Bản đã dùng đúng định dạng một cột và ẩn danh.

Nếu phản biện bản cũ theo tiêu chí đóng góp và thực nghiệm, tôi nghiêng về **weak reject**. Đây là nhận định chuyên môn sau khi đọc bài và kiểm tra artifact, không phải kết quả phản biện AMI hay xác suất thống kê. Ý tưởng kiểm soát các khẳng định thiếu bằng chứng có ích, nhưng các kết quả hiện tại chưa chứng minh hiệu quả phát hiện hoặc xác minh khoảng trống nghiên cứu.

| Điểm yếu | Tại sao ảnh hưởng đến đánh giá | Đã sửa / việc còn cần |
|---|---|---|
| “Ngăn 80%” từ năm trường hợp được chọn | Không đo hiệu quả từ đầu đến cuối; không có nhãn độc lập | Bỏ con số này khỏi đóng góp chính; công bố rõ kết quả audit và nguồn ứng viên |
| “100%” từ sáu tình huống lỗi | Chứng minh logic xử lý các tình huống đã dựng, không chứng minh an toàn thực tế | Tách kiểm thử phần mềm khỏi kết quả khoa học |
| Nhánh đồ thị chưa có lợi ích đo được | Replay bốn miền có 39 hàng ứng viên, đều từ explicit limitations; typed evidence-map bằng 0 | Không tuyên bố đồ thị nâng chất lượng discovery; cần ablation với cùng verifier |
| Không truy hồi được khác với vấn đề chưa được giải quyết | API, abstract và extractor có thể bỏ sót công trình | Tất cả trạng thái “không có witness” chỉ dẫn đến review; không chứng nhận novelty |
| Action verb hoặc co-mention được dùng như dấu hiệu closure | Review, đề xuất khái niệm và thí nghiệm khác phạm vi có thể bị coi là giải pháp | Triển khai hợp đồng phạm vi và yêu cầu một thí nghiệm đáp ứng đủ điều kiện |
| Evidence span có trong nguồn chưa chứng minh entailment | Có thể trích đúng chữ nhưng hiểu sai kết quả hoặc ghép sai thí nghiệm | Tách kiểm tra substring khỏi semantic review và experiment-membership review |
| Nhãn silver nhiễu | BM25 có proxy recall cao hơn nhưng năm matches chưa được xác nhận là closure | Giữ nhãn proxy, chuẩn bị bộ đánh giá độc lập và hard negatives |
| Một record MongoDB chưa rõ nguồn/năm | Có thể làm sai temporal split và score | Công bố hạn chế; bổ sung độ nhạy loại toàn miền MongoDB; chưa gọi đây là sửa metadata |
| Vocabulary saturation bị gán ý nghĩa quá mạnh | Bão hòa từ vựng không chứng minh đầy đủ bằng chứng thực nghiệm | Không dùng Heaps' law để nâng trạng thái thành research gap được chứng nhận |
| Tuyên bố trước đây về prior scientific agents quá rộng | ResearchAgent đã có đánh giá người và model | Viết lại related work, xác định đóng góp hẹp hơn |

## 2. Ý tưởng được triển khai

**Scope-aware evidence control:** biểu diễn câu hỏi bằng nhiệm vụ, miền, điều kiện thí nghiệm, yêu cầu kết quả và ngày cutoff. Một record của **cùng một thí nghiệm** phải đáp ứng tất cả các điều kiện để đóng câu hỏi trong phạm vi đó.

Ví dụ minh họa tổng hợp: phát hiện tấn công chưa biết trên thiết bị biên, F1 >= 0.90, latency <= 10 ms. Các số này chỉ dùng trong test, không phải ngưỡng khoa học đã được xác lập. Đạt F1 trong thí nghiệm A và latency trong thí nghiệm B không tạo thành witness hoàn chỉnh.

Mã tại `../src/scoped_verification.py`:

- Đòi hỏi hợp đồng phạm vi được duyệt; không tự tạo ngưỡng thành công cho câu hỏi mơ hồ.
- Kiểm tra source spans, ngày availability, task/domain, conditions, units, thresholds và membership trong cùng thí nghiệm.
- Giữ residual obligations cho bằng chứng chưa đầy đủ; không ghép các record khác nhau.
- Đòi hỏi ledger đầy đủ cho mọi probe đã khai báo; thiếu query, snapshot, pagination hoặc semantic review dẫn đến review.
- Một witness hoàn chỉnh có thể đóng câu hỏi dù probe khác lỗi; không cần tìm kiếm toàn bộ thế giới để bác bỏ một phát biểu “chưa có” trong phạm vi cụ thể.
- Luôn xuất `novelty_established: false`. `CLOSED_WITHIN_SCOPE` không phải chứng nhận chất lượng nghiên cứu hay novelty.

Đây là controller trên **annotations được tin cậy**. Cờ `reviewed=true` và substring không tự kiểm chứng được độ đúng của người gán nhãn. Mã mới là thành phần opt-in qua API/CLI; chưa thay decision engine lịch sử hoặc tích hợp vào giao diện Streamlit.

## 3. Những gì thực sự đã làm

### Kết quả được kiểm tra từ dữ liệu đã lưu

- Đếm lại 378 records và 3.188 triples; hashes của tám input khớp replay đã lưu.
- IoT gốc: 486 raw -> 191 screened -> 150 retained; 133 raw candidates -> 0 automatically eligible / 5 review / 128 reject.
- Replay đã có: bốn miền, mười cutoff evaluations, chín evaluations có positive controls; 64 control rows có 31 control IDs khác nhau trong miền. Không phải 64 mẫu độc lập.
- Proxy macro Recall@20: ESV-Gap 0.0139; confidence-LACKS 0; BM25-LACKS 0.0933. BM25 này xếp hạng **cùng pool đã extract**, chưa phải baseline discovery độc lập.
- Độ nhạy mới, loại toàn miền MongoDB khỏi aggregate đã lưu: bảy scored evaluations, 61 control rows; ESV-Gap 0.0179, BM25-LACKS 0.0485. Đây là kiểm tra aggregate hậu kiểm, không phải rerun sau sửa metadata.

### Công việc mới

- Controller có API `evaluate_scope(contract, evidence, ledger)` và CLI JSON.
- 25 kiểm thử dựng tổng hợp đã pass; không dùng chúng để tuyên bố hiệu quả khoa học.
- Truy hồi evidence bằng BM25 trên title + abstract gốc, chỉ sử dụng candidate subject/capability trước cutoff, bỏ supporting paper IDs; không sử dụng future controls để tạo query.
- Chuẩn bị 188 cặp từ 39 candidate rows, liên quan 97 paper IDs khác nhau. Candidate generation vẫn phụ thuộc extracted triples; chỉ bước evidence ranking độc lập với extractor.
- Hai CSV được xáo trộn riêng, ẩn score/rank/system judgment; nhãn người hoàn toàn để trống. Có 39 contract templates chưa annotate.
- Bộ chấm agreement từ hai người, kiểm tra thiếu nhãn, source span và reviewer identity; xuất Cohen's kappa và disagreements. Agreement không phải ground truth hoặc system accuracy.
- Bản tiếng Anh mới `main.tex` theo Springer, ẩn danh, chọn định vị audit + scoped-control research implementation.

Chưa làm: independent expert annotation cho scope closure, full-text witness validation, semantic extraction benchmark, calibration, dense neural baseline, end-to-end gap-discovery experiment, triển khai app, nộp CMT hoặc xuất bản artifact. Context24 sử dụng nhãn chuyên gia công khai có sẵn; không tạo nhãn người mới.

## 4. Chạy lại

Từ thư mục `ESV-Gap`, với Python phù hợp:

```powershell
python -B -m unittest discover -s tests -p test_scoped_verification.py -v
python -B paper_ami2026/audit_evidence.py
python -B paper_ami2026/prepare_scope_review.py
python -B -m src.scoped_verification --input paper_ami2026/examples/synthetic_example.json --output paper_ami2026/examples/synthetic_decision.json
```

Lệnh cuối chỉ minh họa fixture tổng hợp. Lệnh chuẩn bị sẽ tái tạo packet; không chạy lại vào bản CSV đã điền nhãn. Hãy sao chép bản gán nhãn sang tên riêng trước khi tái tạo.

Sau khi có hai CSV do hai người đánh giá hoàn thành:

```powershell
python -B paper_ami2026/score_scope_review.py --reviewer-a reviewer_A_completed.csv --reviewer-b reviewer_B_completed.csv --output paper_ami2026/human_agreement.json
```

Không có nhãn thì lệnh chấm từ chối tạo score. Kết quả agreement không thay cho adjudication hoặc đánh giá dự đoán controller. Cần hoàn thiện các scope contracts, chuyển annotations sang schema evidence và freeze system predictions trước khi chấm closure precision/coverage.

## 5. Đường đi để bài mạnh hơn

### Ưu tiên 1: xác định câu hỏi đủ cụ thể và kiểm tra witness

Hai người có chuyên môn đọc độc lập các nguồn trước cutoff để xác định task, conditions và outcome requirements. Không xem kết quả tương lai khi chọn threshold. Tiếp đó annotate các cặp evidence bằng source spans và experiment identity; đọc full text khi abstract thiếu. Người thứ ba/consensus giải quyết disagreement. Đối với điểm pilot này, nên ưu tiên chất lượng nhãn hơn thêm corpus không kiểm soát.

### Ưu tiên 2: baseline và ablation trả lời trực tiếp đóng góp

Freeze tập câu hỏi đủ scope. So sánh lexical co-mention, broad relation/action matching và scoped witness rule trên **cùng evidence pool**; không thay retrieval đồng thời với verifier. Tiếp đó so BM25/dense/hybrid với **cùng scope verifier**. Đánh giá tách biệt witness retrieval recall, closure precision, decision coverage, review workload, latency/cost. Báo cáo mẫu chưa có nhãn và denominator rỗng.

Đánh giá discovery cần thêm pool proposals từ raw-text baseline, explicit limitations và graph, cùng candidate budget và cùng verifier. Nhánh graph không tạo được positive utility thì bỏ khỏi đóng góp chính. Controller hiện tại chỉ suppress/route một câu hỏi, chưa chứng minh tự khám phá câu hỏi mới tốt hơn.

### Ưu tiên 3: tính hợp lệ thời gian và provenance

Kiểm tra lại nguồn/năm MongoDB, đối chiếu DOI/title/preprint và first-online dates, freeze corpus/search snapshots và rerun với phiên bản mới. Giữ cả report cũ và mới cùng hashes. Đây mới là corrected-metadata replay; phép loại toàn miền trong bản sửa không thay thế nó.

Không chọn một tỷ lệ accuracy đẹp làm tiêu chuẩn accept. Mục tiêu là nhãn có thể kiểm chứng, comparator công bằng và giới hạn tuyên bố đúng với kết quả. Nếu scoped rule không cải thiện precision hoặc tăng abstention quá mức, phải báo cáo trade-off hoặc chỉnh phương pháp dựa trên tập development riêng.

## 6. AMI 2026 và quyết định nộp

Theo trang chính thức kiểm tra trong phiên này:

- Deadline: 30/09/2026 23:59 AoE = **01/10/2026 18:59 giờ Việt Nam**.
- Full paper: 12-15 trang; short paper: 6-11 trang, gồm references/appendices.
- Springer LNCS/CCIS một cột, tiếng Anh, double-blind.
- Không nộp khi bài đang được xét ở nơi khác. Thư mục `paper_icai2026` có artifact của một đợt chuẩn bị khác; sự tồn tại của thư mục không chứng minh trạng thái đã nộp. Tác giả cần xác nhận trạng thái thực tế.
- [Submission guidelines](https://ami.ctu.edu.vn/2026/submission), [CFP](https://ami.ctu.edu.vn/2026/cfp), [deadline AoE trên trang chính](https://ami.ctu.edu.vn/2026/).

Phiên bản trước có 11 trang, thuộc **short paper**. Phiên bản hiện tại có thêm phương pháp/thí nghiệm và **chưa xác nhận số trang** do native compiler không hoạt động; phải kiểm tra giới hạn sau khi compiler chạy được. Scope thuộc scientific-agent decision systems, nhưng độ phù hợp chủ đề không bù được thiếu validation. Không tự chuyển track hoặc nộp hộ.

Nếu cần nộp ngay trong hạn, bản mới giúp giảm overclaim và làm contribution có thể kiểm tra. Đánh giá hiện tại vẫn có rủi ro bị reject do thiếu real-literature evaluation của controller. Để thành bài có bằng chứng thực nghiệm mạnh cần hoàn tất các ưu tiên ở trên; công việc con người chưa thể thay bằng nhãn do AI tự tạo.
