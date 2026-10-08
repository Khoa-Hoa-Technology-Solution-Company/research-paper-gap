# Tự phản biện bản đang mở — 01/10/2026

## Kết luận

**Chưa đủ căn cứ để gọi khả năng accept là cao.** Nếu phản biện theo đóng góp thuật toán và đánh giá quyết định closure, tôi vẫn nghiêng về **weak reject**. Nếu được xét như một bài hệ thống nhỏ về thu thập bằng chứng và điều khiển có phạm vi, bài hiện ở vùng **borderline**. Đây là đánh giá chuyên môn của chính trợ lý đã sửa bài, không phải phản biện độc lập, tỷ lệ accept của AMI, hay xác suất được hiệu chỉnh.

Đã làm thêm sau lần review trước: triển khai retrieval trên full text, chạy dữ liệu expert-labelled công khai, báo cáo đủ baseline/ablation, kiểm tra ngân sách và chuyển nhóm dữ liệu. Kết quả mới củng cố một thành phần của hệ thống; chưa giải quyết được thiếu ground truth cho closure.

## Sau phản biện bạn gửi

Phản biện được gửi dựa trên PDF export cũ 11 trang, không bao gồm thí nghiệm Context24 đã thêm vào source. Các nhận xét về thiếu real-literature closure evaluation vẫn đúng. Đã tiếp tục sửa source hiện tại, giữ editor đang mở:

- Định vị scope-control contribution trong title, abstract và RQs; bỏ số liệu retrieval cũ ra khỏi abstract, giữ chúng trong failure analysis để minh bạch.
- Thêm system figure native LaTeX: proposed question → reviewed contract → acquisition → semantic review → single-experiment check → residual obligations/actions. Caption không gọi đây là end-to-end loop đã được đánh giá.
- Chạy lại trên ba miền, loại toàn MongoDB khỏi primary analysis: 325 records/2,545 triples, 8 folds/7 scored, 61 control rows/29 IDs, 38 candidates. ESV R20 0.0179, BM25 0.0485; regenerated candidates/controls/metrics khớp archive. Các kết quả bốn miền giữ ở secondary accounting. **Không gọi đây là provenance repair hoặc clean dataset**.
- Bổ sung các conditional properties và verifier: 8,192 constructed inputs pass; 180 state pairs có union đủ facets nhưng không có individual witness được giữ nonclosed. Thêm checks cho insertion khi witness key không trùng, permutation, repeatability và threshold tightening. Duplicate-key counterexample chứng minh unrestricted monotonicity không đúng.
- Cố định pilot 12 câu hỏi/53 cặp/40 paper IDs và cập nhật scorer để chấm đúng frozen manifest. CSV vẫn trống; scorer từ chối tạo score. Bạn xác nhận chưa có người hoặc nhãn.

Các sửa đổi giải quyết positioning, diagram và technical precision; primary numbers không còn phụ thuộc miền có record tranh chấp. **Verdict chưa chuyển sang strong accept** vì thiếu independent closure labels vẫn chưa được giải quyết. Kiểm chứng quy tắc có điều kiện không chứng minh semantic understanding của một scientific agent.

Tiêu chí tham chiếu: [AMI submission](https://ami.ctu.edu.vn/2026/submission) và [CFP](https://ami.ctu.edu.vn/2026/cfp): originality, technical quality, relevance, clarity, experimental validation, potential impact. Full paper 12–15 trang, short paper 6–11 trang, tính cả tài liệu tham khảo.

## Review theo tiêu chí

| Tiêu chí | Nhận định ở bản hiện tại | Rủi ro reviewer có thể nêu |
|---|---|---|
| Phù hợp | Có liên hệ với điều khiển quyết định của scientific agents | Chưa đo được hành vi agent hay vòng perception–action từ đầu đến cuối |
| Originality | Sự kết hợp scope contract, truy hồi và abstention có ý nghĩa hệ thống | Lexical prior và redundancy penalty là kỹ thuật đơn giản; không đủ căn cứ nhận là thuật toán retrieval mới |
| Kỹ thuật | Quy tắc closure minh bạch, giữ source spans và không ghép thí nghiệm | Controller tin semantic annotations; chứng minh điều kiện chỉ xác nhận logic quy tắc |
| Thực nghiệm | Có 42 claims/31 papers được chuyên gia annotate sẵn, sáu arms và split theo paper | Tập nhỏ; cùng hai labs; lexical metric; không phải official test hoặc end-to-end gap evaluation |
| Độ rõ | Abstract/contribution/limitations đã khớp loại bằng chứng thực sự có | Ba phần retrieval, controller và historical audit vẫn cần đọc kỹ để tránh hiểu là cùng một đánh giá |
| Tác động | Có thể hỗ trợ reviewer tìm ngữ cảnh và theo dõi nghĩa vụ còn thiếu | Chưa có bằng chứng giảm false closure, giảm workload hoặc tăng chất lượng câu hỏi |
| Định dạng | Source tiếp tục dùng LNCS, tác giả/đơn vị ẩn danh | Compiler tích hợp lỗi môi trường; số trang và bố cục bản mới chưa được xác nhận |

## Kết quả đã kiểm tra, kể cả kết quả không thuận lợi

- Context24 dùng expert method-context quotations có sẵn, revision `457d3b5cb4bb8ade34e37458f4900c6eae0959bb`; không tự tạo nhãn người.
- 42 claims, 31 papers; tất cả claims và full text của paper đang đánh giá bị loại khỏi prior training. Gold của claim đang đánh giá chỉ đi vào scorer.
- Mỗi arm dùng tối đa 512 whitespace words, cùng candidate windows. Không thay weights sau khi thấy kết quả.
- Claim-macro ROUGE-L: BM25 **0.1805**, TF-IDF **0.1857**, prior-only **0.2045**, BM25-diverse **0.1805**, hybrid **0.2102**, hybrid-diverse **0.2122**.
- Proposed–BM25 paper-macro difference **0.0393**, descriptive 95% paper-bootstrap **[0.0131, 0.0666]**. Đây không phải claim-macro difference 0.0317.
- Gain so với prior-only và hybrid không rõ: paper-bootstrap intervals của hai comparisons đều chứa 0. Không tuyên bố diversification là cải tiến đã được chứng minh.
- MegaCog: TF-IDF **0.1918** cao hơn proposed **0.1771**.
- Cross-lab transfer: proposed/BM25 **0.1873/0.1850** ở Akamatsu và **0.1504/0.1714** ở MegaCog. Prior không tổng quát ổn định sang nhóm khác.
- Budgets 256 và 768 vẫn cho proposed cao hơn BM25 trên tập tổng; đây là sensitivity post hoc, không model selection.
- 252 predictions được kiểm tra source offsets, đúng quote và word budget; arithmetic recount pass; LCS đối chiếu DP trên 10 prefixes pass. 6 retrieval tests và 25 policy tests pass.
- 109 claims trong official-test file không có gold contexts: không đánh giá official test. Metric của bài khác official scorer; không tuyên bố leaderboard/SOTA.
- Tập 188 cặp local review vẫn chưa có nhãn người. Không biến Context24 method quotations thành nhãn scope closure.

## Bằng chứng còn thiếu để thay đổi verdict

1. **Đánh giá closure có ground truth độc lập.** Hai người có chuyên môn cố định task/conditions/outcome thresholds từ nguồn trước cutoff, annotate source spans và experiment membership, rồi adjudicate disagreement. Không dùng kết quả tương lai để chọn threshold. Có sẵn pilot cố định `pilot_review/`: 12 câu hỏi/53 pairs. Đây là workload pilot, không bảo đảm đủ power hay precision. Bộ đầy đủ vẫn giữ 188 pairs.
2. **So sánh trên cùng evidence pool.** Đánh giá scoped controller với các rules đơn giản và ablations của same-experiment/semantic review/metadata. Freeze predictions trước khi mở gold labels. Báo cáo precision của closure, coverage, abstention và nhóm lỗi; không coi abstain là negative prediction đúng.
3. **Tách hai loại retrieval.** Context24 đang đánh giá tìm passages khi paper đã được cung cấp. Muốn chứng minh tìm được witness phải có tập witness documents được adjudicate; muốn chứng minh discovery phải có candidate-quality evaluation riêng.
4. **Generalization và giá trị vận hành.** Chạy thêm expert-labelled source contexts ở miền đích và một comparator retrieval mạnh nếu có tài nguyên. Đo review effort hoặc thời gian. Cross-lab result hiện tại không cho phép quảng bá prior sang bốn miền audit.
5. **Xuất bản và kiểm tra artifact.** Native compiler phải hoạt động để xác nhận trang/layout. Kiểm tra provenance/time metadata của audit; public artifact và independent reproduction chưa thực hiện.

Không có một mức accuracy tùy ý nào bảo đảm accept. Khi các endpoints trên có kết quả đáng tin và lợi ích so với baseline, verdict mới có cơ sở thay đổi. Tiếp tục chỉnh văn phong hoặc thêm thí nghiệm synthetic không thay được những nhãn độc lập còn thiếu.

## Trạng thái bản nộp

Đã sửa **chính `main.tex` đang mở**; không tạo bản thay thế hoặc xuất PDF riêng. Built-in compiler trả về `Unable to find standard directories for platform`. Đây là lỗi môi trường; không có log biên dịch để chứng nhận source không có lỗi hoặc layout đạt yêu cầu.

`main.pdf` và PDF export trước đó vẫn là bản cũ 11 trang, **không phản ánh sửa đổi hiện tại**. Không dùng chúng làm kết quả của vòng sửa này. Số trang bản source mới đang chưa xác nhận.

Deadline công bố là 30/09/2026 AoE, tương ứng 01/10/2026 18:59 UTC+7. Không có thao tác nộp bài hoặc xác nhận trạng thái CMT trong phiên này.
