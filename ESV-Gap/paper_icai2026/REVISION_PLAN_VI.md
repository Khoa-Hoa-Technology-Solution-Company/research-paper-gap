# Kế hoạch cải thiện ESV-Gap theo phản biện

## 1. Chẩn đoán ngắn gọn

Hai phản biện đồng thuận về điểm mạnh: quy trình có provenance, có thể kiểm toán, có cơ chế fail-closed và minh bạch giới hạn. Điểm yếu quyết định là bằng chứng hiện tại chưa chứng minh hệ thống tìm được research gap hữu ích. Cần giữ riêng ba kết quả temporal: (i) screenshot review nêu 0 recall trên 22 positive controls và 153 tín hiệu; (ii) archived summary cho cùng primary frozen corpus/triples hashes báo 108 positive controls, macro candidate Recall@20 = 0.0238 qua ba cutoff (0/0/0.0714); (iii) replay cùng hash-verified inputs dưới code hiện tại báo 36 positive controls (14/14/8), macro candidate Recall@20 = 0.0417 (0/0/0.125), 6 negative controls và 0 candidates ở cutoff 2022/2023. Đây **không phải cải thiện**: số control/label thay đổi do chưa đối soát code/config drift trong control generation; nguyên nhân chưa được xác định. Screenshot figures cũng chưa đối chiếu được với hai artifact temporal. Cả archived và replay reports cho thấy 0 certified candidates; archived `certificate_precision` null/not estimable, còn replay `certified=0` là trường cố định chứ không phải ước lượng precision. Không đánh đồng candidate retrieval với certification. Ảnh review ghi 153 tín hiệu, trong khi run IoT khác trong repo được báo là 133; xác minh run ID/raw output trước khi thống nhất mẫu số. Review sau cũng xếp ESV-Gap gần với hệ thống sàng lọc/kiểm chứng hơn là hệ khám phá. Đây là vấn đề thực nghiệm và định nghĩa bài toán, không giải được bằng viết lại lời quảng cáo.

Trong repo đã có revision lớn cho một hướng khác: `paper_v2` mô tả benchmark controlled, ranking metrics, ablation, một temporal backtest và các hàng đợi expert review; tuy nhiên chính `RESPONSE_TO_REVIEWERS.md` và `REPRODUCIBILITY.md` nói đánh giá chuyên gia hiện chỉ do một reviewer nội bộ, không mù/độc lập; không có kiểm chứng đa miền; corpus 53 bài thiếu bản ghi truy xuất gốc; và chưa có độc lập đánh giá ngoài. Những báo cáo này không được gộp với run 150 bài hoặc trình bày như thể chúng xác nhận cùng một kết quả. Trước khi sửa manuscript, cần lập bảng đối chiếu run ID, corpus, snapshot date, code/config hash, output và nguồn số liệu cho từng con số.

Thêm một trạng thái cần phân biệt: prototype `paper_icai2026/prototype/scoped_contract.py` hiện kiểm tra tính nhất quán của contract có phạm vi (task, điều kiện và bounds của metric trong cùng experiment; evidence không sau cutoff; span do người gán tin cậy). 10 synthetic tests pass theo báo cáo trong thread. Đây là kiểm tra hợp đồng đầu vào trên dữ liệu tổng hợp; chưa chứng minh được entailment NLP, novelty, chất lượng retrieval, hay cải thiện thực nghiệm ngoài miền.

## 2. Phương án định vị và đóng góp

### Định vị khả thi cho hạn nộp ngày mai

Đề xuất trình bày bài như một nghiên cứu hệ thống về **evidence-grounded research-gap triage**: hệ thống tách *candidate retrieval* (tìm đúng cơ hội cần xem) khỏi *candidate verification* (đánh giá bằng chứng cho từng cơ hội), và chứng minh được điều gì, chưa chứng minh được điều gì. Không gọi framework hiện tại là autonomous scientific discovery nếu recall phát hiện và expert-confirmed utility vẫn chưa đạt. Đổi tiêu đề/abstract/introduction để “triage”, “auditable verification” hoặc “temporal evaluation” là trọng tâm; “discovery” chỉ là mục tiêu tương lai.

Giải pháp có tiềm năng tạo đóng góp nghiên cứu mạnh hơn là protocol đánh giá **retrieval–verification có kiểm soát thời gian, đối chứng từng phần (partial closure), calibration và so sánh giá trị tăng thêm của KG**. Đóng góp này chỉ được tuyên bố như đề xuất/protocol cho đến khi dữ liệu và thí nghiệm mới hoàn tất. Không có thủ thuật thuật ngữ nào bảo đảm được accept.

### Track A — trung thực, có thể hoàn tất trước hạn

1. Xác nhận template, giới hạn 8 trang, tác giả, quyền sử dụng dữ liệu và hạn cuối trên trang ICAI/CMT; trang chính thức hiện ghi 30/09/2026. Dùng đúng một submission.
2. Chọn đúng một manuscript/run làm nguồn kết quả. Không trộn 150-paper IoT result với 53-paper MongoDB-security run, paper_v2 benchmark cũ hoặc các output khác.
3. Đổi claim chính: hệ thống kiểm toán và triage tín hiệu; chưa có bằng chứng đủ để tuyên bố discovery utility hữu ích. Nêu riêng metric/run cụ thể: screenshot (0/22), archived summary (macro candidate Recall@20 = 0.0238; 108 positives), current-code replay trên cùng frozen inputs (macro = 0.0417; 36 positives). Điều tra label/control code drift trước khi so sánh; không mô tả chênh lệch là cải thiện. Certification: không có candidate được chứng nhận; archived precision không estimable, replay count bằng 0 không phải precision estimate.
4. Xóa/đánh dấu là “planned” các câu nói về expert validation độc lập, discovery capability, generalization hoặc đóng góp topology nếu hiện không có kết quả trực tiếp hỗ trợ.
5. Thêm bảng “claims–evidence–limits”, baseline tối thiểu chỉ khi cùng run, cùng corpus, cùng search budget và có output lưu vết. Nếu chưa hoàn tất comparison thì nói rõ.
6. Thêm limitations và reproducibility: corpus retrieval ledger thiếu ở đâu, snapshot date, model/API, cutoff, query, top-k, dữ liệu chỉ abstract hay full text, lỗi/abstention; đính kèm artifact hiện có và hash mà không tạo hash giả.
7. Kiểm tra lại yêu cầu ICAI và độ dài. Track này có thể thành submission minh bạch về engineering/evaluation, nhưng rủi ro bị từ chối vẫn đáng kể vì reviewer đòi evidence discovery tích cực.

### Track B — nghiên cứu mạnh hơn trong 4–6 tuần

| Tuần | Công việc | Tiêu chí đầu ra |
|---|---|---|
| 1 | Chốt task, ontology cho hypothesis có phạm vi, preregistration, snapshot/cutoff và inclusion criteria. Chọn ≥3 miền khoa học có nguồn metadata/abstract ổn định. | Protocol đóng băng; data dictionary; manifest truy xuất |
| 1–2 | Thu thập theo truy vấn đã lưu, deduplicate theo DOI/ID, ghi ngày truy xuất/coverage; lấy các time slice; tạo train/dev/test theo **publication date và paper ID**. | Corpus audit được; không dùng bài tương lai khi tạo candidate |
| 2–3 | Xây gold set theo pooling từ mọi retrieval baseline; ít nhất hai annotator độc lập gắn nhãn relation scope/closure, một adjudicator; blind hệ thống và nguồn candidate. | Agreement (Krippendorff’s α hoặc κ phù hợp), rationale spans, bất đồng được adjudicate |
| 3–4 | Chạy retrieval baselines, verifier, ablation no-KG/KG; tune trên dev only; hiệu chuẩn xác suất/abstention trên calibration split. | Metrics per domain/cutoff; paired bootstrap CIs; full audit traces |
| 4–5 | Blind expert utility study, error analysis, subgroup/domain shift, compute/API cost. | Expert ratings + agreement + uncertainty; cost/latency |
| 5–6 | Independent reproduction, rerun frozen command, paper rewrite, artifact release/license. | Manifest + reproducible outputs + manuscript with claims tied to evidence |

## 3. Ma trận feedback → sửa đổi

| Feedback | Chẩn đoán từ bằng chứng hiện có | Sửa cụ thể | Bằng chứng bắt buộc trước khi claim mạnh hơn |
|---|---|---|---|
| Screenshot nêu 0/22; archived summary trên frozen inputs báo 108 controls, macro Recall@20=.0238 (0/0/.0714); current-code replay trên chính input hashes báo 36 controls, macro=.0417 (0/0/.125), 6 negatives. Candidate count là 0/0/16 ở cả summary và replay. | Không được kết luận replay cải thiện: control-generation labels/denominators thay đổi vì chưa đối soát code/config drift. Cả hai snapshot đều có 0 certified candidates; archived certificate precision null/not estimable. | Ghim hash của corpus/triples, commit/config/prompt và implementation của control-generation; diff từng control ID/label, report side-by-side; sau đó đóng băng test protocol và đánh giá lại. Báo candidate-vs-certificate distinction. | Candidate Recall@k so với baselines chỉ sau khi ground truth/control definitions được đối soát và đóng băng; certification cần historical snapshots + blinded labels. Nếu chưa giải quyết được drift, nêu tất cả artifact và giới hạn, không gộp metrics. |
| Core graph topology components không tạo final hypothesis đã validate | Có thể KG chỉ giúp lưu evidence/provenance hoặc gate, chưa có contribution về phát hiện. | Phân tách retrieval và verification. So sánh cùng candidate pool: graph structure on/off; node/edge features; TransE/Louvain chỉ là baseline khi cùng budget. Chứng minh KG cải thiện recall/ranking hoặc verification quality/cost. | Ablation paired trên held-out set; CIs cho ΔRecall@k/nDCG@k, expert utility, cost; nếu Δ không khác 0 thì mô tả KG là audit/provenance layer, không discovery engine. |
| 150 bài, một domain, một frozen run | Không đủ để suy rộng; không nhất thiết cần số bài tùy tiện lớn hơn nếu positive/negative labels hiếm. | Dùng ≥3 domain, nhiều temporal cutoffs và nhiều query/topic; ghi snapshot, coverage; phân tích theo domain. Xác định cỡ đánh giá từ precision mong muốn/độ rộng CI và khả năng tuyển expert, không chọn N sau khi xem kết quả. | Kết quả per-domain, macro average, leave-one-domain-out; dữ liệu retrieval rõ; ngoại suy giới hạn theo coverage thật. |
| Không chứng minh practical discovery utility, và chưa có positive discovery | Chuyển mục tiêu từ “đã tìm ra gap” thành “tìm candidate hữu ích để chuyên gia xem” cho đến khi xác nhận được. | Đo candidate recall dưới review budget k, thời gian review tiết kiệm, actionable/novelty rating và chất lượng evidence. Tách output hợp lệ của system khỏi utility của hypothesis. | Expert-blind randomized comparison với baselines ở cùng budget; inter-rater reliability; phân tích ứng dụng/effort, không chỉ số candidate được sinh. |
| Nhầm retrieval đồng xuất hiện từ khoá với quan hệ giải quyết gap | Co-mention không phải counterevidence; một paper có thể chỉ giải quyết một điều kiện con. | Retrieval stage lấy pooled candidate documents (lexical/BM25+dense/citation/graph). Verification stage gán SUPPORTS/REFUTES/PARTIAL/MENTIONS/NOT-ENOUGH-EVIDENCE cho một scoped claim, bằng sentence evidence. | Retrieval recall và verifier performance đo riêng. Report evidence-span support và partial-scope error. |
| Thiếu expert-based validation | Internal unblinded rating có nguy cơ confirmation bias; không phải gold truth. | Ít nhất 2 chuyên gia độc lập trên mỗi item và adjudication người thứ 3; che nguồn hệ thống/ranking; chấm scope, closure, novelty-in-corpus, usefulness, evidence sufficiency riêng. | Agreement + confusion by label; blinded audit packet/version; disclose conflicts/affiliations; CI theo clustered item/domain. |
| Threshold/policy lựa chọn thủ công, tùy tiện | Threshold nhìn có vẻ chính xác nhưng chưa chứng minh tradeoff. | Chỉ tune thresholds trên dev/calibration; báo risk–coverage và precision–review workload; đặt abstention khi bằng chứng ngoài phạm vi, incomplete hoặc confidence chưa hiệu chuẩn. | Frozen thresholds, calibration split độc lập, sensitivity analysis; chỉ báo Brier/ECE khi có numeric scores được hiệu chuẩn bằng human labels tách biệt, không lấy verbal confidence của LLM làm xác suất. Không tuyên bố bảo đảm distribution-free nếu không dùng protocol chứng minh điều đó. |
| Giá trị KG chưa rõ, TransE/Louvain có vẻ dư thừa | Topology component không hữu ích nếu không tác động tới outcome/efficiency. | Ablation theo matched candidates và query/document budget: BM25 only, dense only, hybrid, KG-only, hybrid+KG; gate/provenance-only; remove TransE/Louvain; random graph control. | Paired bootstrap/permutation test; candidate overlap and utility; compute/API cost. Report null findings. |
| Paper nêu “100% fail-closed” nhưng không nói tỷ lệ coverage hoặc real-world utility | Safety property hữu ích nhưng không thay discovery quality; một hệ thống abstain mọi lúc cũng pass fail-closed. | Báo coverage, yield, error risk among non-abstained, abstention reasons, fault-injection pass rate riêng. | Risk–coverage curve và failure categories; không nhập “safety pass” với “scientific success”. |

## 4. Ý tưởng đột phá có thể kiểm chứng (không phải novelty claim sẵn có)

Đề xuất **Scope-aware, retrieval-first temporal gap triage**: mỗi giả thuyết được biểu diễn như tuple có method, capability, dataset/setting, population/threat model, metric và thời điểm. Hệ thống tìm evidence trước, rồi phân biệt complete closure, partial closure, mere mention, contradiction và unknown. Nó ưu tiên truy xuất những bài có thể bác bỏ candidate; giữ nguyên `unknown` khi search chưa đủ và abstain khi quality/coverage thấp. Bộ đánh giá chấm riêng:

1. Có lấy được ít nhất một evidence item liên quan không? (retrieval recall@k)
2. Bằng chứng đó có giải quyết đúng **phạm vi** của hypothesis hay chỉ phần con? (verification/partial-closure quality)
3. Hệ thống tìm đúng các future problem-solution cases mà chuyên gia cho là hữu ích trước khi công bố không? (temporal anticipatory utility)
4. KG thêm được gì ngoài retrieval văn bản? (paired incremental-value ablation)

Điểm mới khả dĩ nằm ở protocol đánh giá kết hợp temporal outcomes với expert-judged partial closure và retrieval-budget utility; cần rà soát related work đầy đủ và có kết quả trước khi khẳng định novelty. SciFact và SciFact-Open là tiền lệ cho tách retrieval/evidence rationale và vấn đề evidence chỉ hỗ trợ trường hợp đặc biệt; LBD literature cũng lưu ý temporal prediction là silver/noisy proxy, không phải nhãn novelty tuyệt đối.

## 5. Nguyên tắc biên tập không được vi phạm

- Không bịa thêm experiments, experts, papers, citations, metrics, license hoặc DOI.
- Không viết “guaranteed acceptance”; quyết định thuộc hội đồng.
- Nêu rõ mọi số liệu thuộc run nào và ngày snapshot nào; không trộn repo artifacts khác nhau.
- Nếu giữ paper ở track A, câu kết luận phải giới hạn vào kết quả thực sự quan sát được, kể cả kết quả âm tính.
- ICAI official page: [submission requirements](https://icai.cmcu.edu.vn/icai-2026/join) ghi hạn 30/09/2026 và manuscript tối đa 8 trang gồm references; xác nhận lại trên CMT trước khi nộp.

## Nguồn chính dùng để định hướng protocol

- Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020: [ACL Anthology](https://aclanthology.org/2020.emnlp-main.609/).
- Wadden et al., *SciFact-Open: Towards open-domain scientific claim verification*, Findings EMNLP 2022: [ACL Anthology](https://aclanthology.org/2022.findings-emnlp.347/).
- Henry & McInnes, *Literature Based Discovery: Models, methods, and trends*, Journal of Biomedical Informatics 2017: [DOI](https://doi.org/10.1016/j.jbi.2017.08.011).
- *Literature-based discovery: addressing the issue of the subpar evaluation methodology*, Bioinformatics 2023: [Oxford Academic](https://academic.oup.com/bioinformatics/article/39/2/btad090/7036333).
- Weis & Jacobson, *Learning on knowledge graph dynamics provides an early warning of impactful research*, Nature Biotechnology 2021: [Nature](https://www.nature.com/articles/s41587-021-00907-6).
