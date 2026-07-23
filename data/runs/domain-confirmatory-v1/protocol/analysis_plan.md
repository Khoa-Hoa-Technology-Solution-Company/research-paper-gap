# KG-TABI confirmatory blinded-review analysis plan

> Status: **DRAFT — NOT LOCKED**  
> Run ID: `domain-confirmatory-v1`  
> Khóa file này trước candidate generation và trước khi tuyển reviewer.

## 1. Research questions

- **ARQ1:** Candidate có đủ source support dưới source closure đã khóa không?
- **ARQ2:** Sau closure, novelty rating của KG-TABI khác direct-LLM baseline như
  thế nào?
- **ARQ3:** Grounds đầy đủ có cải thiện rating so với concept-only,
  abstract-bundle-only và shuffled-Grounds không?
- **ARQ4:** Blinding có được duy trì ở mức reviewer không phân biệt system tốt
  hơn chance không?

## 2. Units and populations

- Candidate là đơn vị nội dung.
- Candidate × reviewer là đơn vị rating.
- Domain được báo cáo riêng trước pooled analysis.
- Confirmatory population gồm toàn bộ candidate được chọn bằng rule và seed đã
  khóa; không loại candidate vì rating thấp hoặc claim bất lợi.

## 3. Primary outcome

Primary outcome: `novelty_after_closure_1_to_5`.

Source support được báo cáo đồng thời. Phân tích sensitivity đã khai báo trước:

1. toàn bộ novelty ratings;
2. novelty ratings trong candidate-reviewer rows có
   `source_support_1_to_5 >= 3`;
3. composite success rate:
   `source_support >= 3 AND novelty_after_closure >= 4`.

Không thay primary outcome sau khi mở unblinding key.

## 4. Secondary outcomes

- `source_support_1_to_5`;
- `claim_clarity_1_to_5`;
- `importance_1_to_5`;
- `actionability_1_to_5`;
- `feasibility_1_to_5`;
- `already_addressed_yes_no_uncertain`;
- `unsupported_or_hallucinated_evidence_yes_no_uncertain`;
- system-identity guess accuracy và confidence, nếu collected.

## 5. Sample size and stopping

- Target: 30 candidate/system/domain, 6 systems, tối thiểu 2 domains.
- Tối thiểu 3 independent experts/domain.
- Pilot với 10 candidate/system chỉ báo cáo descriptive statistics và
  uncertainty; không claim superiority.
- Nếu một hệ thống có dưới target, giữ toàn bộ, báo cáo achieved sample size và
  underpowered status; không nới gate.
- Trước confirmatory run, `[[FILL_REQUIRED]]`: thực hiện simulation/power analysis cho planned
  ordinal mixed-effects model và ghi expected detectable effect.

## 6. Data validation before analysis

Analyst kiểm tra khi vẫn blinded:

- candidate IDs giống nhau giữa reviewers;
- không duplicate hoặc unknown IDs;
- ordinal ratings chỉ thuộc 1–5;
- categorical ratings chỉ thuộc `yes/no/uncertain`;
- reviewer IDs là pseudonyms;
- missing data giữ nguyên, không impute trong primary analysis;
- packet và reviewer-file hashes khớp manifest.

Mọi exclusion phải dựa trên rule đã khai báo và ghi vào exclusion log trước
unblinding.

## 7. Agreement before adjudication

- Tính Krippendorff's alpha ordinal cho từng outcome 1–5.
- Tính agreement thích hợp cho categorical outcomes có `uncertain` và missing.
- Báo cáo agreement riêng từng domain và từng outcome.
- Không xóa reviewer vì agreement thấp sau khi xem system effects.
- Adjudication chỉ diễn ra sau khi raw agreement đã được lưu.

## 8. Descriptive reporting

Cho mỗi system × domain:

- candidate count và rating count;
- median, IQR và full distribution cho ordinal outcomes;
- already-addressed rate;
- unsupported/hallucinated-evidence rate;
- missing/uncertain rate;
- candidate-level và reviewer-level raw data đã khử thông tin nhận dạng.

## 9. Confirmatory comparisons

Primary contrast: `kgtabi` versus `direct` trên novelty-after-closure.

Secondary contrasts:

- `full_grounds` versus `concept_only`;
- `full_grounds` versus `abstract_bundle`;
- `full_grounds` versus `shuffled_grounds`.

Planned model cho confirmatory study: cumulative-link mixed-effects model hoặc
một ordinal mixed model tương đương, với fixed effect cho system và domain,
random intercept cho reviewer, và candidate-level clustering thích hợp. Báo cáo
effect size và 95% interval; không chỉ báo cáo p-value.

Nếu model không hội tụ, fallback đã khai báo:

1. domain-stratified bootstrap trên candidate-level summaries;
2. report median difference và percentile 95% interval;
3. không đổi outcome hoặc loại system để đạt significance.

## 10. Multiplicity

- Primary contrast là một kiểm định chính.
- Secondary contrasts dùng Holm correction trong cùng outcome family.
- Các outcome còn lại được gọi là secondary/exploratory và báo cáo đầy đủ,
  không chọn riêng kết quả có lợi.

## 11. Missing and uncertain ratings

- Primary analysis không impute missing ordinal rating.
- Báo cáo missingness theo reviewer, system và domain sau unblinding.
- `uncertain` không được tự chuyển thành `no` hoặc midpoint.
- Sensitivity analysis có thể loại `uncertain`, nhưng primary categorical table
  phải giữ nó như một category riêng.

## 12. Blinding diagnostic

Sau khi rating chính hoàn tất, reviewer đoán generating system. Tính:

```text
A_guess = correct system guesses / non-missing guesses
chance = 1 / number_of_systems
```

Báo cáo interval quanh `A_guess`. Đây là diagnostic; không loại reviewer dựa vào
guess accuracy sau khi xem kết quả.

## 13. Adjudication

Giữ riêng:

1. raw reviewer files;
2. agreement-before-adjudication output;
3. adjudication input;
4. adjudicated output;
5. unblinding key.

Adjudicator xử lý disagreements/uncertain theo rule đã khóa. Không overwrite raw
ratings và không dùng adjudication để thay đổi primary reviewer-level analysis.

## 14. Analysis freeze and unblinding

Trước khi mở key:

- validate và hash reviewer files;
- chạy agreement;
- khóa analysis code/version;
- tạo blinded descriptive report;
- ghi mọi deviations vào `protocol_deviations.md`;
- lưu UTC timestamp và analyst sign-off.

Chỉ packet custodian mới join `candidate_unblinding_key.csv` sau freeze.

## 15. Required outputs

```text
analysis/
  input_hashes.json
  validation_report.json
  agreement_before_adjudication.json
  blinded_descriptive_report.md
  protocol_deviations.md
  unblinded_candidate_ratings.csv
  confirmatory_results.json
  confirmatory_report.md
```

## 16. Fields to complete before lock

- Power/simulation method and result: `[[FILL_REQUIRED]]`.
- Statistical package and version: `[[FILL_REQUIRED]]`.
- Exact model formula: `[[FILL_REQUIRED]]`.
- Reviewer recruitment/eligibility criteria: `[[FILL_REQUIRED]]`.
- Ethics/consent determination: `[[FILL_REQUIRED]]`.
- Analyst ID and sign-off UTC: `[[FILL_REQUIRED]]`.
