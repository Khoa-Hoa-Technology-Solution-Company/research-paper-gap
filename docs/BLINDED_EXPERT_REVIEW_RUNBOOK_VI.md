# Runbook đánh giá mù candidate bằng chuyên gia

Tài liệu này là quy trình thao tác đầy đủ để thực hiện đánh giá mù đối với
candidate do KG-TABI và các hệ thống đối chứng sinh ra. Quy trình này chỉ được
dùng cho candidate từ một forward run đã khóa. Không được dùng các output legacy
ở `data/gaps/` hoặc `data/audits/` làm kết quả xác nhận.

## 0. Điều kiện quan trọng trước khi bắt đầu

Run `microservices-security-v1` và `microservices-security-e2e-v1` hiện có
`0` candidate KG-TABI dưới cấu hình chính. Vì vậy, chưa thể gửi hai run này cho
chuyên gia để đánh giá chất lượng candidate.

Nếu một run xác nhận cũng sinh `0` candidate, phải báo cáo đây là kết quả
underpowered/null. Không được nới gate sau khi nhìn thấy kết quả null. Muốn có
candidate, hãy khóa trước một domain phát triển và một domain xác nhận khác, hoặc
đổi phạm vi nghiên cứu trong protocol trước khi chạy.

Để claim về candidate quality, nên có tối thiểu:

- 2 domain độc lập;
- 3 chuyên gia độc lập;
- cùng số candidate từ mỗi hệ thống;
- source closure giống nhau cho mọi hệ thống;
- một outcome chính được xác định trước khi xem rating.

## 1. Vai trò và quyền truy cập

Nên phân công ba vai trò:

1. **Run owner**: khóa code, config, candidate outputs và tạo manifest.
2. **Packet custodian**: tạo packet mù, giữ unblinding key, liên lạc với reviewer.
3. **Analyst/adjudicator**: chỉ nhận rating đã khóa; không được biết system identity
   trước khi tính agreement.

Một người có thể giữ nhiều vai trò trong pilot, nhưng trong nghiên cứu xác nhận
nên tách packet custodian khỏi analyst nếu có thể.

Không đưa `candidate_unblinding_key.csv` lên Google Drive/Forms cùng packet. Key
phải nằm ở thư mục riêng được mã hóa hoặc do một người không tham gia chấm giữ.

## 2. Chuẩn bị môi trường

Mở PowerShell tại thư mục repository:

```powershell
Set-Location D:\Workspace\research-paper-gap

# Có thể thay bằng đường dẫn Python của máy bạn.
$Python = "C:\Users\leanh\AppData\Local\Programs\Python\Python314\python.exe"
& $Python --version
& $Python -B -m unittest discover -s tests -p "test_*.py" -v
```

Test phải pass trước khi tạo candidate packet. Nếu máy không dùng đường dẫn
trên, thay `$Python` bằng Python 3.11+ có cài các package trong
`requirements.txt`.

Sau khi đã thay toàn bộ `FILL_REQUIRED` và điền các trường `null` bắt buộc trong
`config_snapshot.json`, khóa protocol bằng:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\lock_confirmatory_protocol.ps1 `
  -RunId domain-confirmatory-v1
```

Script tạo `protocol_lock.json`, tự ghi hash của hai Markdown vào config và ghi
hash cuối của config ra lock record. Sau khi đã sinh đủ 6 candidate JSON, chạy
toàn bộ validation + closure + packet generation bằng một lệnh:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_confirmatory_review.ps1 `
  -RunId domain-confirmatory-v1
```

Để chỉ kiểm tra mà chưa gọi closure API hoặc ghi packet:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_confirmatory_review.ps1 `
  -RunId domain-confirmatory-v1 `
  -ValidateOnly
```

Runner sẽ chủ động dừng nếu protocol chưa `locked`, còn `FILL_REQUIRED`, hash không khớp,
candidate bằng 0, schema sai hoặc số lượng giữa các hệ thống không cân bằng.

Kiểm tra biến môi trường nhưng không in secret ra terminal:

```powershell
Get-ChildItem Env: | Where-Object { $_.Name -match 'KEY|TOKEN|SECRET' } |
  Select-Object Name
```

Không commit `.env`, API key, raw provider response hoặc thông tin cá nhân của
reviewer.

## 3. Khóa protocol trước khi sinh candidate

Tạo một file protocol bất biến, ví dụ:

```text
data/runs/domain-confirmatory-v1/protocol/
  protocol_v1.md
  config_snapshot.json
  analysis_plan.md
```

File protocol phải ghi rõ:

- domain, ngày truy xuất, corpus hash và inclusion rule;
- graph projection, entity-resolution rule, Louvain seed/resolution;
- temporal cutoff, event/year thresholds và FDR rule;
- model identifier, prompt version/hash, generation settings;
- danh sách hệ thống và ablation;
- số candidate cần lấy từ mỗi hệ thống;
- quy tắc chọn candidate nếu hệ thống sinh quá nhiều;
- quy trình source closure và ngày kết thúc retrieval;
- outcome chính, outcome phụ và phương pháp thống kê;
- quy tắc xử lý missing rating, `uncertain` và candidate bị loại;
- điều kiện dừng, bao gồm trường hợp candidate count bằng 0.

Không sửa file protocol sau khi đã sinh candidate. Nếu bắt buộc sửa, tăng version
và chạy lại toàn bộ nghiên cứu như một run mới.

## 4. Chuẩn bị candidate outputs

Mỗi hệ thống phải xuất một JSON array. Mỗi phần tử phải có ít nhất:

```json
{
  "Grounds": "graph or evidence grounds",
  "Claim": "testable potential-gap hypothesis",
  "Warrant": "conditional rationale",
  "Bucket": "near_term_feasible"
}
```

Ví dụ cấu trúc thư mục:

```text
data/runs/domain-confirmatory-v1/gaps/
  kgtabi_gaps.json
  direct_llm_gaps.json
  full_grounds_gaps.json
  concept_only_gaps.json
  abstract_bundle_gaps.json
  shuffled_grounds_gaps.json
```

Tất cả file phải dùng cùng corpus, source bundle, schema, giới hạn độ dài và
candidate budget. Không dùng candidate từ các run có gate khác nhau để so sánh.

### Kiểm tra candidate count và schema

```powershell
$GapDir = "data/runs/domain-confirmatory-v1/gaps"
$Files = @(
  "kgtabi_gaps.json",
  "direct_llm_gaps.json",
  "full_grounds_gaps.json",
  "concept_only_gaps.json",
  "abstract_bundle_gaps.json",
  "shuffled_grounds_gaps.json"
)

foreach ($Name in $Files) {
  $Path = Join-Path $GapDir $Name
  if (-not (Test-Path -LiteralPath $Path)) { throw "Missing $Path" }
  $Rows = @(Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json)
  $Invalid = @($Rows | Where-Object {
    [string]::IsNullOrWhiteSpace($_.Grounds) -or
    [string]::IsNullOrWhiteSpace($_.Claim) -or
    [string]::IsNullOrWhiteSpace($_.Warrant) -or
    $_.Bucket -notin @("near_term_feasible", "long_term_or_speculative")
  })
  [pscustomobject]@{File=$Name; Count=$Rows.Count; Invalid=$Invalid.Count} |
    Format-Table -AutoSize
  if ($Invalid.Count -gt 0) { throw "Invalid TABI rows in $Path" }
}
```

Nếu `kgtabi_gaps.json` có count bằng 0, dừng bước đánh giá candidate. Không
điền candidate bằng tay và không lấy candidate của một domain khác.

## 5. Chạy source closure

Source closure phải được chạy trước khi chuyên gia chấm novelty. Công cụ hiện có
chỉ tạo audit retrieval; nó không tự tạo judgment về novelty.

Chạy deterministic closure cho từng hệ thống, hoặc tạo một file hợp nhất đã được
khóa trước. Ví dụ chạy từng file:

```powershell
$RunDir = "data/runs/domain-confirmatory-v1"

& $Python -B -m src.closure_search `
  --input "$RunDir/gaps/kgtabi_gaps.json" `
  --output "$RunDir/gaps/kgtabi_closure_search_audit.json" `
  --manifest "$RunDir/gaps/kgtabi_closure_search_manifest.json" `
  --limit 20 `
  --citation-limit 10 `
  --citation-candidate-limit 5 `
  --deterministic-only
```

Lặp lại cho direct baseline và từng ablation. Ghi lại retrieval date/time, query
variants, source IDs/ranks/URLs, source-bundle hash và mọi lỗi truy xuất.

Reviewers phải nhận cùng loại source bundle cho mọi candidate. Nếu một candidate
có source bundle không hoàn chỉnh, đánh dấu thiếu dữ liệu và không chấm novelty
như thể closure đã hoàn tất.

## 6. Tạo packet mù và unblinding key

Chạy công cụ tại `src/prepare_candidate_blind_review.py`:

```powershell
$RunDir = "data/runs/domain-confirmatory-v1"

& $Python -B -m src.prepare_candidate_blind_review `
  --input "kgtabi=$RunDir/gaps/kgtabi_gaps.json" `
  --input "direct=$RunDir/gaps/direct_llm_gaps.json" `
  --input "full_grounds=$RunDir/gaps/full_grounds_gaps.json" `
  --input "concept_only=$RunDir/gaps/concept_only_gaps.json" `
  --input "abstract_bundle=$RunDir/gaps/abstract_bundle_gaps.json" `
  --input "shuffled_grounds=$RunDir/gaps/shuffled_grounds_gaps.json" `
  --packet "$RunDir/audits/candidate_blind_packet.csv" `
  --key "$RunDir/audits/candidate_unblinding_key.csv" `
  --seed 20260720
```

Sau khi chạy:

1. kiểm tra số dòng và các trường bắt buộc;
2. lưu SHA-256 của packet vào run manifest;
3. sao lưu key ở nơi riêng;
4. không chạy lại script với seed khác sau khi đã gửi packet;
5. không chỉnh sửa claim, grounds, warrant hoặc bucket trong packet.

```powershell
$Packet = "$RunDir/audits/candidate_blind_packet.csv"
$Key = "$RunDir/audits/candidate_unblinding_key.csv"
Get-FileHash -Algorithm SHA256 -LiteralPath $Packet,$Key
Get-Content -LiteralPath $Packet | Select-Object -First 3
Get-Content -LiteralPath $Key | Select-Object -First 3
```

Reviewer chỉ được nhận packet và source bundle. Không gửi key, system ID, tên
model, prompt, generation order hoặc file path gốc.

## 7. Chuẩn bị bộ tài liệu cho mỗi reviewer

Mỗi reviewer nhận:

```text
reviewer_R01/
  instructions.md
  candidate_blind_packet.csv
  source_bundles/
    blind-0001/
    blind-0002/
    ...
  rating_template.csv
```

Không dùng email hoặc tên thật trong `reviewer_id`; dùng mã `R01`, `R02`, `R03`.
Mỗi reviewer phải nhận cùng candidate rows và cùng source bundles.

### Nội dung hướng dẫn reviewer

Reviewer phải làm độc lập, đọc source closure trước khi chấm novelty, không suy
đoán system identity, chấm tất cả candidate và ghi rationale khi chấm 1, 5 hoặc
`uncertain`. Reviewer cũng phải báo rõ candidate nào không đủ source để đánh giá.

## 8. Schema rating

Mỗi dòng rating phải giữ `blind_candidate_id` và có các cột:

```text
blind_candidate_id
source_support_1_to_5
claim_clarity_1_to_5
novelty_after_closure_1_to_5
importance_1_to_5
actionability_1_to_5
feasibility_1_to_5
already_addressed_yes_no_uncertain
unsupported_or_hallucinated_evidence_yes_no_uncertain
reviewer_id
comments
```

Diễn giải thang điểm: 1 = rất yếu, 2 = yếu, 3 = trung bình/chưa rõ, 4 = tốt,
5 = rất tốt/được hỗ trợ mạnh. `novelty_after_closure` chỉ được chấm sau khi đã
xem backward/forward closure. Retrieval rank không phải bằng chứng novelty.

## 9. Thu nhận và khóa rating

Khi nhận file từ reviewer, lưu nguyên trạng bằng tên bất biến và tạo hash:

```powershell
$ReviewRoot = "data/runs/domain-confirmatory-v1/reviewer_returns"
New-Item -ItemType Directory -Force -Path $ReviewRoot | Out-Null
Get-FileHash -Algorithm SHA256 -LiteralPath "$ReviewRoot\reviewer_R01.csv"
```

Kiểm tra mọi reviewer có cùng candidate IDs, không có duplicate/unknown ID,
rating nằm trong 1–5, categorical labels chỉ là `yes`, `no`, `uncertain`, và
missing values được ghi rõ. Không tự điền hoặc overwrite file gốc.

Chỉ sau khi toàn bộ reviewer đã nộp và hash file đã lưu mới tính agreement. Không
mở `candidate_unblinding_key.csv` trước bước này.

## 10. Phân tích trước khi unblind

Repository hiện có packet generator nhưng chưa có script phân tích reviewer
ratings hoàn chỉnh. Có thể dùng R, Python/pandas hoặc spreadsheet, nhưng phải
đóng băng analysis script trước khi mở key.

Phân tích tối thiểu:

1. Krippendorff's alpha cho outcome ordinal;
2. agreement cho nhãn categorical;
3. median và IQR theo hệ thống;
4. tỷ lệ `already_addressed = yes`;
5. tỷ lệ unsupported/hallucinated evidence;
6. missing ratings và candidate exclusions;
7. kết quả riêng từng domain;
8. bootstrap confidence interval và effect size cho outcome chính.

Outcome chính nên được định nghĩa trước, ví dụ:

```text
Novelty after closure, conditional on source_support >= 3
```

Không nên chỉ báo cáo mean; rating 1–5 là ordinal.

## 11. Mở key và gắn system identity

Sau khi lưu hash của reviewer files, tính agreement, đóng băng analysis output và
lưu protocol/analysis script, mới đọc key:

```powershell
$Key = Import-Csv "$RunDir/audits/candidate_unblinding_key.csv"
$Key | Select-Object -First 5
```

Join key với bảng rating bằng `blind_candidate_id`. Không sửa ID hoặc ghi đè
reviewer ratings. Giữ riêng raw ratings, agreement trước adjudication và
adjudicated ratings.

## 12. Báo cáo trong bài báo

Báo cáo riêng từng domain trước pooled analysis. Tối thiểu cần có số candidate
theo hệ thống, số reviewer/chuyên môn, quy trình closure, median/IQR, agreement,
missing/exclusions, confidence intervals/effect sizes và raw reviewer-level data
đã khử thông tin nhạy cảm.

Không được kết luận KG-TABI tốt hơn baseline nếu candidate budget, source bundle,
độ dài output hoặc reviewer exposure khác nhau.

## 13. Điều kiện dừng và lỗi thường gặp

Dừng và báo cáo rõ nếu KG-TABI sinh 0 candidate, số candidate không cân bằng,
closure không đồng nhất, reviewer thấy system identity, packet bị sửa sau khi
gửi hoặc thiếu reviewer nghiêm trọng.

Không được lấy output legacy làm candidate xác nhận, thêm candidate bằng tay,
đổi threshold sau khi xem candidate count, dùng LLM judge thay chuyên gia, dùng
retrieval rank/NLI score như novelty label, hoặc mở key trước khi tính agreement.

## 14. Checklist cuối cùng

- [ ] Protocol, config, prompts, seeds và analysis plan đã hash.
- [ ] Có ít nhất hai domain nếu muốn claim generalization.
- [ ] Candidate count giữa các hệ thống đã cân bằng.
- [ ] Source closure đã hoàn tất và giống nhau giữa hệ thống.
- [ ] Packet mù đã được tạo bằng seed cố định.
- [ ] Key được giữ riêng và không gửi reviewer.
- [ ] Có ít nhất ba reviewer độc lập.
- [ ] Reviewer files được lưu nguyên trạng và hash lại.
- [ ] Agreement đã tính trước adjudication/unblinding.
- [ ] Analysis script và output đã khóa.
- [ ] Key chỉ được mở sau khi các bước trên hoàn tất.
- [ ] Raw ratings, adjudicated ratings và protocol được lưu riêng.
- [ ] Không claim candidate-quality nếu study chỉ là pilot hoặc có 0 candidate.

Các nguyên tắc nền tảng nằm trong
[`BLINDED_CANDIDATE_EVALUATION.md`](BLINDED_CANDIDATE_EVALUATION.md),
[`AUDIT_PACKET_AND_TRACE_CHAIN.md`](AUDIT_PACKET_AND_TRACE_CHAIN.md) và
[`ANNOTATION_PROTOCOL.md`](ANNOTATION_PROTOCOL.md).
