# KG-TABI confirmatory blinded-review protocol v1

> Status: **DRAFT — NOT LOCKED**  
> Run ID: `domain-confirmatory-v1`  
> Không sinh candidate hoặc tuyển reviewer cho đến khi mọi mục
> `[[FILL_REQUIRED]]` đã được
> điền, `config_snapshot.json` đã chuyển sang `locked`, và ba file protocol đã
> được tạo SHA-256.

## 1. Mục tiêu

Đánh giá mù chất lượng candidate do KG-TABI sinh ra so với direct-LLM baseline
và các graph-ground ablation, với source closure và tối thiểu ba chuyên gia độc
lập. Nghiên cứu đánh giá candidate quality; nó không dùng LLM-as-a-judge làm
gold label.

## 2. Thông tin phải điền trước khi khóa

| Trường | Giá trị |
|---|---|
| Domain | `[[FILL_REQUIRED]]` |
| Domain owner | `[[FILL_REQUIRED]]` |
| Retrieval query | `[[FILL_REQUIRED]]` |
| Retrieval start/end UTC | `[[FILL_REQUIRED]]` |
| Corpus manifest path | `[[FILL_REQUIRED]]` |
| Corpus SHA-256 | `[[FILL_REQUIRED]]` |
| Inclusion/exclusion rule | `[[FILL_REQUIRED]]` |
| Analysis cutoff year | `[[FILL_REQUIRED]]` — đồng bộ với `config_snapshot.json` |
| Reviewer eligibility | `[[FILL_REQUIRED]]` |
| Ethics/consent determination | `[[FILL_REQUIRED]]` |
| Protocol lock UTC | `[[FILL_REQUIRED]]` |
| Git commit/tag | `[[FILL_REQUIRED]]` |

## 3. Thiết kế đã khai báo trước

### Hệ thống

1. `kgtabi`: KG-TABI với Grounds đầy đủ.
2. `direct`: direct-LLM baseline.
3. `full_grounds`: graph-ground generation với Grounds đầy đủ.
4. `concept_only`: chỉ cung cấp concept labels.
5. `abstract_bundle`: chỉ cung cấp matched abstract bundle.
6. `shuffled_grounds`: Grounds được hoán vị bằng seed đã khóa.

Mọi hệ thống phải dùng cùng corpus, source bundle, Toulmin schema, candidate
length limit và candidate budget. Tên hệ thống chỉ xuất hiện trong private key.

### Candidate budget

- Confirmatory target: **30 candidate mỗi hệ thống mỗi domain**.
- Tối thiểu hai domain độc lập nếu claim generalization.
- Nếu một hệ thống sinh dưới 30 candidate, giữ toàn bộ và báo cáo underpowered;
  không nới threshold.
- Nếu sinh trên 30 candidate, chọn 30 bằng deterministic seeded sampling với
  seed `20260720`, sau khi đã sort theo stable candidate hash.
- Không chỉnh sửa candidate text sau selection, ngoài việc loại system-name cue
  đã được định nghĩa trước và áp dụng đồng nhất cho mọi hệ thống.

### Reviewers

- Tối thiểu 3 domain experts độc lập.
- Mỗi reviewer chấm toàn bộ candidate trong domain, trừ khi assignment design
  khác đã được power-calibrate và ghi vào protocol trước khi khóa.
- Reviewer không được biết system identity, prompt, model, generation order,
  topology score hoặc unblinding key.
- Reviewer dùng pseudonymous IDs như `R01`, `R02`, `R03`.

## 4. Frozen KG-TABI configuration

Cấu hình máy đọc nằm trong `config_snapshot.json`. Các giá trị mặc định:

- simple undirected projection;
- Louvain seed 42, resolution 1.0;
- global community size ratio `>= 0.05`;
- undirected cut-edge fraction `<= 0.10`;
- temporal minimum 5 events và 4 covered years;
- normalized decline threshold `>= 0.30`;
- negative Sen slope;
- Benjamini–Hochberg FDR `q < 0.05`;
- compatibility và TABI schema theo source version đã hash.

Không đổi configuration sau khi nhìn thấy candidate count.

## 5. Source closure

Mỗi candidate phải có:

- deterministic query variants;
- top 20 retrieval candidates mỗi query;
- backward/forward citations theo giới hạn trong config;
- source IDs, URLs, ranks và retrieval timestamps;
- cùng loại source bundle cho mọi hệ thống;
- human labels về support, contradiction và already-addressed status.

Retrieval rank, NLI score hoặc LLM output không phải novelty judgment.

## 6. Rating outcomes

Outcome chính:

> `novelty_after_closure_1_to_5`, được phân tích theo kế hoạch đã khai báo trong
> `analysis_plan.md`, với source support được báo cáo đồng thời.

Outcome phụ:

- source support;
- claim clarity;
- importance;
- actionability;
- feasibility;
- already addressed;
- unsupported/hallucinated evidence;
- reviewer guess về system identity và guess confidence nếu form triển khai có
  các trường này.

## 7. Blinding

`src.prepare_candidate_blind_review` tạo:

- public reviewer packet: `audits/candidate_blind_packet.csv`;
- private mapping: `audits/candidate_unblinding_key.csv`.

Packet custodian giữ key riêng. Agreement và primary analysis specification phải
được khóa trước khi analyst nhận key. Reviewer chỉ nhận blind packet và source
bundles được đặt tên bằng `blind_candidate_id`.

Blinding check: sau rating chính, reviewer đoán generating system. Guess accuracy
được so với chance `1 / số hệ thống`; kết quả này là diagnostic, không phải lý do
loại reviewer sau khi xem dữ liệu.

## 8. Stop rules

Dừng candidate-quality evaluation cho domain nếu:

- KG-TABI sinh 0 candidate;
- source closure không hoàn tất theo rule đã khóa;
- packet làm lộ system identity;
- một input candidate file sai schema;
- protocol/config/hash thay đổi sau generation;
- không tuyển đủ reviewer tối thiểu.

Mọi stop đều được báo cáo, không thay gate để cứu study.

## 9. Lock protocol

Sau khi điền hết TODO và cập nhật JSON:

```powershell
$ProtocolDir = "data/runs/domain-confirmatory-v1/protocol"
Get-FileHash -Algorithm SHA256 -LiteralPath `
  "$ProtocolDir/protocol_v1.md", `
  "$ProtocolDir/config_snapshot.json", `
  "$ProtocolDir/analysis_plan.md"
```

Ghi hash của `protocol_v1.md` và `analysis_plan.md` vào
`config_snapshot.json.integrity`, rồi đổi `status` thành `locked`. Sau đó tính
hash cuối của `config_snapshot.json` và ghi hash này vào run manifest/Git commit;
không ghi hash của config vào chính nó vì sẽ tạo tự tham chiếu. Commit/tag
snapshot, rồi không sửa ba file trong cùng run ID. Nếu sửa, tạo
`domain-confirmatory-v2`.

## 10. Sign-off

| Vai trò | ID/chữ ký | UTC |
|---|---|---|
| Run owner | `[[FILL_REQUIRED]]` | `[[FILL_REQUIRED]]` |
| Statistical reviewer | `[[FILL_REQUIRED]]` | `[[FILL_REQUIRED]]` |
| Packet custodian | `[[FILL_REQUIRED]]` | `[[FILL_REQUIRED]]` |
