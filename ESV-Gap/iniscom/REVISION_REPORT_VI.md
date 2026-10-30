# GapCast-6G: thay đổi bản submission INISCOM 2027

Ngày cập nhật: 30/10/2026. Bản nộp: `gapcast_iniscom2027_submission.pdf`.

## 1. Bootstrap so với Adamic–Adar

Đã chạy lại thí nghiệm chính từ count tensor và ma trận semantic cố định. Các điểm
ước lượng, chỉ số chính và khoảng bootstrap trước đó được tái lập chính xác. AUC
trung bình của mô hình đầy đủ vẫn là **0,809**, Adamic–Adar **0,659**, logistic chỉ
dùng popularity và growth **0,776**.

Table 4 đã có cột **Learned − Adamic–Adar, concept bootstrap**:

| Cutoff | AUC Adamic–Adar | AUC đầy đủ | Chênh lệch AUC | Khoảng 95% concept bootstrap |
|---|---:|---:|---:|---|
| 2019 | 0,571 | 0,734 | +0,163 | [0,003; 0,290] |
| 2020 | 0,643 | 0,768 | +0,125 | [0,013; 0,247] |
| 2021 | 0,734 | 0,863 | +0,129 | [−0,017; 0,280] |
| 2022 | 0,687 | 0,870 | +0,183 | [0,065; 0,314] |

Khoảng vượt 0 ở **3/4 cutoff**, không phải cả bốn. Cận dưới 2019 gần 0. So với
logistic popularity + growth, chỉ khoảng ở cutoff 2022 vượt 0. Đã sửa diễn giải cho
phù hợp với mức bằng chứng này.

Mỗi so sánh thử 1.000 lần lấy mẫu; với concept bootstrap, trọng số của một cặp bằng
tích số lần hai concept được lấy. Seed của so sánh Adamic–Adar là năm cutoff, với
generator riêng. Khoảng là percentile 95% theo từng cutoff, điều kiện trên mô hình
đã fit; không refit, không điều chỉnh nhiều so sánh và không kiểm định gộp qua các
năm phụ thuộc nhau. Quy tắc và seed được lưu trong `outputs/results_main.json`.

## 2. Độ nhạy theo tập con concept

Đã chạy lại **20 tập con, mỗi tập 60/90 concept**, lấy không hoàn lại trong từng
lần chọn, seed **2027**, giữ nguyên corpus và bốn cutoff. Trên từng vocabulary con,
đã tính lại graph profiles, candidate pairs và fit lại hai logistic model. Tên/ID
concept, số gap, số closure và AUC theo cutoff đều được lưu để tái lập.

| Phương pháp | Trung vị của AUC trung bình 4 cutoff | Min | Max |
|---|---:|---:|---:|
| Mô hình đầy đủ | 0,797 | 0,737 | 0,845 |
| Logistic popularity + growth | 0,766 | 0,714 | 0,828 |
| Single score tốt nhất trong từng tập con | 0,651 | 0,616 | 0,692 |

Mô hình đầy đủ có AUC trung bình cao hơn single score tốt nhất ở 20/20 tập con.
Chênh lệch trung vị theo cặp với logistic rút gọn là 0,024. Số gap trung bình qua
cutoff nằm trong khoảng 178,5–298,75; từng cutoff có 153–361 gap.

Đã thêm **Table 10** và phần giải thích ở Section 5.6. Single score tốt nhất được
chọn hồi cứu trong từng tập con như một comparator mô tả. Các tập con chồng lấn và
dùng chung corpus; phạm vi Min–Max **không phải khoảng tin cậy**, và đây không phải
bằng chứng khái quát sang corpus mới hay loại bỏ hoàn toàn hindsight của lexicon.

## 3. Sáu bài cụ thể cho RIS, AoI và ISAC

Đã quét lại toàn bộ **194.221 work** trong snapshot, kiểm tra không trùng ID, và
đối chiếu count trực tiếp với `data/index.npz`:

| Cặp concept tại cutoff 2022 | Joint mentions trước/đến 2022 | Joint mentions 2023–2025 | Same-sentence 2023–2025 |
|---|---:|---:|---:|
| RIS × semantic communication | 0 | 26 | 23 |
| Age of information × digital twin | 0 | 28 | 15 |
| ISAC × wireless power transfer | 0 | 16 | 12 |

Section 5.7 hiện dẫn hai nghiên cứu cho mỗi cặp, kèm DOI trong References và
OpenAlex ID ngay trong nội dung:

| Cặp | Nghiên cứu và nội dung liên quan | DOI | OpenAlex |
|---|---|---|---|
| RIS × semantic communication | Du et al., *Semantic Communications for Wireless Sensing: RIS-Aided Encoding and Self-Supervised Decoding* (2023): mã hóa phổ sensing bằng RIS và giải mã tự giám sát. | [10.1109/JSAC.2023.3288231](https://doi.org/10.1109/JSAC.2023.3288231) | [W4382119085](https://openalex.org/W4382119085) |
| RIS × semantic communication | Huang et al., *Joint Active and Passive Beamforming for RIS-Aided Semantic Communication* (2024): liên kết mục tiêu phân loại semantic với beamforming phát/RIS. | [10.1109/TVT.2024.3435754](https://doi.org/10.1109/TVT.2024.3435754) | [W4401113937](https://openalex.org/W4401113937) |
| AoI × digital twin | Li et al., *AoI-Aware User Service Satisfaction Enhancement in Digital Twin-Empowered Edge Computing*: tối ưu placement của digital twin bằng thước đo hài lòng dựa trên AoI. | [10.1109/TNET.2023.3324704](https://doi.org/10.1109/TNET.2023.3324704) | [W4387831770](https://openalex.org/W4387831770) |
| AoI × digital twin | Li et al., *AoI-Aware, Digital Twin-Empowered IoT Query Services in Mobile Edge Computing* (2024): tối ưu freshness và độ trễ query trong MEC. | [10.1109/TNET.2024.3395709](https://doi.org/10.1109/TNET.2024.3395709) | [W4396667043](https://openalex.org/W4396667043) |
| ISAC × WPT | Cui et al., *Communication and Sensing Performance Study of NOMA-ISAC System with IRS-Assisted SWIPT* (2024): cấp năng lượng bằng time switching cho xe thực hiện sensing. | [10.23977/jeis.2024.090113](https://doi.org/10.23977/jeis.2024.090113) | [W4393955155](https://openalex.org/W4393955155) |
| ISAC × WPT | Yaswanth et al., *Active and STAR-RIS-Assisted MIMO ISAC Systems With SWIPT* (2025): tối ưu sensing/communication dưới ràng buộc thu năng lượng. | [10.1109/TCCN.2025.3543405](https://doi.org/10.1109/TCCN.2025.3543405) | [W4407690662](https://openalex.org/W4407690662) |

Bài AoI về service satisfaction xuất bản online ngày 20/10/2023, nhưng nằm trong
issue tháng 4/2024; corpus dùng năm 2023, còn BibTeX ghi năm issue 2024 và chú thích
ngày online, phù hợp với [hồ sơ của CityUHK](https://scholars.cityu.edu.hk/en/publications/aoi-aware-user-service-satisfaction-enhancement-in-digital-twin-e/).
Metadata DOI của cả sáu bài được lưu trong `outputs/reference_metadata.json`; tác giả
và trang 85–94 của bài Cui được kiểm tra thêm trên [bản PDF của nhà xuất bản](https://www.clausiuspress.com/assets/default/article/2024/04/04/article_1712248081.pdf).

Đây là sáu minh họa được chọn sau khi xem kết quả. Tổng count vẫn có survey và
incidental mention; đã ghi rõ trong caption và nội dung. Không dùng sáu bài này để
ước lượng precision, Cohen's kappa hoặc gọi là independent human validation.

## 4. Novelty và các claim đã sửa

- Nêu rõ [Science4Cast](https://www.nature.com/articles/s42256-023-00735-0) đã xét
  dự báo liên kết giữa concept chưa nối và các ngưỡng/horizon. Gap persistence với
  horizon cố định là phần bù của link formation; không tuyên bố bài toán ML mới.
- Định vị đóng góp ở benchmark 6G/IoT, measurement, đánh giá baseline có xét phụ
  thuộc, sensitivity và evidence traces. Không tuyên bố vượt các mô hình Science4Cast
  chưa được chạy trên cùng dữ liệu.
- Thay “leakage-free” bằng mô tả đúng ranh giới publication year; thừa nhận vocabulary,
  encoder và metadata được dùng hồi cứu từ snapshot 2026.
- Đổi “We release” thành “We introduce”; thống nhất kế hoạch phát hành sau xuất bản.
  Chưa có public artifact release hay anonymous repository được công bố trong bản nộp.
- Giới hạn mục tiêu ở closure trong horizon xác định, không suy ra “gap còn tồn tại
  bao lâu”, tính mới khoa học, bài toán đã giải quyết hay hiệu quả chọn đề tài.
- Làm rõ same-sentence vẫn giữ được các câu liệt kê/survey; lỗi lexical có thể tương
  quan với popularity/growth nên không bảo đảm chỉ tạo nhiễu vô hướng.
- Gọi bảng 2026–2028 là forward-window illustration tạo tháng 10/2026, không phải
  forecast đăng ký trước năm 2026. Không dùng dữ liệu xuất bản sau 2025 làm input.
- Nêu đúng bảy feature biến thiên và cột prior-pair count bằng 0 ở g=0, chỉ biến thiên
  khi g=2. Expected co-occurrence là frequency-product analogue, không đồng nhất với
  preferential attachment dựa trên graph degree.
- Bổ sung việc AI hỗ trợ literature search trong Acknowledgments, và giữ trách nhiệm
  nội dung thuộc tác giả; không tuyên bố đã có một vòng human validation mới.

## 5. Biên dịch và kiểm tra bản nộp

- PDF ẩn danh **16 trang**, References bắt đầu ở trang 14; có **22 tài liệu tham khảo**.
- Author block dùng Anonymous Authors/Institution; trường PDF Author trống; không có
  tên tác giả, đơn vị hoặc email riêng trong phần thân trước References.
- Giữ self-citation ESV-Gap bình thường ở ngôi thứ ba, sửa thứ tự tác giả theo
  [danh sách chính thức của AMI 2027](https://ami-conf.org/2027/abstracts/).
  Bibliography vẫn có tên thật của tác giả tài liệu; không bảo đảm không thể suy ra danh tính.
- Đã đối chiếu author kit được liên kết từ [CFP INISCOM 2027](https://iniscom.eai-conferences.org/2027/call-for-papers/):
  template sử dụng lớp `llncs`, cùng họ định dạng với nguồn hiện tại. Không tự chỉnh lề
  hoặc cỡ phần thân để giảm số trang.
- Không còn undefined references/citations hoặc overfull/underfull boxes. Tất cả 21
  font resource được nhúng. Còn một cảnh báo tương tác `amsmath`/`llncs` về math accent
  `\vec`; bản thảo không dùng lệnh này.
- PDF được render toàn bộ để rà bố cục. `paper/ALT_TEXT.md` cung cấp mô tả hai hình
  và 11 bảng; không tuyên bố PDF đã được tagged theo chuẩn accessibility.
- Gói nguồn LaTeX loại bỏ nhánh author có tên/email riêng. Các trích dẫn học thuật
  vẫn giữ nguyên tác giả. Gói được giải nén và biên dịch lại để kiểm tra khả năng build.

Các kiểm tra máy đọc được nằm trong `outputs/revision_analysis_checks.json`,
`outputs/submission_pdf_checks.json` và `outputs/source_package_checks.json`.
`src/verify_revision.py` tính lại 9 phương pháp tại cả bốn cutoff, kiểm tra rằng thay
đổi count sau T không làm đổi feature tại T hoặc nhãn train đã hoàn tất, xác nhận
20 lần chọn subset, sáu DOI/ID và toàn bộ citation keys.

Phạm vi còn giới hạn: chưa có mẫu gán nhãn độc lập, expert utility evaluation hoặc
corpus ngoại kiểm. Các giới hạn này được giữ rõ trong Abstract, Discussion và Conclusion.
File được chuẩn bị cục bộ; chưa upload hoặc nộp lên Confy+.
