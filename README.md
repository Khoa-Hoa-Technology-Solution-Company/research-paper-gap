# Hướng dẫn Biên dịch và Sử dụng Mã nguồn Bài báo Khoa học KG-TABI

Tài liệu này hướng dẫn cách sử dụng các tệp nguồn LaTeX và các lệnh cần thiết để biên dịch xuất ra file PDF hoàn chỉnh cho bài báo khoa học **"KG-TABI: Automating Software Engineering Research Gap Detection via Dynamic Knowledge Graphs and Toulmin-Abductive Inference"**.

## 1. Cấu trúc Thư mục

- `paper.tex`: Mã nguồn LaTeX chính chứa nội dung bản thảo bài báo khoa học.
- `references.bib`: Cơ sở dữ liệu tài liệu tham khảo dưới dạng BibTeX.
- `llncs.cls`: Lớp văn bản Springer Lecture Notes in Computer Science (LNCS).
- `splncs04.bst`: Định dạng phong cách thư mục (Bibliography Style) của Springer.
- `fig1.eps`: Hình vẽ sơ đồ kiến trúc hệ thống dạng EPS.
- `paper.pdf`: File tài liệu bài báo đã được biên dịch hoàn thành.

---

## 2. Yêu cầu Hệ thống

Để biên dịch thành công file `.tex` ra PDF, máy tính của bạn cần cài đặt một bộ phân phối TeX (TeX Distribution):
- **Windows:** [MiKTeX](https://miktex.org/) (khuyên dùng) hoặc [TeX Live](https://www.tug.org/texlive/).
- **macOS:** [MacTeX](https://www.tug.org/mactex/).
- **Linux:** [TeX Live](https://www.tug.org/texlive/) (thông qua package manager của hệ điều hành).

---

## 3. Các bước Biên dịch Xuất PDF (Command Line)

Để tất cả các tài liệu tham khảo (`\cite{...}`) và nhãn chéo (`\ref{...}`) hiển thị chính xác, bạn cần chạy chuỗi lệnh biên dịch theo đúng thứ tự sau trong Terminal (CMD / PowerShell / Bash):

```bash
# Bước 1: Biên dịch sơ bộ để tạo các file phụ trợ (.aux)
pdflatex -interaction=nonstopmode paper.tex

# Bước 2: Liên kết danh mục tài liệu tham khảo từ references.bib
bibtex paper

# Bước 3: Cập nhật liên kết tài liệu tham khảo vào văn bản
pdflatex -interaction=nonstopmode paper.tex

# Bước 4: Biên dịch lần cuối để giải quyết triệt để số trang và liên kết chéo
pdflatex -interaction=nonstopmode paper.tex
```

Sau khi chạy xong, file **`paper.pdf`** sẽ được sinh ra (hoặc cập nhật) tại thư mục hiện tại.

---

## 4. Dọn dẹp File Tạm (Không bắt buộc)

Trong quá trình biên dịch, LaTeX sẽ sinh ra một số file phụ trợ (.aux, .log, .out, .bbl, .blg). Bạn có thể xóa chúng đi để thư mục sạch sẽ hơn:

### Trên Windows (PowerShell):
```powershell
Remove-Item paper.aux, paper.log, paper.out, paper.bbl, paper.blg -ErrorAction SilentlyContinue
```

### Trên Windows (Command Prompt - CMD):
```cmd
del paper.aux paper.log paper.out paper.bbl paper.blg
```

### Trên Linux / macOS (Terminal):
```bash
rm -f paper.aux paper.log paper.out paper.bbl paper.blg
```

---

## 5. Xem File PDF Kết quả

Bạn có thể mở nhanh file PDF kết quả từ cửa sổ lệnh:
- **Windows:** `start paper.pdf`
- **macOS:** `open paper.pdf`
- **Linux:** `xdg-open paper.pdf`
