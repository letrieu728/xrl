# Kế hoạch chấm điểm CausalXRL — theo tiêu chí thầy gửi (11/11/2026)

> Cập nhật 13/11/2026: kế hoạch này viết ngày 11/11, trước khi chạy thực
> nghiệm. Cột "hiện trạng" trong các bảng dưới đây phản ánh đúng lúc lập
> kế hoạch (nhiều mục khi đó "chưa có số chạy thật") và được giữ nguyên
> làm lịch sử. Thực nghiệm sau đó đã làm xong: xem `README.md` và
> `SO-SANH-TAC-GIA-VS-NHOM.md` ở gốc repo để biết số chốt.

Bài: CausalXRL: Explainable Reinforcement Learning through Causal Graph Reasoning
— Yanming Zhang, Eric Papenhausen, Klaus Mueller, arXiv:2610.42028v1, ICML 2026, 17 trang.
Repo: github.com/yanmluk/causal-xrl, commit `ae9096e`, 0 stars, 3 commits, MIT.
Nguồn đã kiểm chứng: `CausalXRL-Tom-tat-noi-dung.md`, `CausalXRL-Phan-tich-ma-nguon.md`,
bản dịch đầy đủ 37 trang `causalxrl-ban-dich-day-du/causalxrl-ban-dich-day-du.pdf` (đủ 8/8 hình).

Mốc >85% = mức "Đạt quá yêu cầu". Mốc 70–85% = "Đạt yêu cầu".
Dưới 70% là mất điểm nặng ở các tiêu chí trọng số lớn.

---

## 0. Form mẫu Nhóm 8, HK1 23-24 — đã đọc thật cả 2 file (11/11/2026)

Triệu gửi 2 file mẫu (lưu ngoài repo này, trong thư mục làm việc của nhóm).
Lưu ý quan trọng: **mỗi file đều gộp 2 phần — dịch sách + bài báo — chứ không phải 1 báo cáo duy nhất.**

### a) Bản trình bày giữa kỳ — `Nho_m-8_HK1_23-24_GK__1___1.pptx`, 53 slides, 26 hình
- Slides 1–27: **deck dịch sách** — bìa "DỊCH SÁCH: E-COMMERCE INDUSTRY CHAIN - THEORY AND PRACTICE (CHƯƠNG 8)", mục lục, rồi 3 phần theo đúng 3 chương của bản dịch: Phần 1 TMĐT tập trung vào con người, Phần 2 TMĐT với hàng hóa là trung tâm, Phần 3 TMĐT với công nghệ mới (VR / Metaverse). Mỗi mục 2–4 slides, nhiều slide chỉ có hình.
- Slides 28–53: **deck bài báo AOAN, 26 slides** — đây mới là form deck cho CausalXRL:
  28 bìa (tên bài + GV + 4 thành viên) → 29 mục lục → 30 divider §1 → 31 giới thiệu bài báo (arXiv, hội nghị) → 32 tác giả → 33 divider §2 → 34 chủ đề → 35 đóng góp chính → 36 kiến thức nền → 37–41 mô hình (tổng quan, đặt vấn đề, BERT, span, …) → 42 divider §3 thực nghiệm tác giả → 43 mã nguồn → 44 dữ liệu → 45 kết quả tác giả → 46 divider §4 thực nghiệm của nhóm → 47 tổng quan mã nguồn nhóm (link Kaggle) → 48–50 kết quả nhóm theo từng tập dữ liệu (ảnh chụp màn hình) → 51–52 kết quả mở rộng → 53 kết thúc + mời đặt câu hỏi.
- Form này khớp với form đã dùng cho HELIX (36 slides) và AI-copy (33 slides): bìa → mục lục → giới thiệu → phân tích → thực nghiệm tác giả → thực nghiệm nhóm → mở rộng → kết thúc.

### b) Báo cáo cuối kỳ — `Nho_m-8_HK1_23-24_CK__2__0_x19t.pdf`, 93 trang, gộp 2 báo cáo bằng PDFsam
- Trang 1–41: **báo cáo dịch sách riêng** — bìa "DỊCH SÁCH: E-COMMERCE… (CHƯƠNG 8)", mục lục i–ii, danh sách hình vẽ iii, rồi bản dịch Chương 8 gồm 3 chương con (tr. 1–36), Tài liệu tham khảo tr. 37. Không có Lời nói đầu, không có chương phân tích — chỉ là bản dịch.
- Trang 42–93: **báo cáo bài báo AOAN, 52 trang** — bìa "BÀI BÁO: [tên tiếng Việt]", mục lục i–ii, danh sách bảng iii, danh sách hình vẽ iv, rồi:
  Ch. 1 GIỚI THIỆU BÀI BÁO VÀ CÁC TÁC GIẢ (tr. 1–3) → Ch. 2 PHÂN TÍCH NỘI DUNG BÀI BÁO (tr. 4–18: chủ đề, đóng góp, kiến thức nền, mô hình chi tiết, thực nghiệm của tác giả gồm dữ liệu, baseline, kết quả, ablation, trực quan hóa, ảnh hưởng độ dài câu, case study) → Ch. 3 TRIỂN KHAI THỰC NGHIỆM CỦA SINH VIÊN (tr. 19–25: phần mềm/dữ liệu, quá trình, kết quả chạy, phân tích mở rộng) → Ch. 4 NGHIÊN CỨU MỞ RỘNG (tr. 26–34: 2 mô hình liên quan SSEGCN + AoM) → Ch. 5 KẾT LUẬN (tr. 35) → Tài liệu tham khảo (tr. 36) → **bản dịch toàn văn bài báo đính kèm sau tài liệu tham khảo, không ghi nhãn "Phụ lục"** (tr. 1–9 của bản dịch, bắt đầu lại từ Abstract).
- Cũng không có Lời nói đầu trong file Nhóm 8 này — khác với form Nhóm-5 đã dùng cho HELIX/AI-copy (có Lời nói đầu). Khi làm CausalXRL, giữ Lời nói đầu như HELIX/AI-copy đã làm, vì đó là form Nhóm-5 đang dùng; đừng bỏ theo Nhóm-8.
- **Sửa kế hoạch cũ:** mục lục KG-R1 trước đây dùng khung S-T-P / Marketing-Mix là KHÔNG đúng form này. Với CausalXRL, dùng đúng 5 chương Nhóm-8/NHóm-5 trên: 1 Giới thiệu, 2 Phân tích, 3 Thực nghiệm của sinh viên, 4 Mở rộng, 5 Kết luận — không dùng S-T-P.

### c) "Chương sách 15%" — Triệu chốt 11/11: NHÓM CHỈ CÓ DỊCH BÁO, KHÔNG DỊCH SÁCH
- Triệu nói rõ: nhóm cậu chỉ có dịch báo thôi. Vì vậy với CausalXRL, **không làm dịch sách**.
- Trong mẫu Nhóm-8, dịch sách là 1 báo cáo riêng độc lập (Chương 8 E-Commerce, tr. 1–41 của PDF gộp) — đó là phần của nhóm trước, không phải phần nhóm Triệu phải làm.
- Form cho CausalXRL vì vậy chỉ còn phần bài báo: **báo cáo chính trước, rồi tới bản dịch báo ghép vào cuối** — đúng thứ tự Nhóm-8 phần bài báo: bìa → mục lục → danh sách bảng/hình → Ch. 1–5 → Tài liệu tham khảo → bản dịch toàn văn bài báo (không ghi nhãn "Phụ lục").
- Bản dịch báo đã có sẵn (37 trang, đủ 8/8 hình) — ghép vào cuối đúng chỗ đó, không cần dịch thêm gì.
- Lưu ý còn mở: bảng tiêu chí cuối kỳ cậu gửi trước đó vẫn ghi Bản dịch chương sách 15% — nếu kỳ này nhóm cậu không dịch sách thì 15% đó tính thế nào (bỏ, hay gộp vào dịch báo) thì nên hỏi thầy cho chắc, đừng tự đoán.

---

## 1. Trình bày trên lớp — chiếm 60% điểm giữa kỳ

| Tiêu chí | Trọng số | Muốn >85% thì cần gì | Hiện trạng CausalXRL | Việc phải làm |
|---|---|---|---|---|
| Hình thức báo cáo | 10% | Trình bày rõ ràng, chuyên nghiệp, không lỗi. Biểu đồ, hình ảnh đầy đủ, chú thích chính xác. Bố cục logic, dễ theo dõi. | Chưa có deck. Đã có đủ hình gốc (Hình 1–8) + số Table 1 đã kiểm chứng. | Làm deck theo form HELIX/AI-copy (33–36 slides): bìa, mục lục, đặt vấn đề, phương pháp 3 bước, Table 1 redraw, đồ thị nhân quả mẫu, kết quả chạy thật, hạn chế, mở rộng. Mỗi hình/bảng có chú thích tiếng Việt + nguồn. |
| Kỹ năng trình bày | 10% | Tự tin, rõ ràng, thuyết phục. Giải thích và trả lời chính xác. Dùng hiệu quả ngôn ngữ cơ thể và phương tiện hỗ trợ. | Chưa có kịch bản. | Viết kịch bản thuyết trình slide-by-slide (~15–20 phút, như 2 bài trước), tập dượt 1 lần, chia phần nói theo người. Demo live chỉ nên là đồ thị đã chạy xong, không demo cài R/dbarts trực tiếp trên lớp (dễ fail). |
| Nội dung / Chất lượng sản phẩm | 50% | Nội dung sâu sắc, đầy đủ, logic chặt chẽ. Dữ liệu chính xác, phân tích chi tiết. Sản phẩm có tính sáng tạo, áp dụng tốt vào thực tiễn. Mở rộng, ý tưởng và thực thi đề xuất cải tiến tốt sẽ được cộng thêm điểm (+). | Đã có: tóm tắt đúng Table 1, phân tích repo thật (~4.883 dòng), biết rõ repo KHÔNG train RL, policy là heuristic viết tay. Chưa có: số đo chạy thật của nhóm. | Đây là tiêu chí quyết định. Phải có (1) số chạy thật ở §3, (2) phân tích vì sao CartPole thua DT (91,4% vs 94,1%) nhưng PongDuel/MicroRTS-Lite thắng, (3) phân tích tốc độ học cấu trúc (0,1–0,5s vs PC 1,7–23,2s), (4) đánh giá 8 chuyên gia 8/8 vs 4/8 kèm lưu ý mẫu nhỏ n=8, (5) 1 mở rộng làm được (xem §4). Không bịa số, không trình bày demo chưa chạy thành "tái hiện đầy đủ". |
| Trả lời câu hỏi | 10% | Trả lời đúng tất cả các câu hỏi. | Chưa có ngân hàng Q&A. Nhớ: chỉ bạn cùng lớp hỏi, thầy không hỏi miệng (theo kinh nghiệm các bài trước). | Chuẩn bị 10 câu hỏi dự kiến + đáp án ngắn, mỗi người phụ trách 2–3 câu theo phần mình làm. Câu chắc chắn gặp: "Sao không train RL?", "Sao CartPole thua DT?", "BART cần R thì cài thế nào?", "MicroRTS sao không chạy?", "Đồ thị học được có đúng là nhân quả không?" |
| Mức độ đóng góp mỗi thành viên | 20% | Đóng góp ≥85% so với thành viên đóng góp nhiều nhất trong nhóm. | Chưa có bảng phân công. | Lập bảng đóng góp ngay từ đầu, ghi rõ ai làm gì (dịch, phân tích code, chạy CartPole/LunarLander/PongDuel, vẽ hình, làm deck, thuyết trình, Q&A). Mỗi người phải có phần thực nghiệm hoặc phân tích cụ thể, không chỉ "dịch" hay "làm slide". Cần danh sách thành viên + MSSV để điền bìa. |

Rủi ro lớn nhất ở phần trình bày: Nội dung 50% + Đóng góp 20% = 70% điểm trình bày.
Nếu không chạy thật, Nội dung tối đa chỉ ở mức "Gần đạt" vì "dữ liệu chưa đủ, phân tích sơ sài".

---

## 2. Báo cáo tiểu luận cuối kỳ — chiếm 50% điểm cuối kỳ

| Tiêu chí | Trọng số | Muốn >85% thì cần gì (suy từ bảng b) | Hiện trạng | Việc phải làm |
|---|---|---|---|---|
| Hình thức | 7% | Theo form Nhóm-8 / Nhóm-5 đã xác nhận ở §0: bìa, Lời nói đầu (theo Nhóm-5/HELIX, Nhóm-8 không có nhưng giữ lại), Mục lục, Danh sách bảng + hình, 5 chương (1 Giới thiệu, 2 Phân tích, 3 Thực nghiệm sinh viên, 4 Mở rộng, 5 Kết luận), Tài liệu tham khảo, rồi bản dịch bài báo đính kèm sau. Không lỗi chính tả, bảng/hình đánh số + chú thích đầy đủ. | Chưa có báo cáo viết. | Viết theo đúng 5 chương trên, KHÔNG dùng khung S-T-P / Marketing-Mix của mục lục KG-R1 cũ. |
| Mục đích | 12% | Nêu rõ vấn đề, mục tiêu, phạm vi — nói thật phạm vi ngay từ đầu. | Đã rõ: giải thích agent RL bằng đồ thị nhân quả ở tầng chiến lược, không gắn cạnh vào action như AIM. | Viết Chương 1: vì sao RL là hộp đen, vì sao saliency/SHAP chưa đủ, vì sao AIM không dùng được cho RTS nhiều unit. Ghi rõ phạm vi evaluation-only (xem §3). |
| Phân tích | 20% | Phân tích phương pháp + kết quả của tác giả chi tiết, có số, có so sánh, có phản biện. | Đã có tóm tắt + phân tích code chi tiết. Số Table 1 đã kiểm chứng từ paper. | Chương 2–3: BART/ATE/HTE/CATE, Thuật toán 1 (global) + Thuật toán 2 (local + counterfactual DiCE), vì sao nhanh hơn PC/FCI, vì sao thua DT ở CartPole. Phân tích bug/rủi ro repo: `HTE` assert hardcode 500 draws, `counterfactual()` gọi `self.pi.predict` trong khi `pi` là hàm, `infer_causes_ate_barl` import module không tồn tại, MicroRTS import `stratvis` cũ. |
| Thực nghiệm | 20% | Chạy thật, có số đo của nhóm, có bảng so sánh với paper, nêu rõ điều kiện chạy. | Chưa chạy. Đây là lỗ hổng lớn nhất. | Xem §3 — chạy CartPole → LunarLander → PongDuel, 200 episode/môi trường, CPU-only, < $1. Không tính MicroRTS (thiếu dữ liệu, README ghi liên hệ tác giả). |
| Mở rộng | 5% | Có ý tưởng mở rộng + (tốt nhất) có thực thi thử, dù nhỏ. | Chưa chốt hướng. | Xem §4 — chọn 1 hướng làm được trong 1–2 buổi, không chọn hướng cần train RL hay cần dữ liệu MicroRTS. |
| Bản dịch — Chương sách | 15% (theo bảng tiêu chí cũ) | — | ❌ KHÔNG LÀM — Triệu chốt 11/11: nhóm chỉ có dịch báo, không dịch sách. | Không dịch sách. Chỉ lưu ý: hỏi thầy 15% này kỳ này tính thế nào (bỏ hay gộp vào dịch báo), đừng tự đoán. |
| Bản dịch — Bài báo | 21% | Dịch đầy đủ toàn văn bài báo, giữ công thức, bảng, thuật toán, hình. | ✅ ĐÃ XONG — bản dịch đầy đủ 37 trang A4, 2,4 MB, đủ 8/8 hình, giữ Bảng 1, Thuật toán 1–2, Phụ lục A–D. File: `causalxrl-ban-dich-day-du/causalxrl-ban-dich-day-du.pdf` (+ bản Word 21 trang ở `causalxrl-ban-dich-tieng-viet/`). | Đưa vào Phụ lục A của báo cáo viết. Kiểm tra lại 1 lượt thuật ngữ: BART, ATE/HTE/CATE, DBN, DiCE, counterfactual — giữ tên gốc kèm tiếng Việt lần đầu xuất hiện. Không cần dịch lại. |

Tổng Bản dịch = 36% điểm cuối kỳ — lớn nhất trong các nhóm.
Đã chắc 21% (bài báo), còn 15% (chương sách) là ẩn số phải làm rõ ngay.

---

## 3. Thực nghiệm — phạm vi trung thực, làm được (20% cuối kỳ + 50% trình bày)

Phạm vi: **evaluation-only, không huấn luyện RL.**

- Repo không có code train agent. Policy trong notebook là heuristic viết tay
  (CartPole: luật tuyến tính trên vị trí/góc; LunarLander: PD controller).
  Nói đúng như vậy trong báo cáo — không gọi là "tái hiện huấn luyện".
- Chạy theo thứ tự: CartPole → LunarLander → PongDuel, 200 episode/môi trường
  (LunarLander trong notebook eval chỉ 30 episode vì nặng — ghi rõ nếu giữ 30).
- Đo 2 thứ như paper Table 1: (1) accuracy/F1 dự đoán action của local model
  so với Linear Model + Decision Tree, (2) thời gian học cấu trúc so với PC/FCI.
- Số của tác giả để đối chiếu (Table 1, arXiv:2610.42028v1):
  CartPole 91,4% / F1 91,6% (thua DT 94,1%); LunarLander 88,7% / 86,6%;
  PongDuel 90,2% / 90,7%; MicroRTS-Lite 86,0% / 85,9%.
  Thời gian: CausalXRL 0,1 / 0,3 / 0,5 / 0,5s vs PC 1,7 / 12,6 / 23,2 / 20,9s.
- Số của nhóm sẽ không khớp 100% paper vì policy heuristic + seed + phiên bản
  thư viện khác — ghi rõ điều kiện chạy (máy, seed, số episode) và giải thích
  chênh lệch, không sửa số cho khớp paper.
- Rào cản lớn nhất: BART qua R package `dbarts` + `rpy2` + Java. Cài được bước
  này thì phần còn lại là CPU thuần, < $1. Nếu không cài được dbarts thì
  Thực nghiệm không có số — phải báo sớm để đổi hướng (dùng DecisionTree
  làm baseline thay thế, ghi rõ là thay thế, không giả vờ là BART).
- Không làm MicroRTS: requirements cũ (TF 2.7, torch 1.8, gym-microrts 0.3.2),
  import `stratvis` sót, thiếu dữ liệu — README ghi phải liên hệ tác giả.

---

## 4. Mở rộng — chọn 1 hướng (5% cuối kỳ + điểm cộng ở trình bày)

Chọn 1 trong 3, theo độ khó tăng dần. Khuyến nghị: hướng A, làm được chắc chắn.

**A. Ablation + kiểm tra độ ổn định (khuyến nghị)**
Chạy CartPole với số episode 50 / 100 / 200, hoặc ngưỡng CI 90% / 95%,
xem đồ thị nhân quả và accuracy thay đổi thế nào. Rẻ, CPU-only, có số để
vẽ biểu đồ, đúng tinh thần "phân tích" của bài.

**B. Sửa bug + kiểm chứng lại counterfactual**
Sửa 2 bug đã tìm thấy trong phân tích code (`HTE` assert 500, `counterfactual`
gọi `self.pi.predict` khi `pi` là hàm), chạy lại 1 ví dụ counterfactual trên
CartPole bằng DiCE và ghi kết quả. Chứng minh nhóm đọc code thật, không chỉ
chạy notebook.

**C. Áp dụng sang 1 môi trường gym mới, nhỏ**
Thử cùng pipeline XRL trên 1 môi trường gym đơn giản khác (ví dụ Acrobot
hoặc MountainCar) với biểu diễn là state gốc. Chỉ làm nếu A đã xong và còn
thời gian — vì cần viết policy heuristic mới và kiểm tra BART có hội tụ không.

Không chọn: train agent RL mới, làm MicroRTS, làm human study với 8 chuyên
gia như paper (không có người, không có IRB, không có kinh phí).

**D. Train agent RL thật rồi giải thích nó bằng CausalXRL (Triệu hỏi 11/11 — ĐƯỢC, khả thi, và là hướng mở rộng mạnh nhất)**
- Vì sao được: CausalXRL chỉ cần 1 policy `pi(state) -> action`, không quan tâm pi train kiểu gì. Repo hiện dùng policy heuristic viết tay (CartPole: luật tuyến tính; LunarLander: PD controller), không có code train, không có checkpoint. Thay heuristic bằng agent train thật là đúng interface `XRL(env, pi)`, và lấp đúng chỗ tác giả không công bố code.
- Phạm vi nên làm: **chỉ CartPole trước, thêm LunarLander nếu còn thời gian. Không train PongDuel (multi-agent, ma-gym, khó ổn định) và không train MicroRTS (thiếu dữ liệu).**
- Cách làm: viết DQN (hoặc PPO) đơn giản bằng PyTorch chạy trên gym 0.20 mà repo đang dùng — tránh dùng stable-baselines3 + gymnasium vì sẽ đụng version với repo. CartPole: train tới reward trung bình ~450–500, CPU 10–20 phút, <$1. LunarLander: khó hơn, 30–60 phút CPU, cần tuning, chỉ làm sau khi CartPole xong.
- So sánh để viết báo cáo: chạy XRL 200 episode trên heuristic vs trên agent đã train, so (1) accuracy/F1 dự đoán action, (2) đồ thị nhân quả học được có khác không, (3) thời gian học cấu trúc. Đây mới là kết quả mở rộng có số, không phải chỉ ý tưởng.
- Rủi ro lớn nhất: train xong vẫn phải chạy được BART/R (dbarts) mới có số XRL để so — nếu BART không cài được thì phần train vẫn có giá trị riêng (agent + reward curve) nhưng không ghép được vào Table 1. Vì vậy thứ tự: chạy XRL trên heuristic trước để có baseline, rồi mới train, rồi chạy XRL trên agent đã train. Nếu train không ổn định thì fallback về hướng A (ablation).

---

## 5. Thứ tự làm — để không mất điểm oan

1. **Hỏi rõ ngay (chỉ còn 1 câu):** bài này có thay KG-R1
   không, hay là bài riêng của nhóm. Câu về chương sách đã chốt xong — không dịch sách.
2. Chạy thử CartPole trước (rào cản R/dbarts). Nếu chạy được thì mới cam kết
   mốc Thực nghiệm >85%; nếu không thì báo sớm để đổi phương án.
3. Chốt hướng mở rộng (A/B/C) sau khi CartPole chạy được.
4. Viết báo cáo theo form 5 chương + Phụ lục A (bản dịch bài báo đã có sẵn),
   đồng bộ số liệu giữa báo cáo viết – deck – kịch bản (cùng 1 bộ số, không
   để mỗi chỗ một số như lỗi từng gặp ở HELIX/AI-copy).
5. Làm deck + kịch bản + ngân hàng 10 câu Q&A, chia phần nói theo bảng đóng góp.

## 6. Câu hỏi cần Triệu chốt

- [ ] Chương sách cần dịch là chương nào? Có file/chương mẫu không?
- [ ] CausalXRL có thay KG-R1 không, hay là tiểu luận riêng, KG-R1 vẫn giữ?
- [ ] Danh sách thành viên nhóm + phân công ai chạy môi trường nào?
- [ ] Chốt hướng mở rộng A, B hay C?
- [ ] Có bắt đầu chạy thử CartPole ngay không (cần cài R + dbarts + rpy2)?

---
*Cập nhật: 11/11/2026, từ tiêu chí thầy gửi + file tóm tắt/phân tích/bản dịch đã có.*
