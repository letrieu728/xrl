# CausalXRL — Tái hiện thực nghiệm và mở rộng

Tiểu luận môn **Cơ sở Khoa học Hệ thống Thông tin (CSCHTTT, INT3201_1)**,
Trường Đại học Công nghệ — Đại học Quốc gia Hà Nội.
Nhóm 5: **Lê Quốc Triệu** (MSSV 24022750) và **Nguyễn Sỹ Việt**.

Bài báo được tái hiện: **CausalXRL: Explainable Reinforcement Learning
through Causal Graph Reasoning** — Yanming Zhang, Eric Papenhausen, Klaus
Mueller (Stony Brook University), arXiv:2610.42028v1, ICML 2026.
Repo gốc của tác giả: <https://github.com/yanmluk/causal-xrl> (commit
`ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162`, giấy phép MIT).

Repo này chứa công việc của nhóm: phân tích bài báo và mã nguồn
của tác giả và thực nghiệm tái hiện (Mục 3). Hướng mở rộng cũ
(Mục 4) đã được gỡ khỏi repo ngày 16/11/2026 theo quyết định đổi hướng
của nhóm; toàn bộ nội dung cũ còn trong lịch sử git. Hướng mở rộng mới
đang lên kế hoạch, chưa có kết quả (xem mục "Hướng mở rộng mới" dưới
đây). Mọi số liệu đều là số nhóm chạy thật, có log, CSV hoặc JSON kèm
theo trong repo này hoặc trong thư mục Drive liên kết dưới đây. Số của
tác giả luôn được ghi rõ nguồn và chỉ dùng để đối chiếu.

## Hướng dẫn đọc nhanh (dành cho thầy)

1. **File này (README)** — tóm tắt toàn bộ công việc, kết quả chính và
   hạn chế trong khoảng 10 phút đọc.
2. **`SO-SANH-TAC-GIA-VS-NHOM.md`** — bảng đối chiếu từng con số của tác
   giả với số của nhóm, kèm nguồn và caveat của mỗi số.
3. **`muc3/EXPERIMENT_REPORT.md`** — báo cáo đầy đủ của Mục 3 (tái hiện
   evaluation-only trên 3 môi trường gym, bằng BART thật).
4. **`muc3/microRTS-thu-nghiem.md`** và
   **`muc3/counterfactual-thu-nghiem.md`** — hai báo cáo chuyên sâu về
   MicroRTS/dashboard và về counterfactual.

Gói tổng hợp riêng để thầy đọc nhanh (thư giới thiệu, 9 báo cáo, số liệu,
hình ảnh, paper và bản dịch, các patch) nằm ở thư mục Drive
[CausalXRL - Tong hop cho thay doc](https://drive.google.com/drive/folders/12b-g_BVK9QP6xMs6oZ3gxNX5MJPi7pbk)
(riêng tư, chưa chia sẻ). Bản thuyết trình 15–20 phút: xem
`THUYET-TRINH.md`.

## Kết quả chính

| Hạng mục | Kết quả của nhóm | Đối chiếu |
|---|---|---|
| Tái hiện CartPole (BART thật, seed 99) | 91,62% / F1 91,50% | Paper 91,4% / 91,6% — gần như khớp |
| Tái hiện LunarLander (BART thật, seed 99) | 82,80% / weighted-F1 81,52% | Paper 88,7% / 86,6% — thấp hơn 5,90 / 5,08 điểm % |
| Tái hiện PongDuel | 81,73% / macro-F1 19,51% (**PROXY**) | **Không so được** với paper (xem caveat) |
| MicroRTS-Lite | Chạy thật 100 episode, đồ thị 23 cạnh, dashboard 9 lượt kiểm tra trên 7 route đều 200 | Fidelity theo đúng protocol **không đo được** (đã chẩn đoán tới cấp cơ chế) |
| Counterfactual (DiCE) | 25/25 (heuristic) kiểm chứng lật đúng action | Code gốc của tác giả không chạy được, nhóm đã sửa |
| Nghiên cứu chuyên gia (expert study) | **Không làm** | Ngoài khả năng của sinh viên (xem "Hạn chế") |

Hai điểm đáng chú ý nhất:

1. **CartPole tái hiện gần như khớp paper**, và như paper, Decision Tree
   thắng BART (93,83% so với 91,62%) — đó là kết quả của tác giả, không
   phải lỗi của nhóm.
2. **Thời gian học cấu trúc trong Table 1 của paper là timer mô phỏng**,
   không phải wall-clock BART thật — nhóm phát hiện điều này khi đọc
   notebook `eval_time_*` của tác giả, đo wall-clock thật và ghi rõ hai
   cách đo không so trực tiếp được.

## Bài báo nói gì (tóm tắt)

Học tăng cường (RL) train được agent rất giỏi nhưng là "hộp đen": không
biết vì sao nó chọn hành động đó. CausalXRL giải thích hành vi của agent
RL bằng **đồ thị nhân quả (causal graph)** giữa các biểu diễn chiến lược
mà con người hiểu được, thay vì gắn cạnh nhân quả vào action đơn lẻ như
AIM trước đây.

Phương pháp 3 bước của tác giả (§4, Fig. 1):

1. **Trích biểu diễn hiểu được** — từ trạng thái thô lấy ra các đặc trưng
   chiến lược (ví dụ trong MicroRTS: số unit từng loại, tài nguyên).
2. **Học đồ thị nhân quả toàn cục** — chạy agent 200 episode, lưu
   transition vào replay buffer, dùng BART (R package `dbarts`) ước lượng
   ATE / HTE (CATE) cho từng biến; biến nào có khoảng tin cậy 95% không
   chứa 0 thì là nguyên nhân (Algorithm 1).
3. **Mô hình local + phản thực** — với một trạng thái cụ thể, xác định
   nguyên nhân local, dự đoán hành động tiếp theo có trọng số theo |ITE|,
   và dùng DiCE sinh counterfactual (Algorithm 2).

Kết quả của tác giả (Table 1, arXiv:2610.42028v1 — **số của tác giả, chỉ
để đối chiếu**):

| Môi trường | CausalXRL Acc / F1 | DT Acc | Thời gian học cấu trúc |
|---|---|---|---|
| CartPole | 91,4% / 91,6% | 94,1% | 0,1s (PC 1,7s) |
| LunarLander | 88,7% / 86,6% | 85,1% | 0,3s (PC 12,6s) |
| PongDuel | 90,2% / 90,7% | 88,3% | 0,5s (PC 23,2s) |
| MicroRTS-Lite | 86,0% / 85,9% | 80,3% | 0,5s (PC 20,9s) |

Lưu ý trung thực: ở CartPole, CausalXRL **thua Decision Tree** (91,4% so
với 94,1%) — tác giả thừa nhận, chỉ nói là "competitive". Nghiên cứu
chuyên gia của paper chỉ có 8 người (bài dự đoán hành vi: 8/8 so với 4/8
khi đối chiếu với SHAP), tác giả tự ghi là bằng chứng khám phá, mẫu nhỏ.
F1 đa lớp của paper là **weighted-F1** (§5.2).

Tóm tắt đầy đủ: `docs/CausalXRL-Tom-tat-noi-dung.md`. Phân tích mã nguồn
của tác giả: `docs/CausalXRL-Phan-tich-ma-nguon.md`. Bản dịch tiếng Việt
toàn văn (28 trang, đủ 8/8 hình): thư mục Drive dưới đây.

## Nhóm đã làm gì (trình tự công việc)

1. **Đọc paper + phân tích mã nguồn tác giả** (10/11/2026): tóm tắt,
   phân tích code thật — phát hiện repo tác giả **không có code train
   RL** (policy trong notebook chỉ là heuristic viết tay), hàm
   `counterfactual()` không chạy được, phía MicroRTS import package cũ
   `stratvis` và thiếu checkpoint PPO.
2. **Mục 3 — Thực nghiệm của sinh viên, evaluation-only** (11–12/11):
   chạy CartPole (200 episode) → LunarLander (30 episode, theo notebook
   của tác giả) → PongDuel (200 episode), seed 99, CPU-only, BART thật
   qua R/dbarts. Xem `muc3/EXPERIMENT_REPORT.md`.
3. **Mục 4 — Mở rộng cũ** (11–15/11): train agent RL thật, so sánh XRL
   trained-vs-heuristic, ablation, sensitivity, counterfactual cho
   policy đã train, mô phỏng sửa đồ thị — **đã gỡ khỏi repo ngày
   16/11/2026** theo quyết định đổi hướng mở rộng; nội dung còn trong
   lịch sử git.
4. **Nhóm việc 1** (12/11): khôi phục weighted-F1 cho BART LunarLander
   (81,52%) từ checkpoint đã lưu, căn split baseline CartPole, vẽ đồ thị
   Mục 3 + case study local kiểu Fig. 4 của paper.
5. **Nhóm việc 2** (12/11): MicroRTS 100 episode + sửa dashboard, mở rộng
   counterfactual (6 query CartPole + 4 query LunarLander).
6. **Nhóm việc 3** (12/11): sửa xong route `/get_local_causal`; chẩn
   đoán fidelity MicroRTS tới cấp cơ chế.
7. **Rà soát đồng bộ toàn repo** (13/11): đối chiếu mọi con số lặp lại
   nhiều nơi với file JSON/log nguồn; sửa số PC/FCI MicroRTS trong bảng
   so sánh; viết lại README này.
8. **Đổi hướng mở rộng** (16/11/2026): nhóm quyết định bỏ hướng mở rộng
   cũ và gỡ toàn bộ nội dung của nó khỏi repo (còn trong lịch sử git).
   Hướng mở rộng mới đang lên kế hoạch, chưa có kết quả — xem mục
   "Hướng mở rộng mới" dưới đây.

Bảng đối chiếu đầy đủ mọi con số: `SO-SANH-TAC-GIA-VS-NHOM.md`.

## Kết quả chi tiết

### Mục 3 — Tái hiện evaluation-only (seed 99, BART thật, heuristic của tác giả)

Chi tiết + protocol: `muc3/EXPERIMENT_REPORT.md`. F1 so được với Table 1:
F1 nhị phân ở CartPole, weighted-F1 ở LunarLander/PongDuel.

| Môi trường | BART (nhóm) | Logistic Regression | Decision Tree |
|---|---|---|---|
| CartPole (200 ep, 6.115 trans) | 91,62% / F1 91,50% | 89,62% / 89,26% | 93,83% / 93,64% |
| LunarLander (30 ep, 7.382 trans) | 82,80% / weighted-F1 81,52% | 77,21% / 76,24% | 79,14% / 79,06% |
| PongDuel (200 ep, 74.445 trans) | 81,73% / macro-F1 19,51% (PROXY) | 81,79% / 81,38% | 96,21% / 96,21% |

- CartPole: tái hiện gần như khớp paper (+0,22 điểm % accuracy, −0,10
  điểm % F1). Như paper, Decision Tree thắng BART (93,83% so với
  91,62%).
- LunarLander: weighted-F1 của BART là **81,52%**, khôi phục từ
  checkpoint điểm số đã lưu mà không chạy lại BART (tái tổng hợp cho
  accuracy/macro-F1 sai khác 0,0 so với số gốc) — so được thẳng với
  86,6% của paper (−5,08 điểm %). Macro-F1 64,10% giữ kèm để đối chiếu.
  Per-class F1: do_nothing 87,51%, fire_main 83,06%, fire_left 59,68%,
  fire_right chỉ 26,14%.
- **PongDuel là PROXY** (hồi quy số làm tròn 0–8, 14.995 mẫu ứng viên
  local), không phải fidelity phân loại tương đương paper — không so
  với 90,2%/90,7% của Table 1. Tác giả không có notebook fidelity cho
  PongDuel; baseline PongDuel là chuyển thể theo mẫu
  CartPole/LunarLander.
- Baseline CartPole đã tính lại ngày 12/11 theo đúng protocol notebook
  fidelity của tác giả (xem `muc3/results/baselines_recomputed.md`);
  trên đúng test set của BART: LR 90,43%/90,22%, DT 95,42%/95,32% (xem
  `muc3/results/baselines_aligned.md`). File `metrics.json` cũ giữ
  nguyên làm lịch sử.
- Thời gian học cấu trúc BART (tổng các process riêng lẻ, wall-clock
  thật, **không so trực tiếp với paper** vì số của paper là timer mô
  phỏng — notebook `eval_time_*` của tác giả dùng dữ liệu ngẫu nhiên):
  CartPole 242,19s; LunarLander 1.086,40s; PongDuel 35.531,20s
  (~9,9 giờ).
- Đồ thị toàn cục đã học: CartPole 13 cạnh, LunarLander 44 cạnh,
  PongDuel 53 cạnh (đủ 21/21 target). Ảnh đồ thị + 1 case study local
  kiểu Fig. 4 của paper: thư mục Drive dưới đây (`muc3/results/`).

### MicroRTS-Lite — chạy tối thiểu thật, fidelity theo đúng protocol không đo được

Chi tiết: `muc3/microRTS-thu-nghiem.md`.

- Sửa **22 chỗ** import `stratvis` → `cxrl_microRTS` trong **8 file**;
  lấy được checkpoint PPO `agent-3.pt` công khai (SHA-256 `ad21e903…`,
  load strict OK, 5.433.185 tham số, **không đưa vào repo**).
- Env thật chạy được (JVM + PPO trả action hợp lệ). Dashboard Flask:
  `/get_board_state` → **200** (trả board thật),
  `/compute_global_causal` → **200** sau khi tự sinh thư mục `data/` từ
  dữ liệu 100 episode; `/get_local_causal` → **200** sau sửa ở nhóm 3
  (danh sách cột hard-code của tác giả không khớp scaler học từ dữ liệu
  thật; patch `muc3/dashboard-local-causal-fix.patch`). **9 lượt kiểm
  tra trên 7 route đều trả 200**, với điều kiện `R_LIBS` trỏ tới dbarts
  và gọi `/compute_global_causal` trước.
- Đồ thị toàn cục **23 cạnh** từ **100 episode / 4.924 transition** thật
  (498,3 giây học BART cho 13 lượt fit: 12 target state + target
  action). Hiệu đính: đồ thị bước 3 ghi "10 episode / 537 transition"
  thực ra chỉ có 8 episode duy nhất (109 transition bị đếm trùng do seed
  ở cấp import).
- **Fidelity MicroRTS theo đúng protocol notebook thì không đo được**
  (tập valid rỗng: action chỉ có 1 cause toàn cục `heavy2`, không hàng
  nào có >1 cạnh local vào nút action). Nhóm 3 đã chẩn đoán tới cấp cơ
  chế: cờ local ⊆ nguyên nhân toàn cục nên ngưỡng >1 cờ không thể đạt
  được về mặt toán học; 13/14 cột treatment có 0 cluster HTE có ý nghĩa;
  BART cho target `action` fit kém (R² = 0,107). Không nới ngưỡng, không
  sinh số fidelity mới. Bản không-lọc (ghi rõ là lệch protocol) cho
  BART 25,15% / 30,86% accuracy (2 cách dựng), thua xa baseline đa số
  ~62–64% — con số 86,0%/85,9% của paper không tái tạo được từ repo
  (paper không cho mã hóa action).

### Counterfactual (DiCE) — code gốc không chạy được, nhóm sửa và chạy được

`XRL.counterfactual()` của tác giả không bao giờ chạy được (thiếu
`self.exp`, `self.state_features`, sai kiểu `self.pi`). Nhóm sửa tối
thiểu trên bản copy (patch ở `muc3/xrl-counterfactual-fix.patch`, patch
đa lớp ở `muc3/xrl-multiclass-fix.patch`) và chạy được đầu-cuối:
**CartPole 6 query, 9/9 counterfactual kiểm chứng lật đúng action bằng
policy gốc; LunarLander 4 query (4 action), 16/16 kiểm chứng lật** —
tổng **25/25**. Chi tiết: `muc3/counterfactual-thu-nghiem.md`.

Giới hạn chung: chưa thử PongDuel/MicroRTS; DiCE dùng phương pháp
`random`.

## Hướng mở rộng mới

Hướng mở rộng mới (đang lên kế hoạch, chưa có kết quả): Causal-Guided RL — đóng kín vòng lặp hành động → tự soi chiếu nhân quả → sửa sai, dùng đồ thị nhân quả ngay trong lúc train để cảnh báo/phạt quyết định tự hủy và thưởng cho chuỗi hành động dẫn tới kết quả tốt. Làm pilot nhỏ trước, đăng ký tiêu chí trước, báo cáo cả khi kết quả âm tính — theo bài học từ nhóm 4 (mô phỏng phẫu thuật đồ thị không đạt kiểm chứng), độ tin cậy của đồ thị phải được chứng minh, không được giả định.

## Hạn chế và việc chưa làm (ghi rõ, không giấu)

- **Nghiên cứu chuyên gia (expert study):** cần 8 chuyên gia RTS (>7
  năm, >1.000 giờ), IRB, chạy crossover với SHAP — ngoài khả năng của
  sinh viên trong phạm vi tiểu luận. Nhóm không làm và không thay bằng
  khảo sát bạn cùng lớp rồi ghi thành expert study.
- **Fidelity MicroRTS theo đúng protocol:** tập valid rỗng (xem trên);
  nhóm 3 đã chẩn đoán tới cấp cơ chế và xác nhận không sửa được bằng cấu
  hình. Dashboard thì **đã sửa xong** (9 lượt kiểm tra trên 7 route đều
  200).
- **PongDuel chỉ là proxy** (không có BART phân loại trong repo tác giả);
  LunarLander Mục 3 chỉ 30 episode.
- **Counterfactual** chưa làm cho PongDuel/MicroRTS.
- Số BART dao động ±1–2 cạnh giữa các lần chạy (code gốc không đặt
  seed); không đọc chênh lệch 1–2 cạnh thành kết luận.

## Cấu trúc repo

```
XRL/
├── README.md                  # file này
├── LICENSE                    # MIT (cùng loại với repo tác giả)
├── THUYET-TRINH.md            # bản thông tin dùng trong thuyết trình (15–20 phút)
├── SO-SANH-TAC-GIA-VS-NHOM.md # bảng so sánh tất cả con số tác giả vs nhóm
├── docs/
│   ├── CausalXRL-Tom-tat-noi-dung.md   # tóm tắt paper + Table 1
│   ├── CausalXRL-Phan-tich-ma-nguon.md # phân tích code repo thật
│   ├── CausalXRL-KE-HOACH-CHAM-DIEM.md # kế hoạch chấm điểm (§0–§6, viết 11/11)
│   └── CausalXRL-MUC3-CODEX-HANDOFF.md # quy tắc giao việc làm Mục 3
└── muc3/                      # Mục 3 — Thực nghiệm của sinh viên
    ├── EXPERIMENT_REPORT.md   # báo cáo Mục 3 (đọc file này trước)
    ├── microRTS-thu-nghiem.md # báo cáo MicroRTS (100 ep, dashboard, fidelity)
    ├── counterfactual-thu-nghiem.md # báo cáo counterfactual heuristic (25/25 kiểm chứng lật)
    ├── counterfactual_*.json/.csv # kết quả counterfactual (heuristic/PD) + log chạy
    ├── dashboard-fix.patch    # patch sửa dashboard (nhóm 2)
    ├── dashboard-local-causal-fix.patch # patch sửa /get_local_causal (nhóm 3)
    ├── stratvis-fix.patch     # patch sửa import stratvis (bước 3)
    ├── xrl-counterfactual-fix.patch # patch sửa counterfactual (+ xrl-multiclass-fix.patch)
    ├── scripts/               # collect/metrics/BART fidelity + recompute + MicroRTS + counterfactual
    ├── notes/                 # lỗi đã gặp + cách fix, blocker
    └── results/               # replay.npz + metrics.json + bart_fidelity.json từng môi trường;
                               # microrts/ (đồ thị, fidelity, chẩn đoán, dashboard);
                               # baselines_recomputed/aligned
```

Hướng mở rộng cũ (Mục 4) đã gỡ khỏi repo ngày 16/11/2026; nội dung còn
trong lịch sử git.

File nhị phân (PDF, ảnh PNG, một số `.npz`) không nằm
trong repo này mà ở thư mục Drive dưới đây, vì connector GitHub chỉ push
được file text UTF-8.

## Cách chạy lại

### 1. Mục 3 — Tái hiện evaluation-only (cần R + dbarts)

```bash
# Cài R + dbarts trước (bước khó nhất; dùng rpy2 3.5.16 với R 4.3.3):
sudo apt install r-base-core r-base-dev default-jdk
# trong Python: pip install rpy2; trong R: install.packages("dbarts")
# Mỗi môi trường: collect → metrics (BART từng target) → fidelity, xem
# muc3/scripts/cartpole_*.py, lunarlander_*.py, pongduel_*.py
# Tính lại baseline theo protocol notebook (không cần BART):
python muc3/scripts/recompute_baselines_weighted_f1.py
```

**Lưu ý:** R/dbarts sống ngoài thư mục làm việc nên **VM restart hoặc
thay VM có thể xóa R + dbarts** — khôi phục bằng
`dpkg -i /var/cache/apt/archives/*.deb` rồi cài lại dbarts 0.9.34 từ
CRAN. Mọi kết quả đã lưu trong repo thì không mất.

### 2. MicroRTS + dashboard

Cần Java + Xvfb + checkpoint PPO (lấy bằng `muc3/scripts/fetch_agent3_range.py`,
không đưa checkpoint vào repo). Sửa import: `muc3/stratvis-fix.patch` (bước 3)
và `muc3/dashboard-fix.patch` (nhóm 2) + `muc3/dashboard-local-causal-fix.patch`
(nhóm 3). Quy trình đầy đủ + log: xem `muc3/microRTS-thu-nghiem.md` §10–§11.

### 3. Counterfactual

Áp patch lên bản copy repo tác giả (`muc3/xrl-counterfactual-fix.patch` +
`muc3/xrl-multiclass-fix.patch`), rồi chạy
`muc3/scripts/run_counterfactual_cartpole_multi.py` và
`muc3/scripts/run_counterfactual_lunarlander.py`. Chi tiết:
`muc3/counterfactual-thu-nghiem.md`.

## File nhị phân — thư mục Drive

Toàn bộ file nhị phân của repo nằm ở thư mục Drive
[XRL - file nhi phan](https://drive.google.com/drive/folders/1rK_Br_Kj3Dr-OZb8c5V2AgJeUW8bx_ea)
(cùng cấu trúc thư mục với repo): PDF paper gốc + bản dịch tiếng Việt
(`paper/`, `ban-dich/`); ảnh PNG đồ thị nhân quả (`muc3/results/`); file
dữ liệu `.npz` và SCM/scaler MicroRTS (`muc3/results/microrts/`).

Gói tổng hợp cho thầy đọc (thư giới thiệu + báo cáo + số liệu + hình ảnh
+ paper/bản dịch + patch): [CausalXRL - Tong hop cho thay doc](https://drive.google.com/drive/folders/12b-g_BVK9QP6xMs6oZ3gxNX5MJPi7pbk)
(riêng tư, chưa chia sẻ — việc chia sẻ do nhóm quyết định).

## Phạm vi trung thực — đọc trước khi trích số

- Số của **tác giả** (Table 1, 8 chuyên gia, MicroRTS-Lite) chỉ trích từ
  paper arXiv:2610.42028v1, ghi rõ nguồn. Nhóm **không** làm human study
  và **không** tái tạo được số fidelity MicroRTS 86,0%/85,9% của Table 1.
- Số của **nhóm** là số chạy thật, có log/CSV/checkpoint kèm theo. Số
  BART dao động ±1–2 cạnh giữa các lần chạy (BART không có seed trong
  code gốc).
- Repo tác giả **không train agent RL**; policy trong notebook của tác
  giả là heuristic viết tay — nhóm tái hiện đúng như vậy
  (evaluation-only, không huấn luyện RL).
- Baseline CartPole cũ trong `muc3/results/cartpole/metrics.json` đo
  task khác với notebook của tác giả (state_t → action_t) — đã sửa ngày
  12/11/2026, số chốt theo protocol notebook ở
  `muc3/EXPERIMENT_REPORT.md`; file `metrics.json` cũ giữ nguyên làm
  lịch sử.

## File đã bỏ qua khi đưa lên repo (và lý do)

- `repo-causal-xrl/` — bản copy repo tác giả, chỉ đọc; lấy từ
  https://github.com/yanmluk/causal-xrl (commit `ae9096e`).
- Log stdout dài — chỉ giữ log tóm tắt + CSV + JSON.
- Checkpoint PPO MicroRTS `agent-3.pt` — file nhị phân, không rõ quyền
  tái phân phối; lấy lại bằng script + SHA ghi trong
  `muc3/microRTS-thu-nghiem.md`.

## Lịch sử commit chính

Ngày 16/11/2026: nhóm quyết định bỏ hướng mở rộng cũ; toàn bộ nội dung
Mục 4 đã gỡ khỏi nhánh main (còn trong lịch sử git). Các dòng dưới đây
là lịch sử, giữ nguyên để đối chiếu.

- `060968c` — Initial commit (11/11/2026)
- `3986d74` — Tóm tắt, phân tích mã nguồn, kế hoạch, Mục 4 DQN + so sánh
  XRL + ablation (11/11)
- `9a0f344` — Mục 3: tái hiện evaluation-only 3 môi trường (12/11)
- `55109bf` / `cbcd492` — Sửa số transition PongDuel, ghi blocker
  CartPole đã giải quyết (12/11)
- `5406c94` — Mục 3: baseline đúng protocol notebook + weighted-F1
  (12/11)
- `c90a392` — Mục 4: ablation + sensitivity bằng BART thật (12/11)
- `c2fa6ca` — MicroRTS chạy được + sửa/chạy counterfactual (12/11)
- `585ffff` — Bảng so sánh tác giả vs nhóm + 3 biểu đồ (12/11)
- `2237c0f` — Nhóm 1: weighted-F1 BART LunarLander, căn split CartPole,
  đồ thị Mục 3 (12/11)
- `d798ed8` — Nhóm 2: MicroRTS 100 episode + dashboard, counterfactual
  mở rộng, DQN LunarLander + CartPole seed 1 (12/11)
- `ebd15c8` — Viết lại README mặt tiền + thêm LICENSE MIT (12/11)
- `f372a37` / `f906491` / `6a75818` / `45b1b79` — Sửa README trực tiếp
  trên web (làm rõ trạng thái repo, nội dung + định dạng, 12/11)
- `d105410` — Nhóm 3 (việc làm 12/11, đẩy lên 13/11): GES đủ 4 môi
  trường, sửa /get_local_causal, chẩn đoán fidelity MicroRTS, XRL
  LunarLander, DQN PongDuel âm tính + bỏ qua XRL PongDuel; cập nhật
  README/THUYET-TRINH/SO-SANH
- `7a3c64b` — README: thêm dòng lịch sử commit nhóm 3 (13/11)
- `f31b1bd` — Vẽ lại 3 biểu đồ so sánh: weighted-F1 LunarLander 81,52%,
  thêm cột GES (13/11)
- `dba9d9a` — Rà soát đồng bộ toàn repo: sửa số PC/FCI MicroRTS, sửa
  nhãn accuracy/weighted-F1 ở so sánh XRL LunarLander, viết lại README
  cho thầy đọc (13/11)
- `5449cd2` — Nhóm 4 (15/11): counterfactual cho policy đã train,
  sensitivity LunarLander + trọng số cạnh, mô phỏng sửa đồ thị (không
  kiểm chứng) — nội dung đã gỡ khỏi main ngày 16/11/2026, còn trong
  commit này

## Credits

- Bài báo và code gốc: Yanming Zhang, Eric Papenhausen, Klaus Mueller
  (Stony Brook University) — arXiv:2610.42028v1, repo
  https://github.com/yanmluk/causal-xrl (MIT).
- Thực hiện tái hiện: nhóm 5, môn CSCHTTT — Lê Quốc Triệu
  (MSSV 24022750) và Nguyễn Sỹ Việt.
- Giấy phép của repo này: MIT (xem `LICENSE`), cùng loại với repo tác
  giả.

