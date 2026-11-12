# CausalXRL — Tái hiện thực nghiệm và mở rộng

Tiểu luận môn **Cơ sở Khoa học Hệ thống Thông tin (CSCHTTT, INT3201_1)**,
trường Đại học Công nghệ — ĐHQGHN. Nhóm 5: Lê Quốc Triệu và Nguyễn Sỹ Việt.

Bài báo: **CausalXRL: Explainable Reinforcement Learning through Causal Graph
Reasoning** — Yanming Zhang, Eric Papenhausen, Klaus Mueller (Stony Brook
University), arXiv:2610.42028v1, ICML 2026.
Repo gốc của tác giả: <https://github.com/yanmluk/causal-xrl> (commit
`ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162`, MIT).

> Mọi số liệu trong README này đều là số nhóm chạy
> thật, có log/CSV/checkpoint kèm theo trong repo hoặc trong thư mục Drive
> dưới đây. Số của tác giả luôn được ghi rõ nguồn và chỉ dùng để đối chiếu.

## Đọc nhanh (dành cho thầy)

Nhóm tái hiện CausalXRL ở 3 môi trường gym (CartPole, LunarLander, PongDuel)
bằng BART thật, chạy tối thiểu được MicroRTS (100 episode + một phần
dashboard), sửa và chạy được hàm counterfactual của tác giả (vốn không chạy
được), rồi mở rộng bằng cách **tự train agent DQN thật** — điều repo tác giả
không có — và so sánh khả năng giải thích của agent đã train với policy
heuristic viết tay.

Kết quả đáng chú ý nhất: CartPole tái hiện gần như khớp paper (91,62% so với
91,4%); DQN CartPole đạt trần 500/500 ở cả hai seed nhưng lại **khó giải thích
hơn** heuristic (BART fidelity 0,8050 so với 0,9350); và số thời gian học cấu
trúc của paper hóa ra là timer mô phỏng, không phải wall-clock BART thật —
nhóm đo wall-clock thật và ghi rõ không so trực tiếp được.

**Tự chấm của nhóm: 6,7/10 so với paper** (chấm sau 3 bước tăng điểm ngày
12/11/2026, theo 8 tiêu chí đối chiếu paper; chưa chấm lại sau nhóm việc 1+2
ngày 12/11). Đây là điểm tự đánh giá của nhóm so với nội dung bài báo,
**không phải điểm môn học**. Nghiên cứu chuyên gia (expert study) của paper
nhóm không làm — xem mục "Chưa làm" dưới đây.

## Bài báo nói gì (tóm tắt)

Học tăng cường (RL) train được agent rất giỏi nhưng là "hộp đen": không biết
vì sao nó chọn hành động đó. CausalXRL giải thích hành vi của agent RL bằng
**đồ thị nhân quả (causal graph)** giữa các biểu diễn chiến lược mà con người
hiểu được, thay vì gắn cạnh nhân quả vào action đơn lẻ như AIM trước đây.

Phương pháp 3 bước của tác giả (§4, Fig. 1):

1. **Trích biểu diễn hiểu được** — từ trạng thái thô lấy ra các đặc trưng
   chiến lược (ví dụ trong MicroRTS: số unit từng loại, tài nguyên).
2. **Học đồ thị nhân quả toàn cục** — chạy agent 200 episode, lưu transition
   vào replay buffer, dùng BART (R package `dbarts`) ước lượng ATE / HTE
   (CATE) cho từng biến; biến nào có khoảng tin cậy 95% không chứa 0 thì là
   nguyên nhân (Algorithm 1).
3. **Mô hình local + phản thực** — với một trạng thái cụ thể, xác định nguyên
   nhân local, dự đoán hành động tiếp theo có trọng số theo |ITE|, và dùng
   DiCE sinh counterfactual (Algorithm 2).

Kết quả của tác giả (Table 1, arXiv:2610.42028v1 — **số của tác giả, chỉ để
đối chiếu**):

| Môi trường | CausalXRL Acc / F1 | DT Acc | Thời gian học cấu trúc |
|---|---|---|---|
| CartPole | 91,4% / 91,6% | 94,1% | 0,1s (PC 1,7s) |
| LunarLander | 88,7% / 86,6% | 85,1% | 0,3s (PC 12,6s) |
| PongDuel | 90,2% / 90,7% | 88,3% | 0,5s (PC 23,2s) |
| MicroRTS-Lite | 86,0% / 85,9% | 80,3% | 0,5s (PC 20,9s) |

Lưu ý trung thực: ở CartPole, CausalXRL **thua Decision Tree** (91,4% so với
94,1%) — tác giả thừa nhận, chỉ nói là "competitive". Nghiên cứu chuyên gia
của paper chỉ có 8 người (8/8 so với 4/8 ở bài dự đoán hành vi, đối chiếu
với SHAP). F1 đa lớp của paper là **weighted-F1** (§5.2).

Tóm tắt đầy đủ: `docs/CausalXRL-Tom-tat-noi-dung.md`. Bản dịch tiếng Việt
toàn văn (28 trang, đủ 8/8 hình): thư mục Drive dưới đây.

## Nhóm đã làm gì (trình tự công việc)

1. **Đọc paper + phân tích mã nguồn tác giả** (10/11/2026): tóm tắt,
   phân tích code thật — phát hiện repo tác giả **không có code train RL**
   (policy trong notebook chỉ là heuristic viết tay), hàm `counterfactual()`
   không chạy được, phía MicroRTS import package cũ `stratvis` và thiếu
   checkpoint PPO. Xem `docs/CausalXRL-Phan-tich-ma-nguon.md`.
2. **Mục 3 — Thực nghiệm của sinh viên, evaluation-only** (11–12/11): chạy
   CartPole (200 episode) → LunarLander (30 episode, theo notebook của tác
   giả) → PongDuel (200 episode), seed 99, CPU-only, BART thật qua R/dbarts.
   Xem `muc3/EXPERIMENT_REPORT.md`.
3. **Mục 4 — Mở rộng** (11/11): Hướng D — train DQN thật và so sánh XRL
   trained-vs-heuristic; Hướng A — ablation. Xem `muc4/results/REPORT.md`.
4. **3 bước tăng điểm** (12/11): sửa tính so sánh của Mục 3 (baseline đúng
   protocol + weighted-F1), chạy lại ablation bằng BART thật, chạy tối thiểu
   MicroRTS + sửa counterfactual.
5. **Nhóm việc 1** (12/11): khôi phục weighted-F1 cho BART LunarLander
   (81,52%) từ checkpoint đã lưu, căn split baseline CartPole, vẽ đồ thị
   Mục 3 + case study local.
6. **Nhóm việc 2** (12/11): MicroRTS 100 episode + sửa dashboard, mở rộng
   counterfactual (6 query CartPole + 4 query LunarLander), DQN LunarLander
   và CartPole seed 1.

Bảng đối chiếu đầy đủ mọi con số: `SO-SANH-TAC-GIA-VS-NHOM.md`.
Bản thông tin thuyết trình (15–20 phút): `THUYET-TRINH.md`.

## Kết quả chính đã kiểm chứng

### Mục 3 — Tái hiện evaluation-only (seed 99, BART thật, heuristic của tác giả)

Chi tiết + protocol: `muc3/EXPERIMENT_REPORT.md`. F1 so được với Table 1:
F1 nhị phân ở CartPole, weighted-F1 ở LunarLander/PongDuel.

| Môi trường | BART (nhóm) | Logistic Regression | Decision Tree |
|---|---|---|---|
| CartPole (200 ep, 6.115 trans) | 91,62% / F1 91,50% | 89,62% / 89,26% | 93,83% / 93,64% |
| LunarLander (30 ep, 7.382 trans) | 82,80% / weighted-F1 81,52% | 77,21% / 76,24% | 79,14% / 79,06% |
| PongDuel (200 ep, 74.445 trans) | 81,73% / macro-F1 19,51% (PROXY) | 81,79% / 81,38% | 96,21% / 96,21% |

- CartPole: tái hiện gần như khớp paper (+0,22 điểm % accuracy, −0,10 điểm %
  F1). Như paper, Decision Tree thắng BART (93,83% so với 91,62%).
- LunarLander: weighted-F1 của BART là **81,52%**, khôi phục từ checkpoint
  điểm số đã lưu mà không chạy lại BART (tái tổng hợp cho accuracy/macro-F1
  sai khác 0,0 so với số gốc) — so được thẳng với 86,6% của paper (−5,08
  điểm %). Macro-F1 64,10% giữ kèm để đối chiếu. Per-class F1: do_nothing
  87,51%, fire_main 83,06%, fire_left 59,68%, fire_right chỉ 26,14%.
- **PongDuel là PROXY** (hồi quy số làm tròn 0–8, 14.995 mẫu ứng viên
  local), không phải fidelity phân loại tương đương paper — không so với
  90,2%/90,7% của Table 1. Tác giả không có notebook fidelity cho PongDuel;
  baseline PongDuel là chuyển thể theo mẫu CartPole/LunarLander.
- Baseline CartPole đã tính lại ngày 12/11 theo đúng protocol notebook
  fidelity của tác giả (xem `muc3/results/baselines_recomputed.md`); trên
  đúng test set của BART: LR 90,43%/90,22%, DT 95,42%/95,32% (xem
  `muc3/results/baselines_aligned.md`). File `metrics.json` cũ giữ nguyên
  làm lịch sử.
- Thời gian học cấu trúc BART (tổng các process riêng lẻ, wall-clock thật,
  **không so trực tiếp với paper** vì số của paper là timer mô phỏng —
  notebook `eval_time_*` của tác giả dùng dữ liệu ngẫu nhiên): CartPole
  242,19s; LunarLander 1.086,40s; PongDuel 35.531,20s (~9,9 giờ).
- Đồ thị toàn cục đã học: CartPole 13 cạnh, LunarLander 44 cạnh, PongDuel
  53 cạnh (đủ 21/21 target). Ảnh đồ thị + 1 case study local kiểu Fig. 4
  của paper: thư mục Drive dưới đây (`muc3/results/`).

### MicroRTS-Lite — chạy tối thiểu thật, fidelity theo đúng protocol không đo được

Chi tiết: `muc3/microRTS-thu-nghiem.md`.

- Sửa **22 chỗ** import `stratvis` → `cxrl_microRTS` trong **8 file**; lấy
  được checkpoint PPO `agent-3.pt` công khai (SHA-256 `ad21e903…`, load
  strict OK, 5.433.185 tham số, **không đưa vào repo**).
- Env thật chạy được (JVM + PPO trả action hợp lệ). Dashboard Flask:
  `/get_board_state` → **200** (trả board thật),
  `/compute_global_causal` → **200** sau khi tự sinh thư mục `data/` từ dữ
  liệu 100 episode; `/get_local_causal` **vẫn 500** (danh sách cột hard-code
  của tác giả không khớp scaler học từ dữ liệu thật). Không khẳng định
  dashboard chạy đầy đủ.
- Đồ thị toàn cục **23 cạnh** từ **100 episode / 4.924 transition** thật
  (498,3 giây học BART). Hiệu đính: đồ thị bước 3 ghi "10 episode / 537
  transition" thực ra chỉ có 8 episode duy nhất (109 transition bị đếm trùng
  do seed ở cấp import).
- **Fidelity MicroRTS theo đúng protocol notebook thì không đo được** (tập
  valid rỗng: action chỉ có 1 cause toàn cục `heavy2`, không hàng nào có >1
  cạnh local vào nút action). Bản không-lọc (ghi rõ là lệch protocol) cho
  BART 25,15% / 30,86% accuracy (2 cách dựng), thua xa baseline đa số
  ~62–64% — con số 86,0%/85,9% của paper không tái tạo được từ repo (paper
  không cho mã hóa action).

### Mục 4 — Hướng D: train DQN thật (repo tác giả không có phần này)

Chi tiết: `muc4/results/REPORT.md`. DQN tự viết bằng PyTorch (MLP, replay
buffer, epsilon-greedy, target network, Adam), **không dùng
stable-baselines3**, train bằng gymnasium (gym 0.20 của repo không tương
thích Python 3.12), CPU-only, $0.

- **CartPole, seed 0 (run 9): Double DQN, 2.000 episode: eval greedy 100
  episode 500,00 ± 0,00 (min 500, max 500).** Heuristic cùng điều kiện:
  31,53 ± 11,16 → DQN hơn **+468,47**. Đạt được nhờ **chọn checkpoint theo
  eval định kỳ giữa train** (eval mỗi 50 episode; 10/40 lần eval đạt đúng
  500,00; eval dao động 500 ↔ ~97–146 — Q-instability).
- **CartPole, seed 1 (nhóm 2, đúng hyperparameter run 9): eval cuối 500,00
  ± 0,00 / 100 episode** — 500/500 tái hiện qua seed khác, nhưng quỹ đạo rất
  khác (8/40 eval đạt 500, checkpoint tốt nhất từ ep 250).
- **LunarLander, seed 0 (nhóm 2): checkpoint tốt nhất @ ep 850, eval cuối
  100 episode 222,03 ± 69,29 (min −60,38, max 308,35)** — vượt ngưỡng solved
  ~200, nhưng train dở dang ở ep 873/1.000 do VM restart và policy còn dao
  động; kém chắc chắn hơn CartPole. Chưa làm so sánh XRL cho LunarLander.

### Mục 4 — Hướng D: so sánh XRL (BART thật) trained-vs-heuristic, CartPole

BART thật (R 4.3.3 + `dbarts` 0.9.34 + `rpy2` 3.5.16), pipeline XRL gốc,
200 episode/policy, seed 0, checkpoint DQN run 9.

| | DT acc / F1 | LR acc / F1 | BART acc / F1 | push parents |
|---|---|---|---|---|
| Heuristic (6.330 trans) | 0,9491 / 0,9498 | 0,9111 / 0,9121 | 0,9350 / 0,9340 | 3 (push, x, θ) |
| DQN run 9 (99.800 trans) | 0,9356 / 0,9392 | 0,8195 / 0,8363 | 0,8050 / 0,8368 | 5 (push, x, v, θ, w) |

- Đồ thị: hte 14 vs 14 cạnh, **Jaccard 0,75, khác 4 cạnh** (chỉ DQN có
  `v→push`, `w→push`); ate 15 vs 16 cạnh, **Jaccard 0,6316, khác 7 cạnh**.
- Thời gian học cấu trúc: DQN chậm hơn ~19–20 lần (hte 1.756s vs 86s; ate
  2.306s vs 120s) do có ~15,8 lần transitions.
- Đọc trung thực: **DQN hoàn hảo kém heuristic ở cả 3 thước đo fidelity** —
  biên quyết định của mạng nơ-ron phi tuyến hơn luật tuyến tính của heuristic.
  Giỏi hơn không đồng nghĩa dễ giải thích hơn. Số BART dao động ±1–2 cạnh
  giữa các lần chạy. Bảng cũ (checkpoint run 5) lưu ở
  `muc4/results/xrl-trained-vs-heuristic-run5/`.

### Mục 4 — Hướng A: ablation + sensitivity bằng BART thật

Lưới 50/100/200 episode × hte/ate, seed 0, CartPole: hte 12–13 cạnh, ate 15
cạnh. Sensitivity 90/95/99% (tính lại từ cùng 500 ITE draws, không fit lại;
quyết định 95% tính lại khớp code gốc 0 mismatch): hte 13→13→11 cạnh, ate
17→15→14 cạnh. Kết luận cũ từ surrogate — "hte ổn định từ 100 episode,
Jaccard 1,0" — **không còn đứng**: BART thật cho Jaccard 0,79–0,86, trong
dải nhiễu. Surrogate (RandomForest 200 cây, lần chạy đầu khi R/dbarts chưa
cài được) cho `push: []` trong khi BART thật tìm được 3–4 nguyên nhân cho
`push`. Kết quả BART thật ở `muc4/results/ablation-bart/`; số surrogate giữ
ở `muc4/results/ablation/` làm lịch sử, luôn ghi nhãn surrogate.

### Counterfactual (DiCE) — code gốc không chạy được, nhóm sửa và chạy được

`XRL.counterfactual()` của tác giả không bao giờ chạy được (thiếu `self.exp`,
`self.state_features`, sai kiểu `self.pi`). Nhóm sửa tối thiểu trên bản copy
(patch ở `muc4/xrl-counterfactual-fix.patch`, patch đa lớp ở
`muc4/xrl-multiclass-fix.patch`) và chạy được đầu-cuối: **CartPole 6 query,
9/9 counterfactual kiểm chứng lật đúng action bằng policy gốc; LunarLander
4 query (4 action), 16/16 kiểm chứng lật** — tổng **25/25**. Chi tiết:
`muc4/counterfactual-thu-nghiem.md`. Giới hạn: chỉ policy heuristic/PD, chưa
thử policy DQN, chưa thử PongDuel; DiCE dùng phương pháp `random`.

### Chưa làm (ghi rõ, không giấu)

- **Nghiên cứu chuyên gia (expert study):** cần 8 chuyên gia RTS (>7 năm,
  >1.000 giờ), IRB, chạy crossover với SHAP — ngoài khả năng của sinh viên
  trong phạm vi tiểu luận. Nhóm không làm và không thay bằng khảo sát
  bạn cùng lớp rồi ghi thành expert study.
- **Fidelity MicroRTS theo đúng protocol:** tập valid rỗng (xem trên);
  dashboard còn 1 route 500 (`/get_local_causal`).
- **GES** (baseline thứ ba trong paper) chưa chạy; so sánh XRL cho
  LunarLander/PongDuel chưa làm; DQN PongDuel chưa làm.

## Cấu trúc repo

```
XRL/
├── README.md                  # file này
├── LICENSE                    # MIT (cùng loại với repo tác giả)
├── THUYET-TRINH.md            # bản thông tin dùng trong thuyết trình (15–20 phút)
├── SO-SANH-TAC-GIA-VS-NHOM.md # bảng so sánh tất cả con số tác giả vs nhóm + link biểu đồ
├── docs/
│   ├── CausalXRL-Tom-tat-noi-dung.md   # tóm tắt paper + Table 1
│   ├── CausalXRL-Phan-tich-ma-nguon.md # phân tích code repo thật
│   ├── CausalXRL-KE-HOACH-CHAM-DIEM.md # kế hoạch chấm điểm (§0–§6)
│   └── CausalXRL-MUC3-CODEX-HANDOFF.md # quy tắc giao việc làm Mục 3
├── muc3/                      # Mục 3 — Thực nghiệm của sinh viên
│   ├── EXPERIMENT_REPORT.md   # báo cáo Mục 3 (đọc file này trước)
│   ├── microRTS-thu-nghiem.md # báo cáo MicroRTS (100 ep, dashboard, fidelity)
│   ├── scripts/               # collect/metrics/BART fidelity + recompute + MicroRTS
│   ├── notes/                 # lỗi đã gặp + cách fix, blocker
│   └── results/               # replay.npz + metrics.json + bart_fidelity.json từng môi trường; microrts/
└── muc4/                      # Mục 4 — Mở rộng
    ├── counterfactual-thu-nghiem.md # báo cáo counterfactual (6 query CartPole, 4 query LunarLander)
    ├── scripts/               # train_dqn_cartpole.py, train_dqn_lunarlander.py, run_xrl_comparison.py, ...
    ├── notes/                 # lỗi đã gặp + cách fix, kiểm tra BART
    └── results/
        ├── REPORT.md          # báo cáo tổng hợp Mục 4 (đọc file này trước)
        ├── cartpole-dqn/      # checkpoint run 9 (eval 500/500) + seed1/ + log + CSV
        ├── lunarlander-dqn/   # DQN LunarLander (best @ ep 850, eval 222,03; .pt trên Drive)
        ├── ablation/          # bảng + ảnh đồ thị (surrogate — chỉ còn là lịch sử)
        ├── ablation-bart/     # ablation + sensitivity bằng BART thật (ảnh đồ thị trên Drive)
        ├── so-sanh/           # script vẽ 3 biểu đồ so sánh (ảnh PNG trên Drive)
        ├── xrl-trained-vs-heuristic/        # bảng so sánh mới (checkpoint run 9)
        └── xrl-trained-vs-heuristic-run5/   # bảng cũ lưu trữ (checkpoint run 5)
```

File nhị phân (PDF, ảnh PNG, checkpoint `.pt`, một số `.npz`) không nằm trong
repo này mà ở thư mục Drive dưới đây, vì connector GitHub chỉ push được file
text UTF-8.

## Cách chạy lại

### 1. Mục 3 — Tái hiện evaluation-only (cần R + dbarts)

```bash
# Cài R + dbarts trước (bước khó nhất — xem muc4/notes/bart-check.md):
sudo apt install r-base-core r-base-dev default-jdk
# trong Python: pip install rpy2; trong R: install.packages("dbarts")
# Mỗi môi trường: collect → metrics (BART từng target) → fidelity, xem
# muc3/scripts/cartpole_*.py, lunarlander_*.py, pongduel_*.py
# Tính lại baseline theo protocol notebook (không cần BART):
python muc3/scripts/recompute_baselines_weighted_f1.py
```

**Lưu ý:** R/dbarts sống ngoài thư mục làm việc nên **VM restart sẽ xóa R** —
khôi phục bằng `dpkg -i /var/cache/apt/archives/*.deb` rồi cài lại dbarts
0.9.34 từ CRAN (xem `muc4/notes/bart-errors.md`). Mọi kết quả đã lưu trong
repo thì không mất.

### 2. Train DQN CartPole / LunarLander (CPU-only, $0)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium numpy
python muc4/scripts/train_dqn_cartpole.py --episodes 2000 --eval-every 50 --eval-episodes 50
python muc4/scripts/train_dqn_lunarlander.py --episodes 1000 --eval-every 50 --eval-episodes 50
```

Checkpoint tốt nhất theo eval định kỳ được lưu vào
`muc4/results/cartpole-dqn/dqn_checkpoint.pt`. Đánh giá lại:

```bash
python muc4/scripts/evaluate_cartpole.py --checkpoint muc4/results/cartpole-dqn/dqn_checkpoint.pt
```

### 3. Chạy XRL trên agent đã train (cần BART thật)

```bash
python muc4/scripts/run_xrl_comparison.py --checkpoint muc4/results/cartpole-dqn/dqn_checkpoint.pt --estimator bart --episodes 200 --seed 0
```

### 4. Ablation bằng BART thật

```bash
python muc4/scripts/run_ablation_bart.py   # BART thật (số chốt)
python muc4/scripts/run_ablation.py        # surrogate RandomForest (chỉ để đối chiếu lịch sử)
```

### 5. MicroRTS + dashboard

Cần Java + Xvfb + checkpoint PPO (lấy bằng `muc3/scripts/fetch_agent3_range.py`,
không đưa checkpoint vào repo). Sửa import: `muc3/stratvis-fix.patch` (bước 3)
và `muc3/dashboard-fix.patch` (nhóm 2). Quy trình đầy đủ + log: xem
`muc3/microRTS-thu-nghiem.md` §10–§11.

### 6. Counterfactual

Áp patch lên bản copy repo tác giả (`muc4/xrl-counterfactual-fix.patch` +
`muc4/xrl-multiclass-fix.patch`), rồi chạy
`muc4/scripts/run_counterfactual_cartpole_multi.py` và
`muc4/scripts/run_counterfactual_lunarlander.py`. Chi tiết:
`muc4/counterfactual-thu-nghiem.md`.

## File nhị phân — thư mục Drive

Toàn bộ file nhị phân của repo nằm ở thư mục Drive
[XRL - file nhi phan](https://drive.google.com/drive/folders/1rK_Br_Kj3Dr-OZb8c5V2AgJeUW8bx_ea)
(cùng cấu trúc thư mục với repo): PDF paper gốc + bản dịch tiếng Việt
(`paper/`, `ban-dich/`); ảnh PNG đồ thị nhân quả và curve
(`muc3/results/`, `muc4/results/ablation/graphs/`,
`muc4/results/ablation-bart/graphs/`, `muc4/results/so-sanh/`,
`muc4/results/cartpole-dqn/`); checkpoint DQN `.pt`
(`muc4/results/cartpole-dqn/`, `muc4/results/lunarlander-dqn/`); file dữ
liệu `.npz` và SCM/scaler MicroRTS (`muc3/results/microrts/`).

## Phạm vi trung thực — đọc trước khi trích số

- Số của **tác giả** (Table 1, 8 chuyên gia, MicroRTS-Lite) chỉ trích từ paper
  arXiv:2610.42028v1, ghi rõ nguồn. Nhóm **không** làm human study và
  **không** tái tạo được số fidelity MicroRTS 86,0%/85,9% của Table 1.
- Số của **nhóm** là số chạy thật, có log/CSV/checkpoint kèm theo. Số BART
  dao động ±1–2 cạnh giữa các lần chạy (BART không có seed trong code gốc).
- Ablation **lần đầu** dùng **surrogate RandomForest, không phải BART** —
  không được ghi thành số CausalXRL/BART. Số chốt là số BART thật ở
  `muc4/results/ablation-bart/`.
- Repo tác giả **không train agent RL**; policy trong notebook của tác giả là
  heuristic viết tay. Agent DQN trong repo này là do nhóm train thêm (Hướng D,
  Mục 4 — mở rộng, không phải tái hiện).
- Baseline CartPole cũ trong `muc3/results/cartpole/metrics.json` đo task khác
  với notebook của tác giả (state_t → action_t) — đã sửa ngày 12/11/2026, số
  chốt theo protocol notebook ở `muc3/EXPERIMENT_REPORT.md`; file
  `metrics.json` cũ giữ nguyên làm lịch sử.

## File đã bỏ qua khi đưa lên repo (và lý do)

- `.venv-ablation/` — môi trường Python/R, quá lớn, tái tạo được bằng pip +
  apt (xem "Cách chạy lại").
- `repo-causal-xrl/` — bản copy repo tác giả, chỉ đọc; lấy từ
  https://github.com/yanmluk/causal-xrl (commit `ae9096e`).
- `cartpole-dqn/run2-partial/`, `run6/` — chạy dở dang, không checkpoint,
  không eval, không dùng.
- `xrl-trained-vs-heuristic-dqn-only*/` — các lần chạy DQN-only trung gian;
  kết quả cuối đã merge vào `muc4/results/xrl-trained-vs-heuristic/`.
- Log stdout dài (`comparison_stdout*.log`) — chỉ giữ log tóm tắt
  (`comparison_run.log`, `train_run.log`, `ablation_run.log`) + CSV + JSON.
- Checkpoint PPO MicroRTS `agent-3.pt` — file nhị phân, không rõ quyền tái
  phân phối; lấy lại bằng script + SHA ghi trong `muc3/microRTS-thu-nghiem.md`.
- KG-R1 và form mẫu Nhóm 8 — không thuộc bài CausalXRL, không đưa vào repo.

## Lịch sử commit chính

- `060968c` — Initial commit (11/11/2026)
- `3986d74` — Tóm tắt, phân tích mã nguồn, kế hoạch, Mục 4 DQN + so sánh XRL + ablation (11/11)
- `9a0f344` — Mục 3: tái hiện evaluation-only 3 môi trường (12/11)
- `55109bf` / `cbcd492` — Sửa số transition PongDuel, ghi blocker CartPole đã giải quyết (12/11)
- `5406c94` — Mục 3: baseline đúng protocol notebook + weighted-F1 (12/11)
- `c90a392` — Mục 4: ablation + sensitivity bằng BART thật (12/11)
- `c2fa6ca` — MicroRTS chạy được + sửa/chạy counterfactual (12/11)
- `585ffff` — Bảng so sánh tác giả vs nhóm + 3 biểu đồ (12/11)
- `2237c0f` — Nhóm 1: weighted-F1 BART LunarLander, căn split CartPole, đồ thị Mục 3 (12/11)
- `d798ed8` — Nhóm 2: MicroRTS 100 episode + dashboard, counterfactual mở rộng, DQN LunarLander + CartPole seed 1 (12/11)

## Credits

- Bài báo và code gốc: Yanming Zhang, Eric Papenhausen, Klaus Mueller
  (Stony Brook University) — arXiv:2610.42028v1, repo
  https://github.com/yanmluk/causal-xrl (MIT).
- Thực hiện tái hiện và mở rộng: nhóm 5, môn CSCHTTT — Lê Quốc Triệu
  (MSSV 24022750) và Nguyễn Sỹ Việt.
- Giấy phép của repo này: MIT (xem `LICENSE`), cùng loại với repo tác giả.
