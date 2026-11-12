# XRL — CausalXRL Course Project

Tiểu luận môn CSCHTTT (INT3201_1), trường Đại học Công nghệ — ĐHQGHN.
Bài báo: **CausalXRL: Explainable Reinforcement Learning through Causal Graph
Reasoning** — Yanming Zhang, Eric Papenhausen, Klaus Mueller (Stony Brook
University), arXiv:2610.42028v1, ICML 2026.
Repo gốc của tác giả: https://github.com/yanmluk/causal-xrl (commit
`ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162`, MIT).

> Repo này là **private** (mặc định khi tạo cho bài tiểu luận). Có thể chuyển
> public sau khi nộp bài nếu nhóm muốn.

## Bài báo nói gì (tóm tắt)

Học tăng cường (RL) train được agent rất giỏi nhưng là "hộp đen": không biết
vì sao nó chọn hành động đó. CausalXRL giải thích hành vi của agent RL bằng
**đồ thị nhân quả (causal graph)** giữa các biểu diễn chiến lược mà con người
hiểu được, thay vì gắn cạnh nhân quả vào action đơn lẻ như AIM trước đây.

Phương pháp 3 bước của tác giả:

1. **Trích biểu diễn hiểu được** — từ trạng thái thô lấy ra các đặc trưng
   chiến lược (ví dụ trong MicroRTS: số unit từng loại, tài nguyên).
2. **Học đồ thị nhân quả toàn cục** — chạy agent đã train 200 episode, lưu
   transition vào replay buffer, dùng BART (R package `dbarts`) ước lượng
   ATE / HTE (CATE) cho từng biến; biến nào có khoảng tin cậy 95% không chứa
   0 thì là nguyên nhân.
3. **Mô hình local + phản thực** — với một trạng thái cụ thể, xác định nguyên
   nhân local, dự đoán hành động tiếp theo có trọng số theo |ITE|, và dùng
   DiCE sinh counterfactual.

Kết quả của tác giả (Table 1, arXiv:2610.42028v1 — **số của tác giả, chỉ để
đối chiếu, không phải số của nhóm**):

| Môi trường | CausalXRL Acc / F1 | DT Acc | Thời gian học cấu trúc |
|---|---|---|---|
| CartPole | 91,4% / 91,6% | 94,1% | 0,1s (PC 1,7s) |
| LunarLander | 88,7% / 86,6% | 85,1% | 0,3s (PC 12,6s) |
| PongDuel | 90,2% / 90,7% | 88,3% | 0,5s (PC 23,2s) |
| MicroRTS-Lite | 86,0% / 85,9% | 80,3% | 0,5s (PC 20,9s) |

Lưu ý trung thực: ở CartPole, CausalXRL **thua Decision Tree** (91,4% so với
94,1%) — tác giả thừa nhận, chỉ nói là "competitive". Nghiên cứu chuyên gia
chỉ có 8 người (8/8 so với 4/8 ở bài dự đoán hành vi với SHAP).

## Nhóm đã làm gì

- **Đọc + phân tích paper và repo** — tóm tắt, phân tích mã nguồn thật
  (repo không có code train RL; policy trong notebook chỉ là heuristic viết
  tay), kế hoạch chấm điểm theo form Nhóm 8 / Nhóm 5.
- **Bản dịch tiếng Việt toàn văn** — `ban-dich/causalxrl-ban-dich-bao-cao.pdf`
  (28 trang, đủ 8/8 hình, code và công thức giữ nguyên).
- **Mục 3 — Thực nghiệm của sinh viên (evaluation-only): ĐÃ XONG** — xem
  `muc3/EXPERIMENT_REPORT.md` (kế hoạch giao việc ở
  `docs/CausalXRL-MUC3-CODEX-HANDOFF.md`). Chạy CartPole (200 episode) →
  LunarLander (30 episode) → PongDuel (200 episode), seed 99, CPU-only,
  heuristic của repo tác giả, không train RL. Số chốt ở § "Kết quả chính đã
  kiểm chứng (Mục 3)" dưới đây. Baseline CartPole đã sửa ngày 12/11/2026 theo
  đúng protocol notebook fidelity của tác giả (xem
  `muc3/results/baselines_recomputed.md`); BART không chạy lại. MicroRTS
  đã chạy tối thiểu thật (xem mục MicroRTS-Lite ở § "Kết quả chính đã
  kiểm chứng (Mục 3)" dưới đây).
- **Mục 4 — Mở rộng (xem `muc4/results/REPORT.md`)** — Hướng D (train agent
  RL thật) là chính, Hướng A (ablation) là dự phòng.
- **So sánh tất cả con số tác giả vs nhóm** — xem
  `SO-SANH-TAC-GIA-VS-NHOM.md` (bảng đầy đủ theo từng môi trường + hạng
  mục, mỗi số kèm nguồn và caveat; 3 biểu đồ vẽ bằng
  `muc4/results/so-sanh/ve_bieu_do_so_sanh.py`, ảnh PNG ở thư mục Drive
  dưới đây vì GitHub không nhận file nhị phân qua connector).
- **File nhị phân (ảnh, checkpoint, PDF)** — toàn bộ file nhị phân của
  repo nằm ở thư mục Drive
  [XRL - file nhi phan](https://drive.google.com/drive/folders/1rK_Br_Kj3Dr-OZb8c5V2AgJeUW8bx_ea)
  (cùng cấu trúc thư mục với repo), vì connector GitHub chỉ push được
  file text UTF-8. Trên Drive có: PDF paper gốc + bản dịch tiếng Việt
  (`paper/`, `ban-dich/`); ảnh PNG đồ thị nhân quả và curve
  (`muc4/results/ablation/graphs/`, `muc4/results/ablation-bart/graphs/`
  có 6 ảnh đồ thị BART, `muc4/results/so-sanh/` có 3 biểu đồ so sánh,
  `muc4/results/cartpole-dqn/`); checkpoint DQN `.pt`
  (`muc4/results/cartpole-dqn/`); và file dữ liệu `.npz`
  (`muc3/results/`).

## Kết quả chính đã kiểm chứng (Mục 3)

Evaluation-only, seed 99, heuristic viết tay của repo tác giả (không train
RL). Chi tiết + protocol: `muc3/EXPERIMENT_REPORT.md`. F1 so được với Table 1
của paper: F1 nhị phân ở CartPole, weighted-F1 ở LunarLander/PongDuel
(paper §5.2: weighted khi multiclass).

| Môi trường | BART | Logistic Regression | Decision Tree |
|---|---|---|---|
| CartPole (200 ep) | 91,62% / F1 nhị phân 91,50% | 89,62% / 89,26% | 93,83% / 93,64% |
| LunarLander (30 ep) | 82,80% / macro-F1 64,10% | 77,21% / weighted-F1 76,24% | 79,14% / weighted-F1 79,06% |
| PongDuel (200 ep) | 81,73% / macro-F1 19,51% (PROXY) | 81,79% / weighted-F1 81,38% | 96,21% / weighted-F1 96,21% |

- Như paper, ở CartPole Decision Tree thắng BART (93,83% so với 91,62%).
- **LunarLander chỉ 30 episode** (theo notebook đánh giá của tác giả, vì chạy
  đắt). BART LunarLander chỉ có macro-F1 — **chưa so được** với F1 weighted
  86,6% của Table 1 (dự đoán từng mẫu của BART không được lưu, muốn có
  weighted-F1 phải chạy lại BART).
- **BART PongDuel là PROXY** (hồi quy số làm tròn 0–8, 14.995 mẫu ứng viên
  local), không phải fidelity phân loại tương đương paper — không so với
  90,2%/90,7% của Table 1. Baseline PongDuel là chuyển thể theo mẫu
  CartPole/LunarLander vì tác giả không có notebook fidelity cho PongDuel.
- Thời gian học cấu trúc BART (tổng các process riêng lẻ, không so trực tiếp
  với paper): CartPole 242,19s; LunarLander 1.086,40s; PongDuel 35.531,20s.

### MicroRTS-Lite — đã chạy tối thiểu thật (chưa có fidelity theo Table 1)

- Sửa **22 chỗ** import `stratvis` → `cxrl_microRTS` trong **8 file**,
  lấy được checkpoint PPO `agent-3.pt` công khai (SHA-256 `ad21e903…`,
  load strict OK, 5.433.185 tham số, **không đưa vào repo**).
- Env thật chạy được (JVM + PPO trả action hợp lệ); dashboard Flask `/`
  → 200, nhưng 2 route `/get_board_state` và `/compute_global_causal`
  còn 500 (bug của tác giả + thiếu thư mục `data/`).
- Đồ thị toàn cục **20 cạnh** từ **10 episode / 537 transition** thật.
  **Chưa có fidelity MicroRTS theo Table 1.** Chi tiết xem
  `muc3/microRTS-thu-nghiem.md`.

## Kết quả chính đã kiểm chứng (Mục 4)

### Hướng D — Train DQN thật trên CartPole: ĐẠT 500,00/500

DQN tự viết bằng PyTorch (MLP 4→128→128→2, replay buffer, epsilon-greedy,
target network, Adam), **không dùng stable-baselines3**. Train bằng gymnasium
(gym 0.20 của repo không tương thích Python 3.12 — ghi rõ).

- **Best: run 9 — Double DQN, 2.000 episode, lr 1e-3, MSE, eps-decay 1300:
  eval greedy 100 episode 500,00 ± 0,00 (min 500, max 500).** Heuristic của
  repo cùng điều kiện: 31,53 ± 11,16 → DQN hơn **+468,47**.
- Cách đạt được: **chọn checkpoint theo eval định kỳ giữa train** (eval mỗi
  50 episode, lưu checkpoint tốt nhất). Run 9 có 10/40 lần eval đạt đúng
  500,00; eval dao động 500 ↔ ~97–146 giữa các lần đo cách nhau 50 ep
  (Q-instability) — lấy checkpoint cuối train là sai.
- Các run khác (cùng eval 100 ep): run 1: 246,16; run 5 (Double DQN):
  272,68; run 7: dở dang do VM restart, best 343,26; run 8 (target-update
  10→100): 234,51 — thụt lùi, không dùng.

### Hướng D — So sánh XRL (BART thật) trained-vs-heuristic

BART chạy thật: R 4.3.3 + `dbarts` 0.9.34 + `rpy2` 3.5.16. Pipeline XRL gốc
của repo, 200 episode/policy, seed 0, checkpoint DQN run 9.

| | DT acc / F1 | LR acc / F1 | BART acc / F1 | push parents |
|---|---|---|---|---|
| Heuristic (6.330 trans) | 0,9491 / 0,9498 | 0,9111 / 0,9121 | 0,9350 / 0,9340 | 3 (push, x, θ) |
| DQN run 9 (99.800 trans) | 0,9356 / 0,9392 | 0,8195 / 0,8363 | 0,8050 / 0,8368 | 5 (push, x, v, θ, w) |

- Đồ thị: hte 14 vs 14 cạnh, **Jaccard 0,75, khác 4 cạnh** (chỉ DQN có
  `v→push`, `w→push`); ate 15 vs 16 cạnh, **Jaccard 0,6316, khác 7 cạnh**.
- Thời gian học cấu trúc: DQN chậm hơn ~19–20 lần (hte 1.756s vs 86s; ate
  2.306s vs 120s) do có ~15,8 lần transitions.
- Đọc trung thực: **DQN hoàn hảo kém heuristic ở cả 3 thước đo fidelity** —
  biên quyết định của mạng nơ-ron phi tuyến hơn luật tuyến tính của heuristic,
  nên mô hình nhân quả bắt chước kém hơn. Giỏi hơn không đồng nghĩa dễ giải
  thích hơn. Số BART dao động ±1–2 cạnh giữa các lần chạy (xem REPORT.md).
- Bảng cũ (checkpoint run 5, eval 272,68) lưu nguyên vẹn ở
  `muc4/results/xrl-trained-vs-heuristic-run5/`.

### Hướng A — Ablation: đã chạy lại bằng BART thật (số surrogate chỉ còn là lịch sử)

**Lần đầu — surrogate, không phải BART:** dùng **RandomForest 200 cây thay
BART** (R/dbarts chưa cài được lúc làm ablation). Mọi số đều ghi nhãn
"SURROGATE, NOT BART — chỉ kiểm tra độ ổn định, phải chạy lại bằng BART
thật". Lưới 6 run (50/100/200 ep × hte/ate), seed 0: DT/LR dự đoán action
0,978–0,989. Surrogate nói hte ổn định từ 100 episode (đồ thị 100ep =
200ep, Jaccard 1,0) và ate vẫn thay đổi tới 200 episode (Jaccard 0,647 ở
100ep). CI 90% → 8 cạnh, 95% → 7 cạnh (hte, 200ep).

**Chạy lại bằng BART thật** (R/dbarts 0.9.34): cùng lưới 50/100/200 episode
× hte/ate; hte 12–13 cạnh, ate 15 cạnh; sensitivity 90/95/99% đủ cả
hte/ate. Kết luận cũ từ surrogate — "hte ổn định từ 100 episode, Jaccard
1,0" — **không còn đứng**: BART thật cho Jaccard 0,79–0,86, trong dải nhiễu
giữa các lần chạy. Surrogate cho `push: []` trong khi BART thật tìm được
3–4 nguyên nhân cho `push`. Kết quả ở `muc4/results/ablation-bart/` (thư
mục mới, không ghi đè `ablation/` cũ). Chi tiết xem
`muc4/results/REPORT.md`.

## Cấu trúc thư mục

```
XRL/
├── README.md                  # file này
├── THUYET-TRINH.md            # bản thông tin dùng trong thuyết trình (15–20 phút)
├── SO-SANH-TAC-GIA-VS-NHOM.md # bảng so sánh tất cả con số tác giả vs nhóm + link biểu đồ
├── paper/
│   └── CausalXRL-2610.42028.pdf        # PDF gốc của tác giả
├── ban-dich/
│   └── causalxrl-ban-dich-bao-cao.pdf  # bản dịch tiếng Việt, 28 trang, đủ 8/8 hình
├── docs/
│   ├── CausalXRL-Tom-tat-noi-dung.md   # tóm tắt paper + Table 1
│   ├── CausalXRL-Phan-tich-ma-nguon.md # phân tích code repo thật
│   ├── CausalXRL-KE-HOACH-CHAM-DIEM.md # kế hoạch chấm điểm (§0–§6)
│   └── CausalXRL-MUC3-CODEX-HANDOFF.md # quy tắc giao Codex làm Mục 3
├── muc3/                      # Mục 3 — Thực nghiệm của sinh viên
│   ├── EXPERIMENT_REPORT.md   # báo cáo Mục 3 (đọc file này trước)
│   ├── scripts/               # collect/metrics/BART fidelity + recompute_baselines_weighted_f1.py
│   ├── notes/                 # lỗi đã gặp + cách fix, blocker
│   └── results/               # replay.npz + metrics.json + bart_fidelity.json từng môi trường
└── muc4/                      # Mục 4 — Mở rộng
    ├── scripts/               # train_dqn_cartpole.py, run_xrl_comparison.py, ...
    ├── notes/                 # lỗi đã gặp + cách fix, kiểm tra BART
    └── results/
        ├── REPORT.md          # báo cáo tổng hợp Mục 4 (đọc file này trước)
        ├── cartpole-dqn/      # checkpoint run 9 (eval 500/500) + log + CSV + ảnh
        ├── ablation/          # bảng + ảnh đồ thị (surrogate — đã thay thế)
        ├── ablation-bart/     # ablation + sensitivity bằng BART thật (ảnh đồ thị trên Drive)
        ├── so-sanh/           # script vẽ 3 biểu đồ so sánh (ảnh PNG trên Drive)
        ├── xrl-trained-vs-heuristic/        # bảng so sánh mới (checkpoint run 9)
        └── xrl-trained-vs-heuristic-run5/   # bảng cũ lưu trữ (checkpoint run 5)
```

## Cách chạy lại

### 1. Train DQN CartPole (CPU-only, $0)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium numpy
python muc4/scripts/train_dqn_cartpole.py --episodes 2000 --eval-every 50 --eval-episodes 50
```

Checkpoint tốt nhất theo eval định kỳ được lưu vào
`muc4/results/cartpole-dqn/dqn_checkpoint.pt`. Đánh giá lại:

```bash
python muc4/scripts/evaluate_cartpole.py --checkpoint muc4/results/cartpole-dqn/dqn_checkpoint.pt
```

### 2. Chạy XRL trên agent đã train (cần BART thật)

Cài R + dbarts trước (bước khó nhất — xem `muc4/notes/bart-check.md`):

```bash
sudo apt install r-base-core r-base-dev default-jdk
# trong Python: pip install rpy2, rồi trong R: install.packages("dbarts")
python muc4/scripts/run_xrl_comparison.py --checkpoint muc4/results/cartpole-dqn/dqn_checkpoint.pt --estimator bart --episodes 200 --seed 0
```

**Lưu ý:** R/dbarts sống ngoài thư mục làm việc nên **VM restart sẽ xóa R** —
khôi phục bằng `dpkg -i /var/cache/apt/archives/*.deb` rồi cài lại dbarts
0.9.34 từ CRAN (xem `muc4/notes/bart-errors.md`). Mọi kết quả đã lưu trong
repo thì không mất.

### 3. Ablation (surrogate — không cần R)

```bash
python muc4/scripts/run_ablation.py
```

## Phạm vi trung thực — đọc trước khi trích số

- Số của **tác giả** (Table 1, 8 chuyên gia, MicroRTS-Lite) chỉ trích từ paper
  arXiv:2610.42028v1, ghi rõ nguồn. Nhóm **không** làm human study và
  **không** có số fidelity MicroRTS theo Table 1; MicroRTS chỉ chạy tối
  thiểu (env + dashboard một phần + đồ thị 10 episode, xem
  `muc3/microRTS-thu-nghiem.md`).
- Số của **nhóm** là số chạy thật, có log/CSV/checkpoint kèm theo. Số BART
  dao động ±1–2 cạnh giữa các lần chạy.
- Ablation **lần đầu** dùng **surrogate RandomForest, không phải BART** —
  không được ghi thành số CausalXRL/BART. Ablation đã chạy lại bằng BART
  thật ở `muc4/results/ablation-bart/` (xem REPORT.md); số chốt là số
  BART thật, số surrogate chỉ giữ làm lịch sử.
- Repo tác giả **không train agent RL**; policy trong notebook của tác giả là
  heuristic viết tay. Agent DQN trong repo này là do nhóm train thêm (Hướng D,
  Mục 4 — mở rộng, không phải tái hiện).
- Baseline CartPole cũ trong `muc3/results/cartpole/metrics.json` đo task khác
  với notebook của tác giả (state_t → action_t) — đã sửa ngày 12/11/2026, số
  chốt theo protocol notebook ở `muc3/EXPERIMENT_REPORT.md`; file
  `metrics.json` cũ giữ nguyên làm lịch sử.

## File đã bỏ qua khi đưa lên repo (và lý do)

- `causalxrl-muc4-mo-rong/.venv-ablation/` — môi trường Python/R, quá lớn,
  tái tạo được bằng pip + apt (xem "Cách chạy lại").
- `causalxrl-muc4-mo-rong/repo-causal-xrl/` — bản copy repo tác giả, chỉ đọc;
  lấy từ https://github.com/yanmluk/causal-xrl (commit `ae9096e`).
- `cartpole-dqn/run2-partial/`, `run6/` — chạy dở dang, không checkpoint,
  không eval, không dùng.
- `xrl-trained-vs-heuristic-dqn-only*/` — các lần chạy DQN-only trung gian;
  kết quả cuối đã merge vào `xrl-trained-vs-heuristic/`.
- Log stdout dài (`comparison_stdout*.log`) — chỉ giữ log tóm tắt
  (`comparison_run.log`, `train_run.log`, `ablation_run.log`) + CSV + JSON.
- KG-R1 và form mẫu Nhóm 8 — không thuộc bài CausalXRL, không đưa vào repo.
