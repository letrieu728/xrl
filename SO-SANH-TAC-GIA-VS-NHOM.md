# So sánh tất cả con số: tác giả CausalXRL vs nhóm

Ngày lập: 12/11/2026. Cập nhật nhóm 1, 2 và 3: 12/11/2026; rà soát
đồng bộ toàn repo và sửa số PC/FCI MicroRTS: 13/11/2026 (xem các mục
đánh dấu). Mọi số trong file này đều đọc từ nguồn gốc ghi ở
đầu mỗi mục — không bịa, không lấy số tác giả ghi thành số nhóm.
Số thập phân viết theo kiểu Việt Nam (dấu phẩy).

## Nguồn số liệu

| Bên | Nguồn |
|---|---|
| Tác giả | Paper arXiv:2610.42028v1 (17 trang): Table 1 tr. 8, §5.1–§5.2, §6 (expert study), Appendix D + Hình 8. PDF gốc: `paper/CausalXRL-2610.42028.pdf`. Repo: github.com/yanmluk/causal-xrl, commit `ae9096e`. |
| Nhóm — Mục 3 | `muc3/EXPERIMENT_REPORT.md` (đã sửa 12/11/2026, commit `5406c94`), `muc3/results/baselines_recomputed.json/.md`, `muc3/results/*/metrics.json`, `bart_fidelity.json`, `collection_metadata.json`, `replay.npz`. Seed 99, Windows, BART thật (R/dbarts qua rpy2). |
| Nhóm — Mục 4 | `muc4/results/REPORT.md`, `muc4/results/xrl-trained-vs-heuristic/comparison_table.md`, `muc4/results/xrl-trained-vs-heuristic-lunarlander/comparison_table.md`, `muc4/results/pongduel-dqn/train_summary.md`, `muc4/results/ablation-bart/` (BART thật), `muc3/microRTS-thu-nghiem.md`, `muc4/counterfactual-thu-nghiem.md`. |

## Cách đọc — các nhãn caveat dùng trong toàn file

- **PROXY**: số đo bằng phương pháp thay thế, không tương đương paper.
- **SURROGATE**: số đo bằng RandomForest thay BART — đã bị thay bằng số
  BART thật, chỉ giữ làm lịch sử, không trích làm kết quả.
- **KHÔNG SO ĐƯỢC**: hai số đo khác định nghĩa/cách đo, đặt cạnh nhau chỉ
  để thấy sự khác nhau, không được trừ nhau hay kết luận hơn/kém.
- **Không làm**: hạng mục nhóm chưa hề chạy — ghi rõ, không để trống.

---

## 1. Fidelity dự đoán hành vi — Table 1 của tác giả vs Mục 3 của nhóm

### 1.1. Số của tác giả (Table 1, tr. 8)

F1 đa lớp của tác giả là **weighted-F1** (§5.2: "average='weighted' if
multiclass"). #Rep = số biểu diễn (representations).

| Môi trường | #Rep | LM Acc / F1 (%) | DT Acc / F1 (%) | CausalXRL Acc / F1 (%) |
|---|---:|---:|---:|---:|
| CartPole | 4 | 88,5 / 90,0 | 94,1 / 94,6 | 91,4 / 91,6 |
| LunarLander | 8 | 81,3 / 80,2 | 85,1 / 84,7 | 88,7 / 86,6 |
| PongDuel | 12 | 81,5 / 81,1 | 88,3 / 88,4 | 90,2 / 90,7 |
| MicroRTS-Lite | 14 | 75,3 / 74,5 | 80,3 / 82,2 | 86,0 / 85,9 |

### 1.2. Số của nhóm — Mục 3 (đã sửa theo protocol notebook, BART không chạy lại)

| Môi trường | BART (CausalXRL) Acc / F1 | Logistic Regression Acc / F1 | Decision Tree Acc / F1 |
|---|---|---|---|
| CartPole (200 ep) | 91,62% / F1 nhị phân 91,50% | 89,62% / F1 nhị phân 89,26% (weighted 89,61%, macro 89,60%) | 93,83% / F1 nhị phân 93,64% (weighted 93,83%, macro 93,82%) |
| LunarLander (30 ep) | 82,80% / weighted-F1 **81,52%** (macro-F1 64,10%) — khôi phục đúng từ checkpoint điểm từng action (nhóm 1) | 77,21% / weighted-F1 76,24% (macro 59,91%) | 79,14% / weighted-F1 79,06% (macro 63,25%) |
| PongDuel (200 ep) | **PROXY** 81,73% / macro-F1 19,51% | 81,79% / weighted-F1 81,38% (macro 62,65%) | 96,21% / weighted-F1 96,21% (macro 90,21%) |
| MicroRTS-Lite | **Không đo được** (đã chẩn đoán tới cấp cơ chế ở nhóm 3, xem §6; không đối chiếu Table 1) | **Không đo được** (cùng lý do) | **Không đo được** (cùng lý do) |

### 1.3. Chênh lệch so với Table 1 (chỉ tính khi so được)

| Môi trường | BART: chênh Acc | BART: chênh F1 | LR: chênh Acc / F1 | DT: chênh Acc / F1 |
|---|---|---|---|---|
| CartPole | +0,22 điểm % | −0,10 điểm % (F1 nhị phân) | +1,12 / −0,74 điểm % | −0,27 / −0,96 điểm % |
| LunarLander | −5,90 điểm % | **−5,08 điểm %** (weighted vs weighted, so được — nhóm 1) | −4,09 / −3,96 điểm % (weighted) | −5,96 / −5,64 điểm % (weighted) |
| PongDuel | **KHÔNG SO ĐƯỢC** (PROXY) | **KHÔNG SO ĐƯỢC** (PROXY) | +0,29 / +0,28 điểm % (weighted, chuyển thể) | +7,91 / +7,81 điểm % (weighted, chuyển thể) |
| MicroRTS-Lite | Không đo được | Không đo được | Không đo được | Không đo được |

Con số chênh F1 LunarLander −22,50 điểm % từng báo trước đây đã **rút
lại**: nó so macro-F1 của nhóm với weighted-F1 của paper. **Cập nhật
nhóm 1:** weighted-F1 của BART LunarLander **đã khôi phục được đúng
81,52%** từ checkpoint điểm từng action đã lưu (theo từng class:
do_nothing 87,51%, fire_main 83,06%, fire_left 59,68%, fire_right
26,14%) — câu "không tính được" trước đây đã rút lại. Chênh F1 so với
paper: −5,08 điểm %.

### 1.4. Caveat của Mục 3 (đọc kèm bảng trên)

1. **Policy là heuristic viết tay, không phải agent đã train.** Paper
   §5.1 viết "execute trained RL agents for 200 episodes", nhưng repo
   tác giả không có code train RL nào — notebook chỉ có heuristic:
   CartPole luật tuyến tính, LunarLander PD controller, PongDuel
   strong/weak heuristic. Nhóm dùng đúng các policy này.
2. **LunarLander chỉ 30 episode**, theo notebook đánh giá của tác giả
   (vì chạy đắt), không phải 200 episode như paper viết.
3. **CartPole BART dùng split có phân tầng (stratified)** trong khi
   notebook dùng split không phân tầng — test rows của BART khác test
   rows của baseline tính lại. Độ nhạy với split có phân tầng (cùng
   test rows với BART): LR 90,43%, DT 95,42% accuracy.
4. **PongDuel BART là PROXY**: action là 9 class rời rạc nhưng repo chỉ
   có BART hồi quy — số này là hồi quy số làm tròn 0–8, trên tập con
   14.995 mẫu ứng viên local / test 5.998, khác tập 74.444 cặp của
   baseline. Tác giả không có notebook fidelity cho PongDuel nên
   baseline PongDuel là chuyển thể theo mẫu LunarLander.
5. **Số baseline CartPole cũ đã thay thế** (LR 90,76%, DT 98,20% trong
   `metrics.json` cũ đo task khác: state_t → action_t). Số chốt là số
   ở §1.2. File `metrics.json` cũ giữ nguyên làm lịch sử.
6. **Mask valid-step từng mẫu không được lưu**; baseline tính lại dùng
   toàn bộ cặp t → t+1, biện minh bằng việc số mẫu valid đã lưu bằng
   đúng số cặp ứng viên (CartPole 6.114/6.114, LunarLander 7.381/7.381).

---

## 2. Điều kiện chạy — tác giả vs nhóm (Mục 3)

| Điều kiện | Tác giả (paper + repo) | Nhóm (Mục 3) |
|---|---|---|
| Agent | Paper viết "trained RL agents"; repo không có code train, notebook dùng heuristic | Heuristic của repo tác giả, không train RL (đúng repo) |
| CartPole | 200 episode (paper §5.1) | 200 episode / 6.115 transition, seed 99, CartPole-v0 |
| LunarLander | 200 episode (paper); notebook fidelity dùng 30 episode, notebook timing dùng 20 | 30 episode / 7.382 transition, seed 99, LunarLander-v2 |
| PongDuel | 200 episode (paper); #Rep ghi 12 nhưng notebook dùng 20 cột quan sát + 1 action = 21 cột | 200 episode / 74.445 transition, seed 99, PongDuel-v0 |
| MicroRTS | ~500 episode (tên file `scm500_hte` trong repo gợi ý) | 100 episode / 4.924 transition (nhóm 2; chỉ ở §6, không phải Table 1) |
| BART | R `dbarts` qua rpy2, 4 chain, `nskip=200, ndpost=125, nchain=4, nthread=4, seed=99` | R `dbarts` qua rpy2, 4 chain, 1 thread, từng target chạy process riêng (tránh chết im) |
| Split fidelity | Notebook: test 40%, seed 10 (CartPole không phân tầng, LunarLander phân tầng) | Baseline tính lại theo đúng notebook; BART CartPole dùng split phân tầng (xem caveat §1.4.3) |

---

## 3. Thời gian học cấu trúc (Efficiency) — KHÔNG SO TRỰC TIẾP ĐƯỢC

### 3.1. Số của tác giả (Table 1, giây)

| Môi trường | PC (s) | FCI (s) | CausalXRL (s) |
|---|---:|---:|---:|
| CartPole | 1,7 | 0,8 | 0,1 |
| LunarLander | 12,6 | 5,1 | 0,3 |
| PongDuel | 23,2 | 8,7 | 0,5 |
| MicroRTS-Lite | 20,9 | 7,4 | 0,5 |

### 3.2. Số của nhóm (Mục 3, giây, wall-clock thật)

| Môi trường | PC (s) | FCI (s) | GES (s) — nhóm 3 | BART (s) |
|---|---:|---:|---:|---:|
| CartPole | 0,076 | 0,061 | 0,018 (7 cạnh, 0 cạnh có hướng) | 242,19 (tổng 5 process target riêng lẻ) |
| LunarLander | 0,197 (8 cột state) | 0,799 (8 cột state) | 0,090 (18 cạnh có hướng) | 1.086,40 (tổng 12 process target riêng lẻ) |
| PongDuel | 0,997 (basis 10 cột độc lập) | 0,957 (basis 10 cột độc lập) | 0,346 (34 cạnh: 32 có hướng, 2 vô hướng) | 35.531,20 ≈ 9,9 giờ (tổng 21 process target riêng lẻ) |
| MicroRTS-Lite | 0,165 (states-only) / 0,286 (states+action) — cùng phiên GES | 0,520 (states-only) / 1,009 (states+action) — cùng phiên GES | 0,563 (50 cạnh, states-only) / 0,685 (53 cạnh có hướng, states+action) | 498,3 (13 lượt fit: 12 target state + target action; `light2` hằng bỏ qua; 100 episode — nhóm 2, không phải protocol Table 1) |

Nguồn từng cột (đọc kỹ trước khi trích): PC/FCI ở 3 môi trường gym là
số **trích sẵn** từ `metrics.json` của Mục 3 (phiên chạy khác GES).
PC/FCI của MicroRTS chưa từng chạy ở Mục 3 nên lấy số **cùng phiên
GES** trong `muc3/results/microrts/ges.json` (biến thể states-only:
PC 0,165 s / FCI 0,520 s trên 4.924 hàng; biến thể states+action:
PC 0,286 s / FCI 1,009 s trên 3.200 hàng có action). PC/FCI cùng phiên
GES ở 3 môi trường gym (chạy lại trên đúng ma trận GES, ghi trong
`ges.json` từng môi trường) là: CartPole 0,014 / 0,017 s; LunarLander
0,040 / 0,104 s; PongDuel 0,273 / 0,274 s — nhanh hơn số trích sẵn,
cho thấy thời gian dưới 1 giây phụ thuộc máy/phiên chạy; chỉ nên đọc
bậc độ lớn. GES: causal-learn 0.1.4.8, `local_score_BIC`, 3 lần lặp,
tập cạnh giống hệt nhau, thời gian là median; GES chạy trên cùng dữ
liệu replay như PC/FCI (không chuẩn hóa, không giới hạn số cha).
Sửa 13/11/2026: bản trước ghi nhầm thời gian GES (0,563/0,685) vào cột
PC của MicroRTS — đã sửa theo đúng `ges.json`.

### 3.3. Vì sao không so được

Thời gian của tác giả trong Table 1 **không phải wall-clock BART thật**:
notebook `eval_time_*` trong repo đo bằng một khối mô phỏng (dữ liệu
ngẫu nhiên `np.random.rand` + cộng dồn thời gian tính quantile của
`CITestFast`). Thời gian của nhóm là wall-clock thật của BART. Hai cách
đo cho kết quả ngược hướng nhau (tác giả: CausalXRL nhanh hơn PC/FCI
hàng chục lần; nhóm: BART chậm hơn PC/FCI hàng trăm–hàng chục nghìn
lần) — đó là khác biệt về **cách đo**, không phải bằng chứng ai đúng
ai sai. GES: paper nhắc tới nhưng Table 1 không có cột GES; nhóm **đã
chạy GES** đủ 4 môi trường ở nhóm 3 (xem §3.2) — vẫn là wall-clock
thật, không so trực tiếp với timer mô phỏng của tác giả.

Riêng PC/FCI của nhóm ở PongDuel phải chạy trên basis 10 cột độc lập
(vì 20 cột gốc có cột trùng/one-hot phụ thuộc, ma trận tương quan suy
biến làm Fisher-Z lỗi) — đã ghi rõ, không so thẳng với paper.

---

## 4. Đồ thị nhân quả toàn cục

### 4.1. Mục 3 — đồ thị học bằng BART thật (phương pháp hte)

| Môi trường | Số target | Tổng số cạnh (tổng số nguyên nhân) | Nguồn |
|---|---:|---:|---|
| CartPole | 5 | 13 | `muc3/results/cartpole/metrics.json` (cha của `push`: x, θ, push) |
| LunarLander | 12 | 44 | `muc3/results/lunarlander/metrics.json` |
| PongDuel | 21 | 53 | `muc3/results/pongduel/targets/target_0..20.json` (2 target không có nguyên nhân nào) |
| MicroRTS | 13 state (12 tính được + `light2` hằng) + target action | 23 | `muc3/results/microrts/` — 100 episode / 4.924 transition (nhóm 2), xem §6; đồ thị 20 cạnh/10 episode cũ giữ làm lịch sử (thực ra 8 episode duy nhất + 109 hàng trùng) |

Tác giả không báo tổng số cạnh cho 3 môi trường gym trong paper nên
bảng này không có cột tác giả để đối chiếu. **Cập nhật nhóm 1:** ảnh
đồ thị Mục 3 (CartPole 13, LunarLander 44, PongDuel 53 cạnh) đã vẽ và
lưu trên Drive (thư mục `muc3/results/` trong "XRL - file nhi phan"),
kèm case study local kiểu Hình 4 của paper (hàng replay 1131, góc cột
+0,2075 rad). Chưa phân tích trọng số cạnh.

### 4.2. Mục 4 — đồ thị trained-vs-heuristic (CartPole, BART thật, seed 0, 200 ep/policy)

| Policy (transitions) | hte: cạnh / thời gian | ate: cạnh / thời gian | Cha của `push` (hte / ate) |
|---|---|---|---|
| Heuristic (6.330) | 14 / 85,84 s | 15 / 120,38 s | 3 (push, x, θ) / 4 (thêm w) |
| DQN run 9 (99.800) | 14 / 1.756,30 s | 16 / 2.306,02 s | 5 (push, x, v, θ, w) / 5 |

- Jaccard hte = 0,75 (khác 4 cạnh; chỉ DQN có `v→push`, `w→push`).
- Jaccard ate = 0,6316 (khác 7 cạnh).
- DQN chậm hơn ~19–20 lần do có ~15,8 lần transitions (mọi episode DQN
  đều chạm trần 500 bước).
- Nhiễu BART: số cạnh dao động ±1–2 giữa các lần chạy (heuristic hte
  quan sát 12–14 cạnh qua 3 lần chạy). DQN hte tái hiện giống hệt qua
  2 lần chạy độc lập.

---


### 4.3. Mục 4 — đồ thị trained-vs-heuristic (LunarLander, BART thật, seed 0, 30 ep/policy — nhóm 3)

| Policy (transitions) | hte: cạnh / thời gian | ate: cạnh / thời gian |
|---|---|---|
| Heuristic (7.594) | 48 / 475,45 s | 46 / 723,96 s |
| DQN (13.105) | 51 / 742,91 s | 45 / 1.121,66 s |

- Jaccard hte = 0,5231 (khác 31 cạnh), ate = 0,5690 (khác 25 cạnh).
  Khác biệt ổn định: `fire_left` của DQN có nhiều cha hơn (10 vs 6 ở
  giai đoạn đồ thị).
- DQN chậm hơn chỉ ~1,55 lần (khác CartPole ~19–20 lần) vì số transition
  chỉ gấp ~1,7 lần.
- Nhiễu BART: refit HTE ở giai đoạn fidelity cho đồ thị khác giai đoạn
  đồ thị (Jaccard 0,7547 heuristic, 0,6452 DQN) vì `bart_train` không
  gieo seed — chỉ kết luận "đồ thị khác nhau rõ rệt", không đọc từng
  cạnh. Ảnh 4 đồ thị: Drive, thư mục
  `muc4/results/xrl-trained-vs-heuristic-lunarlander/graphs/`.

## 5. Độ nhạy theo ngưỡng CI (sensitivity) — chỉ so xu hướng

Tác giả (Appendix D + Hình 8, **MicroRTS**): 90% → 95% mất 5 cạnh
(Base xanh→Worker xanh, Worker đỏ→Base xanh, Barrack đỏ→Heavy đỏ,
Worker đỏ→Barrack đỏ, Ranged đỏ→Base xanh); 95% → 99% mất thêm 3 cạnh
(Barrack đỏ→Worker xanh, Heavy xanh→Ranged xanh, Ranged xanh→Heavy
xanh). Số cạnh đếm từ Hình 8: 21 → 16 → 13 (khớp với số cạnh mất trong
text; paper không ghi tổng số cạnh bằng chữ).

Nhóm (**CartPole**, BART thật, 200 episode, cùng model đã fit, tính lại
từ cùng 500 ITE draws, không fit lại; quyết định 95% tính lại khớp
phương thức repo ở cả 25 cặp × 2 phương pháp, 0 mismatch):

| Ngưỡng CI | Tác giả (MicroRTS) | Nhóm hte (CartPole) | Nhóm ate (CartPole) |
|---|---:|---:|---:|
| 90% | 21 cạnh | 13 cạnh | 17 cạnh |
| 95% | 16 cạnh (mất 5) | 13 cạnh (mất 0) | 15 cạnh (mất 2: `w→x`, `w→θ`) |
| 99% | 13 cạnh (mất thêm 3) | 11 cạnh (mất 2: `θ→w`, `w→θ`) | 14 cạnh (mất 1: `w→push`) |

**Khác môi trường nên chỉ so xu hướng** (cạnh giảm hoặc không đổi khi
ngưỡng chặt hơn), không so số tuyệt đối. Nhóm chưa làm sensitivity cho
LunarLander, PongDuel hay MicroRTS. Sensitivity surrogate cũ (90% → 8
cạnh, 95% → 7 cạnh, hte) là **SURROGATE, đã thay thế**, không trích.

---

## 6. MicroRTS + dashboard

| Hạng mục | Tác giả | Nhóm |
|---|---|---|
| Fidelity MicroRTS-Lite (Table 1) | LM 75,3%/74,5%; DT 80,3%/82,2%; CausalXRL 86,0%/85,9% | **Không đo được, đã chẩn đoán tới cấp cơ chế (nhóm 3)** — tập valid rỗng: cờ local ⊆ nguyên nhân toàn cục, BART cho `action` chỉ có 1 nguyên nhân (`heavy2`) nên ngưỡng >1 cờ không thể đạt được; 13/14 cột treatment có 0 cluster HTE có ý nghĩa. Không nới ngưỡng, không sinh số mới. Bản không-lọc lệch protocol giữ làm lịch sử audit |
| Thời gian học cấu trúc | PC 20,9 s / FCI 7,4 s / CausalXRL 0,5 s (timer mô phỏng) | GES 0,563 s / 0,685 s (nhóm 3); PC/FCI cùng phiên GES 0,165/0,520 s (states-only) và 0,286/1,009 s (states+action); BART 498,3 s (13 lượt fit: 12 target state + target action, 100 ep — nhóm 2). Wall-clock thật, **KHÔNG SO ĐƯỢC** với timer mô phỏng |
| Env MicroRTS thật + agent PPO | Có (checkpoint riêng, không công bố trong repo) | **Chạy được**: checkpoint `agent-3.pt` từ nguồn công khai (S3 của repo gym-microrts-paper, 21.734.570 byte, SHA-256 `ad21e903…`, 5.433.185 tham số, `load_state_dict(strict=True)` OK — không chứng minh được giống byte bản riêng của tác giả); JVM + PPO trả 2 action hợp lệ thật; `run_num_steps(3)` chạy được trên map có sẵn và map `case1.xml` |
| Sửa import `stratvis` | (lỗi trong repo tác giả) | Sửa 22 chỗ trong 8 file `.py` (`stratvis.` → `cxrl_microRTS.`), patch `muc3/stratvis-fix.patch` áp sạch lên cây gốc |
| Dashboard Flask | Có (dùng trong expert study) | **Chạy được: 9 lượt kiểm tra trên 7 route đều 200** (nhóm 2–3): `/`, `/utt_to_uht`, `/read_xml_map`, `/get_board_state`, `/compute_global_causal`, `/get_local_causal` (3 step), `/run_env_episode`. `/get_board_state` và `/compute_global_causal` sửa ở nhóm 2; `/get_local_causal` sửa ở nhóm 3 (patch `muc3/dashboard-local-causal-fix.patch`, danh sách cột suy từ dữ liệu thay vì hard-code 10 cột). Cần `R_LIBS` trỏ tới dbarts và gọi `/compute_global_causal` trước. Các route còn lại của tác giả không thử từng cái |
| Đồ thị toàn cục MicroRTS | Có (Hình 3, Hình 8; ~500 episode) | **100 episode / 4.924 transition** thật (nhóm 2), 12/13 target state tính được (target `light2` hằng, bỏ qua đúng luật `infer_DBN`) + target action (1 nguyên nhân `heavy2`), **23 cạnh**. Nhiều cạnh tự vòng; chỉ chứng minh pipeline chạy đầu-cuối, không phải bản sao kết quả tác giả |
| File dữ liệu `data/` | Không có trong repo công khai (README bảo liên hệ tác giả) | Thư mục `data/` **tự sinh từ dữ liệu 100 episode** (nhóm 2) để dashboard chạy; dữ liệu gốc của tác giả vẫn thiếu |

Chi tiết: `muc3/microRTS-thu-nghiem.md` + log. Checkpoint `.pt`
**không** đưa vào repo (không rõ quyền tái phân phối); lấy lại bằng
script Range-fetch + SHA-256 ghi trong báo cáo.

---

## 7. Counterfactual (phản thực, DiCE)

| Hạng mục | Tác giả | Nhóm |
|---|---|---|
| Phương pháp | `XRL.counterfactual()` dùng DiCE, 40 counterfactual thô, rồi lọc simplify + dominating | Chạy được **sau khi sửa code tác giả trên bản copy** (patch `muc4/xrl-counterfactual-fix.patch`): code gốc sập 100% khi gọi (`self.exp` và `self.state_features` không tồn tại, `self.pi` là hàm thuần không có `.predict`) |
| Chạy thử | Không có số counterfactual nào báo trong paper (chỉ mô tả phương pháp + UI) | Bước 3: 1 query CartPole (hàng replay số 6, action 0 → muốn 1, seed 99, 200 episode, 6.115 transition): DiCE sinh **40** → còn **2** counterfactual; cả **2/2 kiểm chứng độc lập bằng hàm policy gốc đều lật 0→1**. **Nhóm 2 (mở rộng): 6 query CartPole → 9/9 lật + 4 query LunarLander (4 action, sửa đa lớp) → 16/16 lật; tổng 25/25** kiểm chứng bằng policy gốc |
| Phạm vi khác | — | **Không làm**: counterfactual cho policy DQN, PongDuel, MicroRTS; DiCE ở đây là phương pháp `random`, không phải `genetic`/`kdtree` |

Chi tiết: `muc4/counterfactual-thu-nghiem.md`.

---

## 8. Expert study (nghiên cứu chuyên gia)

| Hạng mục | Tác giả | Nhóm |
|---|---|---|
| Chuyên gia | 8 người (E1–E8), >7 năm kinh nghiệm RTS, trung bình >1.000 giờ chơi; IRB 1020950; $8/giờ; Zoom, think-aloud, crossover | **Không làm** — không có chuyên gia, không có IRB, không có nghiên cứu người dùng |
| Dự đoán hành vi | CausalXRL 8/8 đúng vs SHAP 4/8 đúng | **Không làm** |
| Điểm Likert 1–5 | Hầu hết chỉ số giải thích >4 ("somewhat agree"); mọi chỉ số tin cậy của mô hình local >4 | **Không làm** |
| Tính chất | Tác giả tự ghi là bằng chứng khám phá (exploratory/suggestive), số người tham gia hạn chế, không đủ power thống kê | — |

---

## 9. Mở rộng của nhóm (tác giả không có — không có số tác giả để đối chiếu)

### 9.1. Hướng D — Train DQN thật trên CartPole (repo tác giả không có code train RL)

DQN tự viết PyTorch (MLP 4→128→128→2), gymnasium CartPole-v1, CPU-only,
$0. Eval greedy 100 episode, seeds 10000–10099 (trừ khi ghi khác).

| Run | Cấu hình | Eval 100 ep (mean ± std, min–max) |
|---|---|---|
| Heuristic của repo (cùng điều kiện) | Luật tuyến tính viết tay | 31,53 ± 11,16 (11–63) |
| Run 1 | DQN, 600 ep | 246,16 ± 10,89 |
| Run 3 | Huber + clip 1.0 | 86,77 ± 31,63 |
| Run 4 | Huber, lr 5e-3, clip 10.0 | 56,81 ± 7,94 |
| Run 5 | Double DQN, 600 ep | 272,68 ± 58,53 (203–500) |
| Run 7 | Double DQN, 1.500 ep — **dở dang** (VM restart ở ep 1040) | Best 343,26 ± 22,30 (xác minh 100 ep) |
| Run 8 | Như run 7, target-update 10→100 — thụt lùi | 234,51 ± 34,59 (mạng cuối train: 89,68) |
| **Run 9 (best)** | **Double DQN, 2.000 ep, lr 1e-3, MSE, eps-decay 1300, eval mỗi 50 ep** | **500,00 ± 0,00 (500–500)** — hơn heuristic **+468,47** |

Run 9: 10/40 lần eval định kỳ đạt đúng 500,00; eval dao động
500 ↔ 108,20 / 96,80 giữa các lần đo cách 50 episode (Q-instability),
loss MSE đỉnh 34,35 ở ep 1900 — nên chọn checkpoint theo eval định kỳ
là bắt buộc. Cả mạng cuối train cũng đạt 500,00 nên kết quả không phụ
thuộc chọn checkpoint. Run 6 dừng giữa chừng, không checkpoint, không
dùng. CartPole seed 1 (nhóm 2): **500,00/500** cả mạng cuối và
checkpoint tốt nhất — tái hiện qua seed khác. **DQN LunarLander
(nhóm 2): 222,03 ± 69,29** (100 ep, checkpoint tốt nhất ep 850, train
dở ở ep 873 do VM restart). **DQN PongDuel (nhóm 3): âm tính** — xem
§9.5.

### 9.2. Hướng D — Fidelity trained-vs-heuristic (BART thật, §4.2)

| Policy | DT acc / F1 nhị phân | LR acc / F1 nhị phân | BART acc / F1 nhị phân (n=200 test steps) |
|---|---|---|---|
| Heuristic (6.330 trans) | 0,9491 / 0,9498 | 0,9111 / 0,9121 | 0,9350 / 0,9340 |
| DQN run 9 (99.800 trans) | 0,9356 / 0,9392 | 0,8195 / 0,8363 | 0,8050 / 0,8368 |

Đọc trung thực: DQN hoàn hảo **kém heuristic ở cả 3 thước đo** — biên
quyết định phi tuyến của mạng khó bắt chước hơn luật tuyến tính. Caveat:
BART-weighted chỉ trên tối đa 200 test steps trong khi DT/LR trên toàn
bộ test set (test set khác nhau); số BART dao động ±1–2 cạnh giữa các
lần chạy. Bảng cũ (checkpoint run 5) lưu ở
`muc4/results/xrl-trained-vs-heuristic-run5/` — không so trực tiếp số
run 5 với run 9 như cùng test set.

### 9.3. Hướng A — Ablation bằng BART thật (CartPole, seed 0, policy heuristic)

| Episode | Phương pháp | Transitions | Cạnh | Thời gian (s) | Cha của `push` |
|---:|---|---:|---:|---:|---|
| 50 | hte | 1.572 | 12 | 19,86 | push, x, θ (3) |
| 50 | ate | 1.572 | 15 | 29,11 | push, x, θ (3) |
| 100 | hte | 3.183 | 13 | 43,54 | push, x, θ, w (4) |
| 100 | ate | 3.183 | 15 | 62,18 | push, x, θ (3) |
| 200 | hte | 6.330 | 13 | 88,87 | push, x, θ (3) |
| 200 | ate | 6.330 | 15 | 124,06 | push, x, θ, w (4) |

- Jaccard so với đồ thị 200 ep: hte 0,7857 (50 ep), 0,8571 (100 ep);
  ate 0,8750 (50 ep), 0,7647 (100 ep). **Không còn kết luận "hte ổn
  định từ 100 episode"** của lần chạy surrogate.
- Đo lặp hte/200 ep (2 process độc lập): 12 vs 13 cạnh, Jaccard 0,7857;
  dải hte 200 ep quan sát: 12–14 cạnh qua 3 lần chạy. Chênh 2–4 cạnh
  nằm trong/gần dải nhiễu, không đọc thành xu hướng.
- So với surrogate: Jaccard BART vs surrogate chỉ 0,31–0,59; surrogate
  cho `push: []` ở cả 3 mức hte trong khi BART tìm 3–4 nguyên nhân ở
  mọi run. **Số surrogate đã thay thế, không trích làm kết quả.**
- Ảnh 6 đồ thị BART: xem §10 (Drive).

---


### 9.4. Hướng D — Fidelity trained-vs-heuristic, LunarLander (BART thật, nhóm 3)

30 episode/policy, seed 0, LunarLander-v3, test set đầy đủ (không giới
hạn 200 step như CartPole). Heuristic mạnh hơn DQN về reward nên so sánh
này không cùng hướng với CartPole (xem §4.3).

| Policy | DT acc / weighted-F1 | LR acc / weighted-F1 | BART acc / weighted-F1 |
|---|---|---|---|
| Heuristic (7.594 trans, n=3.038) | 0,7870 / 0,7881 | 0,7808 / 0,7758 | 0,8348 / 0,8304 |
| DQN (13.105 trans, n=5.242) | 0,6616 / 0,6613 | 0,6167 / 0,5502 | 0,6728 / 0,6265 |

### 9.5. Hướng D — DQN PongDuel: kết quả âm tính + bỏ qua XRL PongDuel (nhóm 3)

Joint action 0–8 (1 policy điều khiển cả 2 paddle), cùng recipe run 9 /
LunarLander. Baseline đo trước khi train (100 ep): heuristic 8,81 ± 1,04;
random 4,90 ± 1,56. Tiêu chí thành công đặt trước: eval cuối ≥9,30 và
≥3/4 eval định kỳ cuối trên 8,81. Train dừng ở 800/2.000 episode do VM
restart; 15 eval định kỳ quanh mức random (tốt nhất 5,42 @ ep 350); eval
cuối của checkpoint tốt nhất **5,44 ± 1,56** — cả 2 điều kiện đều trượt.
Recipe này không học được gì có ý nghĩa trên joint-action PongDuel trong
800 episode. Vì không có policy đã train nên **so sánh XRL PongDuel bỏ
qua**; ước tính so sánh đầy đủ >70 giờ BART liên tục (~28 giờ với hệ số
tốc độ 2,5× của VM), quy mô 30 episode vẫn ~5–6 giờ + rủi ro OOM chưa
kiểm chứng. Bằng chứng: `muc4/results/pongduel-dqn/`.

## 10. Biểu đồ

3 biểu đồ vẽ bằng matplotlib (script text trong repo, ảnh PNG trên
Drive vì connector GitHub không push được file nhị phân):

| Biểu đồ | Nội dung | Ảnh (Drive) |
|---|---|---|
| 1 | Grouped bar Acc/F1 của BART: tác giả vs nhóm, 3 môi trường gym (cột không so được gạch chéo + ghi chú đỏ) | `muc4/results/so-sanh/bieu-do-1-fidelity-bart.png` |
| 2 | Grouped bar thời gian học cấu trúc, thang log, kèm caveat timer mô phỏng vs wall-clock thật | `muc4/results/so-sanh/bieu-do-2-thoi-gian-hoc-cau-truc.png` |
| 3 | Sensitivity 90/95/99%: số cạnh tác giả (MicroRTS) vs nhóm (CartPole hte/ate), chỉ so xu hướng | `muc4/results/so-sanh/bieu-do-3-do-nhay-ci.png` |

- Script vẽ: `muc4/results/so-sanh/ve_bieu_do_so_sanh.py` (chạy lại
  được, số liệu ghi kèm nguồn trong comment đầu file).
- Ảnh biểu đồ trên Drive: thư mục
  [XRL - file nhi phan / muc4 / results / so-sanh](https://drive.google.com/drive/folders/1qzMeZPj6iilK2ylqv1ltirdx74BQFhaU).
- Ảnh 6 đồ thị ablation-BART trên Drive: thư mục
  [XRL - file nhi phan / muc4 / results / ablation-bart / graphs](https://drive.google.com/drive/folders/1gja-Y6YvIJ1pmVuPSJvG5vpC3AWbKhpN).
- Thư mục Drive gốc (toàn bộ file nhị phân của repo):
  [XRL - file nhi phan](https://drive.google.com/drive/folders/1rK_Br_Kj3Dr-OZb8c5V2AgJeUW8bx_ea).

**Mỗi biểu đồ đều có caveat ghi ngay trên hình.** Khi đưa biểu đồ vào
slide/báo cáo, không được cắt bỏ phần ghi chú đỏ.

**Cập nhật 13/11/2026:** 3 biểu đồ đã vẽ lại (script commit
`f31b1bd`): biểu đồ 1 dùng weighted-F1 LunarLander 81,52% (không còn
macro-F1 cũ); biểu đồ 2 thêm cột GES đủ 4 môi trường và nhóm MicroRTS
(PC/FCI MicroRTS 0,165/0,520 s, biến thể states-only, cùng phiên GES);
biểu đồ 3 giữ nguyên số sau khi kiểm chứng lại. Ảnh mới đã thay trên
Drive (giữ nguyên link).

---

## 11. Những gì còn thiếu / chưa làm được (tóm tắt)

1. Fidelity MicroRTS theo Table 1: **không đo được, đã chẩn đoán tới
   cấp cơ chế** (nhóm 3) — không phải "không làm". Thời gian học cấu
   trúc MicroRTS: đã có GES + BART (nhóm 2–3), xem §3.2/§6. Dashboard
   đã sửa xong (9 lượt kiểm tra trên 7 route đều 200).
2. Expert study: **không làm**.
3. ~~Weighted-F1 của BART LunarLander: không tính được~~ — **đã khôi
   phục: 81,52%** (nhóm 1).
4. Fidelity BART phân loại cho PongDuel: chỉ có PROXY hồi quy; repo
   không có BART phân loại.
5. ~~GES: không làm~~ — **đã chạy đủ 4 môi trường** (nhóm 3).
6. ~~Ảnh đồ thị Mục 3, case study local kiểu Hình 4: không làm~~ —
   **đã làm ở nhóm 1** (ảnh trên Drive + case study hàng 1131). Phân
   tích trọng số cạnh: **chưa làm**.
7. Sensitivity cho LunarLander/PongDuel/MicroRTS: **không làm**.
8. Counterfactual cho DQN/PongDuel/MicroRTS: **không làm** (CartPole +
   LunarLander đã làm tới 25/25, nhóm 2).
9. ~~DQN cho LunarLander/PongDuel: không làm~~ — **LunarLander đã làm
   (222,03, nhóm 2); PongDuel đã thử, âm tính** (nhóm 3). So sánh XRL
   PongDuel: **bỏ qua có lý do** (nhóm 3).
10. Mask valid-step từng mẫu của Mục 3 không được lưu. Số baseline DT
    LunarLander cũ trong `metrics.json` (accuracy 78,80%, macro-F1
    62,75%) không tái hiện được đúng bằng script đã commit: accuracy
    khớp `random_state=99`, còn macro-F1 không khớp cả `random_state=1`
    lẫn 99 — số tính lại theo đúng protocol notebook
    (`random_state=1`) là 79,14% / 63,25% (xem
    `muc3/results/baselines_recomputed.md`).
11. Số cạnh Hình 8 của tác giả (21/16/13) là số đếm từ hình, paper chỉ
    ghi số cạnh mất (5 rồi 3) bằng chữ — khi trích thì ghi rõ là đếm
    từ hình.
