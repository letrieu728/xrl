# So sánh tất cả con số: tác giả CausalXRL vs nhóm

Ngày lập: 12/11/2026. Cập nhật nhóm 1, 2 và 3: 12/11/2026; rà soát
đồng bộ toàn repo và sửa số PC/FCI MicroRTS: 13/11/2026; nhóm việc 4:
15/11/2026. Ngày 16/11/2026: gỡ toàn bộ số liệu của hướng mở rộng cũ
khỏi file này (còn trong lịch sử git); chỉ giữ số Mục 3 và hướng mở
rộng mới đang lên kế hoạch (xem §9). Mọi số trong file này đều đọc từ nguồn gốc ghi ở
đầu mỗi mục — không bịa, không lấy số tác giả ghi thành số nhóm.
Số thập phân viết theo kiểu Việt Nam (dấu phẩy).

## Nguồn số liệu

| Bên | Nguồn |
|---|---|
| Tác giả | Paper arXiv:2610.42028v1 (17 trang): Table 1 tr. 8, §5.1–§5.2, §6 (expert study), Appendix D + Hình 8. PDF gốc: `paper/CausalXRL-2610.42028.pdf`. Repo: github.com/yanmluk/causal-xrl, commit `ae9096e`. |
| Nhóm — Mục 3 | `muc3/EXPERIMENT_REPORT.md` (đã sửa 12/11/2026, commit `5406c94`), `muc3/results/baselines_recomputed.json/.md`, `muc3/results/*/metrics.json`, `bart_fidelity.json`, `collection_metadata.json`, `replay.npz`, `muc3/microRTS-thu-nghiem.md`, `muc3/counterfactual-thu-nghiem.md`. Seed 99, Windows, BART thật (R/dbarts qua rpy2). |

## Cách đọc — các nhãn caveat dùng trong toàn file

- **PROXY**: số đo bằng phương pháp thay thế, không tương đương paper.
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

| Môi trường | PC (s) | FCI (s) | BART (s) |
|---|---:|---:|---:|
| CartPole | 0,076 | 0,061 | 242,19 (tổng 5 process target riêng lẻ) |
| LunarLander | 0,197 (8 cột state) | 0,799 (8 cột state) | 1.086,40 (tổng 12 process target riêng lẻ) |
| PongDuel | 0,997 (basis 10 cột độc lập) | 0,957 (basis 10 cột độc lập) | 35.531,20 ≈ 9,9 giờ (tổng 21 process target riêng lẻ) |
| MicroRTS-Lite | Không chạy ở Mục 3 | Không chạy ở Mục 3 | 498,3 (13 lượt fit: 12 target state + target action; `light2` hằng bỏ qua; 100 episode — nhóm 2, không phải protocol Table 1) |

Nguồn từng cột (đọc kỹ trước khi trích): PC/FCI ở 3 môi trường gym là
số **trích sẵn** từ `metrics.json` của Mục 3. Thời gian dưới 1 giây phụ
thuộc máy/phiên chạy; chỉ nên đọc bậc độ lớn. PC/FCI của MicroRTS chưa
từng chạy ở Mục 3.

### 3.3. Vì sao không so được

Thời gian của tác giả trong Table 1 **không phải wall-clock BART thật**:
notebook `eval_time_*` trong repo đo bằng một khối mô phỏng (dữ liệu
ngẫu nhiên `np.random.rand` + cộng dồn thời gian tính quantile của
`CITestFast`). Thời gian của nhóm là wall-clock thật của BART. Hai cách
đo cho kết quả ngược hướng nhau (tác giả: CausalXRL nhanh hơn PC/FCI
hàng chục lần; nhóm: BART chậm hơn PC/FCI hàng trăm–hàng chục nghìn
lần) — đó là khác biệt về **cách đo**, không phải bằng chứng ai đúng
ai sai.

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

## 5. Độ nhạy theo ngưỡng CI (sensitivity) — chỉ so xu hướng

Tác giả (Appendix D + Hình 8, **MicroRTS**): 90% → 95% mất 5 cạnh
(Base xanh→Worker xanh, Worker đỏ→Base xanh, Barrack đỏ→Heavy đỏ,
Worker đỏ→Barrack đỏ, Ranged đỏ→Base xanh); 95% → 99% mất thêm 3 cạnh
(Barrack đỏ→Worker xanh, Heavy xanh→Ranged xanh, Ranged xanh→Heavy
xanh). Số cạnh đếm từ Hình 8: 21 → 16 → 13 (khớp với số cạnh mất trong
text; paper không ghi tổng số cạnh bằng chữ).

Nhóm: **chưa có số sensitivity trong repo hiện tại.** Số sensitivity
CartPole/LunarLander chạy trước đây đã gỡ khỏi repo ngày 16/11/2026
cùng hướng mở rộng cũ (còn trong lịch sử git); khi nào chạy lại thì ghi
kèm mức CI và chỉ so xu hướng với số của tác giả, không so số tuyệt đối
(vì khác môi trường).

---

## 6. MicroRTS + dashboard

| Hạng mục | Tác giả | Nhóm |
|---|---|---|
| Fidelity MicroRTS-Lite (Table 1) | LM 75,3%/74,5%; DT 80,3%/82,2%; CausalXRL 86,0%/85,9% | **Không đo được, đã chẩn đoán tới cấp cơ chế (nhóm 3)** — tập valid rỗng: cờ local ⊆ nguyên nhân toàn cục, BART cho `action` chỉ có 1 nguyên nhân (`heavy2`) nên ngưỡng >1 cờ không thể đạt được; 13/14 cột treatment có 0 cluster HTE có ý nghĩa. Không nới ngưỡng, không sinh số mới. Bản không-lọc lệch protocol giữ làm lịch sử audit |
| Thời gian học cấu trúc | PC 20,9 s / FCI 7,4 s / CausalXRL 0,5 s (timer mô phỏng) | BART 498,3 s (13 lượt fit: 12 target state + target action, 100 ep — nhóm 2); PC/FCI MicroRTS không chạy ở Mục 3. Wall-clock thật, **KHÔNG SO ĐƯỢC** với timer mô phỏng |
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
| Phương pháp | `XRL.counterfactual()` dùng DiCE, 40 counterfactual thô, rồi lọc simplify + dominating | Chạy được **sau khi sửa code tác giả trên bản copy** (patch `muc3/xrl-counterfactual-fix.patch` + patch đa lớp `muc3/xrl-multiclass-fix.patch`): code gốc sập 100% khi gọi (`self.exp` và `self.state_features` không tồn tại, `self.pi` là hàm thuần không có `.predict`) |
| Chạy thử | Không có số counterfactual nào báo trong paper (chỉ mô tả phương pháp + UI) | Bước 3: 1 query CartPole (hàng replay số 6, action 0 → muốn 1, seed 99, 200 episode, 6.115 transition): DiCE sinh **40** → còn **2** counterfactual; cả **2/2 kiểm chứng độc lập bằng hàm policy gốc đều lật 0→1**. **Nhóm 2 (mở rộng): 6 query CartPole → 9/9 lật + 4 query LunarLander (4 action, sửa đa lớp) → 16/16 lật; tổng 25/25** kiểm chứng bằng policy gốc |
| Phạm vi khác | — | **Không làm**: counterfactual cho PongDuel, MicroRTS; DiCE ở đây là phương pháp `random`, không phải `genetic`/`kdtree` |

Chi tiết: `muc3/counterfactual-thu-nghiem.md`.

---

## 8. Expert study (nghiên cứu chuyên gia)

| Hạng mục | Tác giả | Nhóm |
|---|---|---|
| Chuyên gia | 8 người (E1–E8), >7 năm kinh nghiệm RTS, trung bình >1.000 giờ chơi; IRB 1020950; $8/giờ; Zoom, think-aloud, crossover | **Không làm** — không có chuyên gia, không có IRB, không có nghiên cứu người dùng |
| Dự đoán hành vi | CausalXRL 8/8 đúng vs SHAP 4/8 đúng | **Không làm** |
| Điểm Likert 1–5 | Hầu hết chỉ số giải thích >4 ("somewhat agree"); mọi chỉ số tin cậy của mô hình local >4 | **Không làm** |
| Tính chất | Tác giả tự ghi là bằng chứng khám phá (exploratory/suggestive), số người tham gia hạn chế, không đủ power thống kê | — |

---

## 9. Hướng mở rộng mới

Hướng mở rộng mới (đang lên kế hoạch, chưa có kết quả): Causal-Guided RL — đóng kín vòng lặp hành động → tự soi chiếu nhân quả → sửa sai, dùng đồ thị nhân quả ngay trong lúc train để cảnh báo/phạt quyết định tự hủy và thưởng cho chuỗi hành động dẫn tới kết quả tốt. Làm pilot nhỏ trước, đăng ký tiêu chí trước, báo cáo cả khi kết quả âm tính — theo bài học từ nhóm 4 (mô phỏng phẫu thuật đồ thị không đạt kiểm chứng), độ tin cậy của đồ thị phải được chứng minh, không được giả định.

---

## 10. Những gì còn thiếu / chưa làm được (tóm tắt)

1. Fidelity MicroRTS theo Table 1: **không đo được, đã chẩn đoán tới
   cấp cơ chế** (nhóm 3) — không phải "không làm". Thời gian học cấu
   trúc MicroRTS: đã có BART (nhóm 2), xem §3.2/§6. Dashboard
   đã sửa xong (9 lượt kiểm tra trên 7 route đều 200).
2. Expert study: **không làm**.
3. ~~Weighted-F1 của BART LunarLander: không tính được~~ — **đã khôi
   phục: 81,52%** (nhóm 1).
4. Fidelity BART phân loại cho PongDuel: chỉ có PROXY hồi quy; repo
   không có BART phân loại.
5. GES (baseline học cấu trúc thứ ba paper nhắc tới): đã chạy ở nhóm 3
   nhưng **đã gỡ khỏi repo ngày 16/11/2026** cùng hướng mở rộng cũ (còn
   trong lịch sử git); hiện repo không còn số GES.
6. ~~Ảnh đồ thị Mục 3, case study local kiểu Hình 4: không làm~~ —
   **đã làm ở nhóm 1** (ảnh trên Drive + case study hàng 1131). Phân
   tích trọng số cạnh: chưa có trong repo hiện tại (bản nhóm 4 đã gỡ
   ngày 16/11/2026, còn trong lịch sử git).
7. Sensitivity theo ngưỡng CI: chưa có trong repo hiện tại (số
   CartPole/LunarLander cũ đã gỡ ngày 16/11/2026 cùng hướng mở rộng cũ,
   còn trong lịch sử git).
8. Counterfactual cho PongDuel/MicroRTS: **không làm** (CartPole +
   LunarLander heuristic 25/25, nhóm 2).
9. Train agent RL / so sánh XRL trained-vs-heuristic: đã gỡ khỏi repo
   ngày 16/11/2026 cùng hướng mở rộng cũ (còn trong lịch sử git);
   hướng mở rộng mới xem §9.
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
12. Biểu đồ so sánh cũ (kèm cột GES và số sensitivity): đã gỡ khỏi repo
    ngày 16/11/2026 cùng hướng mở rộng cũ; ảnh cũ còn trên Drive, không
    trích số từ đó.
