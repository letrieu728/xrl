# Thuyết trình CausalXRL — Bản thông tin (15–20 phút)

Bài: **CausalXRL: Explainable Reinforcement Learning through Causal Graph
Reasoning** — Yanming Zhang, Eric Papenhausen, Klaus Mueller (Stony Brook),
arXiv:2610.42028v1, ICML 2026.
Repo nhóm: https://github.com/letrieu728/XRL (xem `README.md` và
`muc4/results/REPORT.md` để đối chiếu mọi số dưới đây).

> Quy tắc của bản này: **chỉ dùng số đã kiểm chứng**. Số của tác giả ghi rõ
> "tác giả"; số của nhóm ghi rõ "nhóm". Số nào chưa có thì ghi "chưa có",
> không bịa.

---

## 1. Đặt vấn đề (2 phút)

- Agent RL chơi game, lái xe, làm robot rất giỏi, nhưng là **hộp đen**: không
  biết vì sao nó chọn hành động đó → không tin tưởng, không debug được.
- Các cách giải thích cũ (saliency map, SHAP) chỉ nói được agent "nhìn vào
  đâu", không nói được "vì sao".
- Cách dùng mô hình nhân quả trước đây (AIM, Madumal et al. 2020) có 2 hạn
  chế: cần chuyên gia định trước cấu trúc nhân quả, và gắn cạnh nhân quả trực
  tiếp vào action — không dùng được cho game chiến thuật thời gian thực (RTS)
  nhiều unit, vì quan hệ lúc đó trở thành tất định, không còn thông tin gì.
- **Câu hỏi của bài báo:** làm sao giải thích vì sao agent RL làm vậy, ở tầng
  chiến lược mà con người hiểu được?

## 2. Phương pháp của tác giả — 3 bước (4 phút)

1. **Trích biểu diễn hiểu được.** Từ trạng thái thô lấy ra các đặc trưng chiến
   lược (ví dụ MicroRTS: số unit từng loại, tài nguyên). CartPole dùng luôn 4
   biến trạng thái gốc; LunarLander 8; PongDuel 12; MicroRTS-Lite 14.
2. **Học đồ thị nhân quả toàn cục.** Chạy agent đã train 200 episode, lưu
   transition vào replay buffer, dùng BART (R `dbarts`) ước lượng ATE / HTE
   (CATE) cho từng biến; biến nào có khoảng tin cậy 95% không chứa 0 thì là
   nguyên nhân. Kết quả là một đồ thị có hướng (DAG) giữa các biểu diễn.
3. **Mô hình local + phản thực.** Với một trạng thái cụ thể, xác định nguyên
   nhân local, dự đoán hành động tiếp theo có trọng số theo |ITE|, dùng DiCE
   sinh counterfactual ("nếu lúc đó khác đi thì agent làm gì?"). Có dashboard
   web cho MicroRTS để chuyên gia tự khám phá.

Điểm mấu chốt để nói: **không gắn cạnh vào action**, mà học quan hệ nhân quả
giữa các biểu diễn chiến lược, rồi mới dùng các biểu diễn đó để dự đoán hành
động.

## 3. Kết quả của tác giả — Table 1 (3 phút)

Số của tác giả (arXiv:2610.42028v1, Table 1). Đánh giá 2 thứ: (1) accuracy/F1
dự đoán action của local model so với Linear Model + Decision Tree;
(2) thời gian học cấu trúc so với PC/FCI.

| Môi trường | CausalXRL Acc / F1 | DT Acc | Thời gian (CausalXRL / PC) |
|---|---|---|---|
| CartPole | 91,4% / 91,6% | 94,1% | 0,1s / 1,7s |
| LunarLander | 88,7% / 86,6% | 85,1% | 0,3s / 12,6s |
| PongDuel | 90,2% / 90,7% | 88,3% | 0,5s / 23,2s |
| MicroRTS-Lite | 86,0% / 85,9% | 80,3% | 0,5s / 20,9s |

Nói đúng như tác giả viết:

- Môi trường đơn giản (CartPole): CausalXRL **thua Decision Tree** (91,4% so
  với 94,1%) — chỉ "competitive".
- Môi trường phức tạp (PongDuel, MicroRTS-Lite): thắng rõ.
- Học cấu trúc nhanh hơn PC/FCI hàng chục lần, càng nhiều biểu diễn càng
  chênh lệch.
- Nghiên cứu chuyên gia: 8 chuyên gia RTS (>7 năm kinh nghiệm), bài dự đoán
  hành vi nhóm có CausalXRL đúng **8/8**, nhóm chỉ có SHAP đúng **4/8**.
  Đây là nghiên cứu khám phá, mẫu nhỏ (n=8), không phải chứng minh thống kê.

## 4. Thực nghiệm của nhóm — Mục 3 (3 phút)

Phạm vi: **evaluation-only, không huấn luyện RL**. Repo tác giả không có code
train agent; policy trong notebook là heuristic viết tay. Chạy theo thứ tự
CartPole (200 episode) → LunarLander (30 episode, theo notebook đánh giá của
tác giả) → PongDuel (200 episode), seed 99, CPU-only. Chi tiết:
`muc3/EXPERIMENT_REPORT.md`. MicroRTS đã chạy tối thiểu thật: sửa 22 chỗ
import `stratvis` → `cxrl_microRTS` trong 8 file, lấy checkpoint PPO
`agent-3.pt` công khai (SHA-256 `ad21e903…`, 5.433.185 tham số, không đưa
vào repo), env thật chạy được, **100 episode / 4.924 transition**, đồ thị
toàn cục **23 cạnh**, dashboard Flask đã sửa xong **cả 9 route đều 200**
(kể cả `/get_local_causal`, sửa ở nhóm 3 ngày 12/11/2026). Fidelity
MicroRTS theo Table 1 vẫn chưa có — nhóm 3 đã chẩn đoán tới cấp cơ chế:
tập valid rỗng là bản chất dữ liệu + cơ chế HTE, không sửa được bằng cấu
hình. GES cũng đã chạy đủ 4 môi trường (nhóm 3). Chi tiết:
`muc3/microRTS-thu-nghiem.md`.

Số chốt của nhóm (accuracy / F1 so được với Table 1 — F1 nhị phân ở CartPole,
weighted-F1 ở LunarLander/PongDuel; baseline CartPole đã sửa ngày 12/11/2026
theo đúng protocol notebook fidelity của tác giả):

| Môi trường | BART | Logistic Regression | Decision Tree |
|---|---|---|---|
| CartPole | 91,62% / 91,50% | 89,62% / 89,26% | 93,83% / 93,64% |
| LunarLander | 82,80% / weighted-F1 81,52% (macro-F1 64,10%) | 77,21% / 76,24% | 79,14% / 79,06% |
| PongDuel | 81,73% / macro-F1 19,51% (PROXY) | 81,79% / 81,38% | 96,21% / 96,21% |

Nói thẳng 3 caveat:

- CartPole: Decision Tree thắng BART (93,83% so với 91,62%) — **giống paper**
  (94,1% so với 91,4%), không phải lỗi của nhóm.
- LunarLander: weighted-F1 của BART đã khôi phục được đúng từ checkpoint
  điểm từng action đã lưu: **81,52%**, kém paper (86,6%) **−5,08 điểm %**,
  so được trực tiếp. (Cập nhật nhóm 1, 12/11/2026; trước đó chỉ có
  macro-F1 64,10% và đã ghi là không so được.)
- PongDuel: số BART là **proxy** (hồi quy số làm tròn, 14.995 mẫu ứng viên
  local, trên 74.445 transition), không phải fidelity phân loại tương
  đương paper — đừng so với 90,2%/90,7% của Table 1.

## 5. Mở rộng của nhóm — Mục 4, Hướng D (5 phút — phần hay nhất)

Tác giả không train agent RL. Nhóm train agent thật rồi giải thích nó bằng
chính CausalXRL — đúng interface `XRL(env, pi)` của tác giả.

### 5.1. Train DQN CartPole: 500,00/500

- DQN tự viết bằng PyTorch (MLP 4→128→128→2, replay buffer, epsilon-greedy,
  target network, Adam), không dùng stable-baselines3. CPU-only, $0.
- **Run 9 — Double DQN, 2.000 episode: eval greedy 100 episode
  500,00 ± 0,00 (min 500, max 500).** Heuristic cùng điều kiện:
  31,53 ± 11,16 → DQN hơn **+468,47**.
- Các run trước: run 1: 246,16; run 5: 272,68; run 7: dở dang do VM restart,
  best 343,26; run 8: 234,51 (đổi target-update thì thụt lùi).

**Điểm hay nhất để nói — Q-instability:** eval định kỳ của run 9 dao động
500 ↔ ~97–146 giữa các lần đo cách nhau 50 episode. Policy tốt liên tục bị
train tiếp phá hỏng rồi tìm lại được. Vì vậy **chọn checkpoint theo eval định
kỳ là bắt buộc** — lấy checkpoint cuối train là sai. Run 9 có 10/40 lần eval
đạt đúng 500,00.

### 5.2. So sánh XRL: agent giỏi hơn nhưng khó giải thích hơn

Chạy pipeline XRL gốc bằng **BART thật** (R/dbarts 0.9.34), 200
episode/policy, seed 0, checkpoint DQN run 9.

| | DT acc / F1 | LR acc / F1 | BART acc / F1 | push parents |
|---|---|---|---|---|
| Heuristic (6.330 trans) | 0,9491 / 0,9498 | 0,9111 / 0,9121 | 0,9350 / 0,9340 | 3 (push, x, θ) |
| DQN run 9 (99.800 trans) | 0,9356 / 0,9392 | 0,8195 / 0,8363 | 0,8050 / 0,8368 | 5 (push, x, v, θ, w) |

- Đồ thị: hte 14 vs 14 cạnh, **Jaccard 0,75, khác 4 cạnh** (chỉ DQN có
  `v→push`, `w→push`); ate 15 vs 16 cạnh, **Jaccard 0,6316, khác 7 cạnh**.
- `push` của DQN có cả 5 biến làm nguyên nhân, heuristic chỉ 3 — mạng DQN
  dùng toàn bộ state kể cả vận tốc, còn heuristic chỉ là hàm của vị trí và
  góc cột.
- Thời gian học cấu trúc: DQN chậm hơn ~19–20 lần (hte 1.756s vs 86s; ate
  2.306s vs 120s) do có ~15,8 lần transitions.
- **Kết luận để nói:** DQN hoàn hảo kém heuristic ở cả 3 thước đo fidelity.
  Giỏi hơn không đồng nghĩa dễ giải thích hơn — biên quyết định của mạng
  nơ-ron phi tuyến hơn luật tuyến tính. Lưu ý: số BART dao động ±1–2 cạnh
  giữa các lần chạy; tập valid steps khác nhau giữa 2 checkpoint nên không so
  trực tiếp số run 5 vs run 9 như cùng test set.

### 5.3. Ablation — Hướng A (1 phút, nói nhanh)

Lần đầu dùng **surrogate RandomForest thay BART** (ghi rõ, không phải số
BART); 6 run (50/100/200 ep × hte/ate), DT/LR 0,978–0,989. Đã **chạy
lại bằng BART thật** (R/dbarts 0.9.34): lưới 50/100/200 ep × hte/ate,
hte 12–13 cạnh, ate 15 cạnh; sensitivity 90/95/99% đủ cả hte/ate. Kết luận cũ từ surrogate — "hte ổn định từ 100 episode, Jaccard
1,0" — **không còn đứng**: BART thật cho Jaccard 0,79–0,86, trong dải
nhiễu. Surrogate cho `push: []` trong khi BART thật tìm được 3–4 nguyên
nhân cho `push`. Kết quả ở `muc4/results/ablation-bart/`; chi tiết xem
`muc4/results/REPORT.md`.


### 5.4. So sánh XRL LunarLander + thử DQN PongDuel (nhóm 2–3)

- **DQN LunarLander** (nhóm 2): train 873/1.000 episode (VM restart),
  checkpoint tốt nhất ở ep 850, eval cuối **222,03 ± 69,29** (100 ep).
  Heuristic cùng điều kiện **255,41 ± 74,32** — ở LunarLander heuristic
  mạnh hơn DQN, ngược với CartPole.
- **So sánh XRL LunarLander** (nhóm 3, BART thật, 30 ep/policy): fidelity
  weighted-F1 heuristic DT 0,7870 / LR 0,7808 / BART 0,8348 so với DQN DT
  0,6616 / LR 0,6167 / BART 0,6728. Đồ thị hte 48 vs 51 cạnh (Jaccard
  0,5231), ate 46 vs 45 (Jaccard 0,5690); thời gian học cấu trúc chỉ chậm
  hơn ~1,55 lần. Kết luận: đồ thị khác nhau rõ rệt, nhưng số BART có nhiễu
  (refit cùng dữ liệu cho Jaccard 0,65–0,75) nên chỉ kết luận "khác nhau",
  không đọc từng cạnh.
- **DQN PongDuel: âm tính, nói thẳng.** Joint action 0–8, cùng recipe run 9,
  dừng ở 800/2.000 episode do VM restart; eval cuối của checkpoint tốt nhất
  **5,44 ± 1,56** so với heuristic 8,81 ± 1,04 — không đạt tiêu chí thành
  công đã đặt trước khi train. Vì không có policy đã train nên **so sánh
  XRL PongDuel bỏ qua** (ước tính >70 giờ BART nếu làm đầy đủ). Đây là kết
  quả âm tính được ghi lại đầy đủ, không phải "chưa làm".

## 6. Hạn chế — nói thẳng (1 phút)

- Tác giả: cần biểu diễn trích sẵn; hiệu quả phụ thuộc chất lượng biểu diễn;
  nghiên cứu chuyên gia chỉ n=8, khám phá; không công bố đủ dữ liệu MicroRTS.
- Nhóm: CartPole thua DT ở paper (91,4% vs 94,1%) — đó là số của tác giả,
  không phải lỗi của nhóm; BART cần R/dbarts, cài khó, VM restart xóa R 2 lần
  trong ngày làm; số BART có nhiễu ±1–2 cạnh; ablation đã chạy lại bằng
  BART thật, kết luận cũ từ surrogate không còn đứng; Hướng D đã làm tới
  LunarLander (so sánh XRL xong) và thử PongDuel (DQN âm tính, so sánh XRL
  bỏ qua có lý do); fidelity MicroRTS theo protocol vẫn chưa đo được
  (đã chẩn đoán tới cấp cơ chế).
- Repo tác giả có bug đã tìm thấy: `counterfactual()` gọi `self.pi.predict`
  trong khi `pi` là hàm; `HTE` assert hardcode 500; `infer_causes_ate_barl`
  import module không tồn tại; MicroRTS import `stratvis` sót (nhóm đã sửa
  22 chỗ trong 8 file khi chạy tối thiểu).

## 7. Câu hỏi dự kiến + đáp án ngắn

1. **"Sao không train RL?"** — Repo tác giả không có code train, policy là
   heuristic viết tay. Nhóm đã train DQN thật ở Mục 4 (run 9 đạt 500/500) và
   giải thích nó bằng CausalXRL — đó chính là mở rộng.
2. **"Sao CartPole thua DT?"** — Đó là số của tác giả (91,4% vs 94,1%), tác
   giả thừa nhận chỉ "competitive". Môi trường đơn giản thì cây quyết định đủ
   tốt; CausalXRL thắng ở môi trường phức tạp (PongDuel 90,2%, MicroRTS
   86,0%).
3. **"BART cần R thì cài thế nào?"** — Cài R 4.3.3 + `dbarts` 0.9.34 qua CRAN
   + `rpy2`; chi tiết trong `muc4/notes/bart-check.md`. VM restart xóa R 2
   lần, khôi phục bằng `dpkg -i /var/cache/apt/archives/*.deb` rồi cài lại
   dbarts.
4. **"MicroRTS chạy tới đâu?"** — Nhóm đã chạy tối thiểu thật: sửa 22 chỗ
   import `stratvis` → `cxrl_microRTS` trong 8 file, lấy checkpoint PPO
   `agent-3.pt` công khai (không đưa vào repo), env thật chạy được,
   100 episode / 4.924 transition, đồ thị toàn cục 23 cạnh, dashboard đã
   sửa xong cả 9 route đều 200 (kể cả `/get_local_causal`). Chưa có
   fidelity MicroRTS theo Table 1 — nhóm 3 đã chẩn đoán: tập valid rỗng
   là bản chất dữ liệu + cơ chế HTE (BART cho `action` chỉ có 1 nguyên
   nhân toàn cục nên ngưỡng >1 cờ không thể đạt được), không sửa được
   bằng cấu hình — ghi rõ trong báo cáo.
5. **"Đồ thị học được có đúng là nhân quả không?"** — Không chứng minh tuyệt
   đối. Bằng chứng: dự đoán hành vi tốt (fidelity), học nhanh hơn PC/FCI,
   chuyên gia thấy hợp lý (8/8 vs 4/8), và đồ thị DQN vs heuristic khác nhau
   đúng theo bản chất policy (DQN dùng vận tốc, heuristic không). Số BART có
   nhiễu ±1–2 cạnh giữa các lần chạy.
6. **"Sao DQN 500/500 mà fidelity lại kém heuristic?"** — Vì biên quyết định
   của mạng nơ-ron phi tuyến, mô hình tuyến tính và BART theo trọng số ITE
   bắt chước kém hơn luật tuyến tính. Giỏi hơn không đồng nghĩa dễ giải thích
   hơn — đây chính là điểm hay nhất của mở rộng.
7. **"Ablation dùng surrogate thì có tin được không?"** — Surrogate chỉ để
   kiểm tra độ ổn định, không phải số CausalXRL/BART. Nhóm đã chạy lại
   bằng BART thật (R/dbarts 0.9.34): kết luận cũ "hte ổn định từ 100
   episode, Jaccard 1,0" không còn đứng (BART thật Jaccard 0,79–0,86,
   trong dải nhiễu); surrogate cho `push: []` trong khi BART tìm được
   3–4 nguyên nhân. Số chốt là số BART thật ở
   `muc4/results/ablation-bart/`.
8. **"Sao chọn checkpoint theo eval mà không lấy cuối train?"** — Vì
   Q-instability: eval run 9 dao động 500 ↔ ~97 giữa các lần đo cách nhau 50
   episode. Lấy cuối train có thể vớ phải policy đã bị phá hỏng.

9. **"GES là gì, sao giờ mới chạy?"** — GES là baseline học cấu trúc thứ ba
   paper nhắc tới (cùng PC/FCI) nhưng Table 1 không có cột GES. Nhóm 3 chạy
   GES đủ 4 môi trường bằng causal-learn, wall-clock thật: CartPole 7 cạnh
   (0,018s), LunarLander 18 cạnh (0,090s), PongDuel 34 cạnh (0,346s),
   MicroRTS 50 cạnh (0,563s). Không so trực tiếp với thời gian của tác giả
   vì thời gian đó là timer mô phỏng.
10. **"DQN PongDuel thất bại thì có sao không?"** — Không sao nếu nói thẳng:
    recipe DQN đã thắng CartPole (500/500) không học được gì trên joint
    action 9 class của PongDuel trong 800 episode (5,44 vs heuristic 8,81).
    Nhóm đặt tiêu chí thành công trước khi train, dừng khi không đạt, ghi
    đầy đủ log, và bỏ qua bước so sánh XRL PongDuel vì không có policy đã
    train để giải thích. Kết quả âm tính có bằng chứng vẫn là kết quả.
