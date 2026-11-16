# Thuyết trình CausalXRL — Bản thông tin (15–20 phút)

Bài: **CausalXRL: Explainable Reinforcement Learning through Causal Graph
Reasoning** — Yanming Zhang, Eric Papenhausen, Klaus Mueller (Stony Brook),
arXiv:2610.42028v1, ICML 2026.
Repo nhóm: https://github.com/letrieu728/XRL (xem `README.md` và
`muc3/EXPERIMENT_REPORT.md` để đối chiếu mọi số dưới đây).

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
toàn cục **23 cạnh**, dashboard Flask đã sửa xong **9 lượt kiểm tra
trên 7 route đều 200** (kể cả `/get_local_causal`, sửa ở nhóm 3 ngày
12/11/2026). Fidelity
MicroRTS theo Table 1 vẫn chưa có — nhóm 3 đã chẩn đoán tới cấp cơ chế:
tập valid rỗng là bản chất dữ liệu + cơ chế HTE, không sửa được bằng cấu
hình. Chi tiết:
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

## 5. Hướng mở rộng mới (2 phút — chỉ nói hướng đi, chưa có kết quả)

Hướng mở rộng mới (đang lên kế hoạch, chưa có kết quả): Causal-Guided RL — đóng kín vòng lặp hành động → tự soi chiếu nhân quả → sửa sai, dùng đồ thị nhân quả ngay trong lúc train để cảnh báo/phạt quyết định tự hủy và thưởng cho chuỗi hành động dẫn tới kết quả tốt. Làm pilot nhỏ trước, đăng ký tiêu chí trước, báo cáo cả khi kết quả âm tính — theo bài học từ nhóm 4 (mô phỏng phẫu thuật đồ thị không đạt kiểm chứng), độ tin cậy của đồ thị phải được chứng minh, không được giả định.

## 6. Hạn chế — nói thẳng (1 phút)

- Tác giả: cần biểu diễn trích sẵn; hiệu quả phụ thuộc chất lượng biểu diễn;
  nghiên cứu chuyên gia chỉ n=8, khám phá; không công bố đủ dữ liệu MicroRTS.
- Nhóm: CartPole thua DT ở paper (91,4% vs 94,1%) — đó là số của tác giả,
  không phải lỗi của nhóm; BART cần R/dbarts, cài khó, VM restart xóa R 2 lần
  trong ngày làm; số BART có nhiễu ±1–2 cạnh; fidelity MicroRTS theo
  protocol vẫn chưa đo được (đã chẩn đoán tới cấp cơ chế).
- Repo tác giả có bug đã tìm thấy: `counterfactual()` gọi `self.pi.predict`
  trong khi `pi` là hàm; `HTE` assert hardcode 500; `infer_causes_ate_barl`
  import module không tồn tại; MicroRTS import `stratvis` sót (nhóm đã sửa
  22 chỗ trong 8 file khi chạy tối thiểu).

## 7. Câu hỏi dự kiến + đáp án ngắn

1. **"Sao không train RL?"** — Repo tác giả không có code train, policy là
   heuristic viết tay; nhóm tái hiện đúng như vậy (evaluation-only).
   Hướng mở rộng mới dùng đồ thị nhân quả ngay trong lúc train đang lên
   kế hoạch, chưa có kết quả.
2. **"Sao CartPole thua DT?"** — Đó là số của tác giả (91,4% vs 94,1%), tác
   giả thừa nhận chỉ "competitive". Môi trường đơn giản thì cây quyết định đủ
   tốt; CausalXRL thắng ở môi trường phức tạp (PongDuel 90,2%, MicroRTS
   86,0%).
3. **"BART cần R thì cài thế nào?"** — Cài R 4.3.3 + `dbarts` 0.9.34 qua CRAN
   + `rpy2` 3.5.16. VM restart xóa R 2 lần, khôi phục bằng
   `dpkg -i /var/cache/apt/archives/*.deb` rồi cài lại dbarts.
4. **"MicroRTS chạy tới đâu?"** — Nhóm đã chạy tối thiểu thật: sửa 22 chỗ
   import `stratvis` → `cxrl_microRTS` trong 8 file, lấy checkpoint PPO
   `agent-3.pt` công khai (không đưa vào repo), env thật chạy được,
   100 episode / 4.924 transition, đồ thị toàn cục 23 cạnh, dashboard đã
   sửa xong (9 lượt kiểm tra trên 7 route đều 200, kể cả
   `/get_local_causal`). Chưa có
   fidelity MicroRTS theo Table 1 — nhóm 3 đã chẩn đoán: tập valid rỗng
   là bản chất dữ liệu + cơ chế HTE (BART cho `action` chỉ có 1 nguyên
   nhân toàn cục nên ngưỡng >1 cờ không thể đạt được), không sửa được
   bằng cấu hình — ghi rõ trong báo cáo.
5. **"Đồ thị học được có đúng là nhân quả không?"** — Không chứng minh tuyệt
   đối. Bằng chứng tác giả đưa ra: dự đoán hành vi tốt (fidelity), học
   nhanh hơn PC/FCI (theo thời gian Table 1 — nhóm phát hiện thời gian đó
   đo bằng timer mô phỏng trong notebook của tác giả, không phải wall-clock
   BART thật, nên không dùng làm bằng chứng mạnh), chuyên gia thấy hợp lý
   (8/8 vs 4/8). Số BART có nhiễu ±1–2 cạnh
   giữa các lần chạy.
