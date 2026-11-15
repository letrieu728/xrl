# Counterfactual cho policy DQN CartPole (run 9) — kiểm chứng bằng forward mạng

Ngày chạy: 2026-11-15. Đây là phần việc còn thiếu trong §7 của
`muc4/counterfactual-thu-nghiem.md`: counterfactual cho policy DQN đã
train của Mục 4 (trước đây chỉ có heuristic/PD). Code gốc của tác giả
đã được sửa bằng patch bước 3 (`muc4/xrl-counterfactual-fix.patch`) +
patch đa lớp (`muc4/xrl-multiclass-fix.patch`); **không cần thêm patch
nào cho DQN** — `DummySklearn` chỉ cần `pi(state) -> action`, và policy
DQN là hàm thuần như vậy. Lời gọi `counterfactual()` của tác giả không
bị sửa. Tiêu chí thành công đã đăng ký TRƯỚC khi chạy:
`muc4/counterfactual-dqn-criteria.md`. Kết quả đầy đủ:
`muc4/counterfactual_dqn_cartpole.json` + `.csv`. Script chạy:
`muc4/scripts/run_counterfactual_dqn_cartpole.py`. Log đầy đủ:
`muc4/counterfactual-dqn-cartpole-log.txt`. So sánh với bản heuristic:
`muc4/counterfactual_dqn_comparison.json`, tính bằng
`muc4/scripts/compare_counterfactual_dqn_vs_heuristic.py`.

## 1. Thiết lập chạy

- Policy: DQN run 9 (Double DQN, MLP 4→128→128→2, seed 0, 2.000 episode,
  eps-decay 1.300, checkpoint tốt nhất theo eval định kỳ @ episode 1150).
  Checkpoint `muc4/results/cartpole-dqn/dqn_checkpoint.pt` tải từ Drive
  "XRL - file nhi phan" (73.833 byte, md5
  `a6fa113cae140c50f1d8e469eb77caac`); file `.pt` không commit vào repo.
  Load đúng kiến trúc trong `muc4/scripts/cartpole_policies.py`
  (`QNetwork`, `load_dqn_policy`); các shape state_dict kiểm tra khớp:
  (128,4), (128,128), (2,128).
- Env: `gym` CartPole-v0 (gym 0.21.0), seed **99** (`env.seed(99)` +
  `np.random.seed(99)`), 200 episode — đúng thiết lập bản heuristic.
  Lưu ý: DQN được train trên gymnasium CartPole-v1; ở đây policy chỉ
  đóng vai hàm `pi(state) -> action` trong rollout thu replay của
  CartPole-v0 (động học như nhau; v0 trần 200 bước/episode, v1 trần 500).
  Việc kiểm chứng checkpoint (§2) dùng đúng CartPole-v1 như lúc train.
- `repr_names` như bản heuristic, `repr_integration=False`, BART thật
  qua R/dbarts 0.9.34 (`R_LIBS=~/workspace/R-lib`). DiCE method
  `random`, `total_CFs=40` (lời gọi gốc). Trước mỗi query đặt lại
  `random.seed(99)` + `np.random.seed(99)` như bản heuristic.
- Query: hàng replay [6, 100, 500, 1000, 2000, 3000] — cùng chỉ số hàng
  như bản heuristic; nội dung hàng khác vì replay của policy DQN khác.
  Action mong muốn = 1 − action quan sát.
- Kiểm chứng: MỌI counterfactual cuối đều kiểm chứng lại bằng forward
  trực tiếp mạng DQN (argmax Q), kèm giá trị Q và biên quyết định
  (margin) của query và của CF. Một CF chỉ tính là "lật đúng" khi action
  do mạng trả về bằng action mong muốn VÀ cột action của CF cũng bằng
  action mong muốn (cùng quy ước như bản heuristic).

## 2. Kiểm chứng checkpoint trước khi chạy (cổng kiểm tra — đạt mới chạy)

Trước khi làm counterfactual, checkpoint đã tải về được load và đánh giá
độc lập bằng đúng protocol eval cuối của script train (log:
`muc4/counterfactual-dqn-checkpoint-verify-log.txt`):

- Load được; metadata trong checkpoint khớp run 9 (`best_eval_episode`
  1150, best eval 500,0/50 episode; hyperparams đúng train_summary.json).
- Eval greedy 100 episode, CartPole-v1, seed 10000..10099:
  **500,00 ± 0,00 (min 500, max 500; 100/100 episode đạt 500)** — khớp
  đúng kết quả run 9 đã báo cáo. Cổng kiểm tra đạt, mới chạy DiCE.

## 3. Kết quả

`XRL` dựng xong trong **1.400,6 giây** (~23,3 phút; dưới ngưỡng dự
phòng 30 phút đã đăng ký trước, nên không cần giảm episode); replay
**39.800 transition** (action 0: 20.011, action 1: 19.789) — đúng
200 episode × 199 transition (hàm `play()` của tác giả bỏ transition
cuối mỗi episode; mọi episode đều chạm trần 200 bước của CartPole-v0,
đúng với một policy giải hoàn hảo). Đồ thị nhân quả toàn cục học kèm
có 13 cạnh (ghi trong JSON; đồ thị không được dùng bởi
`counterfactual()`, chỉ ghi nhận). Kiểm tra nhất quán: action trong
replay khớp policy DQN trên 200/200 state lấy mẫu.

| Hàng replay | Quan sát → mong muốn | Biên Q của query | DiCE thô | Còn lại | Kiểm chứng lật | Thời gian |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 → 1 | 0,214 | 40 | 6 | 6/6 | 1,03 s |
| 100 | 1 → 0 | 0,384 | 40 | 6 | 6/6 | 1,48 s |
| 500 | 0 → 1 | 0,298 | 40 | 5 | 5/5 | 0,48 s |
| 1000 | 1 → 0 | 15,446 | 40 | 6 | 6/6 | 0,32 s |
| 2000 | 0 → 1 | 0,287 | 40 | 6 | 6/6 | 0,60 s |
| 3000 | 1 → 0 | 0,606 | 40 | 7 | 7/7 | 0,71 s |

Tổng: **36/36 counterfactual cuối lật đúng action** khi kiểm chứng bằng
forward mạng DQN.

Đối chiếu tiêu chí đăng ký trước (`counterfactual-dqn-criteria.md`):

- **C1 (chính): ĐẠT — 6/6 query (≥ 5/6 yêu cầu) có ≥ 1 CF kiểm chứng
  lật đúng.**
- C2: tỉ lệ CF cuối kiểm chứng lật đúng = 36/36 = 100%; trung bình
  6,0 CF cuối mỗi query (heuristic: 1,5).
- C3: cả 6 query và cả 36 CF cuối đều có trong JSON/CSV, không loại bỏ
  CF nào (cũng không có CF nào thất bại để loại).

Kiểm chứng độc lập lần 2: toàn bộ 36 CF + 6 query được chạy lại bằng
một venv/torch khác (dqn-venv, torch 2.15, khác với torch 2.5.1 lúc
chạy chính) — **36/36 action của CF và 6/6 action của query khớp đúng**
với kết quả đã ghi; tính nhất quán của nhãn `flips` đúng 36/36.

Điểm đáng chú ý: 5/6 query có biên Q rất nhỏ (0,21–0,61) — các state
trên quỹ đạo của policy cân bằng nằm gần biên quyết định, Q hai action
gần bằng nhau. Query hàng 1000 là ngoại lệ (biên 15,45, policy rất tự
tin); nó vẫn lật được, nhưng cần thay đổi lớn nhất (xem §4). Biên của
các CF (Q_mong_muốn − Q_quan_sát): trung vị 0,35, nhỏ nhất 0,001 — đa
số CF chỉ vừa qua biên, đúng bản chất counterfactual tối thiểu; có 1
CF vượt biên rất xa (biên 65,84, hàng 6, sửa `pole_velocity` +1,40).

## 4. So sánh với counterfactual heuristic (cùng 4 chỉ số đăng ký trước)

Baseline: `muc4/counterfactual_cartpole_multi.json` trên remote (6
query, 9 CF cuối, 9/9 lật đúng). Mọi chỉ số dưới đây tính bằng cùng một
script cho cả hai bản. Độ lệch chuẩn (std) của replay heuristic được
lấy bằng cách tái tạo replay bằng đúng hàm `play()` của tác giả (không
BART): tái tạo được đúng 6.115 transition, action 0: 3.139 / 1: 2.976 —
khớp hoàn toàn số đã ghi, nên std dùng cho chuẩn hoá là đáng tin.
Std replay: DQN (0,366; 0,457; 0,060; 0,317), heuristic (0,082; 0,927;
0,099; 1,192) theo thứ tự 4 đặc trưng — hai replay có phân bố state
khác nhau rõ rệt (DQN giữ xe gần tâm, vận tốc nhỏ; heuristic trôi rộng
hơn), vì vậy ngoài L1 chuẩn hoá (mỗi bản chuẩn hoá theo std của chính
nó, như đã đăng ký) còn báo thêm độ lớn thay đổi thô cùng đơn vị để
đối chiếu; cả hai cách đều cho cùng kết luận.

| Chỉ số | Heuristic | DQN run 9 |
|---|---|---|
| Query có ≥ 1 CF lật đúng | 6/6 | 6/6 |
| CF cuối kiểm chứng lật đúng | 9/9 (100%) | 36/36 (100%) |
| CF cuối trung bình / query | 1,5 | 6,0 |
| Số đặc trưng bị sửa / CF (trung bình; trung vị) | 1,0; 1,0 (9/9 CF sửa đúng 1) | 1,28; 1,0 (26 CF sửa 1, 10 CF sửa 2, 0 CF sửa ≥ 3) |
| L1 chuẩn hoá / CF (trung bình; trung vị; min–max) | 0,46; 0,37; 0,09–0,94 | 2,61; 1,70; 0,13–9,94 |
| Đặc trưng hay bị sửa nhất (số CF) | `pole_angle` 6, `cart_position` 3; vận tốc 0 | `pole_velocity` 16, `cart_velocity` 12, `cart_position` 11, `pole_angle` 7 |

Độ lớn thay đổi thô (trung vị |delta| trong số CF có sửa đặc trưng đó):

| Đặc trưng | Heuristic | DQN |
|---|---|---:|
| `cart_position` | 0,026 (n=3) | 0,318 (n=11) |
| `cart_velocity` | — (không CF nào sửa) | 0,756 (n=12) |
| `pole_angle` | 0,055 (n=6) | 0,070 (n=7) |
| `pole_velocity` | — (không CF nào sửa) | 0,949 (n=16) |

Đọc đúng, không phóng đại:

- **Tỉ lệ lật bằng nhau (100% cả hai), nhưng DQN khó lật hơn về độ
  lớn:** trung vị L1 chuẩn hoá của DQN gấp ~4,6 lần heuristic (1,70 so
  với 0,37), trung bình gấp ~5,6 lần; thay đổi thô cũng lớn hơn hẳn ở
  `cart_position` (~12× trung vị) và hai vận tốc (heuristic không bao
  giờ cần sửa vận tốc). CF lớn nhất của DQN (hàng 1000, query tự tin
  nhất) sửa `cart_position` +0,81 và `pole_velocity` −2,45, L1 chuẩn
  hoá 9,94 — gấp hơn 10 lần CF lớn nhất của heuristic.
- **DQN dùng cả 4 đặc trưng, heuristic chỉ dùng 2:** đúng với bản chất
  hai policy — heuristic là hàm tuyến tính chỉ của `cart_position` và
  `pole_angle`, nên counterfactual của nó không bao giờ sửa vận tốc;
  mạng DQN phi tuyến phụ thuộc cả 4 đặc trưng, và hai vận tốc (nhất là
  `pole_velocity`, 16/36 CF) mới là đặc trưng hay bị sửa nhất. Score
  của `counterfactual()` cũng phản ánh đúng điều này (cả 4 đặc trưng
  đều có score ≠ 0 ở hầu hết query DQN, so với chỉ 2 đặc trưng ở
  heuristic).
- **Số CF cuối nhiều hơn hẳn (36 so với 9)** vì bước simplify/dominating
  giữ lại nhiều CF không trội lẫn nhau hơn khi biên quyết định phi
  tuyến; đây là số lượng sau lọc, không phải chất lượng — cả hai bản
  đều xuất phát từ 40 CF thô của DiCE mỗi query.
- Không có bằng chứng DQN "không lật được": với 6 query này, mọi query
  đều lật được. Kết luận chỉ là "cần thay đổi lớn hơn", không phải
  "không lật được".

## 5. Kết luận trung thực

- `counterfactual()` đã sửa (patch bước 3 + patch đa lớp) chạy được
  đầu-cuối cho policy DQN đã train mà không cần sửa thêm code: 6 query,
  36/36 CF kiểm chứng lật đúng bằng forward mạng, tiêu chí C1 đăng ký
  trước đạt 6/6.
- Khác biệt chính so với heuristic không nằm ở tỉ lệ lật (cả hai 100%)
  mà ở độ lớn thay đổi cần thiết (DQN lớn hơn nhiều) và đặc trưng bị
  sửa (DQN cả 4, heuristic chỉ 2) — phù hợp với biên quyết định phi
  tuyến của mạng so với biên tuyến tính của heuristic.
- Giới hạn: chỉ 6 query CartPole, 1 checkpoint (run 9), DiCE phương
  pháp `random` duy nhất; chưa thử LunarLander/PongDuel hay policy DQN
  seed 1; các query là state trên quỹ đạo (on-policy), phần lớn gần
  biên quyết định, nên tỉ lệ lật 100% không suy ra được cho state xa
  quỹ đạo; đồ thị nhân quả học kèm (13 cạnh) không được phân tích hay
  dùng cho kết luận nào ở đây; không có đánh giá con người về tính
  khả thi/thực tế của các CF (một số CF thay đổi vận tốc lớn có thể
  không ứng với state đạt được trong thực tế — không kiểm tra tính
  đạt được này).

## 6. Tái lập

```sh
# venv bước 3 (gym 0.21.0, dice-ml 0.7.2, torch CPU) + R/dbarts 0.9.34
R_LIBS=~/workspace/R-lib <venv>/bin/python \
    muc4/scripts/run_counterfactual_dqn_cartpole.py \
    --checkpoint <duong-dan-toi/dqn_checkpoint.pt> \
    --outdir <outdir>
# --checkpoint: file .pt tải từ Drive (không có trong repo).
# Script import cartpole_policies.py cùng thư mục muc4/scripts/.
# Dựng XRL mất ~23 phút CPU (replay 39.800 transition); mỗi query < 2 s.
```

Trong lần chạy này, script được chạy từ thư mục staging (không chứa
`cartpole_policies.py`) nên `PYTHONPATH` trỏ tới bản sao nội dung file
đó lấy từ remote; kiến trúc mạng và cách load checkpoint là một. So
sánh §4 tái lập bằng:

```sh
<venv>/bin/python muc4/scripts/compare_counterfactual_dqn_vs_heuristic.py \
    --dqn-json muc4/counterfactual_dqn_cartpole.json \
    --heur-json muc4/counterfactual_cartpole_multi.json \
    --out muc4/counterfactual_dqn_comparison.json
```

Chi phí: $0 (toàn bộ CPU máy ảo local, không Modal, không API trả phí).
Không push GitHub, không động Drive trong nhiệm vụ này (chỉ tải
checkpoint về).
