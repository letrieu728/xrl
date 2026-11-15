# Tiêu chí thành công đăng ký TRƯỚC khi chạy — Counterfactual cho policy DQN CartPole

Ngày giờ ghi: 2026-11-15 22:45 +07 (trước khi chạy DiCE cho DQN; checkpoint
đã kiểm chứng xong: eval 500,00/500, xem `counterfactual-dqn-thu-nghiem.md` §2).

## Thiết lập đã chốt trước khi chạy

- Policy: DQN run 9 (MLP 4→128→128→2, Double DQN), checkpoint
  `muc4/results/cartpole-dqn/dqn_checkpoint.pt` trên Drive "XRL - file nhi phan"
  (md5 `a6fa113cae140c50f1d8e469eb77caac`, 73.833 byte).
- Đường chạy: đúng code `XRL.counterfactual()` đã sửa bằng patch bước 3 +
  patch đa lớp (bản copy `causal-xrl-copy` đã kiểm chứng), giống hệt bản
  heuristic: env `gym` CartPole-v0, seed 99, 200 episode, `repr_names` như
  bản heuristic, BART thật qua R/dbarts, DiCE method `random`,
  `total_CFs=40` (lời gọi gốc của tác giả, không sửa).
- Query: 6 hàng replay [6, 100, 500, 1000, 2000, 3000] — cùng chỉ số hàng
  như bản heuristic (nội dung hàng khác vì replay của policy DQN khác).
  Action mong muốn = 1 − action quan sát. Trước mỗi query đặt lại
  `random.seed(99)` + `np.random.seed(99)` như bản heuristic.
- Kiểm chứng: MỌI counterfactual cuối đều kiểm chứng lại bằng forward
  trực tiếp mạng DQN (argmax Q), không tin kết quả DiCE. Một CF chỉ được
  tính là "lật đúng" khi action do mạng DQN trả về cho state CF bằng
  action mong muốn VÀ cột action của CF cũng bằng action mong muốn
  (cùng quy ước như bản heuristic).

## Tiêu chí thành công (đặt trước, không sửa sau khi thấy kết quả)

- **C1 (chính): ≥ 5/6 query (≥ 80%) có ≥ 1 counterfactual cuối được kiểm
  chứng lật đúng action.** Đạt C1 = thí nghiệm thành công; không đạt = báo
  đúng là không đạt, không hạ tiêu chí sau khi chạy.
- **C2 (báo cáo, không đặt ngưỡng đạt/rớt):** tỉ lệ CF cuối kiểm chứng lật
  đúng trên tổng số CF cuối, và số CF cuối trung bình mỗi query. Chỉ báo
  số, không gán nhãn thành công/thất bại từ C2.
- **C3 (tính toàn vẹn):** mọi query (kể cả query 0 CF hoặc 0 CF lật đúng)
  đều xuất hiện trong JSON kết quả; mọi CF cuối (kể cả CF không lật) đều
  được liệt kê, không loại bỏ CF thất bại khỏi báo cáo.

## So sánh vs counterfactual heuristic (đặt trước các chỉ số)

Baseline heuristic (remote `muc4/counterfactual_cartpole_multi.json`,
đã đọc trước khi chạy): 6 query, 9 CF cuối, 9/9 kiểm chứng lật đúng,
mỗi CF chỉ sửa đúng 1 đặc trưng (chỉ `cart_position`/`pole_angle`).

So sánh đúng 4 chỉ số sau, tính từ JSON của cả hai bản bằng cùng một
script đo (số feature bị sửa, độ lớn thay đổi chuẩn hoá theo độ lệch
chuẩn của từng feature trong replay tương ứng, feature hay bị sửa nhất,
tỉ lệ lật). Nếu DQN khó lật hơn thì báo đúng như vậy, không phóng đại
theo hướng nào.

## Dự phòng đăng ký trước

Nếu dựng `XRL` với 200 episode quá chậm (> 30 phút) hoặc sập do BART/bộ
nhớ (replay DQN có thể tới ~40.000 transition vì mỗi episode chạm trần
200 bước của CartPole-v0, so với 6.115 của heuristic), thì giảm số
episode (vẫn đi đường `XRL.counterfactual()` đầy đủ) và ghi rõ số
episode/số transition thực tế đã dùng trong JSON + báo cáo. Không đổi
policy, seed, query rows, hay phương pháp DiCE. Mọi lần chạy (kể cả lần
sập) đều giữ log.
