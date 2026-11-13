# Bước 6 — So sánh XRL cho PongDuel: BỎ QUA

**Quyết định: BỎ QUA, không chạy.** Hai lý do độc lập, lý do (1) đã đủ theo đúng quy tắc nhiệm vụ:

## (1) Lý do quyết định: bước 5 thất bại — không có policy đã train để giải thích

Nhiệm vụ quy định bước 6 "CHỈ làm nếu bước 5 thành công". Bước 5 KHÔNG thành công: DQN PongDuel tốt nhất đạt 5,44/10, thua heuristic 8,81 và dưới xa ngưỡng 9,30 (xem `train_summary.md`). Không có policy đã train nào đáng đưa vào so sánh XRL. Dừng ở đây là đúng quy tắc, không phải thiếu nỗ lực.

## (2) Lý do supporting: ước tính thời gian vượt quá khả thi ngay cả khi bước 5 thành công

Ước tính từ dữ liệu đã biết (không chạy mù, không số bịa):

**Mốc neo — Mục 3 PongDuel (heuristic, đã đo thật):** 200 episode, 74.445 transitions (mean 373,2 bước/ep), 21 mục tiêu (20 state + 1 `encoded_action` số, theo notebook — `one_hot_actions=False`), BART HTE tổng 12 process riêng = **35.531 s (~9,9 giờ) cho MỘT policy, MỘT phương pháp (HTE)**. Fidelity Mục 3 là proxy số (không phải protocol chuẩn của paper) và đã mất thêm 1.089 s.

**So sánh XRL đầy đủ cần:** đồ thị HTE + ATE × 2 policy (heuristic + DQN) + refit HTE cho fidelity × 2 policy + fit BART fidelity + vòng ITE. Tỉ lệ thời gian ATE/HTE đã đo ở 2 môi trường trước: CartPole 1,31–1,40×; LunarLander 1,51–1,52× → lấy 1,5×.

| Hạng mục (quy mô 200 ep như Mục 3) | Ước tính |
|---|---|
| Đồ thị HTE, 2 policy | 2 × 35.531 s ≈ 19,7 giờ |
| Đồ thị ATE, 2 policy (≈1,5× HTE) | ≈ 29,6 giờ |
| Refit HTE cho fidelity, 2 policy | ≈ 19,7 giờ |
| Fit BART fidelity + vòng ITE (test set ~30k hàng) | nhiều giờ nữa (không ước lượng chính xác được; riêng proxy Mục 3 đã 1.089 s cho 1 policy) |
| **Tổng (quy mô đầy đủ)** | **> 70 giờ BART liên tục** |

**Kể cả với hệ số nhanh của VM này:** VM này chạy HTE LunarLander nhanh hơn ~2,5 lần máy của Mục 3 (36 s vs 90 s/mục tiêu). Áp dụng hào phóng hệ số 2,5× cho toàn bộ: tổng vẫn ≈ **28 giờ** — vượt xa ngưỡng 12–15 giờ cho một lần chạy liên tục có checkpoint. Ở quy mô rút xuống 30 episode (như LunarLander): đồ thị ≈ 7,4 giờ (chưa chia hệ số nhanh) / ≈ 3 giờ (có hệ số), cộng fidelity refit + fit/ITE ≈ 5–6 giờ — nhưng (a) hệ số nhanh ngoại suy từ LunarLander, chưa đo trên PongDuel; (b) rủi ro OOM chưa kiểm chứng ở 74k hàng × 21 cột (CartPole DQN đã lên ~4,5 GB RAM ở ~100k hàng × 5 cột); (c) **không có protocol fidelity chuẩn cho PongDuel trong repo** (không có eval notebook PongDuel; Mục 3 phải dùng proxy số và tự ghi "not_exact_paper_fidelity", macro-F1 proxy chỉ 0,195); (d) one-hot 9 action sẽ thành 29 mục tiêu, còn lâu hơn nữa. VM đã restart 3 lần trong ngày 12/11 (lần cuối còn xóa cả R khỏi hệ thống — `R: command not found` lúc 17:54, phải cài lại + build dbarts mới chạy được BART), nên một chuỗi BART 5–28 giờ gần như chắc chắn bị ngắt nhiều lần; checkpoint từng target cứu được đồ thị nhưng fidelity refit phải làm lại từ đầu mỗi lần.

**Kết luận:** bước 6 bỏ qua theo quy tắc (bước 5 thất bại) và cũng không khả thi ở quy mô đầy đủ trong một lần chạy có checkpoint. Không số BART PongDuel nào được bịa ra hay chạy dở dang để "có gì đó" — thư mục `xrl-trained-vs-heuristic-pongduel/` không được tạo.
