# DQN CartPole — runs 4–6 (Hướng D, Mục 4 CausalXRL)

**Mục tiêu:** eval greedy 100 episode đạt 450–500 (seed_offset 10000, như `evaluate()` trong `scripts/train_dqn_cartpole.py`).
**Điểm xuất phát:** run1 (600 ep, lr 1e-3, MSE, eps 1.0→0.02): train mean last100 264,49; **eval 246,16 ± 10,89** (min 230, max 282). Heuristic repo cùng điều kiện: 31,53 ± 11,16.
**Venv:** `~/workspace/dqn-venv` (torch 2.15.0+cpu, gymnasium 1.4.0, Python 3.12.3, CPU-only, $0). Seed 0 cho mọi run.

## Run 4 — Huber + lr 5e-3 + grad-clip 10.0 (600 ep) — THẤT BẠI

- Lệnh: `--seed 0 --episodes 600 --lr 0.005 --loss huber --grad-clip 10.0`, các hyper khác như run1 (eps decay 400 ep, target update 10 ep, batch 64, buffer 50000, warmup 1000).
- Kết quả: train mean last100 **79,68**; **eval 56,81 ± 7,94** (min 40, max 72). Thời gian 513,8 s.
- Diễn biến: mean last100 chỉ 50,3 ở ep 400, đỉnh 91,9 ở ep 550 rồi giảm còn 79,7 ở ep 600. Loss Huber tăng dần 0,0001 → 0,64.
- Kết luận: Huber + lr cao học chậm hơn hẳn MSE lr 1e-3 (run1) lẫn Huber lr 1e-3 + clip 1.0 (run3: eval 86,77). Giả thuyết: gradient của Huber bị chặn ở 1 với sai số lớn nên tín hiệu học yếu, trong khi lr 5e-3 gây dao động — kết hợp tệ cả hai chiều. Không dùng Huber nữa với bài này.
- File: `results/cartpole-dqn/run4/` (training_log.csv, train_summary.json, dqn_checkpoint.pt, train_run.log, reward_curve.png).

## Run 5 — Double DQN, lr 1e-3, MSE (600 ep) — TỐT NHẤT MỚI, VẪN CHƯA ĐẠT 450

- Thay đổi duy nhất so với run1: thêm flag `--double-dqn` vào `scripts/train_dqn_cartpole.py` — target = `r + gamma * Q_target(s', argmax_a Q_policy(s', a)) * (1-done)` (chọn action bằng policy net, đánh giá bằng target net). Giữ lr 1e-3, MSE, grad-clip off, 600 ep, seed 0.
- Kết quả: train mean last100 **163,28**; **eval 272,68 ± 58,53** (min 203, max 500). Thời gian 1159,6 s.
- **Eval 272,68 > run1 (246,16): đây là checkpoint tốt nhất tính đến nay** (+26,52 so với run1; +241,15 so với heuristic 31,53). Lưu ý: eval (272,68) cao hơn hẳn train last100 (163,28) — greedy không khám phá làm tốt hơn nhiều; std lớn (58,53) và max = 500 cho thấy policy đã giải được một số seed nhưng chưa ổn định (phân phối gần như hai cực: một số episode 500, phần còn lại ~200).
- Diễn biến: mean last100 tăng chậm nhưng đều (86,2 ở ep 400 → 125,1 ở ep 450 → 135,9 ở ep 500 → 119,9 ở ep 550 → 163,3 ở ep 600, reward ep 600 = 347). Đường học chưa bão hòa ở ep 600 — lý do chọn tăng episode cho run6.
- Sự cố: lần chạy đầu bị runtime restart của VM giết ở ep ~424 (không checkpoint, không summary — chỉ còn training_log.csv dở). Đã chạy lại từ đầu cùng seed/lệnh; lần chạy lại cho eval 272,68 (lần chạy dở dang ở ep 424 có mean last100 ~101, tương đương quỹ đạo lần chạy lại).
- File: `results/cartpole-dqn/run5/` (training_log.csv, train_summary.json, dqn_checkpoint.pt, train_run.log, reward_curve.png).

## Run 6 — Double DQN + 1200 ep — DỪNG GIỮA CHỪNG THEO YÊU CẦU CHỐT NHANH (không có eval)

- Thay đổi duy nhất so với run5: tăng episode 600 → 1200, giữ nguyên Double DQN + lr 1e-3 + MSE + eps decay 400 ep + target update 10 ep + seed 0.
- Lý do chọn: run5 vẫn đang cải thiện ở cuối train (chưa bão hòa) và eval đã chạm 500 ở một số seed; thêm thời gian học — đặc biệt 800 episode ở eps = 0,02 (khai thác) thay vì 200 — là cách trực tiếp nhất để policy hội tụ. Không chọn eps decay chậm hơn (sẽ trì hoãn chính giai đoạn khai thác đang cho kết quả) hay target update 5 ep (tăng nguy cơ Q-instability — đúng vấn đề làm run1 giảm cuối train).
- **Dừng lúc 13:21 ngày 11/11/2026 theo yêu cầu chốt nhanh của owner (không chạy thêm run/tweak mới). Process đã kill; dừng ở episode 475/1200** (training_log.csv 476 dòng gồm header; log cuối ep 450, mean last100 134,0; mean last100 ở ep 474 ≈ 112,0, max 50 ep cuối 342). **Không có train_summary.json, không có checkpoint, KHÔNG CÓ EVAL cho run6 — không bịa số eval.**
- File: `results/cartpole-dqn/run6/` (chỉ training_log.csv + train_run.log, dở dang).

## Chốt checkpoint tốt nhất — run5 (Double DQN)

- **Best = run5: eval greedy 100 ep 272,68 ± 58,53 (min 203, max 500), train mean last100 163,28.**
- Đã copy `run5/dqn_checkpoint.pt` → `results/cartpole-dqn/dqn_checkpoint.pt` (md5 950d4bcfd82ff675ca14fee5f096438b, khớp bản run5), **đè lên bản run1 cũ**; bản run1 còn nguyên trong `run1/dqn_checkpoint.pt` (md5 fa8d467293f5dc40a52f37c2e0c88504). Đồng bộ kèm theo ở top-level `results/cartpole-dqn/`: `training_log.csv`, `train_summary.json` (thêm trường `best_run: run5` + `checkpoint_source`), `train_run.log`, `reward_curve.png`, `eval_log.txt` (eval run5 + heuristic 31,53, chênh +241,15) — tất cả đều là bản run5.
- Smoke test sau copy: `scripts/cartpole_policies.py` load được checkpoint top-level, 5-episode rewards = [321, 237, 263, 436, 209] — SMOKE TEST PASSED.
- **Kết luận trung thực: KHÔNG đạt mục tiêu 450–500.** Best eval 272,68 < 450. Không được ghi 450+ trong báo cáo. DQN vẫn rõ ràng tốt hơn heuristic repo (+241,15).

## Bảng tổng hợp (eval greedy 100 ep, seed_offset 10000)

| run | thay đổi chính | train last100 | eval mean ± std | min–max | đạt 450–500? |
|---|---|---|---|---|---|
| run1 | baseline (MSE, lr 1e-3) | 264,49 | 246,16 ± 10,89 | 230–282 | Không |
| run3 | Huber lr 1e-3 + clip 1.0 (800 ep) | 134,74 | 86,77 ± 31,63 | 33–118 | Không |
| run4 | Huber lr 5e-3 + clip 10.0 | 79,68 | 56,81 ± 7,94 | 40–72 | Không |
| run5 | Double DQN, lr 1e-3, MSE | 163,28 | 272,68 ± 58,53 | 203–500 | Không — **BEST, đã chốt checkpoint** |
| run6 | Double DQN + 1200 ep | — (dừng ep 475, last100 ~112) | **không có eval (dừng giữa chừng)** | — | — |
| heuristic repo | — | — | 31,53 ± 11,16 | 11–63 | — |
