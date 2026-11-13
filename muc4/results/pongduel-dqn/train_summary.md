# PongDuel DQN (bước 5) — KHÔNG THÀNH CÔNG (kết quả âm tính)

**Ngày làm:** 12/11/2026. **Trạng thái: KHÔNG THÀNH CÔNG — agent không học được gì có ý nghĩa trong 800 episode; dừng theo quy tắc đã chốt trước khi train, không chạy lại, không làm bước 6.**

## Thiết kế (ghi rõ trước khi train)

- **Wrapper:** `pongduel_wrapper.py` — copy đúng wrapper `SingleAgentPongDuelWrapper` trong notebook `PongDuel.ipynb` của repo (cells 1 + 5), chỉ thêm tham số `seed` cho `reset()` (gym 0.21). State: vector phẳng 20 chiều = [obs agent1 (10), obs agent2 (10)], mỗi obs = [agent_row/40, agent_col/30, ball_row/40, ball_col/30, one-hot hướng bóng (6)]. Action: joint scalar 0..8 = a1×3 + a2 (mỗi paddle NOOP/UP/DOWN). Reward: `reward_mode="agent1"` (mặc định của notebook) — reward thưa: 0 mọi bước trừ bước kết thúc 1 round, paddle nào để bóng lọt thì đối thủ +1; episode kết thúc sau 10 round nên reward agent1 ∈ {0..10} và agent1+agent2 luôn = 10. **Hệ quả phải nói rõ: một policy điều khiển CẢ 2 paddle; tối đa reward agent1 bao gồm cả việc phá agent2 (để agent2 bỏ lọt bóng). DQN này là joint policy, không phải agent1 cạnh tranh với agent2 độc lập.**
- **Môi trường:** ma-gym 0.0.13 + gym 0.21.0 (venv `step3-scratch`, Python 3.12.3, torch 2.5.1+cpu). `pip install ma-gym` thuần THẤT BẠI (pip đòi build lại gym 0.21 từ source, setuptools mới từ chối `extras_require`); giải pháp: `pip install --no-deps --target <ma-gym-target> ma-gym`, tái dùng gym 0.21 đã cài sẵn. Ghi rõ để tái hiện.
- **DQN:** `train_dqn_pongduel.py` — copy đúng recipe run 9 / LunarLander, chỉ đổi theo env (MLP 20→128→128→9, gym 0.21 API, log thêm reward agent2 + độ dài episode): Double DQN, lr 1e-3, gamma 0,99, batch 64, buffer 50000, eps 1,0→0,02 (decay 1.300 episode = 65% của 2.000), target-update mỗi 10 episode, warmup 1000, loss MSE, eval greedy mỗi 50 episode × 50 episode + chọn checkpoint tốt nhất. Không tuning hyperparameter cho PongDuel — mục đích là kiểm tra recipe cũ có học được gì trong 2.000 episode hay không.
- **Baseline (đo TRƯỚC khi train, 100 episode, seeds 10000..10099):** heuristic strong/weak của repo: agent1 **8,81 ± 1,04** (min 6, max 10; mean 360,7 bước/ep); random (joint đều, `Random(12345)`): agent1 **4,90 ± 1,56** (min 1, max 8; mean 188,4 bước/ep). Trần là 10, heuristic chỉ còn headroom 1,19. File: `baselines_eval.json`.
- **Tiêu chí thành công (chốt TRƯỚC khi train, file `success_criterion.txt`):** eval cuối của checkpoint tốt nhất (100 ep, cùng seeds) ≥ 9,30 (heuristic + 0,50) VÀ ≥ 3/4 lần eval định kỳ cuối cũng trên 8,81. Nếu không đạt: kết quả âm tính, dừng, bỏ qua bước 6. Ngân sách tối đa 2.000 episode; quy tắc đã ghi trong script: nếu VM restart giết run thì báo cáo run dở dang từ checkpoint tốt nhất đã lưu, KHÔNG chạy lại.

## Chuyện gì đã xảy ra

- **Attempt 1:** bị VM restart giết sau 29 s, mới 14 episode, chưa eval, chưa checkpoint. Log 14 episode giữ lại ở `training_log_attempt1_killed14ep.csv` / `train_run_attempt1_killed.log`.
- **Attempt 2 (run chính):** chạy lại từ đầu, cùng seed 0 / cùng recipe. **14 episode đầu KHÔNG tái hiện giống attempt 1** (vd. ep 1: 4,0/209 bước vs 5,0/157 bước) — nguyên nhân: `env.action_space.sample()` của gym 0.21 dùng RNG riêng của action space, không được `env.seed()` gieo hạt, nên exploration epsilon-greedy khác nhau giữa các process. Train PongDuel vì vậy **không tái hiện bit-chính-xác giữa các lần chạy** (khác gymnasium, nơi `reset(seed=...)` gieo hạt cả action space). Eval greedy thì xác định (không sampling; đã kiểm chứng heuristic lặp seed 0 cho cùng tổng reward). Hạn chế này ghi rõ, không giấu.
- **Attempt 2 bị VM restart giết ở episode 800/2.000** (hàng cuối `training_log.csv` ep 800, eval cuối ep 750; `uptime` sau đó hiện "up 0 min"; runtime báo SIGTERM). Không chạy lại, theo đúng quy tắc đã chốt + đường cong phẳng dưới đây (xem "Vì sao không chạy lại"). Tổng số episode đã tiêu (cả 2 attempt): 814, trong ngân sách ~2.000.

## Kết quả (bằng chứng âm tính)

- **Eval định kỳ greedy (50 ep, seeds 10000..10049), agent1 mean:** ep 50: 4,88; 100: 4,96; 150: 5,10; 200: 5,04; 250: 5,10; 300: 5,06; 350: **5,42 (tốt nhất)**; 400: 4,80; 450: 5,12; 500: 5,18; 550: 5,18; 600: 5,26; 650: 5,20; 700: 5,12; 750: 5,26. **Phẳng ở mức random trong cả 15 lần đo** (eps đi từ 1,0 xuống 0,44); không có xu hướng tăng nào. File: `eval_log.csv`.
- **Train reward:** mean 100 episode cuối (701–800) = 5,60; rolling-100 quanh 5–5,7 suốt run — cũng ở mức random. Loss MSE ~0,03–0,04, ổn định, không sụp. File: `training_log.csv` (800 hàng). Ảnh curve: `training_curve.png` (file nhị phân, trên Drive — xem báo cáo bàn giao).
- **Eval cuối trên checkpoint tốt nhất (ep 350), 100 ep, seeds 10000..10099:** agent1 **5,44 ± 1,56 (min 1, max 9)**, agent2 4,56, mean 179,76 bước/ep. File: `final_eval_best.json` + `final_eval_best_rewards.csv` (script: `scripts/eval_pongduel_best.py`).
- **Đối chiếu tiêu chí: THẤT BẠI CẢ HAI.** (1) 5,44 < 9,30 (thua heuristic 8,81 tới 3,37 điểm; chỉ hơn random 4,90 có 0,54 — và "tốt nhất" này đã qua chọn lọc trên 15 lần eval nên còn bị winner's curse). (2) Không lần eval định kỳ nào vượt 8,81 (cao nhất 5,42). **Kết luận: recipe DQN run-9/LunarLander KHÔNG học được gì có ý nghĩa trên PongDuel joint-action trong 800 episode.** Mạng cuối train không đánh giá được (không được lưu vì run bị giết; không bịa) — chỉ có checkpoint tốt nhất.

## Vì sao không chạy lại 2.000 episode cho "đủ ngân sách"

1. Quy tắc không-chạy-lại đã chốt trước khi train (trong script + `success_criterion.txt`) để bảo vệ ngân sách; chạy lại 2.000 episode nữa sẽ nâng tổng lên 2.814, vượt ngân sách ~2.000 để "cố thêm chút nữa" — đúng điều nhiệm vụ cấm.
2. Bằng chứng phẳng: eval GREEDY (đo policy đã học, không phụ thuộc eps exploration) không rời khỏi dải random qua 15 lần đo; để đạt 9,30 cần nhảy vọt +4 điểm trong 1.200 episode còn lại — không có dấu hiệu nào cho thấy điều đó (khác LunarLander, nơi eval greedy đã tăng rõ từ ep 150, và CartPole run 9, nơi eval ep 100 đã 399).
3. VM đã restart 3 lần trong ngày 12/11; chạy lại ~45–60 phút nữa vẫn có thể bị giết lại.
4. Kể cả nếu DQN có học được, bước 6 vẫn gần như chắc chắn bị bỏ qua vì ước tính thời gian BART PongDuel vượt khả thi (xem `step6_estimate_skipped.md`) — giá trị thêm của việc chạy lại là thấp.

## Đọc kết quả trung thực

- Thất bại này là của **recipe + bài toán như đã đặt ra** (joint-action 9 class, reward thưa, 2.000 episode, không reward shaping, không self-play), không phải bằng chứng "PongDuel không học được bằng RL". Reward thưa (chỉ ~10 tín hiệu reward trên ~180–370 bước/episode) + không gian joint action + trần heuristic 8,81/10 làm bài toán này khó hơn hẳn CartPole/LunarLander với cùng recipe. Hướng khác (shaping, tách 2 agent, train dài hơn nhiều) nằm ngoài ngân sách/phạm vi bước 5 và không được thử.
- File nhị phân duy nhất của run này: `best_checkpoint.pt` (ep 350, 84.952 byte) + `training_curve.png` — ở thư mục làm việc, cần upload Drive; KHÔNG có `dqn_checkpoint.pt`/`final_checkpoint.pt`/`train_summary.json` (script chỉ ghi các file đó lúc kết thúc bình thường — run bị giết nên không có; không bịa ra).
- Chi phí: $0, CPU-only. Thời gian train tới ep 800: ~16–17 phút CPU (elapsed trong log 976 s ở ep 750 + eval cuối ep 750; wall-clock của session dài hơn do VM restart).

## Cây file (thư mục này)

```
results/pongduel-dqn/
├── train_summary.md (file này)
├── baselines_eval.json (heuristic 8,81 / random 4,90, 100 ep)
├── success_criterion.txt (tiêu chí chốt trước khi train)
├── training_log.csv (800 episode, attempt 2)
├── eval_log.csv (15 lần eval định kỳ)
├── train_run.log (stdout attempt 2, tới ep 800)
├── training_log_attempt1_killed14ep.csv + train_run_attempt1_killed.log (attempt 1)
├── final_eval_best.json + final_eval_best_rewards.csv (eval cuối checkpoint ep 350)
├── step6_estimate_skipped.md (ước tính bước 6 + lý do bỏ qua)
└── [Drive] best_checkpoint.pt (ep 350), training_curve.png
```
