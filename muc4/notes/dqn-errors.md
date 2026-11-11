# DQN CartPole — errors, fixes, and run history

Date: 2026-11-11. Environment: Linux VM, Python 3.12.3 (`/usr/bin/python3`),
4 CPUs (1.5 shared), ~7 GB RAM. Venv: `~/workspace/dqn-venv`
(torch 2.15.0+cpu, gymnasium 1.4.0, numpy 2.5.4, matplotlib 3.11.2).
Training used **gymnasium, not gym 0.20** (gym 0.20 is incompatible with
Python 3.12) — training only; the delivered policy is a plain
`pi(state) -> action` function pluggable into `XRL(env, pi)`.

## Error 1 — pip: gymnasium not found on the PyTorch index (fix 1, worked)
- Symptom: `pip install torch --index-url https://download.pytorch.org/whl/cpu
  gymnasium numpy matplotlib` failed with
  `ERROR: Could not find a version that satisfies the requirement gymnasium`
  (the custom `--index-url` applied to all packages; the PyTorch index does
  not host gymnasium; also read timeouts to download.pytorch.org).
- Fix: install torch alone from the PyTorch CPU index, then install
  gymnasium/numpy/matplotlib separately from PyPI (default index).

## Error 2 — pip: "No space left on device" installing torch (fix 1, worked)
- Symptom: `OSError: [Errno 28] No space left on device` while installing
  torch into `/tmp/dqn-venv`.
- Cause: `/tmp` is a 512 MB tmpfs; the torch wheel + install needs ~1 GB.
  (`df` showed 148 GB free on `/home/hatch`, so the disk itself was fine.)
- Fix: recreated the venv at `~/workspace/dqn-venv` (home disk) and set
  `TMPDIR=~/workspace/.tmp-pip` for pip's temp files. Torch 2.15.0+cpu
  then installed cleanly.

## Slowdown (not an error) — training stalled ~10 min around episode 250–270
- Symptom: run 1 produced no new log lines for ~12 min wall time
  (episodes 250→267) while the process was alive at ~84% CPU.
- Cause: a second, unrelated job on the same VM (an R ablation run,
  `.venv-ablation/.../run_ablation.py`) was competing for the shared CPUs.
- Fix: none needed — waited; speed recovered (72 episodes / 2 min) once
  the other job finished. Run 1 total time 1671 s includes this stall.

## Outcome — 450–500 target NOT reached (honest result, no fabrication)
Three runs, all seed 0, CartPole-v1, Q-network MLP 4→128→128→2,
replay 50k, batch 64, gamma 0.99, Adam:

| Run | Hyperparameters | Episodes | Train mean, last 100 ep | Greedy eval, 100 ep (mean ± std, min–max) |
|-----|-----------------|----------|------------------------|--------------------------------------------|
| 1 (BEST, delivered) | lr 1e-3, MSE loss, eps 1.0→0.02 over 400 ep, target update every 10 ep | 600 | **264.49** | **246.16 ± 10.89, 230–282** |
| 2 (tweak 1, killed at ep 404) | lr 5e-4, Huber loss, grad-clip 1.0, eps 1.0→0.01 over 250 ep, target every 5 ep | 404 (of 800) | 119.0 (last 100 of partial) | not evaluated (killed early: plateau ~117 at ep 400, no plausible path to 450) |
| 3 (tweak 2) | lr 1e-3, Huber loss, grad-clip 1.0, eps 1.0→0.02 over 400 ep, target every 10 ep | 800 | 134.74 | 86.77 ± 31.63, 33–118 |

- Run 1 peaked at mean-last-100 = 343.8 (ep 500) then fell to 264.5 by
  ep 600 as the MSE loss grew (0.45 → 2.17) — late-training Q-value
  instability. Its greedy eval (246.16) is the best of the three.
- Tweaks 1 and 2 (Huber loss + gradient clipping) stabilised the loss
  (run 3 loss fell to ~0.13) but slowed learning far more than expected:
  Huber caps per-sample gradients and grad-clip 1.0 caps update size, so
  both runs learned much slower than run 1's MSE schedule. That is why
  they ended lower, not higher.
- Both allowed hyperparameter tweaks were used (limit: 2). No further
  tuning was attempted. A plausible next attempt (NOT run — would be a
  3rd tweak): Huber loss with a higher lr (e.g. 5e-3) or a looser clip
  (e.g. 10.0), or Double DQN; left for future work.
- Heuristic baseline (CausalXRL repo formula), same 100-episode eval:
  mean **31.53 ± 11.16, min 11, max 63**. The delivered DQN beats it by
  **+214.63** mean reward. The DQN is genuinely trained (not the
  heuristic) and clearly better than the repo baseline, but it does NOT
  meet the 450–500 target — do not claim it does.

## Minor observation (not an error)
- Run 3 episode 1 gave reward 31 vs 11 in runs 1–2 with the same seed 0.
  Cause not established (suspected gymnasium action-space seeding
  nondeterminism); immaterial — a single random-exploration episode.

## No unresolved blocker
- No `notes/dqn-blockers.md` was created: the pipeline works end to end
  (install → train → checkpoint → independent eval → plot → smoke test),
  and the only shortfall is the performance target, reported honestly
  above rather than being a technical blocker.

## Artefacts (delivered = run 1, the best run)
- `results/cartpole-dqn/dqn_checkpoint.pt` — state_dict + hyperparams +
  final eval (run 1)
- `results/cartpole-dqn/training_log.csv` — 600 episodes (run 1)
- `results/cartpole-dqn/train_run.log` — full stdout (run 1)
- `results/cartpole-dqn/eval_log.txt` — independent re-eval: DQN
  246.16 ± 10.89; heuristic 31.53 ± 11.16
- `results/cartpole-dqn/reward_curve.png` — run 1 curve, final 100-ep
  mean 264.5
- `results/cartpole-dqn/run1/`, `run2-partial/`, `run3/` — archived logs,
  CSVs and checkpoints of each run (for audit, not the deliverable)
- `scripts/train_dqn_cartpole.py`, `scripts/cartpole_policies.py`
  (heuristic + `load_dqn_policy`, smoke test passed: 5-episode rewards
  255, 265, 234, 241, 226), `scripts/evaluate_cartpole.py`,
  `scripts/plot_reward_curve.py`

## Run 7–9 (11/11/2026, làm chỉn chu phần D — eval định kỳ + best checkpoint)
- Thêm `--eval-every/--eval-episodes/--eval-seed-offset` vào
  `scripts/train_dqn_cartpole.py`: eval greedy mỗi 50 episode (50 ep,
  seeds 10000..10049 — tập con của eval cuối 100 ep), lưu checkpoint
  tốt nhất theo eval vào `best_checkpoint.pt`, cuối train promote lên
  `dqn_checkpoint.pt`, mạng cuối train giữ ở `final_checkpoint.pt`,
  `eval_log.csv` ghi toàn bộ eval định kỳ. `scripts/plot_eval_curve.py`
  (mới) vẽ train + eval định kỳ trên 1 hình.
- Run 7 (Double DQN, 1500 ep, eps-decay 1000): **bị VM restart giết ở
  ep 1040/1500** (uptime "up 0 min", không traceback, không TRAIN DONE;
  `training_log.csv` ep 1–1040, `eval_log.csv` 20 lần eval). Nhờ best
  checkpoint lưu tăng dần nên không mất: best @ ep 900, eval 50 ep
  344,78 ± 23,85; xác minh 100 ep 343,26 ± 22,30 (min 308, max 421).
- Run 8 (1500 ep, chỉ đổi 1 yếu tố vs run 7: target-update-every
  10 → 100): best @ ep 900, eval 50 ep 233,26 ± 29,83; xác minh 100 ep
  234,51 ± 34,59 (min 169, max 395); mạng cuối train 89,68 ± 2,09.
  Loss chỉ 0,01–0,15 (target quá cũ). Kết luận: thụt lùi, không dùng.
- Run 9 (Double DQN, 2000 ep, eps-decay 1300): **ĐẠT MỤC TIÊU —
  eval greedy 100 ep 500,00 ± 0,00 (min 500, max 500)**; mạng cuối
  train cũng 500,00. 10/40 lần eval định kỳ đạt đúng 500,00 (ep 1150,
  1200, 1350, 1400, 1450, 1500, 1700, 1800, 1850, 2000); lần đầu 500 ở
  ep 1150 (checkpoint được promote). Train mean last100 326,35, thời
  gian 2603,3 s (~43 phút). Q-instability: eval dao động 500 ↔ ~97–
  146 giữa các lần đo cách nhau 50 ep (vd 500 @ ep 1500 → 108,20 @
  ep 1550 → 500 @ ep 1700; loss MSE đỉnh 34,35 @ ep 1900) — policy tốt
  liên tục bị train tiếp phá hỏng rồi tìm lại được, nên chọn checkpoint
  theo eval định kỳ là bắt buộc. Smoke test trên checkpoint chính:
  5/5 episode = 500.
- Sửa 1 lỗi ghi chép (không đổi số liệu train): `train_summary.json`
  của run 8 trước đây chỉ ghi eval 100 ep của mạng cuối train (89,68);
  đã bổ sung `best_checkpoint_eval_100ep` = 234,51 (nguồn:
  train_run.log FINAL EVAL) và bổ sung `final_eval_100ep` vào metadata
  của best checkpoint run 7 (343,26), run 8 (234,51), run 9 (500,00)
  — chỉ thêm metadata, không đổi weights (md5 checkpoint run 9 đổi
  afcbf854… → a6fa113c… do thêm key; reload + eval lại vẫn 500,00).
  Script train cũng đã sửa để `train_summary.json` ghi
  `final_eval_100ep` = eval của checkpoint được promote (trước đây ghi
  nhầm eval của mạng cuối train).
- Run 9 là best hiện tại (`train_summary.json` top-level ghi
  `best_run: run9`); checkpoint chính md5 khớp run9/dqn_checkpoint.pt.
