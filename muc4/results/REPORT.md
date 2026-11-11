# Mục 4 — Mở rộng CausalXRL — Báo cáo tổng hợp

**Bài:** CausalXRL, arXiv:2610.42028v1, repo yanmluk/causal-xrl commit `ae9096e`
**Phạm vi:** §4 kế hoạch chấm điểm. Hướng D (train RL thật) là chính, Hướng A (ablation) là dự phòng.
**Ngày làm:** 11/11/2026 (train run 7–9 + so sánh XRL chạy lại cùng ngày, sau 2 lần VM restart). **Không làm:** Hướng B/C, human study, MicroRTS, PongDuel, LunarLander.
**Lưu ý:** Mục 3 (baseline heuristic qua BART) đang do Codex làm song song ở gói riêng — báo cáo này không làm lại §3.

## 1. Kết quả chính

### Hướng D — Train DQN thật trên CartPole: ĐẠT MỤC TIÊU — EVAL 500,00/500 (run 9)
- DQN tự viết bằng PyTorch (MLP 4→128→128→2, replay buffer, epsilon-greedy, target network, Adam), **không dùng stable-baselines3**. Train bằng gymnasium (chỉ cho bước train; gym 0.20 của repo không tương thích Python 3.12 — ghi rõ).
- **Run tốt nhất (run 9, Double DQN, 2.000 episode, lr 1e-3, MSE, eps-decay 1.300): train mean 326,35 ở 100 episode cuối; eval greedy 100 episode: 500,00 ± 0,00 (min 500, max 500)** — policy giải hoàn hảo CartPole-v1 trên cả 100 episode đánh giá (seeds 10000..10099). Checkpoint tốt nhất hiện tại, đã copy lên `results/cartpole-dqn/dqn_checkpoint.pt` (`train_summary.json` ghi `best_run: run9`; md5 khớp `run9/dqn_checkpoint.pt`, đã kiểm tra lại). **Cả mạng cuối train cũng đạt 500,00 ± 0,00 trên 100 episode** (final-net eval = best-net eval = 500,00), nên kết quả không phụ thuộc vào việc chọn checkpoint. Phần so sánh XRL dưới đây dùng checkpoint run 9 này.
- **Cách đạt được (đúng gợi ý trong báo cáo cũ): chọn checkpoint theo eval định kỳ giữa train** — script `train_dqn_cartpole.py` nay có `--eval-every 50 --eval-episodes 50` (eval greedy mỗi 50 episode, lưu checkpoint tốt nhất theo eval vào `best_checkpoint.pt`, cuối train promote lên `dqn_checkpoint.pt`; mạng cuối train giữ riêng ở `final_checkpoint.pt`; `eval_log.csv` ghi toàn bộ eval định kỳ). Run 9 có **10/40 lần eval định kỳ đạt đúng 500,00** (ep 1150, 1200, 1350, 1400, 1450, 1500, 1700, 1800, 1850, 2000) — lần đầu đạt 500 ở ep 1150, checkpoint đó chính là checkpoint được promote. Nếu chỉ lấy mạng cuối train như các run trước thì cũng đạt 500 lần này, nhưng eval định kỳ mới là thứ bắt được policy tốt (xem phân tích Q-instability dưới).
- **Q-instability (bằng chứng rõ nhất):** eval định kỳ của run 9 dao động dữ dội giữa các lần đo cách nhau 50 episode — 500,00 ở ep 1500 rồi 108,20 ở ep 1550, rồi lại 500,00 ở ep 1700/1800/1850, rồi 96,80 ở ep 1900, rồi 500,00 ở ep 2000; loss MSE dao động 0,1–34 (đỉnh 34,35 ở ep 1900; thường 2–16 ở nửa sau train). Policy tốt liên tục bị train tiếp phá hỏng rồi lại tìm lại được — vì vậy **chọn checkpoint theo eval định kỳ là bắt buộc**, không phải tối ưu phụ. Run 1 từng cho thấy cùng mẫu hình ở quy mô nhỏ hơn (đỉnh rolling-100 train 355,6 quanh ep 400–500 rồi tụt).
- Các run trước (để so sánh, cùng điều kiện eval 100 episode, seeds 10000..10099):
  - Run 1 (600 ep, vanilla DQN, MSE, lr 1e-3): train mean 264,49; eval **246,16 ± 10,89** (min 230, max 282).
  - Run 5 (600 ep, Double DQN, lr 1e-3): train mean 163,28; eval **272,68 ± 58,53** (min 203, max 500) — từng là tốt nhất, nay bị run 9 vượt xa.
  - Run 7 (1.500 ep, Double DQN, eps-decay 1000, eval mỗi 50 ep): **DỞ DANG — bị VM restart giết ở ep 1040/1500** (uptime hiện "up 0 min", không traceback, không TRAIN DONE; `training_log.csv` có ep 1–1040, `eval_log.csv` 20 lần eval). Nhờ checkpoint tốt nhất được lưu tăng dần nên không mất: best @ ep 900, eval 50 ep 344,78 ± 23,85, **xác minh lại 100 ep: 343,26 ± 22,30 (min 308, max 421)** — đã là best lúc đó, nhưng run 9 vượt qua.
  - Run 8 (1.500 ep, Double DQN, **chỉ đổi 1 yếu tố so với run 7: target-update-every 10 → 100**): best @ ep 900, eval 50 ep 233,26 ± 29,83, **xác minh lại 100 ep: 234,51 ± 34,59 (min 169, max 395)**; mạng cuối train tụt còn 89,68 ± 2,09. **Kết luận: target update thưa hơn học chậm hơn hẳn mà vẫn sụp cuối train — hướng này là thụt lùi, không dùng.** (Loss run 8 chỉ 0,01–0,15 vì target quá cũ, Q fit target cố định rất nhanh nhưng lan truyền giá trị chậm.)
  - Run 9 chỉ đổi so với run 7: train dài hơn (2.000 ep, eps-decay 1.300 giữ tỉ lệ 2/3 như run 7) — và quỹ đạo exploration khác ngay từ đầu do eps-decay dài hơn (eval ep 100 đã 399,64 rồi tụt; tính hỗn loạn của DQN: cùng seed 0 nhưng đổi eps-decay là đổi toàn bộ quỹ đạo). Run 3 (Huber + clip 1.0, eval 86,77) và run 4 (Huber lr 5e-3 + clip 10.0, eval 56,81) vẫn là các hướng đã thử và thất bại — không thử lại.
  - Run 6 cũ: dừng giữa chừng ở ep ~450–475, không checkpoint, không eval — dở dang, không dùng (giữ nguyên trạng thái này).
- Heuristic của repo đánh giá cùng điều kiện: **31,53 ± 11,16 (min 11, max 63)** — DQN run 9 hơn **+468,47** (500,00 − 31,53), và **mọi episode rollout 200 episode của DQN đều đạt đúng 500 bước** (mean_ep_len 500,0 — xem phần XRL).
- Thời gian: run 9 ~43 phút CPU (2.603 s); run 7 ~6 phút tới ep 1040; run 8 ~17 phút (994 s). Chi phí $0.
- **Kết luận: MỤC TIÊU 450–500 ĐÃ ĐẠT — VƯỢT MỨC TRẦN (500,00/500).** Không còn là "chưa đạt" như báo cáo cũ.

### Hướng D — So sánh XRL (BART thật) trained-vs-heuristic: XONG — CHẠY LẠI VỚI CHECKPOINT RUN 9 (500/500)
- **BART CÀI ĐƯỢC và chạy được thật**: R 4.3.3 (`apt install r-base-core`) + R package `dbarts` 0.9.34 (CRAN) + `rpy2` 3.5.16 trong venv `.venv-ablation` (đã thêm torch 2.15.0+cpu). Chi tiết + mọi lỗi đã fix trong `notes/bart-check.md`, `notes/bart-errors.md`. **VM restart 2 lần trong ngày 11/11 đã xóa R khỏi hệ thống 2 lần** (R sống ngoài `~/workspace`); lần 1 khôi phục bằng `apt-get install r-base-core` (APT RC=0), lần 2 khôi phục nhanh bằng `dpkg -i /var/cache/apt/archives/*.deb` (34 file .deb còn trong cache, R 4.3.3 về trong ~30 s), rồi cài lại dbarts 0.9.34 từ CRAN trong cả 2 lần. Repo dùng bản copy bền trong thư mục này (`repo-causal-xrl/`, commit `ae9096e`) vì `/tmp` cũng bị xóa khi VM replace (bản `/tmp/causal-xrl-muc4` đã mất — lần chạy lại này dùng `--repo repo-causal-xrl`).
- **Bảng cũ (checkpoint run 5, eval 272,68) đã lưu trữ nguyên vẹn ở `results/xrl-trained-vs-heuristic-run5/`** (comparison_table + per_run_details + graphs + logs) — không xóa, chỉ thay bằng bảng mới. Bảng dưới đây là bảng mới, checkpoint run 9.
- **Lần chạy lại này bị VM restart giết 1 lần:** full run (11:05–~11:52, log cuối 11:38) chết giữa DQN ate; `partial_results.json` đã lưu tăng dần nên giữ được heuristic đầy đủ + DQN hte (14 cạnh, 1.738,58 s). DQN-only rerun (11:55–13:08, `--policies dqn`, cùng checkpoint run 9, seed 0, 200 episode) làm lại DQN hte + ate + fidelity, rồi `scripts/merge_xrl_comparison.py` (đã cập nhật provenance cho lần merge run 9) ghép thành bảng cuối. **Kiểm chứng chéo quan trọng: DQN hte của lần chạy bị giết và của rerun cho TẬP CẠNH GIỐNG HỆT NHAU (14 cạnh, symdiff 0)** — đồ thị hte của DQN ổn định giữa 2 lần chạy, không phải nhiễu. Provenance từng số ghi trong `per_run_details.json` > `provenance`.
- Pipeline XRL gốc của repo (`infer_causes_hte` 70/30, `infer_causes_ate` 50/50), 200 episode/policy, seed 0, gymnasium CartPole-v1. DQN = checkpoint run 9 (eval 500,00 ± 0,00). Script: `scripts/run_xrl_comparison.py` (+ `scripts/merge_xrl_comparison.py`). Kết quả đầy đủ: `results/xrl-trained-vs-heuristic/` (`comparison_table.csv/.md`, `per_run_details.json`, `graphs/` 4 ảnh, log các lần chạy; log lần chạy bị giết `comparison_run.log` 11:38 + log DQN-only rerun ở `results/xrl-trained-vs-heuristic-dqn-only-run9/`).
- **(1) Fidelity dự đoán action (kiểu eval notebook của repo: bước nào local graph có >1 nguyên nhân cho `push`, chia 60/40, random_state=10; BART-weighted tối đa 200 bước test):**

| policy | valid steps | DT acc / F1 | LR acc / F1 | BART acc / F1 (n=200) | global parents của `push` |
|---|---|---|---|---|---|
| heuristic | 6.329 (100,0%) | 0,9491 / 0,9498 | 0,9111 / 0,9121 | 0,9350 / 0,9340 | push, x, θ (3) |
| DQN (run 9) | 54.472 (54,6%) | 0,9356 / 0,9392 | 0,8195 / 0,8363 | 0,8050 / 0,8368 | push, x, v, θ, w (5) |

- **(2) Đồ thị nhân quả học được (200 episode):**

| policy | hte: số cạnh / thời gian infer | ate: số cạnh / thời gian infer |
|---|---|---|
| heuristic (6.330 transitions) | 14 cạnh / 85,84 s | 15 cạnh / 120,38 s |
| DQN (99.800 transitions) | 14 cạnh / 1.756,30 s | 16 cạnh / 2.306,02 s |

- So sánh đồ thị: **hte Jaccard 0,7500, khác nhau 4 cạnh** (chỉ heuristic có: `v→θ`, `w→v`; chỉ DQN có: `v→push`, `w→push`); **ate Jaccard 0,6316, khác nhau 7 cạnh** (chỉ heuristic có: `θ→x`, `θ→v`, `θ→w`; chỉ DQN có: `x→θ`, `v→push`, `w→x`, `w→θ`). Điểm khác biệt lớn nhất giữ nguyên như bảng run 5: **action `push` của DQN có cả 5 biến làm nguyên nhân (x, v, θ, w, push) ở cả hte lẫn ate, còn heuristic chỉ có 3 (hte) / 4 (ate, thêm w)** — đúng với bản chất: policy heuristic chỉ là hàm của x và θ, còn mạng DQN dùng toàn bộ state kể cả vận tốc. Cả hai đồ thị đều có cạnh `push→push` (action ở t dự báo action ở t+1, do state có tính liên tục) và các cạnh tự hồi quy của state.
- **(3) Thời gian học cấu trúc:** DQN chậm hơn heuristic ~20 lần (hte 1.756 s vs 86 s; ate 2.306 s vs 120 s), chủ yếu vì DQN sống lâu hơn nên có ~15,8 lần transitions (99.800 vs 6.330) — và mọi episode DQN đều chạm trần 500 bước (mean_ep_len 500,0). (Bảng run 5 cũ ghi "~9–13 lần" — con số đó bản thân nó đã nói quá: thực tế bảng run 5 là 8,6–9,1 lần (hte 801/88, ate 1.039/121); với checkpoint run 9 thì tỉ lệ là ~19–20 lần. Đã sửa ở đây, không giữ số cũ.)
- **Đọc kết quả trung thực:** mô hình nhân quả dự đoán action của DQN KÉM hơn heuristic ở cả 3 thước đo — DT 0,9356 vs 0,9491; LR 0,8195 vs 0,9111; BART-weighted 0,8050 vs 0,9350 — và kém rõ hơn bảng run 5 (khi đó BART 0,8850 vs 0,9200). Tức là policy DQN hoàn hảo (mạng nơ-ron) có biên quyết định phi tuyến mà mô hình tuyến tính và BART theo trọng số ITE bắt chước kém hơn so với luật tuyến tính của heuristic — không phải DQN "kém giải thích được" theo nghĩa tuyệt đối, mà là action của nó phụ thuộc phức tạp hơn vào state. **Lưu ý khi so với bảng run 5: tập valid steps khác nhau rất nhiều** (DQN run 9 chỉ 54,6% bước có >1 nguyên nhân local cho `push`, so với 95,5% ở run 5; heuristic 100,0% so với 87,7%) — vì vậy các số fidelity run 5 vs run 9 không so trực tiếp được như cùng một test set; chênh lệch một phần do tập bước được đánh giá đã đổi, không chỉ do policy đổi. Đọc đúng: trong cùng bảng run 9, DQN kém heuristic ở cả 3 thước đo trên cùng protocol.
- **Cảnh báo reproducibility (quan trọng, ghi rõ trong báo cáo):** chạy lại pipeline BART với cùng seed 0 cho số cạnh lệch ±1–2 cạnh giữa các lần chạy. Bằng chứng lần này: heuristic hte 13 (bảng run 5) → 14 (bảng run 9); heuristic ate 17 → 15; fidelity BART heuristic 0,9200 → 0,9350. Dải nhiễu quan sát được tới nay: heuristic hte 12–14, ate 15–19 (đã cập nhật — bảng cũ ghi hte 12–13, ate 17–19 là hẹp hơn thực tế). Ngược lại, **DQN hte tái hiện chính xác tuyệt đối** (14 cạnh, tập cạnh giống hệt) qua 2 lần chạy độc lập cùng checkpoint — khác biệt đã nêu (push có 5 vs 3 nguyên nhân, Jaccard 0,63–0,75) vượt xa mức nhiễu này. Số trong bảng trên là bảng merge cuối (heuristic: full run bị giết; DQN: DQN-only rerun; xem provenance trong `per_run_details.json`).
- **Vấn đề RAM đã gặp và fix (không giấu):** process chết im lặng 2 lần khi chạy DQN ở bảng run 5 (nghi OOM: RAM cạn, không traceback) — đã fix bằng cách bỏ model BART ngay sau mỗi target (xem báo cáo cũ). Lần chạy lại này RAM process lên tới ~4,5 GB lúc DQN ate (BART trên ~100k transitions), host có lúc chỉ còn ~250–460 MB free, nhưng **không chết OOM** — fix cũ vẫn giữ được process sống; lần chết duy nhất lần này là do VM restart (uptime "up 0 min"), không phải OOM, và `partial_results.json` đã cứu được heuristic + DQN hte.

### Hướng A — Ablation / độ ổn định: XONG HẾT, KHÔNG BLOCKER (nhưng dùng SURROGATE, không phải BART)
- R/dbarts **không** được cài trong gói này (Codex Mục 3 đang thử hướng đó song song). Ablation dùng **SurrogateRegressor RandomForest 200 cây** thay BART, giữ nguyên interface và logic chia train/test của repo (`infer_causes_hte` 70/30, `infer_causes_ate` 50/50). **Mọi số ablation đều ghi nhãn "SURROGATE, NOT BART/dbarts — chỉ kiểm tra độ ổn định, phải chạy lại bằng BART thật khi có". Không được ghi thành số CausalXRL/BART.**
- Lưới 6 run (episode 50/100/200 × hte/ate), seed 0, policy heuristic đúng công thức repo, gymnasium CartPole-v1:

| run | số cạnh | thời gian infer | DT acc/F1 | LR acc/F1 |
|---|---|---|---|---|
| hte 50 | 5 | 23,6 s | 0,978 / 0,978 | 0,978 / 0,977 |
| ate 50 | 12 | 22,5 s | 0,978 / 0,978 | 0,978 / 0,977 |
| hte 100 | 7 | 43,2 s | 0,986 / 0,986 | 0,986 / 0,986 |
| ate 100 | 14 | 32,9 s | 0,986 / 0,986 | 0,986 / 0,986 |
| hte 200 | 7 | 71,9 s | 0,983 / 0,984 | 0,989 / 0,989 |
| ate 200 | 14 | 59,1 s | 0,983 / 0,984 | 0,989 / 0,989 |

- **Độ ổn định:** hte ổn định từ 100 episode (đồ thị 100ep = 200ep, Jaccard 1,0; 50ep thiếu 2 cạnh push). ate vẫn thay đổi tới 200 episode (Jaccard 0,647 ở 100ep). Đồ thị hte 200ep: `x→x, v→v, v→w, θ→θ, w→w, push→v, push→w`.
- **Độ nhạy CI (200ep):** hte 90% → 8 cạnh (thêm `w→θ`), 95% → 7 cạnh; ate không đổi (14 cạnh).
- Đối chiếu vật lý thật CartPole (16 cạnh nguyên nhân trực tiếp): hte thận trọng — 7/7 cạnh đúng nhưng thiếu 9; ate 200ep đúng 12/14 (dương tính giả `x→v`, `w→x`).

## 2. CHƯA LÀM ĐƯỢC — ghi rõ, không giấu
- **LunarLander:** chưa làm (chỉ làm sau khi CartPole D xong — CartPole D nay ĐÃ đạt mục tiêu 500/500 nên LunarLander có thể mở rộng sau nếu cần, nhưng không thuộc phạm vi lần làm này).
- **Hướng B/C:** không làm (D và A đã xong trong phạm vi).
- **So sánh XRL trained-vs-heuristic bằng BART thật: ĐÃ XONG với checkpoint run 9 (xem §1, Hướng D)** — bảng run 5 cũ lưu ở `results/xrl-trained-vs-heuristic-run5/`. Còn lại 1 điểm yếu đã biết: số BART dao động ±1–2 cạnh giữa các lần chạy (heuristic ate 15–19 qua các lần chạy; xem cảnh báo reproducibility ở §1); nếu cần số "chốt" cho báo cáo thì lấy bảng ở §1 (lần chạy cuối đã merge).
- **Run 7 dở dang do VM restart (ep 1040/1500):** không chạy lại vì run 9 đã vượt xa kết quả của nó (500,00 vs 343,26); checkpoint tốt nhất của run 7 vẫn lưu ở `results/cartpole-dqn/run7/` để đối chiếu.

## 3. Điều kiện chạy
- VM: Python 3.12.3, 4 CPU, ~7 GB RAM, CPU-only, $0. **Ngày 11/11 VM restart 2 lần** (lần 1 giết run 7 ở ep 1040/1500; lần 2 giết full run XRL giữa DQN ate) — mọi file trong `~/workspace` còn nguyên; chỉ R (`/usr/lib/R`) và `/tmp/causal-xrl-muc4` bị xóa, đã khôi phục (xem §1 Hướng D, XRL).
- DQN: torch 2.15.0+cpu, gymnasium 1.4.0, numpy 2.5.4; venv tại `~/workspace/dqn-venv` (ngoài thư mục goal, tái dùng được). Train script nay hỗ trợ eval định kỳ + best checkpoint (`--eval-every`, `--eval-episodes`, `--eval-seed-offset`; xem `scripts/train_dqn_cartpole.py`).
- XRL/BART: R 4.3.3 + dbarts 0.9.34 + rpy2 3.5.16 + torch 2.15.0+cpu trong venv `.venv-ablation` (trong thư mục này); repo dùng bản copy `repo-causal-xrl/` (commit `ae9096e`). Cần `R_HOME=/usr/lib/R` (script tự đặt mặc định). Xem `notes/bart-check.md` (hiện trạng) và `notes/bart-errors.md` (mọi lỗi + fix + nhiều lần VM replacement/restart, 2 lần R bị xóa ngày 11/11, 2 lần chết process nghi OOM).
- Ablation: venv `.venv-ablation` trong thư mục này (sklearn 1.9.1, gymnasium 1.4.0, networkx 3.7, pandas 3.1.1, numpy 2.5.4). Repo dùng bản copy `repo-causal-xrl/` (commit `ae9096e`), không sửa file repo (bản clone `/tmp/causal-xrl-muc4` đã mất sau VM restart).
- Seed 0 cho cả hai hướng. Heuristic CartPole đúng công thức repo: `1 if 0.4*(state[0]/2.4)+0.6*(state[2]/2.095)>=0 else 0`.

## 4. Cây file
```
causalxrl-muc4-mo-rong/
├── notes/
│   ├── dqn-errors.md, ablation-errors.md, dqn-run4-6.md
│   ├── ablation-blockers.md (không blocker)
│   ├── bart-check.md, bart-errors.md (BART cài được; lỗi + fix + VM replace + OOM + 2 lần R bị xóa 11/11)
│   └── (không có bart-blockers.md — BART không còn blocker)
├── scripts/
│   ├── train_dqn_cartpole.py (đã thêm --eval-every/--eval-episodes/--eval-seed-offset: eval định kỳ + best checkpoint), cartpole_policies.py, evaluate_cartpole.py, plot_reward_curve.py, plot_eval_curve.py (mới: train + eval định kỳ trên 1 hình)
│   ├── surrogate_bart.py, run_ablation.py
│   ├── run_xrl_comparison.py, merge_xrl_comparison.py (đã cập nhật provenance cho merge run 9), redraw_xrl_graphs.py, smoke_bart.py
│   └── (XRL comparison: BART thật)
├── repo-causal-xrl/ (bản copy repo yanmluk/causal-xrl commit ae9096e, chỉ đọc; bản /tmp/causal-xrl-muc4 đã mất sau VM restart)
├── results/
│   ├── cartpole-dqn/ (dqn_checkpoint.pt = run 9 Double DQN 2000 ep, eval 500,00/500; training_log.csv, eval_log.csv, train_run.log, eval_log.txt, reward_curve.png, eval_curve.png, train_summary.json ghi best_run: run9; run1/, run2-partial/, run3/, run4/, run5/, run6/ (dở dang, không dùng), run7/ (dở dang do VM restart ở ep 1040/1500, best @ ep 900 eval 343,26), run8/ (target-update 100, best eval 234,51), run9/ (best, đầy đủ))
│   ├── ablation/ (ablation_table.csv/.md, ci_sensitivity.md, per_run_details.json, ablation_run.log, graphs/ 6 PNG)
│   ├── xrl-trained-vs-heuristic/ (BẢNG MỚI, checkpoint run 9: comparison_table.csv/.md, per_run_details.json, partial_results.json (full run bị VM restart giết), comparison_run.log (full run bị giết) + comparison_stdout.log, graphs/ 4 PNG — BART thật)
│   ├── xrl-trained-vs-heuristic-run5/ (BẢNG CŨ lưu trữ: checkpoint run 5, eval 272,68 — comparison_table + per_run_details + graphs + logs, nguyên vẹn)
│   ├── xrl-trained-vs-heuristic-dqn-only-run9/ (DQN-only rerun cho bảng mới: comparison_run.log, partial_results.json, comparison_table.csv/.md, per_run_details.json, graphs/ 2 PNG DQN)
│   ├── xrl-trained-vs-heuristic-dqn-only/ (DQN-only rerun cũ cho bảng run 5 — lưu trữ)
│   └── REPORT.md (file này)
```

## 5. Checklist bàn giao
- [x] DQN CartPole: checkpoint + reward curve + eval curve + log train/eval — **ĐẠT MỤC TIÊU 450–500: run 9 eval 500,00 ± 0,00 / 100 episode (min 500, max 500)**
- [x] Heuristic vs DQN eval cùng điều kiện (31,53 vs 500,00 run 9; run 7: 343,26; run 8: 234,51; run 5: 272,68; run 1: 246,16; run 6 dừng giữa chừng, không checkpoint)
- [x] Ablation 6 run + CI sensitivity + ảnh đồ thị (surrogate, ghi nhãn rõ)
- [x] XRL/BART trained-vs-heuristic — XONG VỚI CHECKPOINT RUN 9 (BART thật; DT/LR/BART-weighted + 2 đồ thị/policy + thời gian; xem §1 Hướng D và `results/xrl-trained-vs-heuristic/`; bảng run 5 cũ lưu ở `results/xrl-trained-vs-heuristic-run5/`)
- [ ] LunarLander — chưa làm (nay có thể mở rộng sau vì CartPole đã đạt mục tiêu, ngoài phạm vi lần này)
- [x] Không số bịa, không số paper ghi thành số nhóm, không GPT judge, $0
