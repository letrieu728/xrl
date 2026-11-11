# BART install — error log (2026-11-11)

## Fix 1 — R + dbarts: OK (không lỗi)
- `apt-get install -y r-base-core` → R 4.3.3 ("Angel Food Cake") cài thành công.
- `R install.packages("dbarts")` → dbarts 0.9.34, `* DONE (dbarts)`.
- Smoke test R: `bart(x, y, nskip=50, ndpost=25, nchain=2)` chạy được,
  `dim(fit$yhat.train)` = 50x50. gfortran có sẵn.
- `pip install rpy2` (mặc định) → rpy2 3.6.9 cài được (wheel), nhưng
  `import rpy2.robjects` lỗi:
  `ffi.error: symbol 'R_getVar' not found in library '/usr/lib/R/lib/libR.so'`
  — rpy2 3.6.x cần symbol R_getVar chỉ có từ R 4.4+, trong khi Ubuntu 24.04
  chỉ có R 4.3.3.
- Fix thử: hạ rpy2 về dòng 3.5.x (repo pin rpy2==3.5.0; 3.5.12+ hỗ trợ
  Python 3.12, dùng 3.5.16 mới nhất dòng 3.5).

## Fix 2 — rpy2 3.6.9 không tương thích R 4.3.3: ĐÃ FIX BẰNG rpy2 3.5.16
- Lỗi: `ffi.error: symbol 'R_getVar' not found in library '/usr/lib/R/lib/libR.so'`
  (rpy2 3.6.x cần R >= 4.4).
- Lần thử 2a: `pip install rpy2==3.5.16` đè lên 3.6.9 → lỗi import trộn file:
  `ImportError: cannot import name 'SexpVectorCCompatibleAbstract' from
  'rpy2.rinterface_lib.sexp'` (file cũ 3.6.9 còn sót).
- Lần thử 2b (fix đúng): `pip uninstall -y rpy2`, xóa tay thư mục
  `site-packages/rpy2*`, cài lại sạch `rpy2==3.5.16`.
- Kết quả: `import rpy2.robjects` OK, rpy2 3.5.16, R 4.3.3,
  `library(dbarts)` chạy được qua rpy2.
- Lưu ý: `ro.r('requireNamespace("dbarts", quietly=TRUE)')[0]` trả về None
  trong rpy2 (không dùng cách check đó); dùng `library(dbarts)` trực tiếp.

## Fix 3 — repo code dùng `sklearn.tree._tree.DTYPE` đã bị xóa (sklearn 1.9.1)
- Lỗi khi chạy BART thật lần đầu (fit/predict/ATE/ITE đều OK, chỉ fail ở
  `is_causal_HTE`): `AttributeError: module 'sklearn.tree._tree' has no
  attribute 'DTYPE'` tại `bart_regressor.py:32` (`HTE.cluster`) và
  `decision_tree_cluster`. Đây là lỗi tương thích sklearn, không phải lỗi BART.
- Fix (không sửa file repo): shim trong script của nhóm — trước khi import
  repo, đặt `sklearn.tree._tree.DTYPE = np.float32` (giá trị cũ của DTYPE).
  Áp dụng trong `scripts/smoke_bart.py` và `scripts/run_xrl_comparison.py`
  (`load_repo_infer_network`).
- Kết quả smoke test sau fix: fit 2.5s (n=300); ITE shape (500, 50) —
  assert 500 draws của repo HTE thỏa; ATE col0 = 1.010, CI (0.890, 1.131)
  (hiệu ứng thật +1/SD — đúng); is_causal_HTE: col0 True, col1 True,
  col2 (noise) False — phân biệt đúng.
- Kết luận: BART THẬT CHẠY ĐƯỢC end-to-end. Không cần surrogate cho Mục 4
  trừ khi bước chạy so sánh gặp blocker mới (ghi vào bart-blockers.md nếu có).

## Gián đoạn runtime (không phải lỗi BART) — lần chạy so sánh đầu tiên
- Lần chạy 1 (`run_xrl_comparison.py --estimator bart`, bắt đầu 05:37):
  heuristic hte OK (13 cạnh, infer 241,29 s), heuristic ate OK (17 cạnh,
  infer 323,50 s), DQN rollout OK (48.909 transitions, mean 245,5 bước/ep).
- VM/runtime restart lúc ~05:52 đã kill process khi đang chạy DQN hte.
  Không còn process; chỉ còn log + 2 ảnh đồ thị heuristic
  (`results/xrl-trained-vs-heuristic/graphs/`) + các dòng log đầy đủ
  (edge list + thời gian) trong `comparison_run.log`.
- Không mất BART hay venv; checkpoint DQN tốt nhất vẫn là run1
  (md5 fa8d467293f5dc40a52f37c2e0c88504, eval 246,16). run5 của agent khác
  cũng chết ở ep 400, chưa có checkpoint — không ảnh hưởng lựa chọn
  checkpoint (run1 vẫn tốt nhất).
- Khắc phục: script đã sửa để lưu tăng dần (`partial_results.json` sau mỗi
  policy) và chạy fidelity ngay sau mỗi policy (scm hte còn trong RAM),
  rồi chạy lại toàn bộ từ đầu (seed 0, kết quả heuristic phải giống hệt).

## VM replacement (~05:52, 2026-11-11) — môi trường ngoài ~/workspace bị xóa
- VM bị thay thế (uptime reset về 0). Mất: R (`/usr/lib/R`), dbarts
  (`/usr/local/lib/R/site-library`), repo clone `/tmp/causal-xrl-muc4`
  (tmpfs). Còn nguyên: mọi thứ dưới `~/workspace` (venv `.venv-ablation`
  với rpy2 3.5.16 + torch, scripts, results, notes, checkpoints).
- Đây KHÔNG phải lỗi cài đặt — lần cài đầu đã thành công. Khôi phục bằng
  đúng lệnh cũ (mất ~5 phút): `apt-get install -y r-base-core`,
  `R install.packages("dbarts")`, `git clone ... && git checkout ae9096e`.
- Rủi ro còn lại: R/dbarts sống ngoài `~/workspace` nên VM replacement sau
  sẽ lại xóa chúng; lệnh khôi phục đã ghi ở đây và trong `notes/bart-check.md`.
  Không chuyển R vào `~/workspace` (build R từ source tốn ~10+ phút và
  ~500 MB, không cần thiết khi apt reinstall chỉ mất vài phút).

## Bug trong script so sánh của nhóm (không phải lỗi BART) — đã fix
- Lần chạy 2 (sau khôi phục VM) crash lúc 06:11 ở
  `fidelity_eval_notebook`, dòng `y_test_bart = y[np.isin(idx_test, use_idx)]`:
  `IndexError: boolean index did not match ... size of axis is 6329 but size
  of corresponding boolean axis is 2532` — `y` là nhãn toàn bộ eval set
  (6329 bước), còn mask chỉ dài bằng test set (2532).
- Fix 1 dòng: `y_test_bart = y_test[np.isin(idx_test, use_idx)]`
  (lấy nhãn test tại đúng các vị trí của `use_idx` trong test set).
- Vòng lặp ITE BART-weighted đã chạy xong trước khi crash (200 bước, ~24 s);
  chỉ phần tính acc cuối bị lỗi. Chạy lại toàn bộ từ đầu sau fix.
- Số heuristic đã có trước crash (lần chạy 2): hte 12 cạnh / 239,41 s;
  ate 19 cạnh / 314,49 s; fidelity DT 0,9491/0,9498, LR 0,9111/0,9121
  (n_test=2532); valid_steps 6330/6330 (100% — bộ lọc local-graph của
  notebook không lọc được bước nào ở đây, xem REPORT cập nhật).

## Chết process khi chạy DQN ate (lần chạy 3) — nghi OOM, đã fix bằng giảm RAM
- Lần chạy 3 chết không traceback lúc ~06:38–06:48, ngay sau DQN hte xong
  (15 cạnh, infer 859,48 s), trong lúc chạy DQN ate. RAM lúc 06:43 chỉ còn
  158 MB trống, process chiếm 62,6% (~5 GB). Không có bằng chứng OOM trực
  tiếp trong dmesg/journal (không đọc được log kernel), nhưng chết im lặng
  không traceback + RAM cạn là đúng mẫu OOM-kill: scm hte DQN (5 model
  BART trên ~34k dòng train) vẫn giữ trong RAM trong khi ate fit thêm 5
  model nữa và giữ hết trong dict scm.
- Fix trong `scripts/run_xrl_comparison.py`: (1) `run_ate` bỏ mỗi model ngay
  sau khi lấy causes xong (`del model; gc.collect()`), chỉ giữ ATE table
  (surrogate mới có); (2) fidelity bỏ model BART sau mỗi fit; (3) sau mỗi
  policy, bỏ scm hte của policy đó khỏi RAM sau khi lưu tăng dần;
  (4) chạy với `--methods ate hte` (ate trước, hte sau) để đỉnh RAM là
  max(ate, hte+fidelity) thay vì tổng hai scm.
- Số DQN hte lần chạy 3 còn nguyên trong log + ảnh `graph_dqn_hte_200ep.png`:
  15 cạnh, infer 859,48 s. Chạy lại toàn bộ từ đầu sau fix (lần chạy 4);
  số heuristic/DQN hte có thể lệch ±1–2 cạnh giữa các lần chạy (xem ghi chú
  reproducibility trong REPORT cập nhật).

## Chết process lần 2 khi chạy fidelity DQN (lần chạy 4) — nghi OOM, đã fix
- Lần chạy 4 chết im lặng (không traceback) lúc ~07:21, trong lúc fit model
  BART đầu tiên của fidelity DQN (5 global causes: push, x, v, θ, w — DQN
  có action phụ thuộc cả 5 biến, khác heuristic chỉ 3). Lúc chết RAM còn
  ~1,1 GB trống, process ~4,7 GB (scm hte DQN 5 model lớn vẫn giữ).
- Fix trong `scripts/run_xrl_comparison.py` (`fidelity_eval_notebook`):
  trước khi fit các model BART fidelity, bỏ fitted model nặng khỏi mọi
  target của scm hte TRỪ target action (index 4) — local_treatments chỉ
  cần cây HTE nhỏ, ITE chỉ gọi trên target action. Real BART: `_bart_machine`;
  surrogate: `_model`. Đỉnh RAM lúc fidelity còn ~2–3 GB thay vì ~5,7 GB.
- Thêm lưu tăng dần sau MỖI method (không chỉ sau mỗi policy) vào
  `partial_results.json`, để chết process cũng không mất method đã xong.
- Số đã có trước khi chết (lần chạy 4, checkpoint run5 Double DQN):
  heuristic ate 18 cạnh/121,27 s, hte 13 cạnh/86,16 s, fidelity DT
  0,9491/0,9498 LR 0,9111/0,9121 BART 0,9350/0,9340 (n=200);
  DQN ate 16 cạnh/1050,37 s, hte 17 cạnh/818,97 s, fidelity DT 0,9513/0,9494
  LR 0,8615/0,8542 (n_test=19401); chỉ thiếu BART-weighted của DQN.
  Ảnh đồ thị cả 4 run đã lưu (graphs/). Chạy lại toàn bộ từ đầu sau fix.

## VM replacement lần 2 (~07:53) + khôi phục nhanh bằng dpkg cache
- VM replace lần 2: mất R + dbarts + /tmp (lần này cả /tmp bị xóa sạch).
  Boot replay của hệ thống (pid 804, `apt-get update`) giữ lock apt hơn
  8 phút chưa xong nên không apt-get install được.
- Khôi phục nhanh KHÔNG cần apt: `dpkg -i /var/cache/apt/archives/*.deb`
  (34 file .deb của lần cài trước còn nguyên trong cache) — R 4.3.3 +
  gfortran về trong 4 giây; sau đó `R install.packages("dbarts")` lại
  (0.9.34). Ledger os-intent đã ghi nhận lần cài R trước nên boot replay
  sau này cũng sẽ tự cài lại R (nhưng chậm).
- Bài học: sau VM replacement, khôi phục R bằng `dpkg -i` từ cache là
  đường nhanh nhất; đừng chờ apt lock. `~/workspace` (venv, repo copy,
  scripts, results) không bao giờ mất.

## VM restart 2 lần trong ngày 11/11 + chạy lại XRL với checkpoint run 9
- VM restart lần 3 (giết run 7 train ở ep 1040/1500) và lần 4 (giết
  full run XRL 11:05–11:38 giữa DQN ate): mất R + dbarts + /tmp cả 2
  lần. Khôi phục: lần 3 `apt-get install -y r-base-core gfortran`
  (APT RC=0, R 4.3.3); lần 4 `dpkg -i /var/cache/apt/archives/*.deb`
  (34 file .deb, R 4.3.3 về trong ~30 s). Sau mỗi lần, cài lại dbarts
  0.9.34 từ CRAN (`R install.packages("dbarts")`, biên dịch vài phút).
  `/tmp/causal-xrl-muc4` đã mất hẳn — lần chạy lại dùng
  `--repo repo-causal-xrl` (bản copy bền trong thư mục này).
- Full run XRL bị giết (checkpoint run 9): heuristic xong đầy đủ
  (hte 14 cạnh/85,84 s, ate 15/120,38 s, fidelity DT 0,9491/0,9498
  LR 0,9111/0,9121 BART 0,9350/0,9340, valid 6.329) + DQN hte
  (14 cạnh/1.738,58 s) nhờ `partial_results.json` lưu tăng dần;
  DQN ate + fidelity mất. DQN-only rerun (11:55–13:08,
  `--policies dqn`, cùng checkpoint, seed 0, 200 ep) làm lại DQN
  hte (14 cạnh/1.756,30 s — TẬP CẠNH GIỐNG HỆT lần bị giết, symdiff 0)
  + ate (16 cạnh/2.306,02 s) + fidelity (valid 54.472 = 54,6%,
  DT 0,9356/0,9392, LR 0,8195/0,8363, BART 0,8050/0,8368, n=200).
  Merge bằng `scripts/merge_xrl_comparison.py` (provenance đã cập
  nhật cho merge run 9): hte Jaccard 0,75 symdiff 4; ate Jaccard
  0,6316 symdiff 7. Bảng cũ (checkpoint run 5) lưu nguyên ở
  `results/xrl-trained-vs-heuristic-run5/`.
- RAM lần chạy lại: process lên tới ~4,5 GB lúc DQN ate (BART trên
  ~100k transitions), host có lúc chỉ còn ~250–460 MB free, nhưng
  KHÔNG chết OOM (fix bỏ model sau mỗi target vẫn giữ được); lần
  chết duy nhất là VM restart. DQN rollout 200 ep = 99.800
  transitions, mean_ep_len 500,0 (mọi episode chạm trần 500).
- Dải nhiễu BART quan sát tới nay (cập nhật): heuristic hte 13–14,
  ate 15–19 cạnh giữa các lần chạy cùng seed 0; fidelity BART
  heuristic 0,9200 (bảng run 5) → 0,9350 (bảng run 9).
