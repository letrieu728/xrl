# BART / R / dbarts — kiểm tra hiện trạng (2026-11-11, ~12:25 +07)

## Kết quả kiểm tra

| Hạng mục | Kết quả |
|---|---|
| `R` trên PATH | **KHÔNG CÓ** — `which R` trống, `R --version` → command not found, `/usr/lib/R` không tồn tại |
| `rpy2` trong `.venv-ablation` | **KHÔNG CÓ** — `ModuleNotFoundError: No module named 'rpy2'` |
| `rpy2` trong `~/workspace/dqn-venv` | **KHÔNG CÓ** — `ModuleNotFoundError: No module named 'rpy2'` |
| R package `dbarts` | Không thể có — chưa có R |
| Codex Mục 3 (BART) — thư mục/notes | **KHÔNG TÌM THẤY** output nào. Chỉ có file handoff `CausalXRL-MUC3-CODEX-HANDOFF.md` trong thư mục goal; không có thư mục `results/` hay `notes/` nào của Mục 3, không có venv/log nào cho thấy R/dbarts đã từng cài được |
| Repo clone `/tmp/causal-xrl-muc4` | Còn, commit cần xác minh bằng `git rev-parse HEAD` khi chạy |
| OS | Ubuntu 24.04.5 LTS, Python 3.12.3, root (sudo không cần mật khẩu) |
| Tài nguyên lúc kiểm tra | 4 CPU, RAM 7,9 GB (chỉ còn ~600 MB free lúc kiểm tra — có agent khác đang train run4 song song), disk 147 GB trống |

## Kết luận (cập nhật sau khi cài — 2026-11-11)

BART **CÀI ĐƯỢC và CHẠY ĐƯỢC thật** (không cần surrogate cho Mục 4):

- R 4.3.3 (`apt install r-base-core`) + dbarts 0.9.34 (CRAN) + rpy2 3.5.16
  (xem chi tiết lỗi + fix trong `notes/bart-errors.md`).
- Smoke test `scripts/smoke_bart.py`: fit/predict/ATE/ITE/is_causal_HTE đều
  đúng — ATE col0 = 1.010, CI (0.890, 1.131) với hiệu ứng thật +1/SD;
  ITE (500, 50) nên assert 500 draws của repo thỏa; is_causal_HTE phân biệt
  đúng True/True/False.
- So sánh XRL trained-vs-heuristic bằng BART thật đang chạy:
  `scripts/run_xrl_comparison.py --estimator bart`, 200 episode/policy,
  seed 0, checkpoint DQN tốt nhất lúc chạy (xem `results/xrl-trained-vs-heuristic/`).
- Venv chạy BART: `.venv-ablation` (đã thêm rpy2 3.5.16 + torch 2.15.0+cpu);
  cần `R_HOME=/usr/lib/R` (script tự đặt mặc định nếu chưa có).
