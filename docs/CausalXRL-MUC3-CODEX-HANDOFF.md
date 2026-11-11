# CausalXRL — Mục 3 (Thực nghiệm) — Bộ quy tắc giao cho Codex

**Giao cho:** Codex (agent thực thi)
**Phạm vi duy nhất:** Mục 3 trong `CausalXRL-KE-HOACH-CHAM-DIEM.md` — chạy thực nghiệm evaluation-only của CausalXRL.
**Không làm:** báo cáo viết, deck, kịch bản, bản dịch, mở rộng (§4, kể cả hướng D train RL thật — đó là việc khác, không thuộc gói này).

---

## 0. Bối cảnh đã kiểm chứng — đọc trước khi làm gì

- **Paper:** CausalXRL: Explainable Reinforcement Learning through Causal Graph Reasoning — Yanming Zhang, Eric Papenhausen, Klaus Mueller, Stony Brook. arXiv:2610.42028v1, ICML 2026, 17 trang.
- **Repo chính thức:** https://github.com/yanmluk/causal-xrl — commit `ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162`, 3 commits, 0 stars, MIT.
- **Tài liệu đã có trong cùng thư mục này (đọc hết trước khi chạy):**
  - `CausalXRL-Tom-tat-noi-dung.md` — tóm tắt paper + Table 1
  - `CausalXRL-Phan-tich-ma-nguon.md` — phân tích code repo thật, đã đọc từng file
  - `CausalXRL-KE-HOACH-CHAM-DIEM.md` — kế hoạch chấm điểm, §3 là phạm vi của gói này
  - `CausalXRL-2610.42028.pdf` — PDF gốc
- **Sự thật quan trọng nhất:** repo **không có code train agent RL**. Policy trong notebook chỉ là **heuristic viết tay** (CartPole: luật tuyến tính trên vị trí/góc; LunarLander: PD controller). Nhiệm vụ của gói này là **evaluation-only**: chạy policy có sẵn qua pipeline XRL của tác giả, đo kết quả, so với paper. **Không train RL, không viết agent mới, không thay policy.**

## 1. Mục tiêu — thành công nghĩa là gì (acceptance criteria)

Hoàn thành khi **tất cả** các mục sau đều có bằng chứng chạy thật, không phải số copy từ paper:

1. **CartPole chạy xong** — pipeline XRL đầy đủ: `XRL(env, pi)` → `infer_from_env(nepisodes=200)` → `infer_DBN()` → `DBN_to_causal()` → local model → đo accuracy/F1 dự đoán action so với Linear Model + Decision Tree, và đo thời gian học cấu trúc so với PC/FCI.
2. **LunarLander chạy xong** — cùng pipeline, cùng 2 phép đo. Ghi rõ nếu notebook eval gốc chỉ dùng 30 episode (vì nặng) thì nhóm dùng bao nhiêu và vì sao.
3. **PongDuel chạy xong** — cùng pipeline, cùng 2 phép đo. Đây là môi trường multi-agent (ma-gym), khó nhất trong 3 môi trường; nếu sau 3 lần fix vẫn không chạy được thì dừng ở đây và báo cáo rõ (xem §6), không được bỏ qua trong im lặng.
4. **Bảng kết quả của nhóm** — 1 file CSV + 1 bảng Markdown, cùng format với Table 1 của paper, có đủ: accuracy/F1 của LM / DT / CausalXRL và thời gian học cấu trúc PC / FCI / CausalXRL cho từng môi trường đã chạy được.
5. **Ít nhất 1 đồ thị nhân quả học được** (global causal graph) cho CartPole, lưu thành file ảnh, và ít nhất 1 ví dụ local causal model.
6. **Báo cáo chạy** (`results/REPORT.md`) ghi rõ: máy chạy, phiên bản Python/thư viện, seed, số episode, thời gian chạy, toàn bộ số đo, và giải thích chênh lệch so với paper (số nhóm sẽ không khớp 100% paper vì policy heuristic + seed + version thư viện khác — đó là bình thường, ghi rõ, không sửa số cho khớp).

**Số của tác giả để đối chiếu (Table 1, arXiv:2610.42028v1 — chỉ dùng để so sánh, không được ghi thành số của nhóm):**

| Môi trường | CausalXRL Acc | CausalXRL F1 | DT Acc | CausalXRL time | PC time | FCI time |
|---|---|---|---|---|---|---|
| CartPole | 91,4% | 91,6% | 94,1% | 0,1s | 1,7s | 0,8s |
| LunarLander | 88,7% | 86,6% | 85,1% | 0,3s | 12,6s | 5,1s |
| PongDuel | 90,2% | 90,7% | 88,3% | 0,5s | 23,2s | 8,7s |
| MicroRTS-Lite | 86,0% | 85,9% | 80,3% | 0,5s | 20,9s | 7,4s |

**Không làm MicroRTS-Lite.** Lý do đã kiểm chứng: requirements cũ (TF 2.7, torch 1.8, gym-microrts 0.3.2), import `stratvis` sót trong `app.py`, thiếu dữ liệu (README ghi phải liên hệ tác giả). Ghi rõ "không làm MicroRTS" trong báo cáo chạy, không thử chạy nó.

## 2. Thứ tự làm — bắt buộc theo thứ tự này

1. **Phase 0 — Đọc + freeze (không chạy gì cả).** Clone repo đúng commit `ae9096e`. Đọc `CausalXRL-Phan-tich-ma-nguon.md` và các file: `cxrl/lib/experience/experience.py`, `cxrl/lib/causal/infer_network.py`, `cxrl/lib/models/bart_regressor.py`, `cxrl/lib/xrl/xrl.py`, `cxrl/lib/r_to_py/r_to_py.py`, `cxrl/requirements.txt`, các notebook `cxrl/jupyter/examples/CartPole.ipynb`, `LunarLander.ipynb`, `PongDuel.ipynb` và `cxrl/jupyter/eval/eval_fidelity_Cart Pole.ipynb`, `eval_fidelity_Lunar Lander.ipynb`. Ghi lại vào `notes/phase0.md`: pipeline chạy như thế nào, file nào làm gì, notebook nào dùng cho việc gì.
2. **Phase 1 — Môi trường + rào cản R/dbarts.** Cài theo `cxrl/requirements.txt` (Python ≥3.8, gym==0.20.0, rpy2==3.5.0, causal-learn, dice-ml, ma-gym từ commit đã pin). Chạy `setup_R()` để cài R package `dbarts` (+ `hexbin`, `rJava` — cần Java). **Đây là rào cản lớn nhất, đã biết trước.** Nếu bước này fail, đó là việc của Phase 1, không được nhảy sang chạy notebook khi chưa có BART.
3. **Phase 2 — CartPole trước.** Chạy example notebook CartPole, rồi eval notebook CartPole. Đây là môi trường dễ nhất, là bằng chứng pipeline chạy được. Chỉ khi CartPole có số đo đầy đủ mới sang LunarLander.
4. **Phase 3 — LunarLander.** Chạy example + eval. Ghi rõ số episode thực dùng.
5. **Phase 4 — PongDuel.** Chạy example. Lưu ý: không có notebook eval PongDuel trong repo — phải tự viết eval theo đúng flow của 2 notebook eval đã có (giữ steps có local graph >1 cause cho action, so DT vs LogisticRegression vs BART weighted by |ITE|).
6. **Phase 5 — Tổng hợp.** Viết bảng kết quả, vẽ đồ thị nhân quả, viết `results/REPORT.md`.

## 3. Quy tắc tự tìm hiểu — tự build — tự fix đến khi success

Đây là phần quan trọng nhất của gói này. Codex phải làm việc như một kỹ sư, không phải như người chạy lệnh thuê:

### 3.1. Tự tìm hiểu (không hỏi owner những gì tự đọc được)
- Trước khi hỏi bất cứ gì, đọc code thật, đọc lỗi thật, đọc README thật, đọc notebook thật. Mọi câu trả lời về "file này làm gì", "hàm này nhận gì", "notebook này chạy ra sao" đều có trong repo hoặc trong `CausalXRL-Phan-tich-ma-nguon.md`.
- Khi gặp lỗi, đọc **toàn bộ** traceback, không chỉ dòng cuối. Xác định: lỗi ở file nào, dòng nào, do thiếu package, sai version, sai import, hay logic bug.
- Khi không chắc một API dùng thế nào (gym 0.20, rpy2, dbarts, dice-ml, causal-learn), tra docs chính thức của thư viện đó, không đoán.
- Chỉ hỏi owner khi: cần quyết định vượt quá phạm vi (§6), cần tiền/tài nguyên, hoặc đã thử hết cách mà vẫn fail sau giới hạn ở §3.3.

### 3.2. Tự build — tự fix (vòng lặp bắt buộc)
Mỗi lần chạy fail, làm đúng vòng lặp này, không bỏ bước:
1. **Ghi lại lỗi** vào `notes/errors.md`: lệnh đã chạy, toàn bộ traceback, phiên bản thư viện liên quan.
2. **Chẩn đoán nguyên nhân gốc** (root cause), không fix triệu chứng. Ví dụ: lỗi `ModuleNotFoundError: ai_analyzer` không phải do thiếu pip install, mà do hàm `infer_causes_ate_barl()` import module không tồn tại trong repo — cách fix đúng là không gọi hàm đó, không phải đi tìm package `ai_analyzer`.
3. **Fix tối thiểu, đúng chỗ.** Sửa đúng 1 vấn đề mỗi lần, không sửa lan man nhiều file cùng lúc. Nếu phải sửa code repo, sửa trong bản copy làm việc, **không commit lên repo gốc**, và ghi rõ trong `notes/fixes.md`: file nào, dòng nào, sửa gì, vì sao.
4. **Chạy lại từ đầu bước đó** (không chạy tiếp từ giữa khi state đã bẩn). Kiểm tra kết quả thay đổi.
5. **Lặp lại** cho đến khi bước đó success, hoặc chạm giới hạn ở §3.3.

### 3.3. Giới hạn và điểm dừng (để không fix vô hạn)
- **Mỗi lỗi cụ thể tối đa 3 lần fix.** Sau 3 lần fix mà cùng một lỗi vẫn còn, dừng lại, ghi vào `notes/blockers.md` (lỗi gì, đã thử 3 cách nào, kết quả mỗi cách), rồi chuyển sang bước tiếp theo có thể làm được, hoặc sang phương án fallback ở §5. Không được fix lần thứ 4 cùng một lỗi mà không có giả thuyết mới.
- **Mỗi môi trường tối đa 1 ngày làm việc.** Hết 1 ngày mà môi trường đó chưa chạy được thì ghi blocker, báo cáo, chuyển sang môi trường tiếp theo (trừ CartPole — CartPole là bắt buộc, nếu CartPole fail thì dừng toàn bộ và báo cáo, vì đó là bằng chứng pipeline không chạy được).
- **Không bao giờ:** sửa số kết quả cho khớp paper; chạy với dữ liệu giả rồi ghi thành số thật; bỏ qua lỗi bằng `try/except: pass` rồi ghi success; cài package lạ từ nguồn không rõ để "cho chạy được".

### 3.4. Bằng chứng chạy thật (không có bằng chứng = chưa làm)
- Mọi kết quả phải có ít nhất 1 trong: file log chạy, file CSV số đo, ảnh đồ thị, notebook đã chạy có output. Lưu vào `results/` (xem §4).
- Mỗi lần chạy thành công, lưu ngay: lệnh đã chạy, thời gian chạy, output đầy đủ. Không chạy lại "cho chắc" rồi ghi đè log cũ — log cũ là bằng chứng.
- Số đo phải có đơn vị, số episode, seed, và điều kiện chạy kèm theo. Số không có điều kiện kèm theo là số không dùng được.

## 4. Cây thư mục đầu ra — bắt buộc

```
cauxrl-muc3/
├── notes/
│   ├── phase0.md          # pipeline + file nào làm gì (Phase 0)
│   ├── errors.md          # mọi lỗi gặp, kèm traceback đầy đủ
│   ├── fixes.md           # mọi chỗ đã sửa, file/dòng/lý do
│   └── blockers.md        # lỗi sau 3 lần fix vẫn còn + đã thử gì
├── scripts/
│   └── *.py / *.sh        # mọi script tự viết (eval PongDuel, đo thời gian, vẽ đồ thị)
├── results/
│   ├── cartpole/          # log, CSV, ảnh đồ thị CartPole
│   ├── lunarlander/       # log, CSV, ảnh đồ thị LunarLander
│   ├── pongduel/          # log, CSV, ảnh đồ thị PongDuel (nếu chạy được)
│   ├── results_table.csv  # bảng kết quả tổng hợp, format như Table 1
│   ├── results_table.md   # bảng Markdown tương ứng
│   └── REPORT.md          # báo cáo chạy: điều kiện, số đo, so sánh paper, chênh lệch, blocker
└── CausalXRL-MUC3-CODEX-HANDOFF.md  # file này
```

## 5. Bug và rủi ro đã biết trước — đừng mất thời gian phát hiện lại

Đã phân tích code thật, liệt kê sẵn để Codex xử lý thẳng, không phải mò lại từ đầu:

1. **`infer_causes_ate_barl()` import module không tồn tại** (`ai_analyzer.lib.bayesian.barl` không có trong repo). Cách xử lý: **không gọi hàm này**. Dùng `infer_causes_hte()` hoặc `infer_causes_ate()` hoặc `infer_causes()` (CITestFast) như notebook vẫn dùng.
2. **`counterfactual()` gọi `self.pi.predict(...)` nhưng `pi` là hàm thường** — trong `XRL.__init__`, dòng `self.pi = DummySklearn(pi)` đang bị comment, `self.pi = pi` được dùng. Một policy dạng hàm sẽ raise `AttributeError` ở bước simplify counterfactual. Cách xử lý: bọc policy bằng `DummySklearn` (class đã có sẵn trong `xrl.py`) trước khi gọi `counterfactual()`, hoặc chỉ gọi counterfactual khi policy là sklearn-like object. Ghi rõ cách đã chọn vào `notes/fixes.md`.
3. **`HTE.__init__` có `assert hte.shape[0] == 500` hardcode.** Nếu số mẫu ITE khác 500 thì assert fail. Cách xử lý: kiểm tra số mẫu thực tế trước, nếu khác 500 thì sửa assert thành kiểm tra đúng số mẫu (ghi rõ vào `notes/fixes.md`), không xóa assert trong im lặng.
4. **`xrl_test.py` tham chiếu `xrl.replay` nhưng class hiện tại dùng `replay_buffer`.** Test đó đã cũ, không dùng làm chuẩn. Dựa vào notebook, không dựa vào test.
5. **MicroRTS: không làm.** `app.py` import `from stratvis.lib...` trong khi package thật là `cxrl_microRTS` — sẽ ImportError. Requirements 2021–2022, thiếu dữ liệu. Không fix MicroRTS trong gói này.
6. **LunarLander eval notebook dùng NEPISODES = 30, không phải 200.** Ghi rõ số episode thực dùng trong báo cáo, không ghi 200 nếu chỉ chạy 30.
7. **BART qua R là rào cản lớn nhất.** Cần R + Java + `dbarts` + `rpy2` cùng version tương thích. Nếu `rpy2` không import được, kiểm tra `R_HOME`, version R, version Python trước khi cài lại. Đây là bước tốn thời gian nhất — dành thời gian cho nó, đừng vội.

## 6. Khi nào thì dừng và hỏi owner — không tự quyết

Dừng lại và báo cáo (ghi vào `notes/blockers.md` + báo owner) khi gặp 1 trong các trường hợp sau, không tự ý quyết thay:

- CartPole sau 3 lần fix vẫn không chạy được (đây là blocker toàn bộ gói).
- BART/dbarts không cài được sau khi đã thử đủ cách (cài R, Java, `rpy2`, kiểm tra `R_HOME`, thử version khác) — lúc đó cần owner quyết: đổi hướng dùng DecisionTree làm baseline thay thế (ghi rõ là thay thế, không giả vờ là BART), hay dừng thực nghiệm.
- Cần tốn tiền (tiền API, tiền thuê máy, tiền mua dữ liệu) — gói này là **CPU-only, $0**. Không chi tiền khi chưa hỏi.
- Cần đổi phạm vi: làm thêm môi trường mới, train RL, làm MicroRTS, làm human study — tất cả đều **ngoài phạm vi gói này**.
- Cần tải dữ liệu lớn (hàng GB) hoặc liên hệ tác giả xin dữ liệu — hỏi trước, không tự tải.

## 7. Quy tắc trung thực — vi phạm là fail cả gói

- **Không bịa số.** Số nào không chạy ra thì ghi "chưa chạy", không ghi số ước lượng, không copy số paper thành số nhóm.
- **Không trình bày demo chưa chạy thành "tái hiện đầy đủ".** Ghi đúng: chạy được môi trường nào, chưa chạy được môi trường nào, vì sao.
- **Không giấu blocker.** Lỗi sau 3 lần fix vẫn còn phải ghi vào `notes/blockers.md` và báo cáo, không được bỏ qua trong im lặng.
- **Không dùng GPT judge hay LLM-as-Judge** để đánh giá. Đo bằng accuracy/F1/exact match như paper Table 1.
- **Tách rõ:** số của tác giả (trích paper, ghi nguồn) vs số của nhóm (chạy thật, ghi điều kiện). Hai loại số không bao giờ trộn vào cùng một bảng mà không ghi rõ cột nào là của ai.

## 8. Checklist bàn giao — Codex tự tick trước khi báo xong

- [ ] CartPole: log + CSV + ảnh đồ thị + accuracy/F1 + thời gian học cấu trúc
- [ ] LunarLander: log + CSV + ảnh đồ thị + accuracy/F1 + thời gian học cấu trúc (ghi rõ số episode)
- [ ] PongDuel: log + CSV + ảnh đồ thị + accuracy/F1 + thời gian học cấu trúc, hoặc blocker ghi rõ vì sao không chạy được
- [ ] `results/results_table.csv` + `results_table.md` đủ 3 môi trường (hoặc ghi rõ môi trường nào thiếu và vì sao)
- [ ] `results/REPORT.md` có: điều kiện chạy, số đo, so sánh với Table 1 paper, giải thích chênh lệch, danh sách blocker
- [ ] `notes/errors.md` + `notes/fixes.md` + `notes/blockers.md` đầy đủ
- [ ] Không có số bịa, không có số paper ghi thành số nhóm, không có tiền phát sinh
- [ ] MicroRTS không làm, đã ghi rõ lý do trong REPORT.md

---

*File này là quy tắc giao việc cho Codex làm Mục 3. Owner: Lê Quốc Triệu. Ngày tạo: 11/11/2026.*
