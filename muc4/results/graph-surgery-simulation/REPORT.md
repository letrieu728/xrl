# Mô phỏng phản thực bằng sửa đồ thị (graph surgery) — CartPole, 1 bước

**Bản thu hẹp. Ngày chạy: 2026-11-15. Chi phí: $0 (CPU local, không Modal, không API).**

Tiêu chí thành công được đặt TRƯỚC khi chạy, xem `PREREGISTRATION.md`
trong cùng thư mục — báo cáo này chỉ đối chiếu kết quả với các tiêu chí
đó, không sửa tiêu chí sau khi thấy kết quả.

## 0. Kết luận (đọc trước)

Theo quy tắc quyết định đặt trước, thí nghiệm này **KHÔNG đạt mức
"mô phỏng sửa đồ thị được kiểm chứng"**: 2/7 tiêu chí không đạt
(C-A3, C-B2). Kết quả từng phần như sau — không gộp thành một tuyên bố
chung chung:

| Hạng mục | Kết quả | Tiêu chí |
|---|---|---|
| Fidelity mô hình thay thế (held-out) | Đạt cổng chất lượng, nhưng `push'` của mô hình đồ thị (91,98%) kém xa baseline (97,47%) | F1 ✓, F2 ✓ |
| Loại (a) can thiệp state → action bước sau | Mô hình đồ thị 90,0% vs baseline 87,5%: hơn đúng 1/40 query — **không có khác biệt rõ** | C-A1 ✓, C-A2 ✓, C-A3 ✗ |
| Loại (b) can thiệp action → state kế tiếp | Mô hình đồ thị tốt hơn rõ ở vận tốc (MAE 0,0230 vs 0,0315), nhưng kém hơn ở (x', θ') | C-B1 ✓, C-B2 ✗, C-B3 ✓ |

Hai thất bại đều có cơ chế rõ ràng, và cả hai đều do **cạnh mà đồ thị
BART bỏ sót**, không phải do can thiệp:

1. **C-A3 không đạt** vì tiền đề của nó không xảy ra: baseline KHÔNG lật
   giả trên query can thiệp v/w (0/20), và cả hai mô hình cùng bỏ lỡ 3
   lần lật action thật do động lực học gây ra. Đồ thị không tránh được
   cái mà baseline cũng không mắc phải.
2. **C-B2 không đạt** vì đồ thị thiếu cạnh `w→θ'` (vật lý thật:
   θ' = θ + τ·w). Baseline dùng w nên dự đoán θ' tốt hơn. Nhưng về
   **tác động của can thiệp** (độ thay đổi Δ), mô hình đồ thị chính xác
   tuyệt đối: Δ thật của x', θ' bằng 0 và mô hình cũng dự đoán 0.

**Không tuyên bố đồ thị là nhân quả thật.** Xem caveat §7 — bắt buộc đọc
kèm mọi con số dưới đây.

Hướng này KHÁC với counterfactual DiCE 25/25 đã có (DiCE sửa feature để
lật dự đoán của một classifier). Hướng này can thiệp lên đồ thị rồi lan
truyền qua các phương trình cấu trúc từng node, và kiểm chứng bằng env
thật.

## 1. Phương pháp

### 1.1. Ngữ nghĩa đồ thị — DBN trễ 1 bước, không phải đồ thị đồng thời

Đồ thị dùng ở đây là đồ thị BART Mục 3 (13 cạnh, seed 99), lấy từ
`muc3/results/cartpole/metrics.json` (`bart_targets`), lưu lại tại
`cartpole_graph_bart_muc3.json` kèm provenance. Theo code gốc
(`muc3/scripts/cartpole_bart_target.py`: `replay_t → replay_tp1`), mỗi
quan hệ là `nguyên nhân(t) → đích(t+1)`:

| Đích (t+1) | Cha (t) |
|---|---|
| x' | x, v, θ |
| v' | v, push |
| θ' | v, θ |
| w' | θ, w, push |
| push' | x, θ, push |

Hệ quả quan trọng: "action dự đoán" của mô hình là `push'` — action ở
bước `t+1`. Mô hình không có phương trình nào dự đoán action đồng thời.
Kiểm chứng loại (a) vì vậy là: đặt env về state đã can thiệp, step 1
bước với action quan sát được, rồi chạy policy thật trên state kế tiếp
thật để được `a'_{thật}`, so với `push'` dự đoán. Vẫn đúng 1 bước env,
không rollout. Cột chẩn đoán phụ `policy_action_at_intervened_state`
trong JSON từng query ghi action của policy tại chính state đã can
thiệp (theo nghĩa đồng thời, không qua đồ thị).

Vì DBN chỉ có 1 lớp, baseline "không lan truyền" khác mô hình đồ thị
**chỉ ở tập feature** (mọi feature so với chỉ cha) — không có lan truyền
nhiều bước để ablation. Thí nghiệm này kiểm tra tính thưa theo cha
(parent-restriction), không kiểm tra lan truyền nhiều bước.

### 1.2. Mô hình thay thế và baseline

- Mô hình đồ thị: RandomForest riêng cho từng node đích (300 cây,
  `random_state=99`), MỖI NODE CHỈ DÙNG CÁC CHA của nó. `push'` dùng
  classifier, 4 tọa độ state dùng regressor.
- Baseline: cùng họ RandomForest, cùng seed và siêu tham số, nhưng mỗi
  node dùng MỌI feature (x, v, θ, w, push) — không cắt cạnh.
- Code interface chung (đồ thị + mô hình từng node + `intervene()`):
  `muc4/scripts/graph_surgery.py`. Script chạy CartPole:
  `muc4/scripts/run_graph_surgery_cartpole.py`.

### 1.3. Dữ liệu và chia tập

- Replay Mục 3, seed 99, 200 episode, 6.115 transition
  (SHA-256 `da5e73d1…89f09`, ghi đủ trong `summary.json`). Kiểm tra khớp:
  tái tạo đúng 200 episode (199 biên), heuristic khớp action 100%.
- Chỉ dùng cặp trong cùng episode: 5.915 cặp. **199 cặp qua biên episode
  mà code BART gốc vô tình đưa vào (dịch hàng đơn thuần) bị loại** khi
  huấn luyện surrogate, vì chúng không phải transition thật. Tập cạnh
  vẫn lấy nguyên như đã học — đây là khác biệt có chủ ý so với pipeline
  gốc, ghi rõ tại đây.
- Chia THEO EPISODE (tránh rò rỉ giữa hàng kề nhau): 160 episode train
  (4.730 cặp) / 40 episode test (1.185 cặp), `SPLIT_SEED=123`.
- Query lấy từ hàng TEST, không dùng hàng train.

### 1.4. Query can thiệp (80 query, `QUERY_SEED=20261115`)

- Loại (a) can thiệp state: 40 query = 4 feature × 10 (5 ngẫu nhiên + 5
  gần ngưỡng ngã mỗi feature). Chỉ can thiệp đúng 1 feature; action `a_t`
  giữ nguyên. Giá trị đặt vào nằm trong miền quan sát được:
  x ∈ {−0,15; −0,08; 0; 0,08; 0,15}, v ∈ {−2; −1; 0; 1; 2},
  θ ∈ {−0,19; −0,10; 0; 0,10; 0,19}, w ∈ {−2,4; −1,2; 0; 1,2; 2,4};
  khoảng cách tối thiểu tới giá trị gốc 0,05 (x/θ) / 0,5 (v/w).
  Miền quan sát: x ∈ [−0,195, 0,198] (rất hẹp — heuristic giữ xe gần
  tâm), v ∈ [−2,52, 2,55], θ ∈ [−0,209, 0,207], w ∈ [−3,11, 3,08].
- Loại (b) can thiệp action: 40 query, lật action (20 ngẫu nhiên + 20
  gần ngưỡng).
- Gần ngưỡng ngã: |θ| ≥ 0,15 rad (ngưỡng ngã |θ| > 0,20944 rad). Thực tế:
  loại (a) 21/40, loại (b) 23/40 query có base state gần ngưỡng
  (vượt mức tối thiểu đặt trước 15/40 và 20/40, vì một số query ngẫu
  nhiên cũng rơi vào vùng này).
- Phân phối giá trị can thiệp thực tế và từng query xem
  `queries_type_a.json` / `queries_type_b.json`.

### 1.5. Kiểm chứng bằng env thật

- Env `CartPole-v1`, gymnasium 1.4.0, đặt trực tiếp `env.state`, step 1
  bước. Policy thật: heuristic của tác giả, đúng công thức repo
  (`1 if 0.4*(x/2.4)+0.6*(θ/2.095)>=0 else 0`).
- Kiểm tra sanity: step từ 100 state replay với action replay tái tạo
  `next_states` với sai số tối đa 2,4e-07 (sai số float) — vật lý 1 bước
  của CartPole-v1 khớp replay CartPole-v0.
- Mỗi query có thêm 1 bước tham chiếu không can thiệp (từ state gốc với
  action gốc) để đo "lật thật" và "thay đổi thật". 3/40 query loại (a)
  kết thúc ở trạng thái terminated sau bước step (state gần ngưỡng);
  `next_state` vẫn hợp lệ và được dùng.
- DQN run 9 chỉ là chẩn đoán tùy chọn (§6), không nằm trong tiêu chí.

## 2. Fidelity của mô hình thay thế (đo TRƯỚC, trên held-out)

n = 1.185 cặp thuộc 40 episode test.

| Đích | Mô hình đồ thị | Baseline (mọi feature) | Persistence |
|---|---|---|---|
| x' — MAE / R² | 0,00080 / 0,99974 | 0,00091 / 0,99971 | 0,01471 / 0,95086 |
| v' — MAE / R² | 0,00130 / 0,99998 | 0,00148 / 0,99997 | 0,19402 / 0,95755 |
| θ' — MAE / R² | 0,00170 / 0,99953 | 0,00101 / 0,99982 | 0,01915 / 0,94389 |
| w' — MAE / R² | 0,00521 / 0,99996 | 0,00735 / 0,99992 | 0,26802 / 0,95035 |
| push' — accuracy / F1 | 0,9198 / 0,9164 | 0,9747 / 0,9740 | 0,9089 / 0,9059 |

- **F1 (accuracy push' ≥ 0,85): ĐẠT** (0,9198). Nhưng phải nói rõ: mô
  hình đồ thị chỉ hơn persistence (luôn dự đoán `push'=push_t`, 0,9089)
  có 1,1 điểm %, trong khi baseline đạt 0,9747. Cha (x, θ, push) bỏ mất
  tín hiệu từ v, w về action bước sau — đúng hạn chế đã ghi trong
  preregistration (đường v/w → động lực học → a' không được mô hình hóa).
  Số này tương đương số BART đã công bố (0,9162) nhưng split
  ở đây là theo episode, khó hơn split theo hàng, nên không so trực tiếp.
- **F2 (R² mỗi regressor ≥ 0,85): ĐẠT** — tất cả R² ≥ 0,9995. Động lực
  học CartPole gần như xác định nên cả hai mô hình đều dự đoán state kế
  tiếp rất tốt trên dữ liệu quan sát. Điểm đáng chú ý duy nhất: θ' của
  mô hình đồ thị kém hơn baseline (MAE 0,00170 vs 0,00101) vì thiếu cha
  w — xem §4, cơ chế này lặp lại dưới can thiệp.

Fidelity quan sát tốt không bảo đảm fidelity dưới can thiệp — đó chính
là điều §3–§4 kiểm tra.

## 3. Loại (a): can thiệp STATE → action bước sau

n = 40. Tỉ lệ `push'` dự đoán khớp `a'_{thật}`:

| Mô hình | Khớp | Tỉ lệ |
|---|---|---|
| Đồ thị (chỉ cha) | 36/40 | 90,0% |
| Baseline (mọi feature) | 35/40 | 87,5% |
| Chênh lệch | +1 query | +2,5 điểm % |

- **C-A1 (≥ 80%): ĐẠT.** **C-A2 (≥ baseline − 5 điểm %): ĐẠT.**
  Nhưng chênh lệch chỉ là 1 query (A026 — query duy nhất baseline sai mà
  mô hình đồ thị đúng; cả hai cùng sai A003, A014, A033, A034). **Không
  kết luận mô hình đồ thị tốt hơn baseline ở loại (a)** — khác biệt này
  không rõ ràng.
- Theo feature bị can thiệp (mỗi feature 10 query):

| Can thiệp | Đồ thị khớp | Baseline khớp | Lật thật (a' đổi) |
|---|---|---|---|
| x | 9/10 | 9/10 | 1/10 |
| v | 9/10 | 9/10 | 1/10 |
| θ | 10/10 | 9/10 | 5/10 |
| w | 8/10 | 8/10 | 2/10 |

  Can thiệp θ lật action bước sau trong 50% số query — cả hai mô hình
  đều theo kịp tốt. Can thiệp v/w ít lật action thật hơn.

- **C-A3: KHÔNG ĐẠT.** Trên 20 query can thiệp v/w (không phải cha của
  `push'`): tỉ lệ lật dự đoán của mô hình đồ thị là 0% (theo cấu trúc),
  nhưng của baseline cũng là **0%** — dưới ngưỡng 10% đặt trước. Đồng
  thời action thật lật trong 3/20 query (15%), và cả hai mô hình cùng
  khớp 17/20 (85%) — tức cả hai cùng bỏ lỡ đúng các lần lật thật qua
  đường động lực học. Kết luận đúng: tính thưa của đồ thị không mang lại
  lợi thế đo được ở đây, và cả hai mô hình đều mù trước đường
  v/w → s' → a'. Đây là hạn chế của mô hình 1 lớp, không phải bằng chứng
  cho hay chống đồ thị.
- Chẩn đoán phụ (không qua đồ thị): can thiệp state làm lật action tức
  thời của heuristic tại chính state đó trong 9/40 query (22,5%) —
  con số chính xác tầm thường từ công thức heuristic, chỉ ghi để trả lời
  câu hỏi theo nghĩa đồng thời.

## 4. Loại (b): can thiệp ACTION → state kế tiếp

n = 40. MAE so với state kế tiếp thật:

| Tọa độ | Mô hình đồ thị | Baseline | Chênh lệch |
|---|---|---|---|
| x' | 0,00112 | 0,00123 | −9% (đồ thị tốt hơn) |
| v' | 0,00305 | 0,00301 | +1% (ngang nhau) |
| θ' | 0,00232 | 0,00145 | +60% (đồ thị kém hơn) |
| w' | 0,04298 | 0,05997 | −28% (đồ thị tốt hơn rõ) |
| **Trung bình 4 tọa độ** | **0,01237** | **0,01641** | **−25%** |
| (x', θ') — không có cha push | 0,00172 | 0,00134 | +29% |
| (v', w') — có cha push | 0,02301 | 0,03149 | −27% |

- **C-B1 (MAE đồ thị ≤ baseline × 1,10): ĐẠT** (0,01237 < 0,01805).
- **C-B3 ((v',w') ≤ baseline × 1,10): ĐẠT**, với biên độ lớn — tính thưa
  giúp rõ rệt ở tọa độ vận tốc dưới can thiệp action.
- **C-B2 ((x',θ') ≤ baseline): KHÔNG ĐẠT** (0,00172 > 0,00134). Toàn bộ
  khoảng cách nằm ở θ' (x' đồ thị còn tốt hơn). Cơ chế: đồ thị thiếu
  cạnh `w→θ'`, mà vật lý thật là θ' = θ + τ·w (phép tích phân Euler
  tường minh — xem dưới),
  nên baseline dùng w dự đoán θ' tốt hơn. Đây là lỗi bỏ sót cạnh của
  BART, đã manh nha ở fidelity quan sát (§2: MAE θ' 0,00170 vs 0,00101).

Tác động của can thiệp (độ thay đổi trung bình |Δ| so với không can
thiệp) — phân tích mô tả, không phải tiêu chí:

| Tọa độ | Δ thật | Δ đồ thị dự đoán | Δ baseline dự đoán |
|---|---|---|---|
| x' | 0,00000 | 0,00000 | 0,00004 |
| v' | 0,38974 | 0,38730 | 0,38720 |
| θ' | 0,00000 | 0,00000 | 0,00001 |
| w' | 0,57942 | 0,53946 | 0,52557 |

Δ thật của x' và θ' bằng 0 chính xác: bộ tích phân Euler mặc định của
gymnasium cập nhật vị trí bằng vận tốc CŨ (`x = x + τ·x_dot` trước khi
cập nhật `x_dot`), nên trong 1 bước, x' và θ' không phụ thuộc action.
Đồ thị BART đúng khi không có cạnh push→x', push→θ', và mô hình đồ thị
dự đoán Δ = 0 chính xác tuyệt đối. Baseline dự đoán thay đổi giả rất nhỏ
(≤ 4e-05, không đáng kể). **C-B2 thất bại ở MAE tuyệt đối (do thiếu
cạnh w→θ'), không phải ở tác động can thiệp** — hai điều này phải được
phân biệt khi trích dẫn.

## 5. Quan sát phụ (mô tả, KHÔNG phải tiêu chí đặt trước)

MAE state kế tiếp dưới can thiệp STATE (loại (a), không nằm trong tiêu
chí — tiêu chí loại (a) chỉ xét action): mô hình đồ thị 0,0112 vs
baseline 0,1927. Khoảng cách nằm gần hết ở w': baseline sai trung bình
1,45–1,48 khi v hoặc w bị can thiệp về giá trị cực biên (±2,0 / −2,4),
với 5 query sai tới 2,2–3,0, trong khi mô hình đồ thị sai ≤ 0,044 trên
cùng các query đó (vd A034, A039, A017, A040, A016). Diễn giải (giả
thuyết, chưa kiểm tra độc lập): tổ hợp can thiệp nằm ngoài phân phối
hợp dù từng biên nằm trong miền quan sát, và mô hình mọi-feature rơi vào
lá cây không phù hợp; mô hình chỉ-cha ít bị ảnh hưởng hơn vì nó không
nhìn các feature bị can thiệp không phải cha. Quan sát này ủng hộ động
cơ của graph surgery nhưng KHÔNG được nâng thành kết luận đã kiểm chứng
— nó không có trong preregistration và n nhỏ.

## 6. Chẩn đoán DQN (tùy chọn, không nằm trong tiêu chí)

Checkpoint: `muc4/results/cartpole-dqn/run9/best_checkpoint.pt` trên
Drive ("XRL - file nhi phan"), tải về thư mục làm việc (KHÔNG commit
file `.pt`), SHA-256
`1c7e1f7fac3ebe8eab0182015e7c3825ec74b2f788683eae3d72169cc35f52e0`.
Đã xác minh metadata: run 9, `best_eval` 500,0, `final_eval_100ep`
500,0, hidden 128, env CartPole-v1. Script:
`muc4/scripts/dqn_agreement_diagnostic.py` (chạy bằng venv có torch);
kết quả chi tiết: `dqn_diagnostic.json`.

- Đồng thuận DQN-vs-heuristic tại state kế tiếp thật: 71,25% (57/80).
- Đồng thuận `push'` của mô hình đồ thị vs action DQN tại state kế tiếp
  thật: 67,5% (54/80).
- Diễn giải: DQN dùng toàn bộ state (kể cả vận tốc) nên khác heuristic
  (chỉ x, θ) là điều đã biết từ Mục 4 (đồ thị DQN có 5 cha cho `push`).
  Mô hình đồ thị học trên replay của heuristic nên không kỳ vọng khớp
  DQN. Chẩn đoán này chỉ ghi nhận, không dùng cho kết luận nào.

## 7. Caveat (bắt buộc — đọc kèm mọi con số)

- **Không chứng minh đồ thị là nhân quả thật.** Đồ thị học từ dữ liệu
  quan sát của một policy duy nhất (heuristic), không phải từ thí nghiệm
  can thiệp. Giả định không nhiễu ẩn (causal sufficiency) chưa được kiểm
  tra; nhiễu ẩn (vd trạng thái ẩn của policy, tương quan theo thời gian
  trong episode) có thể tạo cạnh giả.
- **Nhiễu BART ±1–2 cạnh.** Tập cạnh dùng ở đây là một lần chạy Mục 3
  (13 cạnh); heuristic hte dao động 12–14 cạnh qua các lần chạy, và đồ
  thị Mục 4 seed 0 (14 cạnh) khác vài cạnh. Kết quả có thể đổi nếu dùng
  tập cạnh khác — đặc biệt cạnh `w→θ'` (vắng mặt ở đây) mà §4 cho thấy
  là có thật về mặt vật lý.
- **Can thiệp ngoài phân phối.** Giá trị can thiệp nằm trong miền quan
  sát theo từng biên, nhưng tổ hợp can thiệp có thể nằm ngoài phân phối
  hợp mà mô hình từng thấy (bằng chứng: §5). RandomForest không ngoại suy.
- **Miền x rất hẹp.** Replay heuristic giữ |x| ≤ 0,20 nên không có bằng
  chứng nào về hành vi của mô hình khi x lớn; can thiệp x chỉ trong
  [−0,15, 0,15].
- **Mô hình 1 lớp.** `push'` được dự đoán trực tiếp từ cha ở bước t,
  không lan truyền qua s' dự đoán; đường can thiệp state → động lực học
  → action bước sau không được mô hình hóa (§3, C-A3).
- **Một env, một policy, một lần chạy mô hình.** n = 40+40 query; khác
  biệt 1 query (loại (a)) không có ý nghĩa thống kê. Không chạy MicroRTS
  (xem §8). Không lặp lại với seed khác.
- **199 cặp qua biên episode** đã lọt vào lúc học đồ thị gốc; surrogate
  ở đây loại chúng ra (§1.3). Đồ thị và surrogate vì vậy không dùng đúng
  một tập dữ liệu.

## 8. Phạm vi chưa làm

- **MicroRTS: KHÔNG chạy.** Code `graph_surgery.py` viết theo interface
  chung (đồ thị + mô hình từng node + `intervene()`) để adapter MicroRTS
  có thể dùng sau, và ghi rõ trong code là chưa chạy MicroRTS. Không có
  số liệu MicroRTS nào trong báo cáo này.
- Không rollout nhiều bước (đúng phạm vi thu hẹp).
- Không chạy lại BART, không học lại đồ thị.
- Không so với đồ thị Mục 4 seed 0 (14 cạnh) — chỉ dùng đồ thị Mục 3
  theo nhiệm vụ.

## 9. Tái chạy và file

```sh
# Đặt replay.npz (Drive, thư mục "XRL - file nhi phan"/muc3/results/cartpole/)
# vào muc3/results/cartpole/replay.npz, rồi:
python muc4/scripts/run_graph_surgery_cartpole.py \
    --replay muc3/results/cartpole/replay.npz
# Chẩn đoán DQN (tùy chọn, cần torch + checkpoint tải về, không commit .pt):
python muc4/scripts/dqn_agreement_diagnostic.py \
    --results-dir muc4/results/graph-surgery-simulation \
    --checkpoint /đường/dẫn/tới/dqn_run9_best_checkpoint.pt
```

- Seed: split 123, model 99, query 20261115 (ghi trong `summary.json`).
- Phiên bản: Python 3.12.3, scikit-learn 1.9.1, gymnasium 1.4.0,
  numpy 2.5.4. Thời gian chạy lần cuối: 63,6 s.
- Tính xác định: chạy lại lần 2 cho `summary.json`,
  `queries_type_a.json`, `queries_type_b.json`, `fidelity.json` giống hệt
  lần 1 (trừ `elapsed_seconds`), sau khi sửa bookkeeping
  `steps_beyond_terminated` để hết cảnh báo "step after terminated"
  (sửa này không đổi con số nào — đã kiểm chứng bằng diff).
- Minh bạch về lần chạy bị hủy: lần chạy đầu tiên (trước khi có kết quả
  nào — log trống) bị hủy sau ~28 phút vì vòng lặp dự đoán từng hàng với
  `n_jobs=-1` quá chậm dưới tải CPU chung. Sửa thành dự đoán theo batch
  + `n_jobs=1` lúc dự đoán; đã kiểm chứng batch và từng hàng cho kết quả
  giống hệt (sai số tối đa 0,0) trên mẫu thử trước khi chạy lại. Không
  xem kết quả nào trước khi sửa; tiêu chí không đổi.

File trong thư mục này:

| File | Nội dung |
|---|---|
| `PREREGISTRATION.md` | Tiêu chí đặt trước khi chạy (không sửa sau) |
| `cartpole_graph_bart_muc3.json` | Đồ thị 13 cạnh + provenance |
| `REPORT.md` | Báo cáo này |
| `summary.json` | Tóm tắt số liệu + đánh giá tiêu chí |
| `fidelity.json` | Fidelity held-out từng node |
| `queries_type_a.json` | 40 query loại (a), từng query + kiểm chứng env |
| `queries_type_b.json` | 40 query loại (b), từng query + kiểm chứng env |
| `dqn_diagnostic.json` | Chẩn đoán DQN (tùy chọn) |
| `run.log` | Log lần chạy cuối (giống hệt lần 1 trừ thời gian) |

Script: `muc4/scripts/graph_surgery.py` (interface chung),
`muc4/scripts/run_graph_surgery_cartpole.py` (thí nghiệm),
`muc4/scripts/dqn_agreement_diagnostic.py` (chẩn đoán tùy chọn).

File nhị phân (không stage lên repo, để trong thư mục làm việc
`~/workspace/xrl-g4c-work/`): `graph_surgery_summary.png` (biểu đồ tóm
tắt 2 panel: fidelity/match action + MAE loại (b) theo tọa độ),
`dqn_run9_best_checkpoint.pt` (checkpoint tải về từ Drive, không commit).
