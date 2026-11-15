# Preregistration — Mô phỏng phản thực bằng sửa đồ thị (graph surgery), bản thu hẹp

**Viết trước khi chạy thí nghiệm. Không sửa tiêu chí sau khi thấy kết quả.**

- Ngày viết: 2026-11-15 (giờ +07, trước lần chạy đầu tiên của script thí nghiệm).
- Repo: `letrieu728/XRL`, nhánh `main` tại thời điểm đọc (commit `fa263a52`,
  xem `cartpole_graph_bart_muc3.json` để biết provenance đầy đủ).
- Môi trường: CHỈ CartPole, CHỈ 1 bước. KHÔNG rollout nhiều bước.
  KHÔNG chạy MicroRTS — code viết theo interface chung để sau này chuyển
  được, nhưng MicroRTS chưa chạy (ghi rõ trong code và báo cáo).
- Chi phí: $0 tuyệt đối — CPU máy ảo local, không Modal, không API trả phí.

## 1. Câu hỏi

Đồ thị BART Mục 3 (13 cạnh) có dùng được để mô phỏng can thiệp 1 bước
không — tức là: mô hình thay thế (surrogate) mỗi node chỉ dùng các cha
theo đồ thị, khi bị can thiệp (sửa đồ thị) rồi lan truyền, có dự đoán
khớp với môi trường CartPole thật ít nhất ngang baseline dùng mọi
feature hay không?

Đây là hướng KHÁC với counterfactual DiCE 25/25 đã có (DiCE sửa feature
để lật dự đoán của một classifier; hướng này can thiệp lên đồ thị rồi
lan truyền qua các phương trình cấu trúc từng node).

## 2. Ngữ nghĩa đồ thị (quyết định trước, theo code gốc của nhóm)

Đồ thị là DBN trễ 1 bước, KHÔNG phải đồ thị đồng thời. Bằng chứng:
`muc3/scripts/cartpole_bart_target.py` dùng `replay_t = [s_t, a_t][:-1]`
để dự đoán `replay_tp1[:, target] = [s_{t+1}, a_{t+1}][1:]`, và
`muc3/EXPERIMENT_REPORT.md` ghi quan hệ nguyên nhân là `cause(t) ->
target(t+1)`. Do đó:

- Node đích là biến ở bước `t+1`; cha là biến ở bước `t`.
- "Action dự đoán" của mô hình là `push'` = action ở bước `t+1`,
  với các cha `(x, θ, push)` ở bước `t`. Mô hình KHÔNG có phương trình
  nào dự đoán action đồng thời `push_t` từ `s_t`.
- Kiểm chứng bằng env thật cho loại (a) vì vậy là: đặt env về state đã
  can thiệp, step 1 bước với action quan sát được `a_t` (giữ cố định —
  chỉ can thiệp đúng 1 biến state), lấy `s'_{thật}`, rồi chạy policy
  thật (heuristic) trên `s'_{thật}` để được `a'_{thật}`, so với
  `push'` dự đoán. Vẫn là đúng 1 bước env, không rollout.
- Cột chẩn đoán phụ (không nằm trong tiêu chí): action của policy thật
  tại chính state đã can thiệp, `heuristic(s_t^{do})`, và nó có lật so
  với `a_t` hay không. Cột này trả lời câu hỏi theo nghĩa đồng thời
  theo đúng chữ của nhiệm vụ, nhưng nó không đi qua đồ thị (heuristic
  là công thức đã biết, cho đáp án chính xác tầm thường) nên không dùng
  làm bằng chứng cho đồ thị.

## 3. Mô hình thay thế và đo fidelity TRƯỚC

- Dữ liệu: replay Mục 3, seed 99, 200 episode, 6.115 transition
  (`muc3/results/cartpole/replay.npz` trên Drive; bản local dùng để chạy
  đã kiểm tra khớp: 199 biên episode, heuristic khớp action 100%).
- Cặp DBN: chỉ cặp trong cùng episode (`next_states[i] == states[i+1]`,
  5.915 cặp). 199 cặp qua biên episode mà code BART gốc vô tình đưa vào
  (dịch hàng đơn thuần) bị LOẠI khi huấn luyện surrogate, vì chúng không
  phải transition thật. Đồ thị (tập cạnh) vẫn lấy nguyên như đã học.
- Chia train/test THEO EPISODE (nhóm episode, tránh rò rỉ giữa các hàng
  kề nhau): xáo trộn 200 episode bằng seed, 160 episode train / 40 test.
- Mô hình đồ thị: RandomForest riêng cho từng node đích, MỖI NODE CHỈ
  DÙNG CÁC CHA của nó theo đồ thị BART:
  `x'←(x,v,θ)`, `v'←(v,push)`, `θ'←(v,θ)`, `w'←(θ,w,push)`,
  `push'←(x,θ,push)` (classifier; còn lại regressor).
- Baseline: cùng họ RandomForest, cùng seed và siêu tham số, nhưng MỖI
  NODE DÙNG MỌI FEATURE `(x,v,θ,w,push)` — không cắt cạnh, không lan
  truyền theo đồ thị.
- Fidelity đo trên test episodes: regressor = MAE/RMSE/R² từng tọa độ;
  classifier = accuracy/F1. Kèm tham chiếu persistence
  (`push'` = `push_t`; `s'` = `s_t`).

### Tiêu chí fidelity (cổng chất lượng, đo trước khi diễn giải query)

- **F1**: accuracy của classifier `push'` (mô hình đồ thị) trên held-out
  ≥ 0,85. Căn cứ: số đã công bố trên cùng bài toán `[s_t,a_t]→a_{t+1}`
  là BART 0,9162 / cây quyết định 0,9542 (split theo hàng, khác split
  của thí nghiệm này); 0,85 thấp hơn cả hai để chừa room cho split theo
  episode khó hơn, và cao hơn hẳn mức may rủi 0,50. Nếu thấp hơn cả
  persistence (~0,91) thì ghi rõ trong báo cáo.
- **F2**: R² held-out của MỖI regressor tọa độ ≥ 0,85.
- Nếu F1/F2 không đạt: vẫn chạy query, nhưng KHÔNG kết luận "đồ thị
  dùng được", và mọi kết quả mô phỏng phải diễn giải trong giới hạn
  fidelity đó, ghi rõ trong báo cáo.

## 4. Thiết kế query (80 query, seed cố định, ghi trước khi chạy)

- Nguồn state: hàng replay thuộc TEST episodes (không dùng hàng train,
  tránh mô hình đã nhớ).
- Loại (a) — can thiệp STATE: 40 query = 4 feature × 10. Mỗi feature:
  giá trị đặt vào lấy trong miền quan sát được (xem bảng miền bên dưới),
  cách giá trị gốc một khoảng tối thiểu (|Δ| ≥ 0,05 với x/θ; ≥ 0,5 với
  v/w; nếu không đạt thì lấy mẫu lại). Chỉ can thiệp đúng 1 feature;
  action `a_t` giữ nguyên giá trị quan sát được; cạnh vào của node bị
  can thiệp bị cắt (node đó không còn được dự đoán từ cha nữa — với
  node ở bước `t` thì điều này nghĩa là giá trị can thiệp được đưa
  thẳng vào các phương trình của bước `t+1`).
- Loại (b) — can thiệp ACTION: 40 query, lật action (`do(push)=1-a_t`),
  state giữ nguyên. 20 query ngẫu nhiên + 20 query từ state gần ngưỡng
  ngã.
- State gần ngưỡng ngã: `|θ| ≥ 0,15` rad (ngưỡng ngã của env là
  |θ| > 0,20944 rad; replay có 951 hàng thỏa |θ| ≥ 0,15). Loại (a) đảm
  bảo ≥ 15/40 query có base state gần ngưỡng; loại (b) 20/40.
- Miền quan sát được (từ replay Mục 3, sẽ in lại từ dữ liệu trong log):
  x ∈ [−0,20, 0,20] (heuristic giữ xe gần tâm nên miền x rất hẹp),
  v ∈ [−2,53, 2,55], θ ∈ [−0,209, 0,208], w ∈ [−3,11, 3,08].
  Giá trị can thiệp: x ∈ {−0,15; −0,08; 0; 0,08; 0,15},
  v ∈ {−2,0; −1,0; 0; 1,0; 2,0}, θ ∈ {−0,19; −0,10; 0; 0,10; 0,19},
  w ∈ {−2,4; −1,2; 0; 1,2; 2,4} — tất cả nằm trong miền quan sát được
  theo từng biên. Caveat: trong miền theo từng biên KHÔNG đồng nghĩa
  với trong phân phối hợp (joint) — tổ hợp can thiệp có thể vẫn nằm
  ngoài phân phối mà mô hình từng thấy; ghi rõ trong báo cáo.
- Seed: `QUERY_SEED = 20261115` cho lấy mẫu query; `SPLIT_SEED = 123`
  cho chia episode; `MODEL_SEED = 99` cho RandomForest. Tất cả ghi
  trong JSON kết quả.

## 5. Kiểm chứng bằng env thật (gymnasium, 1 bước)

- Env: `CartPole-v1` (gymnasium 1.4.0; vật lý 1 bước giống CartPole-v0
  dùng lúc thu replay, chỉ khác trần episode — kiểm tra sanity: step
  từ state replay với action replay phải tái tạo `next_states`).
- Loại (a): `env.state = s_t^{do}`; step với `a_t` (quan sát được);
  `a'_{thật} = heuristic(s'_{thật})`. So `push'` dự đoán vs `a'_{thật}`
  (tỉ lệ khớp) và `s'` dự đoán vs `s'_{thật}` (MAE).
- Loại (b): `env.state = s_t`; step với action đã lật; so `s'` dự đoán
  vs `s'_{thật}` (MAE từng tọa độ + MAE trung bình).
- Policy thật chính: heuristic của tác giả
  `1 if 0.4*(x/2.4)+0.6*(θ/2.095)>=0 else 0` (công thức trong
  `muc4/scripts/cartpole_policies.py`). DQN run 9 CHỈ là chẩn đoán tùy
  chọn (nếu tải được checkpoint từ Drive, chỉ tải về, không commit
  `.pt`): báo cáo tỉ lệ đồng thuận DQN-vs-heuristic, KHÔNG nằm trong
  tiêu chí thành công.

## 6. Tiêu chí thành công (đặt trước — không sửa sau khi thấy kết quả)

Loại (a) — can thiệp state, dự đoán action bước sau:

- **C-A1**: tỉ lệ `push'` của mô hình đồ thị khớp `a'_{thật}` ≥ 0,80.
  (Thấp hơn fidelity quan sát 0,9162 của BART để chừa room cho suy giảm
  dưới can thiệp; cao hơn hẳn may rủi 0,50.)
- **C-A2**: tỉ lệ khớp của mô hình đồ thị ≥ baseline − 0,05
  (không-kém-hơn với biên 5 điểm %: dùng ít feature hơn nhưng không
  mất quá 5 điểm %).
- **C-A3** (phân biệt): trên các query can thiệp v hoặc w (không phải
  cha của `push'`), mô hình đồ thị có tỉ lệ lật dự đoán = 0 theo cấu
  trúc (cha không đổi → dự đoán không đổi). Chỉ tuyên bố "tính thưa
  của đồ thị tránh được lật giả" nếu baseline lật ≥ 10% số query này
  VÀ tỉ lệ khớp của mô hình đồ thị trên các query này ≥ baseline − 0,05.
  Ngược lại, nếu baseline khớp tốt hơn rõ rệt trên các query này thì đó
  là bằng chứng đồ thị BỎ MẤT đường v/w → s' → a' (lan truyền qua
  động lực học), phải báo cáo như một hạn chế, không giấu.

Loại (b) — can thiệp action, dự đoán state kế tiếp:

- **C-B1**: MAE trung bình 4 tọa độ của mô hình đồ thị ≤ MAE của
  baseline × 1,10 (không-kém-hơn 10%).
- **C-B2**: MAE trên (x', θ') — hai tọa độ KHÔNG có cha là `push` theo
  đồ thị, nên mô hình đồ thị dự đoán thay đổi bằng 0 — ≤ MAE của
  baseline trên cùng hai tọa độ.
- **C-B3**: MAE trên (v', w') — hai tọa độ CÓ cha là `push` — ≤ MAE của
  baseline × 1,10.

Quy tắc quyết định (đặt trước):

- "Mô phỏng sửa đồ thị được kiểm chứng" CHỈ khi F1 ∧ F2 ∧ C-A1 ∧ C-A2 ∧
  C-B1 ∧ C-B2 ∧ C-B3 đều đạt.
- Ngược lại: báo cáo từng tiêu chí đạt/không đạt, kết luận từng phần
  ("đạt ở loại (b) nhưng không ở loại (a)", v.v.), không gộp thành một
  tuyên bố chung chung. Kết luận "đồ thị tốt hơn baseline" CHỈ khi
  khác biệt vượt biên đã đặt ở trên.
- KHÔNG tuyên bố đồ thị là nhân quả thật. Mọi kết quả đều mang caveat
  §7.

## 7. Caveat bắt buộc (ghi trong báo cáo)

- Đồ thị học từ dữ liệu quan sát (replay của 1 policy), không phải từ
  thí nghiệm can thiệp; giả định không nhiễu ẩn (causal sufficiency)
  chưa được kiểm tra.
- Nhiễu BART: số cạnh dao động ±1–2 giữa các lần chạy (heuristic hte
  quan sát 12–14 cạnh); tập cạnh dùng ở đây là 1 lần chạy Mục 3.
  Đồ thị Mục 4 seed 0 (heuristic hte 14 cạnh) khác đồ thị này vài cạnh.
- Can thiệp có thể ngoài phân phối hợp dù từng biên nằm trong miền
  quan sát; RandomForest không ngoại suy.
- 199 cặp qua biên episode đã lọt vào lúc học đồ thị gốc (dịch hàng
  đơn thuần); surrogate ở đây loại chúng ra — một khác biệt có chủ ý
  so với pipeline gốc.
- Mô hình đồ thị dự đoán `push'` trực tiếp từ cha ở bước `t`, không lan
  truyền qua `s'` dự đoán; đường can thiệp state → động lực học → action
  bước sau không được mô hình hóa.
