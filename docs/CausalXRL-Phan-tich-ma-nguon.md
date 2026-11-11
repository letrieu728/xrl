# Phân tích mã nguồn CausalXRL — yanmluk/causal-xrl
Commit: ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162 (3 commits, 0 stars, 0 forks, MIT, tạo 17/05/2026)
Đã clone và đọc trực tiếp ngày 2026-11-10. Tổng ~4.883 dòng Python (không tính notebook).

## 1. Cây thư mục
```
causal-xrl/
├── README.md
├── cxrl/                          # core library, gym-like envs
│   ├── README.md
│   ├── requirements.txt           # pip freeze đầy đủ, gym==0.20.0, rpy2==3.5.0, causal-learn==0.1.4.3
│   ├── jupyter/examples/          # CartPole.ipynb, LunarLander.ipynb, PongDuel.ipynb
│   ├── jupyter/eval/              # eval_fidelity_*.ipynb, eval_time_*.ipynb (CartPole, LunarLander)
│   └── lib/
│       ├── experience/experience.py   # play(): thu thập rollout
│       ├── causal/infer_network.py    # infer_causes*, infer_DBN, DBN_to_causal, infer_from_env
│       ├── models/bart_regressor.py   # BARTRegressor (wrapper R dbarts qua rpy2)
│       ├── r_to_py/r_to_py.py         # setup_R(): cài R packages hexbin, dbarts, rJava
│       ├── utils/utils.py             # convert_to_numpy, convert_and_expand
│       └── xrl/xrl.py                 # XRL, DummySklearn, local_causal_network, counterfactual
└── cxrl_microRTS/                 # MicroRTS + dashboard Flask
    ├── README.md, app.py (458 dòng), requirements.txt (cũ: TF 2.7, torch 1.8, gym-microrts 0.3.2)
    ├── maps/*.xml (case1-4, default_map)
    ├── static/ templates/         # frontend dashboard
    └── lib/
        ├── gym_microrts_rl/       # gym_microrts_rl.py 891 dòng, env_tools, counterfactual_action,
        │                          # uncertainty_map, obs_gen_xml, xml_to_obs, is_valid_action
        ├── xrl_causal/xrl_causal.py  # local_causal_network(_hte) cho MicroRTS
        ├── causal_layout/         # layout đồ thị nhân quả
        ├── agent_eval/agen_eval.py
        ├── players_resource/
        └── models/bart_regressor.py (bản copy, 377 dòng)
```

## 2. Luồng chạy chính (core `cxrl`)
`XRL(env, pi, repr_names)` → `infer_from_env(env, pi, nepisodes=200)` → `play()` → `infer_DBN()` → `DBN_to_causal()` → `XRL.causalG, scm, replay_buffer`.

1. **Thu thập kinh nghiệm — `experience.py: play()` (51 dòng)**
   - Chạy `nepisodes=200` episode (mặc định), mỗi step: `action = env.action_space.sample()` nếu `pi=None`, ngược lại `pi(state)`.
   - `naction_apply` lần lặp lại cùng action. Thu `states, actions, rewards, states_new`. Bỏ step cuối mỗi episode (`if not done` mới append).
   - Lưu ý: không có training RL ở đây. `pi` là policy có sẵn / heuristic do notebook tự viết, không phải agent được train trong repo.

2. **Học đồ thị nhân quả động — `causal/infer_network.py` (300 dòng)**
   - `infer_DBN(replay_buffer, method='ate')`: ghép `(states, actions)` tại t và t+1 thành `replay_t / replay_tp1`, one-hot action nếu cần. Với mỗi biến đích `i` ở t+1, gọi `infer_causes_hte(replay_t, target)` để tìm các biến gây ra nó, thêm cạnh `state_X_t → state_Y_t+1` (hoặc `action_..._t`) vào `nx.DiGraph`, lưu model BART vào `scm[i]`.
   - `infer_causes_hte`: chia train/test 70/30, chuẩn hoá cột số, fit BART, rồi `bart.is_causal_HTE()` cho từng cột.
   - `infer_causes_ate`: chia 50/50, `bart.ATE()` cho từng cột, giữ cột nào mà 95% CI của ATE không chứa 0 (`lci > 0 or uci < 0`).
   - `infer_causes` (method='expectation'): dùng `bart.CITestFast()` — so sánh MSE có/không có cột đó.
   - `DBN_to_causal(dbn)`: bỏ hậu tố `_t+1`, gộp DBN 2 lớp thời gian thành 1 đồ thị nhân quả (có thể có chu trình).
   - Có hàm `infer_causes_ate_barl` import `ai_analyzer.lib.bayesian.barl` — module này **không có trong repo**, hàm này sẽ lỗi nếu gọi.

3. **Mô hình hiệu ứng nhân quả — `models/bart_regressor.py` (345 dòng)**
   - `BARTRegressor`: wrapper Python cho R package `dbarts` qua `rpy2`. R code nhúng trong `__init__`: `bart(X, y, nskip=200, ndpost=125, nchain=4, nthread=4, seed=99)`.
   - `fit / predict / ATE / ITE`: tạo cặp phản thực X1 (treatment) vs X0 (control), `ATE = mean(y1 - y0)`, CI 95% từ phân vị 2,5% / 97,5% của posterior.
   - `is_causal_HTE`: tính ITE cho từng điểm, gom cụm bằng `DecisionTreeRegressor(max_depth=4, min_samples_leaf=10% n)`, 1 cụm được coi là causal nếu CI 95% của ITE trong cụm không chứa 0. Lưu ý: `HTE.__init__` có `assert hte.shape[0] == 500` — hardcode, sẽ fail nếu số posterior draws khác 500.
   - `CITest / CITestFast`: bỏ cột treatment, so sánh MSE dự báo có/không có cột.
   - `decision_tree_cluster`: gom cụm bằng cây quyết định.

4. **Lớp giải thích — `xrl/xrl.py` (229 dòng)**
   - `XRL.__init__`: nhận `env, pi, repr_names`, gọi `infer_from_env` ngay trong constructor — tức khởi tạo XRL là đã chạy 200 episode + học toàn bộ DBN, tốn thời gian.
   - `local_causal_network(query_state)`: với 1 state đơn, với mỗi biến `i` có trong `scm`, gọi `scm[i].local_treatments(query)` (tức `is_state_causal` cho từng cột), thêm cạnh `cột_gây → cột_đích` vào đồ thị local.
   - `counterfactual(query, desired_action)`: dùng `dice_ml` (DiCE) sinh 40 counterfactual, rồi `__simplify_counterfactuals` (thử trả từng feature về giá trị gốc, giữ lại nếu action vẫn là desired) và `__dominating_counterfactuals` (chỉ giữ counterfactual “chiếm ưu thế” theo độ gần). Trả về DataFrame + score = tỉ lệ counterfactual thay đổi mỗi feature.
   - `DummySklearn`: bọc `pi` thành model kiểu sklearn (`predict_proba` trả 1 cho action đã chọn, 0 cho còn lại) để DiCE dùng được. Trong `XRL.__init__`, dòng `self.pi = DummySklearn(pi)` đã bị comment, `self.pi = pi` — nhưng `counterfactual()` vẫn gọi `self.pi.predict(...)` trong `__simplify_counterfactuals`, nên nếu `pi` là hàm thuần thì sẽ lỗi AttributeError ở bước này. Đây là bug tiềm ẩn / code chưa dọn sạch.
   - `render_network / render_causal_network`: vẽ bằng networkx spring layout.

5. **Cầu nối R — `r_to_py/r_to_py.py` (26 dòng)**
   - `setup_R()`: cài R packages `hexbin, dbarts, rJava` qua CRAN mirror 1 nếu chưa có. Đây là bước cài đặt dễ fail nhất trên máy mới (cần R + Java).

## 3. Notebook — nơi duy nhất có “agent”
- `cxrl/jupyter/examples/CartPole.ipynb, LunarLander.ipynb, PongDuel.ipynb`: chỉ demo `XRL(env, pi=policy)`, không train.
- Policy là **heuristic viết tay**, không phải RL đã train:
  - CartPole: `x0 = state[0]/2.4; x2 = state[2]/2.095; return 1 if 0.4*x0 + 0.6*x2 >= 0 else 0`.
  - LunarLander: PD controller viết tay theo x, vx, angle, vy, leg contact.
- `cxrl/jupyter/eval/eval_fidelity_*.ipynb` (26 cell): chạy XRL 200 episode (LunarLander chỉ 30 episode trong notebook, vì nặng), lọc các step mà local graph có >1 nguyên nhân cho action, rồi so accuracy dự đoán action t+1 của 3 mô hình: DecisionTree, LogisticRegression, và BART có trọng số theo |ITE| từ local causes. Đây chính là Table 1 trong paper.
- `eval_time_*.ipynb`: đo thời gian học cấu trúc (PC/FCI/GES so với BART).

## 4. MicroRTS — `cxrl_microRTS/` (riêng biệt, cũ hơn, nhiều rủi ro)
- `app.py` (458 dòng, Flask): các route `/get_board_state`, `/read_xml_map`, `/download_map`, dashboard xem trạng thái bàn cờ 16×16×27, counterfactual action map, uncertainty map.
- `gym_microrts_rl.py` (891 dòng): wrapper env MicroRTS, `RLAgent`, `unit_ctf_steps`, `unit_intention_analysis`, obs 16×16×27.
- **Lỗi import đã thấy:** `app.py` import `from stratvis.lib.gym_microrts_rl...` trong khi thư mục thật là `cxrl_microRTS/lib/...` — tên package cũ `stratvis` còn sót, chạy `app.py` như hiện tại sẽ lỗi ImportError nếu không đổi tên / cài package `stratvis`. Một số file khác cũng có thể còn import cũ, cần grep `stratvis` trước khi chạy.
- `requirements.txt` rất cũ: TensorFlow 2.7, torch 1.8, gym-microrts 0.3.2, Flask 2.2.2, Python thời 2021–2022. Không tương thích trực tiếp với env hiện đại.
- README gốc ghi rõ: một số file / pretrained / data MicroRTS bị lược khỏi repo công khai vì kích thước / bản quyền — “contact the authors for access”.

## 5. Test
- `cxrl/lib/causal/test/infer_network_test.py` (43 dòng): test tổng hợp — y = X0+X1+noise thì causes phải là [0,1]; test DBN với dữ liệu giả.
- `cxrl/lib/xrl/test/xrl_test.py` (53 dòng): test counterfactual trên CartPole với policy có trọng số đã train sẵn (hardcode weights).
- `cxrl/lib/models/test/bart_regressor_test.py` (142 dòng), `utils/test/utils_test.py` (84 dòng).
- Test nhỏ, chỉ kiểm tra logic trên dữ liệu giả, không test end-to-end với repo MicroRTS.

## 6. Đánh giá tổng thể
| Tiêu chí | Kết luận |
|---|---|
| Repo có thật, đúng tác giả | Có — yanmluk/causal-xrl, commit ae9096e, 3 commits |
| Có code train RL không? | **Không.** Không có code train agent. Policy là heuristic viết tay trong notebook. Repo chỉ học đồ thị nhân quả + giải thích cho policy có sẵn. |
| Có code đánh giá không? | Có — notebook eval fidelity / time cho CartPole, LunarLander (PongDuel chỉ có example, không có eval notebook riêng trong repo đã clone). |
| Chạy được không? | Core `cxrl` chạy được nếu cài được R + dbarts + rpy2 + gym 0.20 + ma-gym. Đây là bước khó nhất. MicroRTS khó hơn nhiều: requirements cũ, import `stratvis` sót, thiếu data. |
| Điểm mạnh | Code core gọn (~1.000 dòng thật), luồng rõ: play → infer_DBN → BART ATE/HTE → local graph → DiCE counterfactual. Test tổng hợp có sẵn. |
| Điểm yếu / rủi ro | (1) BART phụ thuộc R/dbarts qua rpy2 — khó cài nhất. (2) `HTE` assert hardcode 500 draws. (3) `XRL.counterfactual` gọi `self.pi.predict` trong khi `pi` là hàm — bug tiềm ẩn. (4) `infer_causes_ate_barl` import module không tồn tại. (5) MicroRTS: import cũ, requirements 2021, thiếu data. (6) Repo 0 stars, 3 commits, không có CI. |
| Chi phí tái hiện | CPU-only, < $1 nếu chạy local / Modal CPU. 200 episode × 4 môi trường gym, mỗi lần infer DBN phải fit BART cho từng biến đích — thời gian chủ yếu ở BART (R), không phải GPU. |
| Phạm vi tái hiện trung thực | Chỉ 3 môi trường gym (CartPole, LunarLander, PongDuel), 200 episode/môi trường, so accuracy dự đoán action với DecisionTree + Linear (Table 1). Phần MicroRTS không tính vì thiếu data. Số sẽ không khớp 100% paper vì policy là heuristic viết tay. |

## 7. Cách chạy tối thiểu (core, không MicroRTS)
```bash
git clone https://github.com/yanmluk/causal-xrl.git
cd causal-xrl
python -m venv .venv && source .venv/bin/activate
python -m pip install -r cxrl/requirements.txt   # chú ý: ma-gym cài từ git, rpy2 cần R đã cài sẵn
python -c "from cxrl.lib.r_to_py import setup_R; setup_R()"  # cài dbarts, rJava trong R
jupyter notebook cxrl/jupyter/examples/   # CartPole.ipynb trước
jupyter notebook cxrl/jupyter/eval/      # eval_fidelity_Cart Pole.ipynb
```
Thứ tự ưu tiên khi làm: CartPole → LunarLander → PongDuel. Đừng bắt đầu từ MicroRTS.
