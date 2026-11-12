# Bước 3 (fallback) + Nhóm 2 — Sửa và chạy `XRL.counterfactual` cho CartPole và LunarLander

Ngày chạy bước 3: 2026-11-12. Cập nhật nhóm 2: 2026-11-12 (6 query
CartPole, 4 query LunarLander, sửa đa lớp). Code gốc: `yanmluk/causal-xrl`
`ae9096ea`, file `cxrl/lib/xrl/xrl.py`. Mọi sửa đổi chỉ làm trên **bản
copy**; patch bước 3: `muc4/xrl-counterfactual-fix.patch` (đã kiểm chứng
áp sạch lên cây gốc); patch nhóm 2: `muc4/xrl-multiclass-fix.patch`
(áp lên trên patch bước 3, đã kiểm chứng: cây gốc + patch bước 3 +
patch đa lớp = đúng bản copy đang chạy). Log đầy đủ bước 3:
`muc4/counterfactual-thu-nghiem-log.txt`. Log nhóm 2: xem §6.
Kết quả bước 3: `muc4/counterfactual_result.json` + `.csv`. Script chạy
bước 3: `muc4/scripts/run_counterfactual_cartpole.py`.

## Vị trí của fallback này — đọc trước

Theo kế hoạch, fallback **chỉ** chuyển sang nếu phần MicroRTS tắc ở
checkpoint/data. Thực tế MicroRTS **không tắc** (xem
`muc3/microRTS-thu-nghiem.md`: checkpoint tìm được, env chạy, dashboard
trả trang, đồ thị học được). Fallback dưới đây vì vậy **không phải
phương án thay thế bắt buộc** — nó được thực hiện thêm để kiểm chứng
lỗi `counterfactual()` đã biết từ lần chấm trước, vì lỗi đó nằm trong
code gốc và chưa từng được chạy thử. Mọi kết quả dưới đây đều là của
**code đã sửa**, không phải code gốc chạy được — code gốc sập 100% khi
gọi `counterfactual()` (xem §1).

## 1. Lỗi trong code gốc (kiểm chứng bằng đọc code)

`XRL.__init__` (`xrl.py`) chỉ gán `self.env`, `self.pi`,
`self.causalG`, `self.scm`, `self.replay_buffer`. Nhưng
`counterfactual()` và helper của nó dùng thêm 3 thứ không bao giờ tồn tại:

1. `self.pi = pi` (hàm thuần) trong khi dòng `self.pi = DummySklearn(pi)`
   bị comment; `__simplify_counterfactuals` gọi `self.pi.predict(...)`
   → hàm thuần không có `.predict` → `AttributeError`.
2. `self.exp` (đối tượng DiCE) không được gán ở đâu → `counterfactual()`
   sập ngay ở `self.exp.generate_counterfactuals(...)`.
3. `self.state_features` không được gán ở đâu →
   `__simplify_counterfactuals` sập ở vòng lặp `for s in self.state_features`.

Ngoài ra, test của chính tác giả `cxrl/lib/xrl/test/xrl_test.py` cũng
không chạy được với code hiện tại: nó dùng thuộc tính `xrl.replay`
(không tồn tại — tên thật là `replay_buffer`, dạng tuple) và truyền
`repr_names` chỉ 4 tên trong khi `infer_DBN` assert phải đủ 5 (4 state +
1 action). Test này không được sửa trong Bước 3 (ngoài phạm vi); thí
nghiệm dưới đây dùng đúng thiết lập của notebook `CartPole.ipynb` của
tác giả.

## 2. Sửa đã áp dụng (trên bản copy)

Chỉ sửa file `cxrl/lib/xrl/xrl.py`, 3 nhóm thay đổi (xem patch để biết
dòng chính xác):

**Sửa 1 — `__init__`: gán `DummySklearn` cho policy.**
Bỏ comment và dùng `self.pi = DummySklearn(pi)`. Class `DummySklearn`
do chính tác giả viết sẵn trong cùng file, bọc policy thành API
sklearn (`predict`/`predict_proba`) mà cả helper lẫn DiCE yêu cầu.

**Sửa 2 — `__init__` + hàm mới `_build_dice_explainer`: dựng
`self.exp` và `self.state_features` từ replay buffer.**
Sau `infer_from_env`, lấy `(states, actions, ...)` từ
`self.replay_buffer`, dựng DataFrame cột = tên state + cột outcome
`action`, rồi:
`dice_ml.Data(dataframe=..., continuous_features=state_features,
outcome_name="action")`,
`dice_ml.Model(model=self.pi, backend="sklearn")`,
`Dice(dice_data, dice_model, method="random")`.
API này đọc trực tiếp từ wheel `dice-ml==0.7.2` (đúng bản ghim trong
`cxrl/requirements.txt`): `Data`/`Model`/`Dice` trong `dice_ml/__init__`,
backend `'sklearn'`, method `'random'` → `DiceRandom`, và chữ ký
`generate_counterfactuals(query, total_CFs, desired_class, ...)` —
khớp với lời gọi đã có sẵn trong `counterfactual()` (kể cả
`posthoc_sparsity_param=None`, được `DiceRandom` xử lý bằng cách bỏ qua
bước post-hoc). Lời gọi `generate_counterfactuals` trong
`counterfactual()` **không bị sửa**. Lưu ý: cột outcome trong DataFrame
DiCE luôn đặt tên `action` vì `counterfactual()` và các helper hard-code
tên cột này (dù `repr_names` của notebook gọi action là `push`); đồ thị
nhân quả vẫn dùng `repr_names` như cũ.

**Sửa 3 — `__dominating_counterfactuals`: 2 lỗi tương thích pandas 2.x**
(phát hiện khi chạy, không phải đoán trước):
- Groupby key `k1` là tuple khi có ≥2 cột khác nhau; code gốc bọc thành
  `[tuple]` khiến `Index[[tuple]]` lỗi `IndexError`. Sửa: chuẩn hóa
  tuple → list (giữ nguyên trường hợp 1 cột: scalar → `[scalar]`).
- `cf_df.loc[dom]` với `dom` là set bị pandas ≥1.5 cấm (`TypeError`).
  Sửa: `cf_df.loc[sorted(dom)]` — cùng các hàng, thứ tự xác định.

Shim **ngoài code tác giả** (trong script chạy): khôi phục
`DataFrame.append` qua `pd.DataFrame._append` vì dice-ml 0.7.2 còn dùng
`.append` (đã bỏ ở pandas 2.0) tại `dice_random.py:124` và
`public_data_interface.py:568`; và shim `sklearn.tree._tree.DTYPE`
như ở báo cáo MicroRTS. Cả hai đều không sửa file thư viện.

## 3. Thiết lập chạy

- Env: `gym` CartPole-v0 (gym 0.21.0), seed **99** (`env.seed(99)` +
  `np.random.seed(99)`), 200 episode — đúng thiết lập notebook tác giả
  và đúng seed/policy của `muc3/scripts/cartpole_collect.py` (Mục 3).
- Policy: heuristic trong notebook tác giả, `1` nếu
  `0.4*(x/2.4) + 0.6*(theta/2.095) >= 0`, ngược lại `0` — trùng công thức
  `heuristic_policy` của Mục 4.
- `repr_names = ['cart_position', 'cart_velocity', 'pole_angle',
  'pole_velocity', 'push']`, `repr_integration=False`.
- BART thật qua R/dbarts 0.9.34 (rpy2). DiCE không nhận seed qua lời gọi
  gốc (tác giả không truyền `random_seed`), nên script đặt
  `random.seed(99)` + `np.random.seed(99)` ngay trước khi gọi
  `counterfactual()`; chạy lặp lại cho kết quả giống hệt (xem §4).
- Query: hàng replay số 6 (theo đúng quy ước `replay[6:7]` trong test
  của tác giả), kèm cột action quan sát để đi qua nhánh assert của
  `counterfactual()`.

## 4. Kết quả

`XRL` dựng xong trong ~86 giây; replay **6.115 transition**
(action 0: 3.139, action 1: 2.976; giống hệt qua các lần chạy).
Đồ thị nhân quả toàn cục của CartPole cũng được học kèm (11–12 cạnh;
số cạnh dao động nhẹ giữa các lần chạy vì `infer_causes_hte` dùng
`train_test_split` không seed — ghi nhận, không ảnh hưởng
counterfactual).

Query (hàng 6, action quan sát = 0, push trái):

| cart_position | cart_velocity | pole_angle | pole_velocity | action |
|---:|---:|---:|---:|---:|
| 0,090188 | 1,214400 | −0,056350 | −1,710042 | 0 |

Điểm policy của query: `0,4·(0,090188/2,4) + 0,6·(−0,056350/2,095) =
−0,001106 < 0` → action 0. Đúng.

`counterfactual(query, desired_action=1)`:
- DiCE sinh **40** counterfactual thô (`total_CFs=40` như lời gọi gốc).
- Sau `__simplify_counterfactuals` + `__dominating_counterfactuals`
  còn **2** counterfactual.
- Score trả về: `cart_position: 0,5; pole_angle: 0,5; cart_velocity: 0,0;
  pole_velocity: 0,0`.

| # | cart_position | cart_velocity | pole_angle | pole_velocity | action | Thay đổi so với query |
|---|---|---:|---:|---:|---:|---|
| 1 | 0,090188 | 1,214400 | **−0,047481** | −1,710042 | 1 | chỉ `pole_angle`: −0,056350 → −0,047481 |
| 2 | **0,097737** | 1,214400 | −0,056350 | −1,710042 | 1 | chỉ `cart_position`: 0,090188 → 0,097737 |

Kiểm chứng độc lập bằng **hàm policy gốc** (không qua DiCE/DummySklearn):
cả **2/2** counterfactual đều lật action 0 → 1. Kiểm tra bằng tay: CF1
cho điểm policy `+0,001434 ≥ 0`; CF2 cho `+0,000152 ≥ 0` — cả hai đều là
lật biên tối thiểu đúng 1 đặc trưng, đúng bản chất counterfactual cần có.

Tính tái lập: chạy lại toàn bộ (dựng `XRL` mới + cùng seed) cho cùng
replay (6.115 transition), cùng query, cùng 2 counterfactual và cùng
score. 3 lần chạy trước đó sập lần lượt ở: `DataFrame.append`
(shim), tuple key (Sửa 3), set indexer (Sửa 3) — xem log để biết lỗi
đầy đủ từng lần.

## 5. Kết luận trung thực (bước 3 — xem §7 cho cập nhật nhóm 2)

- Code gốc **không bao giờ** chạy được `counterfactual()` (thiếu
  `self.exp`, `self.state_features`, sai kiểu `self.pi`).
- Với 3 nhóm sửa tối thiểu ở §2 (đều ghi rõ trong patch), cộng 2 shim
  tương thích ở script chạy, `counterfactual()` chạy được đầu-cuối cho
  CartPole và cho 2 counterfactual tối thiểu, đã kiểm chứng độc lập.
- Giới hạn: chỉ thử 1 query (hàng replay số 6) với 1 policy (heuristic);
  chưa thử policy DQN của Mục 4, chưa thử LunarLander/PongDuel; DiCE ở
  đây là phương pháp `random` (mặc định của constructor `Dice`), không
  phải `genetic`/`kdtree`.

## 6. Nhóm 2 — Sửa đa lớp + nhiều query CartPole + LunarLander (2c)

### 6.1. Sửa đa lớp (`xrl-multiclass-fix.patch`, trên bản copy)

Code sau patch bước 3 vẫn giả định nhị phân: `DummySklearn` trả
`predict_proba` 2 cột `[[1−a, a]]`, chỉ đúng khi action ∈ {0, 1}.
Patch nhóm 2 thêm tham số `n_classes=2` cho `DummySklearn.__init__`,
trả one-hot trên `n_classes` cột; `XRL.__init__` truyền
`env.action_space.n` (mặc định 2 nếu env không có). Hành vi nhị phân
không đổi chút nào: one-hot 2 cột trùng đúng `[[1−a, a]]` — đã kiểm
chứng bằng hồi quy row 6 ở §6.2 (kết quả giống hệt bước 3). Không sửa
lời gọi `generate_counterfactuals` hay logic simplify/dominating.

### 6.2. CartPole: 6 query (script `muc4/scripts/run_counterfactual_cartpole_multi.py`)

Cùng thiết lập §3 (seed 99, 200 episode, heuristic notebook). `XRL`
dựng trong 87,4 giây; replay **6.115 transition** (0: 3.139, 1: 2.976)
— giống hệt bước 3; đồ thị nhân quả 10 cạnh. Mỗi query đặt lại
`random.seed(99)` + `np.random.seed(99)` trước khi gọi
`counterfactual()`; mỗi counterfactual cuối đều kiểm chứng lại bằng
hàm policy gốc. Kết quả đầy đủ:
`muc4/counterfactual_cartpole_multi.json` + `.csv`.

| Hàng replay | Action quan sát → mong muốn | DiCE thô | Còn lại | Kiểm chứng lật | Score (chỉ đặc trưng ≠ 0) |
|---:|---:|---:|---:|---:|---|
| 6 | 0 → 1 | 40 | 2 | 2/2 | cart_position 0,5; pole_angle 0,5 |
| 100 | 1 → 0 | 40 | 1 | 1/1 | pole_angle 1,0 |
| 500 | 0 → 1 | 40 | 2 | 2/2 | cart_position 0,5; pole_angle 0,5 |
| 1000 | 1 → 0 | 40 | 1 | 1/1 | pole_angle 1,0 |
| 2000 | 0 → 1 | 40 | 1 | 1/1 | pole_angle 1,0 |
| 3000 | 1 → 0 | 40 | 2 | 2/2 | cart_position 0,5; pole_angle 0,5 |

Tổng: **9/9** counterfactual cuối lật đúng action khi kiểm chứng bằng
policy gốc. Mẫu nhất quán: chỉ `cart_position` và `pole_angle` xuất
hiện trong score (đúng — policy heuristic chỉ phụ thuộc 2 đặc trưng
này); `cart_velocity`/`pole_velocity` luôn 0. Hồi quy row 6 so với
bước 3: cùng số lượng, giá trị state chênh lệch tối đa 0,0, cùng score
— patch đa lớp không làm đổi hành vi nhị phân. Thời gian mỗi lần gọi
`counterfactual()`: 0,08–0,11 giây.

### 6.3. LunarLander: 4 query, 4 action (script `muc4/scripts/run_counterfactual_lunarlander.py`)

- Env: `gym` LunarLander-v2 (gym 0.21.0, cần Box2D — xem §6.4), seed
  99, **30 episode** (đúng `NEPISODES` của notebook
  `eval_fidelity_Lunar Lander.ipynb` của tác giả).
- Policy: PD của chính notebook tác giả, copy nguyên văn (điều khiển
  theo x/y/vx/vy/angle/angular_velocity + tiếp xúc chân).
- `repr_names` = 8 tên state + `action`, `repr_integration=False`,
  `one_hot_actions=False`. `XRL` dựng trong 246,5 giây (BART thật qua
  R/dbarts); replay **7.005 transition**, phân phối action
  {0: 3.640, 1: 386, 2: 2.680, 3: 299}; đồ thị nhân quả 27 cạnh.
- Query: hàng đầu tiên của mỗi action quan sát, action mong muốn =
  (quan sát + 2) mod 4. **Cả 4 query đều chạy được bằng
  `counterfactual()` gốc đã sửa (attempt A)** — không cần phương án dự
  phòng B (DiCE với outcome dạng chuỗi). Kết quả đầy đủ:
  `muc4/counterfactual_lunarlander.json` + `.csv`.

| Hàng replay | Action quan sát → mong muốn | DiCE thô | Còn lại | Kiểm chứng lật | Score (chỉ đặc trưng ≠ 0) |
|---:|---:|---:|---:|---:|---|
| 0 | 0 → 2 | 40 | 8 | 8/8 | y 0,33; vy 0,40; x 0,07; left_contact 0,07; right_contact 0,13 |
| 184 | 1 → 3 | 40 | 4 | 4/4 | angle 0,40; x 0,20; vx 0,20; angular_velocity 0,20 |
| 40 | 2 → 0 | 40 | 2 | 2/2 | y 0,50; vy 0,50 |
| 211 | 3 → 1 | 40 | 2 | 2/2 | angle 0,50; angular_velocity 0,50 |

Tổng: **16/16** counterfactual cuối lật đúng action khi kiểm chứng
bằng policy PD gốc. Thời gian mỗi lần gọi: 0,25–0,49 giây. Đây là bằng
chứng đầu tiên trong repo rằng `counterfactual()` đã sửa chạy được cho
env 4 action, không chỉ nhị phân.

### 6.4. Môi trường nhóm 2

Cùng venv bước 3 + `box2d-py` 2.3.8 (build bằng SWIG 4.2.0) và
`pygame` 2.6.1 để `LunarLander-v2` chạy được trên gym 0.21. R/dbarts:
chạy với `R_LIBS=~/workspace/R-lib` sau VM restart (xem báo cáo MicroRTS
§11). Log chạy: `muc4/counterfactual-cartpole-multi-log.txt`,
`muc4/counterfactual-lunarlander-log.txt`.

## 7. Kết luận trung thực (cập nhật sau nhóm 2)

- Code gốc **không bao giờ** chạy được `counterfactual()` (thiếu
  `self.exp`, `self.state_features`, sai kiểu `self.pi`); code sau patch
  bước 3 vẫn giả định nhị phân.
- Với patch bước 3 + patch đa lớp (§6.1), `counterfactual()` chạy được
  đầu-cuối cho CartPole (6 query, 9/9 CF kiểm chứng lật) và LunarLander
  4 action (4 query, 16/16 CF kiểm chứng lật), đều kiểm chứng bằng
  policy gốc, không qua DiCE.
- Giới hạn còn lại: chưa thử policy DQN của Mục 4 (chỉ heuristic/PD);
  chưa thử PongDuel; DiCE vẫn là phương pháp `random`; query LunarLander
  chỉ lấy hàng đầu tiên của mỗi action, chưa quét nhiều hàng như CartPole.
