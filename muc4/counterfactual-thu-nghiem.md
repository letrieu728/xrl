# Bước 3 (fallback) — Sửa và chạy `XRL.counterfactual` cho CartPole

Ngày chạy: 2026-11-12. Code gốc: `yanmluk/causal-xrl`
`ae9096ea`, file `cxrl/lib/xrl/xrl.py`. Mọi sửa đổi chỉ làm trên **bản
copy**; patch: `muc4/xrl-counterfactual-fix.patch` (đã kiểm chứng áp
sạch lên cây gốc). Log đầy đủ: `muc4/counterfactual-thu-nghiem-log.txt`.
Kết quả: `muc4/counterfactual_result.json` + `.csv`. Script chạy:
`muc4/scripts/run_counterfactual_cartpole.py`.

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

## 5. Kết luận trung thực

- Code gốc **không bao giờ** chạy được `counterfactual()` (thiếu
  `self.exp`, `self.state_features`, sai kiểu `self.pi`).
- Với 3 nhóm sửa tối thiểu ở §2 (đều ghi rõ trong patch), cộng 2 shim
  tương thích ở script chạy, `counterfactual()` chạy được đầu-cuối cho
  CartPole và cho 2 counterfactual tối thiểu, đã kiểm chứng độc lập.
- Giới hạn: chỉ thử 1 query (hàng replay số 6) với 1 policy (heuristic);
  chưa thử policy DQN của Mục 4, chưa thử LunarLander/PongDuel; DiCE ở
  đây là phương pháp `random` (mặc định của constructor `Dice`), không
  phải `genetic`/`kdtree`.
