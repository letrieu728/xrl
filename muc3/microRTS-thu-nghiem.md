# Bước 3 — MicroRTS: thử nghiệm tối thiểu khả thi (repo tác giả)

Ngày chạy: 2026-11-12. Repo tác giả: `yanmluk/causal-xrl`, commit
`ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162`, clone mới, mọi sửa đổi chỉ làm
trên **bản copy** (repo gốc không bị sửa). Log đầy đủ:
`muc3/microRTS-thu-nghiem-log.txt`. Patch sửa import:
`muc3/stratvis-fix.patch` (đã kiểm chứng áp sạch lên cây gốc).

## Kết luận trước

MicroRTS **chạy được thật ở mức tối thiểu khả thi**, sau khi sửa import
`stratvis`, lấy lại checkpoint PPO từ nguồn công khai chính thức, và dựng
môi trường tương thích (Python 3.12 + Java + Xvfb + một số shim tương
thích — liệt kê đầy đủ ở §5, không giấu). Cụ thể đã chạy được:

- Trích đặc trưng chiến lược từ cả 5 map XML có sẵn; vòng lặp XML→obs→XML
  cho kết quả giống hệt.
- Agent PPO thật (checkpoint S3) truy vấn action trong env MicroRTS thật
  (JVM + CoacAI), chạy nhiều bước trên cả map có sẵn lẫn map `case1.xml`
  tùy chỉnh.
- Dashboard Flask **trả trang thật**: `GET /` → 200, trang `index.html`
  18.741 byte; 2 route tính toán thuần (`/utt_to_uht`, `/read_xml_map`) → 200.
- Học đồ thị toàn cục bằng `infer_causes_hte` của tác giả trên **537
  transition thật từ 10 episode**: 20 cạnh DBN, 12/13 target tính được
  bằng BART, 1 target hằng (bỏ qua theo đúng luật của `infer_DBN`).

Vẫn tắc / hỏng thật (không che): route `/get_board_state` lỗi 500 do bug
`RLAgent()` thiếu đối số của tác giả; route `/compute_global_causal`
lỗi 500 do thiếu các file `data/*.npz/*.pickle` (không có trong repo);
`infer_DBN` nguyên bản sập toàn bộ khi 1 target làm BART lỗi (đã dùng
driver theo từng target để lấy được đồ thị đầy đủ — xem §4.4). Đồ thị học
được chỉ từ 10 episode, **không** tương đương quy mô của tác giả
(`scm500_hte` gợi ý 500 episode); không dùng để đối chiếu Table 1 của
paper (86,0%/85,9% — đó là expert study, không phải thứ chạy ở đây).

## 1. Sửa import `stratvis.*` (A1)

`grep` trên cây gốc: **22 chỗ** dùng `stratvis.` trong **8 file .py**
(`static/package*.json` còn giữ tên `stratvis` nhưng đó là tên gói JS,
không sửa). Sửa cơ học `stratvis.` → `cxrl_microRTS.` trên bản copy:

| File | Số chỗ |
|---|---:|
| `cxrl_microRTS/app.py` | 9 |
| `cxrl_microRTS/lib/gym_microrts_rl/gym_microrts_rl.py` | 3 |
| `cxrl_microRTS/lib/gym_microrts_rl/counterfactual_action.py` | 2 |
| `cxrl_microRTS/lib/gym_microrts_rl/uncertainty_map.py` | 2 |
| `cxrl_microRTS/lib/models/bart_regressor.py` | 2 |
| `cxrl_microRTS/lib/xrl_causal/xrl_causal.py` | 2 |
| `cxrl_microRTS/lib/gym_microrts_rl/test/test.py` | 1 |
| `cxrl_microRTS/lib/models/__init__.py` | 1 |

Sau sửa: 0 `stratvis` còn lại trong file .py. Diff đầy đủ:
`muc3/stratvis-fix.patch`. Lưu ý thêm, phát hiện khi kiểm chứng: file
`cxrl_microRTS/lib/` không có `__init__.py` (vẫn import được nhờ
namespace package của Python 3), và `test/test.py` import `RLAgent` từ
package `gym_microrts_rl` trong khi `__init__.py` của package đó rỗng —
import này vẫn lỗi kể cả sau khi sửa tên (không nằm trong phạm vi chạy
dashboard, chỉ ghi nhận).

## 2. Checkpoint PPO `agent-3.pt` (A2) — tìm thấy ở nguồn công khai

Đường dẫn code yêu cầu (`gym_microrts_rl.py:42`, tương đối theo thư mục
làm việc): `lib/gym_microrts_rl/trained_models/ppo_gridnet_coacai/agent-3.pt`.
File này **không có** trong repo tác giả (thư mục `trained_models/` không
tồn tại; `find` toàn repo: 0 file `.pt`/`.npz`/`.pickle`).

### 2.1. Các nguồn đã thử

| Nguồn | Cách kiểm tra | Kết quả |
|---|---|---|
| GitHub code search `ppo_gridnet_coacai` | connector GitHub `search_code` | 53 hit (code/script, vd. `Farama-Foundation/MicroRTS-Py`, `vwxyzjn/gym-microrts-paper`); không hit nào là file `.pt` |
| Repo `vwxyzjn/gym-microrts-paper` (clone, commit `1cbc43e2`) | liệt kê cây file | 0 file `.pt`; README (dòng ~130) công bố link S3 `https://microrts.s3.amazonaws.com/microrts/gym-microrts-paper/trained_models.zip` (622.324.025 byte, Last-Modified 12/04/2021) |
| File `trained_models.zip` trên S3 | đọc central directory bằng HTTP Range (không tải toàn bộ 622 MB) | 76 entry; **có đúng** `trained_models/ppo_gridnet_coacai/agent-3.pt` (nén 20.425.832 byte, gốc 21.734.570 byte). Manifest đầy đủ: `muc3/trained_models_zip_manifest.json` |
| Repo `Farama-Foundation/MicroRTS-Py` (clone) | liệt kê file `.pt` | chỉ có `experiments/gym-microrts-static-files/agent_sota.pt` (kiến trúc khác) — **không dùng** |
| Hugging Face | tìm kiếm `site:huggingface.co microrts` + từ khóa gridnet/PPO | không tìm thấy model MicroRTS nào |
| CleanRL (repo/docs) | tìm kiếm web | CleanRL chỉ công bố code `ppo_gridnet`/`ppo_microrts`, không công bố checkpoint `.pt` tương ứng |

### 2.2. Kiểm chứng checkpoint lấy về

Chỉ tải đúng 1 entry `agent-3.pt` bằng Range request
(`muc3/scripts/fetch_agent3_range.py`), lưu ở scratch ngoài repo:

- Kích thước: 21.734.570 byte; SHA-256:
  `ad21e903f7ced10aa172d41abb2ab9859e54c6082214efbeaffc47916ae660db`.
- Kiến trúc khớp **từng lớp** với class `Agent` của CausalXRL
  (`gym_microrts_rl.py:70-85`): Conv2d 27→16→32, Linear 32·6·6→256,
  actor 256→19968 (= 256×78), critic 256→1 — giống hệt class `Agent`
  trong `ppo_gridnet_coacai.py` của repo Gym-μRTS (nơi `nvec[1:].sum()=78`).
  Quy ước đặt tên cũng khớp: `download_models.py` của repo đó đổi tên
  `agent.pt` từ wandb thành `trained_models/{exp_name}/agent-{seed}.pt`
  (seed 3 → `agent-3.pt`).
- Kiểm chứng bằng code (`muc3/scripts/verify_agent3_checkpoint.py`):
  `load_state_dict(strict=True)` **OK**, 10 tensor, 5.433.185 tham số,
  forward trên obs 0 cho logits hữu hạn (1, 19968) và value −1,146.
- Thận trọng: đây là checkpoint seed-3 GridNet-CoacAI do nhóm Gym-μRTS
  công bố, đúng tên file/đường dẫn/kiến trúc mà code CausalXRL yêu cầu;
  **không thể chứng minh giống byte** với bản copy riêng của tác giả
  CausalXRL vì họ không công bố hash. Báo cáo này chỉ khẳng định nguồn
  công khai chính thức + khớp kiến trúc + load được.

**Checkpoint KHÔNG được commit vào repo XRL** (file nhị phân, không rõ
bản quyền tái phân phối). Ai cần chạy lại: dùng script Range-fetch kèm
theo + URL S3 + SHA-256 ở trên để tự lấy, đặt đúng đường dẫn tương đối
mà code yêu cầu. Không có file nhị phân nào cần upload tay cho Bước 3.

## 3. Các file/thư mục còn thiếu (kiểm chứng bằng `ls`/`grep`)

| Thứ code yêu cầu | Vị trí yêu cầu | Trạng thái |
|---|---|---|
| `map_path.txt` | mở ở cấp module trong `app.py:23`, `gym_microrts_rl.py:685`, `uncertainty_map.py:9`, `counterfactual_action.py:14` | **Không có** → import sập nếu không có shim (xem §5) |
| `data/strategy_state_epwt.npz`, `data/scm500_hte.pickle`, `data/scaler500_hte.pickle` | `app.py` route `/compute_global_causal` | **Không có** (thư mục `data/` không tồn tại) → route này 500 |
| `data/ep_play_front_data.npz` | `app.py` route `/run_env_episode` | **Không có** → route này cũng sẽ 500 (không thử thêm vì cùng nguyên nhân) |
| Map XML | `cxrl_microRTS/maps/` | **Có**: `case1–case4.xml` + `default_map.xml`, 16×16, 6–19 unit |

README của `cxrl_microRTS` cũng ghi rõ một số file/data đã lược vì
kích thước/bản quyền, liên hệ tác giả để lấy bản đầy đủ — khớp với
những gì kiểm chứng được.

## 4. Chạy thử theo từng tầng (A3)

Mọi lệnh + output trong `muc3/microRTS-thu-nghiem-log.txt`. Driver chạy
nằm ở `muc3/scripts/` (bản sao đúng script đã chạy, đường dẫn tuyệt đối
trỏ về venv/bản copy thí nghiệm — cần sửa đường dẫn khi chạy lại).

### 4.1. Tầng 1 — map XML → đặc trưng (không cần env/JVM)

`xml_to_obs` + `rawobs_to_state` + `obs_to_xml` chạy tốt sau khi sửa
import. Bảng đặc trưng chiến lược (12 số đếm, thứ tự
base/barrack/worker/light/heavy/ranged × player 1/2):

| Map | Unit | base1 | barr1 | work1 | light1 | heavy1 | rang1 | base2 | barr2 | work2 | light2 | heavy2 | rang2 | RSC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case1 | 10 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 5,5 |
| case2 | 11 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 5,5 |
| case3 | 9 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 6,5 |
| case4 | 19 | 1 | 1 | 9 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 7,0 |
| default_map | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 5,5 |

Vòng lặp `obs_to_xml` → `xml_to_obs` trên case1 cho `raw_obs` và
`player_urt_mat` **giống hệt** bản gốc. Dữ liệu JSON:
`muc3/map_strategy_states.json`. 5 snapshot tĩnh này **không** phải
trajectory — đồ thị ở §4.4 học từ episode chạy thật, không phải từ 5 map.

### 4.2. Tầng 2 — import module agent + chạy env thật

- Import `gym_microrts_rl` **không** shim `map_path.txt`: checkpoint load
  qua được (dòng 122, chỉ cảnh báo `FutureWarning` của torch), rồi sập ở
  dòng 685: `FileNotFoundError: .../cxrl_microRTS/map_path.txt`.
- Với shim `map_path.txt` (nội dung = đường dẫn tới
  `.../site-packages/gym_microrts/microrts/maps/16x16`, nơi duy nhất
  `MicroRTSGridModeVecEnv` đọc `maps/16x16/custom.xml`): import OK,
  `agent1` 5.433.185 tham số.
- Tạo `RLAgent` khi chưa có Java: `JVMNotFoundException: No JVM shared
  library file (libjvm.so)`. Sau khi cài OpenJDK 21 (JRE đầy đủ, không
  phải bản headless) + chạy dưới Xvfb (route `render()` của env cần X11):
  JVM khởi động, `JNIVecClient` chạy, `action_space.nvec =
  [256, 6, 4, 4, 4, 4, 7, 49]` (tổng phần tử sau phần tử đầu = 78, khớp
  `nvec_sum` của `Agent`).
- `manual_get_valid_action()` trên map `basesWorkers16x16.xml`: trả về
  **2 action hợp lệ thật** từ policy PPO, shape (2, 8), vd.
  `[17, 2, 2, 3, 2, 0, 2, 39]`.
- `run_num_steps(3)`: map có sẵn → 3 raw state tại gamestep `[0, 30, 50]`
  (3,1 giây); map tùy chỉnh từ `case1.xml` (viết `custom.xml` qua đúng
  luồng chuyển đổi của `app.py::runenv_steps`) → 3 raw state tại
  `[0, 8, 14]` (1,4 giây). Số đếm chiến lược của 3 snapshot case1 giống
  nhau (thay đổi `utt` không làm đổi tổng số đếm) — ghi đúng như đo được.
- Thoát tiến trình sau khi JVM đã chạy bị treo ở `jpype._core._JTerminate`
  (thread Java không phải daemon); các script kết thúc bằng `os._exit(0)`
  sau khi đã lưu kết quả — đây là workaround của harness, không phải sửa
  code tác giả, và không ảnh hưởng kết quả (đã xác nhận bằng chạy
  `python -u` + faulthandler: công việc hoàn tất trước khi treo).

### 4.3. Tầng 3 — dashboard Flask

Chạy bằng Flask `test_client` (không cần mở server thật để chứng minh
app trả trang; import module `app.py` đã kéo toàn bộ chuỗi import +
checkpoint + `map_path.txt`):

| Route | Kết quả |
|---|---|
| `GET /` | **200**, 18.741 byte, nội dung `index.html` (bắt đầu `<!DOCTYPE html>`) |
| `POST /utt_to_uht` (utt toàn 2) | **200**, uht toàn 10 (đúng: Base = 10 HP) |
| `POST /read_xml_map` (case1.xml) | **200**, `raw_obs` 5×16, `player_urt_mat` 16×16 |
| `GET /get_board_state` | **500** — `TypeError: RLAgent.__init__() missing 3 required positional arguments: 'obs_', 'rawobs_', 'mappath_'` (`app.py:28` gọi `RLAgent()` không đối số trong khi `__init__` bắt buộc 3 đối số). **Bug của tác giả**, xác nhận lúc chạy. |
| `GET /compute_global_causal` | **500** — `FileNotFoundError: data/strategy_state_epwt.npz` (thiếu file, xem §3). |

Các route còn lại (`/runenv_steps`, `/run_ctf_steps`, `/get_counterfactual_map`,
`/get_uncertainty_map`, `/unit_*`, `/get_local_causal`, …) không được thử
từng cái: chúng hoặc cần `data/` đã mất, hoặc cần state từ frontend.
Hai khối chức năng cốt lõi của chúng (chạy env nhiều bước; học đồ thị)
đã được thử trực tiếp ở §4.2 và §4.4. Không khẳng định dashboard "chạy
đầy đủ" — chỉ khẳng định đúng những route đã đo ở bảng trên.

### 4.4. Tầng 4 — đồ thị toàn cục từ episode thật

Thu trajectory bằng đúng `RLAgent.run_episodes()` của tác giả, map
`basesWorkers16x16.xml`, PPO vs CoacAI: **10 episode, 537 transition**
(2 episode đầu: 109 transition/98,8 giây; 8 episode sau: 428
transition/313,9 giây). Mỗi transition là cặp state (t, t+1) 13 chiều
(12 số đếm + timestep).

- Gọi thẳng `infer_DBN(method='hte')` trên 109 transition đầu: chạy được
  10/13 target rồi **sập** ở target `heavy2`:
  `R RuntimeError: sigma estimate must be greater than 0.000000`
  (dbarts lỗi khi target hiệu số gần như hằng — `heavy2` chỉ đổi 4/109
  mẫu; `infer_DBN` không có cơ chế bỏ qua target lỗi).
- Driver theo từng target (`muc3/scripts/microrts_learn_graph.py`) gọi
  đúng hàm `infer_causes_hte` của tác giả cho từng target trên 537
  transition gộp, ghi rõ kết quả từng target (không điền `[]` âm thầm
  cho target lỗi):

| Target | Số mẫu đổi ≠ 0 / 537 | Kết quả |
|---|---:|---|
| base1 | 8 | causes = [] |
| barrack1 | 40 | base1, barrack1, worker2, timestep |
| worker1 | 368 | worker1, base2, timestep |
| light1 | 18 | causes = [] |
| heavy1 | 50 | causes = [] |
| ranged1 | 40 | ranged1 |
| base2 | 20 | base1, base2 |
| barrack2 | 20 | base1, barrack2 |
| worker2 | 40 | base2, worker2, timestep |
| light2 | 0 | **target hằng** → bỏ qua theo đúng luật `infer_DBN`, causes = [] |
| heavy2 | 20 | heavy2, timestep |
| ranged2 | 37 | timestep |
| timestep | 537 | worker1, barrack2 |

Kết quả: **20 cạnh DBN**, chuyển sang đồ thị nhân quả bằng đúng hàm
`DBN_to_causal` của tác giả cũng cho 20 cạnh (đã kiểm chứng hàm gốc cho
cùng tập cạnh với driver). Dữ liệu JSON đầy đủ:
`muc3/microrts_graph10.json`.

Thận trọng khi đọc đồ thị này: (1) chỉ 10 episode — các target hiếm
(base1, light1) gần như không có tín hiệu; (2) nhiều cạnh tự vòng
(vd. `barrack1 → barrack1`) vì target là hiệu số (độ thay đổi phụ thuộc
mức hiện tại) và bộ lọc tự vòng trong `DBN_to_causal` của tác giả thực
chất không lọc được gì (so sánh chuỗi sai — ghi nhận, không sửa); (3)
cạnh từ `timestep` phản ánh quá trình game theo thời gian, không nên đọc
như quan hệ nhân quả giữa các loại unit. Đồ thị này chứng minh **pipeline
học chạy được đầu-cuối**, không phải bản sao kết quả của tác giả.

## 5. Môi trường + shim tương thích (để chạy lại được)

- Python 3.12.3, venv mới; R 4.3.3 + dbarts 0.9.34 (cài lại từ CRAN vào
  `~/workspace/R-lib` sau khi VM restart xóa `/usr/local`); OpenJDK 21
  (JRE đầy đủ) + Xvfb. Freeze đầy đủ: `muc3/step3-venv-freeze.txt`.
  Phiên bản chính: gym 0.21.0, gym-microrts 0.3.2, stable-baselines3 1.0,
  torch 2.5.1+cpu (thay torch 1.8.0 trong requirements 2021 — không cài
  được trên Python 3.12), JPype1 1.7.1, numpy 1.26.4, pandas 2.1.4,
  scikit-learn 1.5.2, Flask 2.2.2, rpy2 3.5.17, dice-ml 0.7.2 (dùng ở
  fallback, xem báo cáo Mục 4).
- Shim **ngoài code tác giả** (trong venv/script chạy, đều ghi rõ lý do):
  1. `gym` 0.21.0: vá 1 chuỗi specifier `opencv-python>=3.` → `>=3.0`
     trong bản copy sdist để `pip` build được; vá `registration.py`
     trong bản đã cài (`entry_points().get` → `.select`, API đã đổi ở
     Python 3.10+).
  2. `np.int = int` trong script chạy (gym-microrts 0.3.2 dùng `np.int`,
     đã bỏ ở NumPy 1.24; đúng hướng dẫn thay thế của NumPy).
  3. `sklearn.tree._tree.DTYPE = np.float32` trong script chạy (code BART
     của tác giả dùng thuộc tính đã bỏ ở scikit-learn ≥ 1.6; giá trị cũ
     chính là `np.float32`).
  4. File `map_path.txt` tự tạo trong bản copy (xem §4.2); checkpoint
     đặt đúng đường dẫn tương đối trong bản copy.
  5. Không cài TensorFlow (requirements ghi TF 2.7) — không module nào
     trong chuỗi chạy MicroRTS import TF; DiCE chạy ở backend sklearn.
- VM restart giữa chừng (khoảng 03:54Z) đã xóa `/tmp` và gói R
  `dbarts`; bản copy + checkpoint + log trong workspace còn nguyên, đã
  clone lại repo gốc và cài lại dbarts, không mất kết quả nào.

## 6. Blocker còn lại

1. Route `/get_board_state` hỏng do bug gọi `RLAgent()` của tác giả —
   cần sửa code tác giả (thêm đối số hoặc constructor mặc định) mới chạy
   được; Bước 3 không sửa vì ngoài phạm vi "tối thiểu khả thi" đã đạt.
2. Các route nhân quả của dashboard cần file `data/` đã lược — chỉ tác
   giả mới có (hoặc phải tự sinh SCM/scaler từ pipeline §4.4 rồi lưu
   đúng tên file — chưa làm, vì dashboard đã chứng minh trả trang được
   và pipeline học đã chạy độc lập).
3. Đồ thị toàn cục cần nhiều episode hơn (tác giả dùng ~500) mới ổn định;
   với 10 episode, các target hiếm vẫn mong manh trước lỗi sigma của BART.
4. Không đối chiếu được Table 1 / expert study của paper từ code này:
   paper dùng dashboard tương tác + elicitation (Appendix C/D), không có
   script đánh giá tự động trong repo.
