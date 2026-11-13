# Bước 3 + Nhóm 2 — MicroRTS: thử nghiệm tối thiểu khả thi và mở rộng (repo tác giả)

Ngày chạy bước 3: 2026-11-12. Cập nhật nhóm 2: 2026-11-12 (rollout
100 episode, đồ thị 100 episode, fidelity, sửa 2 route dashboard).
Cập nhật nhóm 3a: 2026-11-12 (sửa route `/get_local_causal`, chẩn đoán
cơ chế fidelity, GES trên dữ liệu MicroRTS — xem §9.5, §10.4, §12).
Repo tác giả: `yanmluk/causal-xrl`, commit
`ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162`, clone mới, mọi sửa đổi chỉ làm
trên **bản copy** (repo gốc không bị sửa). Log đầy đủ bước 3:
`muc3/microRTS-thu-nghiem-log.txt`. Log nhóm 2:
`muc3/results/microrts/`. Patch sửa import:
`muc3/stratvis-fix.patch` (đã kiểm chứng áp sạch lên cây gốc). Patch sửa
dashboard nhóm 2: `muc3/dashboard-fix.patch`. Patch sửa
`/get_local_causal` nhóm 3a: `muc3/dashboard-local-causal-fix.patch`
(đã kiểm chứng áp sạch; xem §10.4).

## Hiệu đính quan trọng đối với báo cáo bước 3 (đọc trước)

1. **"10 episode / 537 transition" của bước 3 thực ra chỉ có 8 episode
   duy nhất.** Module `gym_microrts_rl.py` của tác giả seed
   `random`/`numpy`/`torch` bằng `seed=5` ở cấp import, nên mỗi tiến
   trình mới đều chạy lại cùng một dãy episode. Đã kiểm chứng bằng số:
   2 episode đầu (109 transition) giống byte với 109 hàng đầu của file
   8 episode (428 transition). Đồ thị 20 cạnh ở §4.4 học trên 537 hàng
   trong đó 109 hàng bị đếm trùng. Nhóm 2 đã sửa bằng cách reseed sau
   import (`--seed` riêng cho mỗi batch) và kiểm chứng các batch khác
   nhau thật (§7). Con số đúng cho bước 3: **8 episode duy nhất, 428
   transition** (+ 109 transition trùng lặp).
2. **86,0%/85,9% MicroRTS-Lite trong Table 1 của paper là behavior
   prediction (§5), KHÔNG phải expert study (§6).** Câu ở bản bước 3
   gọi đó là "expert study" là sai; §6 của paper là đánh giá định tính
   (rating của chuyên gia), còn Table 1 đo dự đoán hành vi. Nhóm 2 đã
   thử đo fidelity theo đúng định nghĩa này — kết quả ở §9: không đo
   được theo đúng protocol của notebook (tập valid rỗng), và bản
   không-lọc (ghi rõ là lệch protocol) cho kết quả thấp hơn nhiều so
   với paper.
3. Hai blocker §6 của bước 3 đã được xử lý ở nhóm 2: route
   `/get_board_state` đã sửa → 200; thư mục `data/` đã tự sinh từ dữ
   liệu nhóm 2 → `/compute_global_causal` → 200. Route
   `/get_local_causal` còn 500 sau nhóm 2 (nguyên nhân mới, ghi rõ ở
   §10.3); **nhóm 3a đã sửa → 200** (xem §10.4). Kết quả fidelity ở §9
   đã được nhóm 3a chẩn đoán tới cấp cơ chế (§9.5): tập valid rỗng là
   bản chất dữ liệu + tính chất của máy HTE, không sửa được bằng cấu
   hình — không có số fidelity mới.

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

> **Cập nhật nhóm 2 (2026-11-12):** hai route trên đã được xử lý (§10);
> đồ thị đã học lại trên 100 episode thật (§8); fidelity đã đo thử (§9).
> Câu cuối của đoạn trên đã được hiệu đính ở đầu báo cáo: 86,0%/85,9% là
> behavior prediction, không phải expert study; và "10 episode" thực ra
> là 8 episode duy nhất + 2 episode trùng lặp.
>
> **Cập nhật nhóm 3a (2026-11-12):** route `/get_local_causal` còn lại
> đã sửa → 200 với đồ thị cục bộ thật (§10.4); cả 9 route đã kiểm chứng
> đều 200. Fidelity đã chẩn đoán xong tới cấp cơ chế (§9.5): không có
> cách sửa cấu hình nào cho tập valid khác rỗng, và không có số mới.

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
| `data/strategy_state_epwt.npz`, `data/scm500_hte.pickle`, `data/scaler500_hte.pickle` | `app.py` route `/compute_global_causal` | **Không có** (thư mục `data/` không tồn tại) → route này 500 ở bước 3; **nhóm 2 đã tự sinh** từ dữ liệu 100 episode, xem §10 (file sinh ra lưu trên Drive, không commit vào repo) |
| `data/ep_play_front_data.npz` | `app.py` route `/run_env_episode` | **Không có** → route này cũng sẽ 500 (không thử thêm vì cùng nguyên nhân); **nhóm 2 đã tự sinh** từ 1 episode chạy thật, xem §10 |
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
**Nhóm 2 đã thử thêm `/get_local_causal` và `/run_env_episode`** sau khi
sinh `data/` — kết quả ở §10.

### 4.4. Tầng 4 — đồ thị toàn cục từ episode thật

Thu trajectory bằng đúng `RLAgent.run_episodes()` của tác giả, map
`basesWorkers16x16.xml`, PPO vs CoacAI: **10 episode, 537 transition**
(2 episode đầu: 109 transition/98,8 giây; 8 episode sau: 428
transition/313,9 giây). **Hiệu đính (nhóm 2, đã kiểm chứng bằng số):
2 episode đầu trùng lặp hoàn toàn với 2 episode đầu của dãy 8 episode
do seed ở cấp import — thực chất chỉ có 8 episode duy nhất.** Mỗi
transition là cặp state (t, t+1) 13 chiều (12 số đếm + timestep).

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

## 6. Blocker còn lại (cập nhật sau nhóm 2)

1. ~~Route `/get_board_state` hỏng~~ — **đã sửa** (nhóm 2, §10): 200,
   trả dữ liệu board thật. Lưu ý: không có code JS nào trong frontend
   gọi route này (board lấy từ `/read_xml_map`), nên việc sửa chỉ được
   kiểm chứng bằng `test_client`.
2. ~~Các route nhân quả cần file `data/`~~ — **đã tự sinh** (nhóm 2,
   §10): `/compute_global_causal` → 200, `/run_env_episode` → 200.
   ~~Route `/get_local_causal` 500~~ — **đã sửa** (nhóm 3a, §10.4):
   danh sách cột hard-code của tác giả (10 cột) không khớp scaler học
   từ dữ liệu thật (8 cột); nhóm 3a suy danh sách cột từ chính dữ liệu
   huấn luyện theo đúng quy tắc của `infer_causes_hte`, không vá
   scaler, route trả 200 với đồ thị cục bộ thật ở cả 3 step đã thử.
3. Đồ thị toàn cục nay học từ 100 episode (§8), vẫn ít hơn quy mô
   ~500 episode mà tên file `scm500_hte` gợi ý; 500 episode ước ~5,5 giờ
   rollout trên máy này nên nhóm 2 dừng ở 100 và ghi rõ.
4. Không đối chiếu được con số Table 1 (86,0%/85,9%) của paper từ code
   này: paper không cho mã hóa action cho MicroRTS, code repo không có
   cột action; bản đo tương tự của nhóm 2 (§9) cho thấy theo đúng
   protocol notebook thì tập valid rỗng, còn bản không-lọc thì mô hình
   nhân quả kém xa baseline đa số. Nhóm 3a đã chẩn đoán tới cấp cơ chế
   (§9.5) và xác nhận không có cách sửa cấu hình nào cho tập valid khác
   rỗng. Con số của paper không tái tạo được từ repo.
5. Expert study (§6 của paper) vẫn chưa làm: cần người đánh giá thật,
   không có trong phạm vi nhóm 2.

## 7. Nhóm 2 — Rollout 100 episode thật, có ghi action (2a)

Mục tiêu ban đầu là ~500 episode theo quy mô tác giả; đo thực tế
~40–50 giây/episode nên 500 episode cần ~5,5 giờ. Nhóm 2 dừng ở
**100 episode** và ghi rõ con số thật — không ghi 500 khi chưa đủ.

### 7.1. Script và định nghĩa action

Script `muc3/scripts/microrts_rollout_actions.py` cài đặt lại trung
thành vòng lặp `RLAgent.run_episodes()` của tác giả (render, `get_action`,
đóng gói Java, step, phát hiện đổi trạng thái chiến lược, kết thúc),
chỉ thêm **ghi action** (đánh dấu `[G2-REC]` trong code) và tắt in
`multi_categoricals` từng bước của tác giả (chỉ ảnh hưởng stdout).
Map `maps/16x16/basesWorkers16x16.xml`, PPO vs CoacAI, `max_steps=4000`.

Định nghĩa nhãn action (do nhóm 2 đặt ra, vì repo không có mã hóa
action — xem §9.1): tại mỗi bước, `valid_actions[:,1]` là loại action
(`types.py` của gym-microrts: 0 none, 1 move, 2 harvest, 3 return,
4 produce, 5 attack); nhãn = loại xuất hiện nhiều nhất trong các action
hợp lệ (hòa thì lấy loại nhỏ nhất); không có action hợp lệ nào thì nhãn
−1. Nhãn của một transition là nhãn của bước đã tạo ra `s_{t+1}`.

### 7.2. Sửa lỗi trùng dãy episode + gián đoạn VM restart

Như hiệu đính ở đầu báo cáo, seed ở cấp import làm mọi tiến trình chạy
lại cùng một dãy episode. Script nhóm 2 thêm `--seed` (mặc định 5) và
reseed `random`/`numpy`/`torch` sau import; seed được lưu trong mỗi file
`.npz`. Đã kiểm chứng seed 1000 cho episode khác hẳn dãy seed-5
(`seedtest.npz`, loại khỏi dữ liệu gộp).

Chạy theo 10 batch × 10 episode, id episode 2–101 (id 0–1 dành cho pilot
đã loại). VM restart giữa chừng (khoảng 07:55Z) giết batch07 khi mới
xong 3 episode chưa lưu file; batch07 được chạy lại toàn bộ với cùng
seed/start-ep, 70 episode của batch00–06 còn nguyên. Sau restart, Java/R
bị xóa và được khôi phục bằng apt replay của hệ thống; `dbarts` 0.9.34
còn trong `~/workspace/R-lib` (xem §11).

| Batch | Episode | Seed | Transition |
|---|---|---:|---:|
| batch00 | 2–11 | 5 | 520 |
| batch01 | 12–21 | 1001 | 534 |
| batch02 | 22–31 | 1002 | 506 |
| batch03 | 32–41 | 1003 | 544 |
| batch04 | 42–51 | 1004 | 488 |
| batch05 | 52–61 | 1005 | 449 |
| batch06 | 62–71 | 1006 | 487 |
| batch07 | 72–81 | 1007 | 418 |
| batch08 | 82–91 | 1008 | 493 |
| batch09 | 92–101 | 1009 | 485 |
| **Tổng** | **100 episode** | | **4.924 transition, 118.822 env step** |

Tính khác nhau giữa các batch đã kiểm chứng bằng số (không batch nào
trùng dãy state của batch khác). Manifest đầy đủ:
`muc3/results/microrts/merge_manifest.json`. File `.npz` từng batch và
file gộp là file nhị phân, lưu trên Drive (xem cuối §10).

### 7.3. Phân phối nhãn action (đo được)

| Nhãn | Ý nghĩa | Số transition |
|---:|---|---:|
| −1 | không có action hợp lệ (unit bận / thay đổi do địch) | 1.724 (35,0%) |
| 0 | none | 2.002 |
| 1 | move | 575 |
| 2 | harvest | 6 |
| 3 | return | 1 |
| 4 | produce | 614 |
| 5 | attack | 2 |

Trong 3.200 transition có action, nhãn 0 chiếm 62,6%; loại 2/3/5 gần
như không xuất hiện (tổng 9). Đây là đặc tính thật của dữ liệu, và là
lý do baseline "đoán nhãn đa số" trong §9 đã cao (~62–64%).

## 8. Nhóm 2 — Đồ thị toàn cục 100 episode (2a)

Học bằng đúng `infer_causes_hte` của tác giả, driver theo từng target
(1 tiến trình/target, checkpoint riêng — 1 target lỗi không làm mất
các target khác), trên toàn bộ 4.924 transition, `state_integration=True`
(quy ước của dashboard). Target = hiệu số `s_{t+1} − s_t`.

| Target | Số mẫu đổi ≠ 0 / 4.924 | Causes | Thời gian |
|---|---:|---|---:|
| base1 | 32 | base1 | 34,3 s |
| barrack1 | 349 | base1, ranged2 | 44,4 s |
| worker1 | 3.244 | base1, worker1, base2, barrack2, heavy2 | 41,7 s |
| light1 | 284 | [] | 38,0 s |
| heavy1 | 466 | ranged2 | 38,5 s |
| ranged1 | 427 | ranged1 | 38,0 s |
| base2 | 198 | [] | 42,3 s |
| barrack2 | 200 | base2, barrack2 | 40,2 s |
| worker2 | 400 | worker1, base2, worker2 | 44,8 s |
| light2 | 0 | **target hằng** → causes = [] theo luật `infer_DBN` | 0,0 s |
| heavy2 | 208 | base2, worker2, heavy2, timestep | 42,5 s |
| ranged2 | 404 | base2, ranged2, timestep | 41,4 s |
| timestep | 4.924 | barrack2 | 31,0 s |

Kết quả: **23 cạnh DBN** (qua đúng hàm `DBN_to_causal` của tác giả, cũng
23 cạnh nhân quả), tổng thời gian học **498,3 giây** cho 13 target.
Dữ liệu đầy đủ: `muc3/results/microrts/microrts_graph.json`. So với đồ
thị 10-episode của bước 3 (20 cạnh, trên dữ liệu có trùng lặp): tập cạnh
thay đổi đáng kể (vd. `worker1` nay có 5 causes thay vì 3; `light1`/
`heavy1`/`base2` mất causes) — đồ thị không ổn định giữa hai lần chạy,
đúng như cảnh báo ở bước 3. Các thận trọng ở §4.4 (cạnh tự vòng do
target hiệu số, cạnh từ `timestep`) vẫn áp dụng.

Hiệu quả học cấu trúc: paper báo 0,5 s cho MicroRTS-Lite (Table 1) —
không thể đối chiếu, vì code repo không có cài đặt PC/FCI để so sánh và
thời gian BART đo được (498,3 s / 13 target, ~38 s/target) lớn hơn nhiều
trăm lần. Báo cáo này chỉ ghi thời gian đo được, không khẳng định khớp
paper.

## 9. Nhóm 2 — Fidelity MicroRTS: đo thử và kết quả âm tính (2a)

### 9.1. Vì sao không thể sao chép đúng Table 1

Paper §5.1.1 định nghĩa behavior prediction: so sánh action suy ra từ
mô hình nhân quả cục bộ với action thật của agent; action chỉ phụ thuộc
vào các nút cha trực tiếp trong đồ thị cục bộ; mỗi cha một BART + gộp
có trọng số theo độ mạnh cạnh cục bộ; baseline LM/DT dự đoán action
trực tiếp từ đặc trưng state; tập test lọc lấy các experience mà action
xuất hiện dưới dạng biến nội sinh. §5.2: Accuracy + F1
(`average='weighted'` nếu đa lớp).

Nhưng: paper **không cho mã hóa action** cho MicroRTS-Lite, và code
`xrl_causal` của repo **không có cột action nào**. `#Rep=14` trong
Table 1 gợi ý 13 đặc trưng state + 1 action, nhưng đó chỉ là suy đoán.
Vì vậy nhóm 2 đo một **bản tương tự (analogue)** với mã hóa action tự
đặt ở §7.1, theo sát cấu trúc notebook đánh giá của tác giả
(`eval_fidelity_Cart Pole.ipynb`: chia 60/40 `random_state=10`,
DT/LR `random_state=1`, trọng số `|mean ITE|`, ngưỡng 0,5 cho nhị phân
→ làm tròn tới mã action gần nhất cho đa lớp). Mọi lệch protocol được
ghi ở §9.4. **Không khẳng định các số dưới đây là số Table 1.**

### 9.2. Kết quả theo đúng protocol notebook: tập valid rỗng

Notebook lọc hàng "valid" = đồ thị cục bộ có >1 cạnh đi vào nút action
(từ các nút không phải action). Nhóm 2 học thêm 1 target `action`
(X = [state_t, label_t] 14 cột, y = label_{t+1}, 2.832 cặp liên tiếp
trong episode, cả hai đều có action): chỉ tìm được **1 cause toàn cục
duy nhất: `heavy2`**. Kiểm tra `local_treatments` của BART action trên
từng hàng (query đã chuẩn hóa bằng scaler của action, đúng quy ước của
`infer_causes_hte` và dashboard):

- Construction A (2.832 cặp): 2.501 hàng flag 0 cột, 331 hàng flag đúng
  1 cột (`heavy2`), **0 hàng flag >1 cột**.
- Construction B (3.200 hàng cùng-dòng): 2.856 hàng flag 0 cột, 344 hàng
  flag đúng 1 cột (`heavy2`), **0 hàng flag >1 cột**.

Đây là kết quả thật của dữ liệu, không phải lỗi code: cùng cơ chế trên
BART của target `worker1` (5 causes) flag 2–3 cột ở 303/1.000 hàng mẫu.
Kết luận: **theo đúng protocol notebook, fidelity MicroRTS không đo
được — tập test rỗng.** File `muc3/results/microrts/fidelity.json` ghi
đúng kết quả này. (Bản chạy đầu tiên dùng query chưa chuẩn hóa cũng cho
0 hàng valid, nhưng đó là sai quy ước của wrapper BART này; bản chuẩn
hóa mới là bản chuẩn. File chạy đầu được giữ lại làm vết kiểm toán:
`fidelity_v1_rawquery.json`.)

### 9.3. Bản không-lọc (lệch protocol, ghi rõ): mô hình nhân quả thua xa

Để vẫn có con số tham khảo, nhóm 2 chạy cùng cấu trúc đánh giá trên
**toàn bộ** hàng ứng viên, không lọc valid — lệch protocol notebook,
ghi rõ trong file `fidelity_nofilter.json` (`valid_mode: nofilter`).
Với chỉ 1 cause, trọng số ITE suy biến: "BART" ở đây thực chất là một
BART hồi quy duy nhất trên [cause + covariates], dự đoán làm tròn tới
mã action gần nhất. Chia 60/40, `random_state=10`:

| Construction | Tập test | Majority | DT acc / F1w | LM acc / F1w | BART acc / F1w |
|---|---|---:|---:|---:|---:|
| A: dự đoán label_{t+1} từ [state_t, label_t] | 1.133 | 64,25% | 52,07% / 52,02% | 63,64% / 53,33% | **25,15% / 27,62%** |
| B: dự đoán label_t từ state_t | 1.280 | 62,03% | 57,19% / 56,64% | 64,06% / 56,97% | **30,86% / 32,57%** |

BART thua cả baseline đa số: dự đoán hồi quy dồn quanh giá trị trung
bình (~0,95, khoảng −0,08–1,97 ở construction A, đã kiểm chứng độc lập),
làm tròn chỉ ra 0/1/2 và không bao giờ dự đoán nhãn 4 (produce, ~20%
tập test). LM/DT cũng chỉ ngang baseline đa số. Cảnh báo: LR của
sklearn không hội tụ trong 1.000 vòng lặp ở cả hai construction (ghi
trong log) — số LM là số chạy được như vậy, theo đúng cấu trúc notebook
(dùng đặc trưng thô, không chuẩn hóa). F1 macro (thêm, không phải của
paper): A — DT 19,21%, LM 15,63%, BART 10,57%; B — DT 23,32%,
LM 21,14%, BART 12,93%.

Đọc đúng kết quả này: với mã hóa action thô (loại action phổ biến nhất
của unit) và 13 số đếm state thô, **mô hình nhân quả của tác giả không
dự đoán được hành vi của PPO** — khác hẳn 86,0%/85,9% của paper. Khoảng
cách này có thể do mã hóa action của tác giả khác (không công bố), do
quy mô dữ liệu, hoặc do cả hai; báo cáo này không suy đoán thêm.

### 9.4. Mọi lệch protocol (tóm tắt)

1. Mã hóa action tự đặt (§7.1); paper/repo không cho mã hóa nào.
2. Lọc transition có action (`n_valid>0`) — tương tự câu lọc của paper
   §5.1 nhưng không giống hệt (paper lọc theo "action xuất hiện dưới
   dạng biến nội sinh" trong đồ thị đã học).
3. Ngưỡng nhị phân 0,5 của notebook → làm tròn tới mã gần nhất (đa lớp).
4. Bản §9.3 bỏ lọc valid (lệch lớn nhất, ghi rõ trong tên file + JSON).
5. Dữ liệu 100 episode, không phải ~500 của tác giả.
6. Checkpoint PPO là bản công khai seed-3 (§2), không chứng minh được
   giống byte bản của tác giả.

### 9.5. Nhóm 3a — Chẩn đoán cơ chế: vì sao tập valid rỗng (không sửa được)

Nhóm 3a kiểm tra lại toàn bộ chuỗi lọc (global causes, local
treatments, scaling, ngưỡng, số mẫu từng bước) trên đúng BART action
đã lưu. Chẩn đoán đầy đủ, kèm mọi số đếm:
`muc3/results/microrts/fidelity_diagnosis.md` (JSON:
`fidelity_diagnosis.json`; script:
`muc3/scripts/fidelity_diagnosis_g3a.py`,
`muc3/scripts/fidelity_refit_probe_g3a.py`). Kết quả chính:

1. **Cơ chế quyết định.** Trong code tác giả, `local_treatments` đọc
   đúng các cây HTE mà `is_causal_HTE` đã khớp lúc học toàn cục: một
   cột chỉ được flag cục bộ nếu nó có ít nhất một cụm HTE có ý nghĩa,
   tức là nếu nó đã là cause toàn cục. Do đó **tập flag cục bộ của mọi
   hàng ⊆ tập cause toàn cục**. Với đúng 1 cause (`heavy2`), ngưỡng
   `>1` không thể đạt được. Kiểm tra trực tiếp: 13/14 cột có 0 cụm có
   ý nghĩa; chỉ `heavy2` có 1/8 cụm có ý nghĩa (178/1.416 hàng test).
   Số đếm §9.2 được chạy lại và khớp đúng (A: 2.501 flag 0 / 331 flag
   1; B: 2.856 / 344; đối chứng `worker1`: 303/1.000 hàng >1 flag).
2. **Scaling không phải nguyên nhân.** Query chuẩn hóa bằng scaler
   test (tái tạo đúng phép chia) cho kết quả gần như giống hệt scaler
   train (A: 329 hàng flag 1; B: 342; vẫn 0 valid). Query thô flag 0
   hàng — cây HTE sống trong không gian chuẩn hóa, còn `timestep` thô
   (0–2.237) đẩy mọi hàng vào lá không có ý nghĩa. Lưu ý: notebook gym
   của tác giả tự nó truyền query thô (không nhất quán có sẵn trong
   code tác giả; vô hại với CartPole cỡ ~1, triệt tiêu flag với
   MicroRTS). Ngưỡng `>1` của nhóm 2 khớp notebook; không có lỗi
   ngưỡng, lỗi cột action, hay thiếu bước.
3. **Tín hiệu yếu là thật.** BART action khớp kém trên 2.832 cặp
   (R² = 0,107; accuracy làm tròn 26,9%, thua baseline đa số 65,3%);
   cây quyết định rẻ tiền cũng thua đa số (52,1% so với 64,3%). Học
   lại BART action trên cùng dữ liệu với seed 1 hoặc với biến thể gym
   (1 scaler, chia 70/30) cho **0 cause** — yếu hơn, không mạnh hơn.
4. **Không sửa.** Không nới ngưỡng, không đổi seed để ép số, không
   đổi protocol. Bản không-lọc (§9.3) được giữ nguyên, với lưu ý bổ
   sung từ chẩn đoán: 1.004/1.133 (A) và 1.135/1.280 (B) hàng test rơi
   vào nhánh fallback không trọng số ITE, nên số "BART" không-lọc
   không được đọc như hiệu năng mô hình nhân quả. Hướng duy nhất có cơ
   sở (thu thêm dữ liệu ~500 episode) vượt phạm vi bước này và không
   đảm bảo cho ≥2 cause — chưa thử, không khẳng định.

## 10. Nhóm 2 — Sửa 2 route dashboard + sinh thư mục `data/` (2b)

### 10.1. Sửa `/get_board_state` (patch, trên bản copy)

Bug tác giả: `app.py:28` gọi `RLAgent()` không đối số trong khi
`__init__` bắt buộc 3 đối số. Patch `muc3/dashboard-fix.patch` chỉ sửa
route này: tạo `RLAgent` với obs 0 đúng shape, `rawobs` 0, map
`basesWorkers16x16.xml`, gọi `envs.reset()` một lần, đặt `obs`/`rawobs`
bằng board khởi đầu thật rồi gọi các hàm `get_*_raw()`. Không sửa code
thư viện. Patch đã kiểm chứng: `app.py` sau sửa-stratvis + patch = đúng
bản copy đang chạy.

### 10.2. Sinh thư mục `data/` từ dữ liệu nhóm 2

Script `muc3/scripts/make_dashboard_data.py` copy artifact của §8 vào
`cxrl_microRTS/data/` đúng tên file hard-code mà `app.py` đọc:
`strategy_state_epwt.npz` (khóa `strategy_states`, 4.924×13 float64),
`scm500_hte.pickle` (dict 13 BART, 12 fitted + `light2=None`,
377.659.346 byte), `scaler500_hte.pickle` (dict 12 scaler, thiếu
`light2` vì target hằng). **Cảnh báo tên file: "500" là tên của tác
giả, KHÔNG phải số episode của nhóm 2 (100).** Đừng trích tên file làm
bằng chứng quy mô. Manifest SHA-256:
`muc3/results/microrts/dashboard_data_manifest.json`.

Thêm (bonus): 1 episode chạy thật qua `RLAgent.run_an_episode()` để
sinh `data/ep_play_front_data.npz` (1.160 bước, 8.324.972 byte) cho
route `/run_env_episode`.

Các file `data/` là file nhị phân, **không commit vào repo**; lưu trên
Drive (thư mục `muc3/results/microrts/`, xem link ở cuối mục này) cùng
file `.npz` rollout và SCM. Ai chạy lại dashboard: tải các file này về
đúng `cxrl_microRTS/data/` + áp patch dashboard. Lưu ý: file SCM thô
377.659.346 byte bị Drive từ chối vì vượt quota lưu trữ, nên trên Drive
chỉ có bản nén gzip `scm_hte.pkl.gz` (137.623.060 byte, SHA-256
`502540ff815b538ff85f5543114dd3f6a443ba78279d46a0d2918dbfd478c422`) —
giải nén ra được đúng file SCM (SHA-256 `6ccf1710…`, xem manifest).
SCM cũng tái tạo được từ file batch `.npz` trên Drive + script
`merge_graph_targets.py` trên repo.

### 10.3. Kết quả kiểm chứng route (Flask `test_client`)

Script `muc3/scripts/test_dashboard_routes.py`, CWD phải là
`cxrl_microRTS/` (đường dẫn `data/` tương đối), chạy dưới Xvfb. Kết quả
đầy đủ: `muc3/results/microrts/dashboard_routes.json`.

| Route | Kết quả nhóm 2 |
|---|---|
| `GET /` | 200 (18.741 byte) |
| `POST /utt_to_uht` | 200 |
| `POST /read_xml_map` | 200 |
| `GET /get_board_state` | **200** (trước: 500) — trả board thật: 7 loại unit, 8 ô có unit, 16×16 |
| `GET /compute_global_causal` | **200** (trước: 500) — `{"error_code": "0"}`. Lưu ý: route này của tác giả chỉ load file `data/` và trả mã lỗi, không tự tính/vẽ đồ thị — 200 nghĩa là file `data/` load được, không hơn |
| `POST /get_local_causal` (step_id 0/10/100) | **500** — `ValueError: X has 10 features, but StandardScaler is expecting 8 features`. Nguyên nhân: `app.py` hard-code `num_cols_hte` 10 cột `[0,1,2,3,4,5,8,10,11,12]`, nhưng scaler học từ dữ liệu thật chỉ có 8 cột `[1,2,3,4,5,8,11,12]` (cột `base1`, `heavy2` có ≤2 giá trị phân biệt nên `infer_causes_hte` không đưa vào scaler). Nhóm 2 **không** vá scaler để ép chạy. Đây là blocker còn lại của dashboard |
| `POST /run_env_episode` | 200 (bonus) — trả 1.160 rawobs / 1.160 gamestep từ file ep-play đã sinh |

Không khẳng định dashboard "chạy đầy đủ": chỉ khẳng định đúng các route
đã đo ở bảng trên, và `/get_local_causal` vẫn hỏng với nguyên nhân đã
ghi. Frontend không có code JS nào gọi `/get_board_state`, nên việc sửa
route này chỉ được kiểm chứng bằng `test_client`.

> **Nhóm 3a:** bảng trên là kết quả của nhóm 2, giữ nguyên làm lịch sử.
> Route `/get_local_causal` đã được nhóm 3a sửa — xem §10.4 (bảng kiểm
> chứng mới, file kết quả mới, file nhóm 2 không bị ghi đè).

**File nhị phân nhóm 2 trên Drive** (thư mục "XRL - file nhi phan" /
`muc3/results/microrts/`): file `.npz` 10 batch rollout, file gộp
`microrts_merged.npz`, `scm_hte.pkl.gz` (SCM đã gộp, chứa đủ 13 mô hình
BART — bản nén gzip vì bản thô 378 MB bị Drive từ chối do quota; không
upload riêng checkpoint từng target vì trùng lặp),
`scaler_hte.pkl`/`strategy_states.npz`, `ep_play_front_data.npz`. Link
thư mục Drive: xem README của repo.

### 10.4. Nhóm 3a — Sửa `/get_local_causal` (patch, trên bản copy)

Nguyên nhân gốc (lấy traceback thật bằng `test_client`, không đoán):
`app.py` hard-code `num_cols_hte` 10 cột, nhưng mọi scaler trong
`scaler500_hte.pickle` đều chỉ có 8 đặc trưng, vì `infer_causes_hte`
chỉ đưa cột có >2 giá trị phân biệt vào scaler. Đã kiểm chứng: cả 13
target state dùng chung một danh sách 8 cột `[1,2,3,4,5,8,11,12]` (cột
`base1`, `base2`, `barrack2` có 2 giá trị; `heavy2` có 2 giá trị;
`light2` hằng 0), mọi scaler đều `n_features_in_ = 8` và giống hệt
nhau; suy lại danh sách từ `strategy_states` bằng đúng quy tắc >2 giá
trị phân biệt cho ra cùng 8 cột.

Cách sửa (patch `muc3/dashboard-local-causal-fix.patch`, chỉ sửa route
này trong `app.py`, không sửa scaler hay code thư viện): thay danh sách
hard-code bằng danh sách suy từ `strategy_states` với **cùng quy tắc
>2 giá trị phân biệt** mà `infer_causes_hte` dùng. Vì scaler được học
trên đúng mảng này, danh sách suy ra luôn khớp scaler — kể cả khi chạy
lại với dữ liệu huấn luyện khác. Patch đã kiểm chứng áp sạch lên bản
copy đang chạy (kết quả sau áp giống hệt file đã sửa) và áp được cả lên
`app.py` gốc của tác giả (phần sửa không chồng lấn patch stratvis /
dashboard của nhóm 2).

Kiểm chứng (cùng script `test_dashboard_routes.py` của nhóm 2, dưới
Xvfb, `R_LIBS=~/workspace/R-lib` — xem lưu ý R bên dưới; file nhóm 2
không bị ghi đè, kết quả mới ở
`muc3/results/microrts/dashboard_routes_g3a.json` +
`dashboard_routes_g3a.log`):

| Route | Kết quả nhóm 3a |
|---|---|
| `GET /` | 200 (18.741 byte, như nhóm 2) |
| `POST /utt_to_uht`, `POST /read_xml_map` | 200 |
| `GET /get_board_state` | 200 (7 loại unit, 8 ô có unit, 16×16 — như nhóm 2) |
| `GET /compute_global_causal` | 200 (`{"error_code": "0"}`) |
| `POST /get_local_causal` step_id 0 | **200** — 6 nút, 7 cạnh, 7 hiệu ứng |
| `POST /get_local_causal` step_id 10 | **200** — 8 nút, 7 cạnh, 7 hiệu ứng |
| `POST /get_local_causal` step_id 100 | **200** — 6 nút, 5 cạnh, 5 hiệu ứng |
| `POST /run_env_episode` | 200 (1.160 rawobs / 1.160 gamestep) |

Nội dung JSON trả về có ý nghĩa, không phải 200 rỗng (đầy đủ:
`muc3/results/microrts/local_causal_g3a.json`; đã đối chiếu khớp với
chạy offline trực tiếp `local_causal_network_hte` + vòng ATE). Ví dụ
step 0: nút `base1, barrack2, worker1, heavy2, base2, worker2`; cạnh
`[0,0], [7,2], [10,2], [6,7], [7,7], [8,8], [10,10]`; hiệu ứng ATE
`−1,77 / 0,83 / 1,58 / 0,25 / −0,75 / −0,54 / −0,53`. Lưu ý trung thực:
đồ thị cục bộ chứa cạnh tự-vòng (vd `base1→base1`) do code tác giả
không lọc cạnh tự thân ở đường local (khác với `DBN_to_causal` ở đường
toàn cục) — nhóm 3a giữ nguyên hành vi của tác giả, không lọc thêm.

Yêu cầu môi trường (đã kiểm chứng cả hai chiều): vòng tính hiệu ứng
ATE cần `dbarts` nạp được trong phiên R — chạy với
`R_LIBS=~/workspace/R-lib` (dbarts 0.9.34, đúng thiết lập nhóm 2 ở
§11) thì route trả 200; chạy không có `R_LIBS` thì phần đồ thị cục bộ
(vẽ từ cây HTE, không cần R) vẫn tính được nhưng ATE sập với
`no applicable method for 'predict' applied to an object of class
"bart"` → route 500. Patch không nhúng đường dẫn R; ai chạy lại
dashboard phải giữ `R_LIBS` như mọi script nhóm 2. Thứ tự gọi vẫn như
thiết kế của tác giả: `/compute_global_causal` trước (nạp globals),
rồi mới `/get_local_causal`.

## 11. Môi trường nhóm 2 (bổ sung §5)

- Cùng venv bước 3 (Python 3.12.3, torch 2.5.1+cpu, …); freeze mới:
  `muc3/results/microrts/group2-venv-freeze.txt`. Thêm `box2d-py`
  2.3.8 + `pygame` 2.6.1 (cho LunarLander ở Mục 4, không ảnh hưởng
  MicroRTS) và SWIG 4.2.0 (apt, để build box2d).
- R 4.3.3 + dbarts 0.9.34: sau VM restart, R bị xóa khỏi `/usr/local`
  và được khôi phục bằng apt replay; `dbarts` còn trong
  `~/workspace/R-lib`, các script nhóm 2 chạy với
  `R_LIBS=~/workspace/R-lib`. Không cài lại dbarts từ CRAN lần này.
- Xvfb `:99` chạy lại sau restart; Java OpenJDK 21 khôi phục bằng apt
  replay. Rollout batch07 bị gián đoạn bởi restart (xem §7.2); không
  mất batch nào đã lưu.

## 12. Nhóm 3a — GES trên dữ liệu MicroRTS (baseline học cấu trúc)

Cùng đợt với GES trên 3 env gym (xem `muc3/EXPERIMENT_REPORT.md` mục
GES và script `muc3/scripts/run_ges.py`), nhóm 3a chạy GES thuần của
causal-learn 0.1.4.8 (BIC chuẩn, `maxP=None`, đo wall-clock thật —
không dùng timer mô phỏng của notebook `eval_time`) trên
`microrts_merged.npz`. MicroRTS không có notebook timing hay tiền lệ
PC/FCI, nên chạy 2 biến thể theo quy ước của env gym, và **bỏ cột hằng
`light2`** (phương sai 0, hồi quy BIC không xác định):

| Biến thể | Ma trận | GES (trung vị, 3 lần) | Số cạnh | PC / FCI cùng phiên |
|---|---|---:|---:|---|
| Chỉ state, mọi hàng | 4.924 × 12 | 0,563 s | 50 (47 có hướng, 3 vô hướng) | 0,165 s / 0,520 s |
| State + action, hàng có action | 3.200 × 13 | 0,685 s | 53 (53 có hướng) | 0,286 s / 1,009 s |

Kết quả đầy đủ (danh sách cạnh, điểm BIC, tóm tắt cột, SHA-256 file
nguồn): `muc3/results/microrts/ges.json`. Cảnh báo đọc số: GES/PC/FCI
học đồ thị **đồng thời** trên [state(, action)] tại thời điểm t, trong
khi đồ thị BART ở §8 là DBN (t → t+1, hiệu số trạng thái, 23 cạnh) —
số cạnh không so sánh trực tiếp được; so sánh có ý nghĩa là thời gian
chạy (GES/PC/FCI dưới 1 giây so với 498,3 giây học BART cho cả 14
target).
BIC giả định tuyến tính–Gauss; các cột đếm/nhị phân vi phạm giả định
này, giống như Fisher-Z của PC/FCI. Không chuẩn hóa dữ liệu (giống
PC/FCI ở env gym).
