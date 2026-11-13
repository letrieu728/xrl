# Chẩn đoán fidelity MicroRTS: vì sao tập valid rỗng (nhóm 3a, 2026-11-12)

Câu hỏi: theo đúng protocol lọc của notebook đánh giá (hàng valid =
đồ thị cục bộ có **>1** cạnh đi vào nút action, từ các nút không phải
action), cả hai construction đều cho **0 hàng valid**. Đây là lỗi cấu
hình/scale/thiếu bước (sửa được), hay bản chất dữ liệu (không ép số)?

## Kết luận trước

**Bản chất dữ liệu + tính chất cấu trúc của máy HTE của tác giả — không
sửa được bằng cấu hình, scale hay ngưỡng.** Bằng chứng quyết định:

1. BART của target `action` chỉ có **1 cause toàn cục duy nhất**
   (`heavy2`), và trong máy HTE của tác giả, **tập flag cục bộ của mọi
   hàng luôn là tập con của tập cause toàn cục** (cùng một cây HTE quyết
   định cả hai — xem §2). Với 1 cause toàn cục, ngưỡng `>1` là **không
   thể đạt được về mặt toán học**, bất kể query được scale kiểu gì.
2. Kiểm tra trực tiếp cây HTE: 13/14 cột điều trị có **0 cụm (cluster)
   có ý nghĩa**; chỉ `heavy2` có 1/8 cụm có ý nghĩa (178/1.416 hàng
   test). Đổi scaler (train/test/thô) chỉ đổi số hàng flag `heavy2` đi
   2 hàng, không tạo thêm cột nào.
3. Học lại BART action trên cùng 2.832 cặp với seed khác hoặc với biến
   thể gym (1 scaler, chia 70/30) cho **0 cause** — tín hiệu còn yếu
   hơn, không mạnh hơn. Không có biến thể protocol nào thử được cho ≥2
   cause.
4. Bản thân BART action khớp dữ liệu rất yếu (R² = 0,107; accuracy làm
   tròn 26,9%, thua cả baseline đa số 65,3%). Một cây quyết định rẻ tiền
   cũng thua baseline đa số. Nhãn action tiếp theo gần như không dự đoán
   được từ state + action hiện tại với biểu diễn này.

**Không áp dụng "sửa" nào, không nới ngưỡng, không đổi seed để ép số.**
Bản không-lọc (`fidelity_nofilter.json`) được giữ nguyên kèm giải thích
(§6): nó cũng bị chi phối bởi cùng một tín hiệu cục bộ rỗng (đa số hàng
test rơi vào nhánh fallback không trọng số).

Số liệu máy-đọc được: `fidelity_diagnosis.json` (cùng thư mục).
Script chẩn đoán: `muc3/scripts/fidelity_diagnosis_g3a.py`,
`muc3/scripts/fidelity_refit_probe_g3a.py` (đường dẫn scratch
hard-code như script nhóm 2; chạy với `R_LIBS=~/workspace/R-lib`).

## 1. Phễu lọc: số mẫu qua từng bước

Dữ liệu: `microrts_merged.npz`, 100 episode, 4.924 transition
(SHA-256 `0a9d406b…73cbe45`, xem `ges.json` cùng thư mục).

| Bước | Số mẫu |
|---|---:|
| Transition (tất cả) | 4.924 |
| Có action (`n_valid > 0`) | 3.200 |
| Construction A: cặp liên tiếp trong episode, cả hai đều có action | 2.832 |
| Construction B: hàng cùng-dòng có action | 3.200 |
| A: flag 0 cột (query chuẩn hóa, scaler train) | 2.501 |
| A: flag đúng 1 cột (`heavy2`) | 331 |
| A: flag >1 cột (**valid**) | **0** |
| B: flag 0 cột | 2.856 |
| B: flag đúng 1 cột (`heavy2`) | 344 |
| B: flag >1 cột (**valid**) | **0** |

Phân phối nhãn action (tất cả 4.924 transition): −1 (không có action
hợp lệ): 1.724; 0: 2.002; 1: 575; 2: 6; 3: 1; 4: 614; 5: 2. Nhãn đích của
construction A (label_{t+1}): 0: 1.850; 1: 407; 2: 6; 3: 1; 4: 567;
5: 1. Ba nhãn 2/3/5 gần như không xuất hiện.

## 2. Cơ chế quyết định: flag cục bộ ⊆ cause toàn cục

Code tác giả (`cxrl_microRTS/lib/models/bart_regressor.py`):

- `is_causal_HTE(X, c)` tính ITE cho cột `c`, khớp một cây quyết định
  HTE lên ITE đó, và ghi `hte_per_treat[c] = HTE(...)`. Cột `c` là cause
  toàn cục **khi và chỉ khi** có ít nhất một cụm (lá cây) có khoảng tin
  cậy 95% của ITE không chứa 0 (`is_cls_causal`).
- `local_treatments(X)` → `is_state_causal(X, c)` → cây HTE của cột `c`
  phân cụm hàng query; hàng được flag **khi và chỉ khi** cụm của nó có
  ý nghĩa.

Vì vậy một cột chỉ có thể được flag cục bộ nếu nó có ít nhất một cụm có
ý nghĩa — tức là nếu nó đã là cause toàn cục. **Tập flag cục bộ của mọi
hàng ⊆ tập cause toàn cục.** Với 1 cause toàn cục, không hàng nào có thể
đạt `count > 1`. Đây không phải giả thuyết: bảng dưới là kiểm tra trực
tiếp từng cây HTE của BART action đã lưu, trên đúng nửa test đã dùng
lúc học (`test_size=0.5, random_state=99`, chuẩn hóa bằng scaler test).

| Cột điều trị | Số cụm | Cụm có ý nghĩa | Hàng test trong cụm có ý nghĩa / 1.416 |
|---|---:|---|---:|
| base1 | 8 | — | 0 |
| barrack1 | 5 | — | 0 |
| worker1 | 9 | — | 0 |
| light1 | 8 | — | 0 |
| heavy1 | 9 | — | 0 |
| ranged1 | 6 | — | 0 |
| base2 | 6 | — | 0 |
| barrack2 | 8 | — | 0 |
| worker2 | 8 | — | 0 |
| light2 (hằng 0) | 1 | — | 0 |
| **heavy2** | **8** | **[5]** | **178** |
| ranged2 | 5 | — | 0 |
| timestep | 7 | — | 0 |
| action | 6 | — | 0 |

Chỉ `heavy2` có cụm có ý nghĩa, và chỉ 12,6% hàng test rơi vào cụm đó —
khớp với tỷ lệ flag ~11–12% ở §1 (331/2.832; 344/3.200). Cột `action`
tự thân cũng không bao giờ được flag, nên việc loại cột 13 khỏi phép
đếm không ảnh hưởng kết quả.

## 3. Scaling không phải nguyên nhân (đã kiểm tra cả 3 cách)

BART action được học trên dữ liệu đã chuẩn hóa (`infer_causes_hte`:
scaler riêng cho nửa train và nửa test; scaler **train** là scaler được
lưu). Nhóm 2 chuẩn hóa query bằng scaler train (đúng quy ước của
dashboard). Nhóm 3a kiểm tra thêm scaler test (tái tạo đúng phép chia)
và query thô:

| Construction / scaling | Flag 0 cột | Flag 1 cột | Flag >1 (valid) | Cột từng được flag |
|---|---:|---:|---:|---|
| A / scaler train (đã lưu) | 2.501 | 331 | **0** | heavy2 |
| A / scaler test (tái tạo) | 2.503 | 329 | **0** | heavy2 |
| A / thô (không chuẩn hóa) | 2.832 | 0 | **0** | — |
| B / scaler train | 2.856 | 344 | **0** | heavy2 |
| B / scaler test | 2.858 | 342 | **0** | heavy2 |
| B / thô | 3.200 | 0 | **0** | — |

Scaler train và test gần như giống nhau (chênh lệch trung bình lớn nhất
2,99 và độ lệch chuẩn lớn nhất 11,89 đều ở cột `timestep`, giá trị
~771/215; các cột còn lại chênh <0,08) — đổi scaler chỉ đổi 2 hàng.
Query thô flag **0 hàng**: cây HTE sống trong không gian đã chuẩn hóa
(ngưỡng tách quanh 0 ± vài đơn vị), còn `timestep` thô chạy 0–2.237
(trung vị 601), nên mọi hàng thô rơi vào cùng lá không có ý nghĩa.

Lưu ý về protocol gốc: notebook fidelity của tác giả cho gym
(`XRL.local_causal_network`) **truyền query thô** vào `local_treatments`
trong khi BART được học trên dữ liệu đã chuẩn hóa — một sự không nhất
quán có sẵn trong code tác giả. Với CartPole (đặc trưng cỡ ~1) nó vẫn
cho hàng valid; với MicroRTS (cột `timestep` cỡ hàng trăm) nó triệt
tiêu mọi flag. Nhóm 2 đã đúng khi dùng query chuẩn hóa cho MicroRTS;
nhưng ngay cả với scaling đúng, §2 cho thấy valid vẫn bằng 0.

Ngưỡng lọc cũng đã kiểm tra: notebook đếm cạnh vào nút action từ nút
không phải action và yêu cầu `count > 1` (tức ≥2); script nhóm 2 dùng
đúng `>1` trên flag của BART action, tương đương với cách notebook đếm
cạnh vào nút action trong đồ thị cục bộ. Không có lỗi ngưỡng.

## 4. Đối chứng: cùng cơ chế, target khác thì flag bình thường

BART của target `worker1` (5 cause toàn cục), cùng pipeline
`local_treatments` + scaler của nó, trên 1.000 hàng mẫu (chuẩn hóa):
163 hàng flag 0 cột, 534 flag 1, 284 flag 2, 19 flag 3 — **303 hàng
>1 flag**. Cơ chế flag, unpickle mô hình, và scaling đều hoạt động;
vấn đề nằm riêng ở BART action. (Với query thô, đối chứng này cũng sụp
xuống 0/500 hàng >1 — củng cố §3.)

Kiểm tra thêm: `ITE(heavy2)` trên mô hình đã unpickle trả về 500 mẫu
hậu nghiệm thật (hàng 0: các giá trị đầu 0,20 / 0,12 / 0,27 / 0,15 /
0,50), và `predict` hoạt động — mô hình đã lưu không bị hỏng do
unpickle.

## 5. Tín hiệu action yếu: BART khớp kém, baseline rẻ tiền cũng thua đa số

- BART action (dự đoán trên 2.832 cặp, không gian scaler train):
  R² = 0,107; tương quan = 0,329; accuracy sau làm tròn = 26,9% so với
  baseline đa số 65,3%. Dự đoán dồn quanh trung bình (trung bình 0,94,
  độ lệch chuẩn 0,46, khoảng −0,09–1,94) trong khi nhãn đích có độ lệch
  chuẩn 1,57 — mô hình không bao giờ dự đoán được nhãn 4 (produce).
- Cây quyết định rẻ tiền trên construction A (đặc trưng thô, chia 60/40
  seed 10): accuracy 52,1% so với đa số 64,3%. Độ quan trọng đặc trưng
  cao nhất là `timestep` (0,464), rồi `worker1` (0,143) — nhưng liên
  quan dự đoán không đồng nghĩa với hiệu ứng điều trị dị biệt có ý
  nghĩa: cây HTE của `timestep` không có cụm nào có ý nghĩa (§2).
- Các cột state phần lớn là số đếm thưa, nhiều giá trị 0 (vd `light1`
  0 ở 3.879/4.924 hàng; `light2` hằng 0 ở mọi hàng); `heavy2` nhị phân
  (1 ở 1.790/4.924 hàng). Với biểu diễn này, nhãn action tiếp theo gần
  như không mang tín hiệu nhân quả mà HTE có thể phát hiện ở cỡ mẫu này.

## 6. Học lại: đổi seed / đổi biến thể protocol cho 0 cause (yếu hơn, không mạnh hơn)

Học lại BART action trên **cùng 2.832 cặp** (dbarts 0.9.34, `R_LIBS`
như nhóm 2), chỉ để đo độ mong manh — không dùng để thay protocol:

| Biến thể | Cause toàn cục | Thời gian |
|---|---|---:|
| Gốc nhóm 2: MicroRTS, chia 50/50, seed 99, 2 scaler | `heavy2` (1) | 21,2 s |
| MicroRTS, chia 50/50, **seed 1**, 2 scaler | **0** | 21,8 s |
| Kiểu gym (`cxrl`): chia 70/30, seed 99, **1 scaler** | **0** | 13,6 s |

Cả hai biến thể đều cho 0 cụm HTE có ý nghĩa ở **mọi** cột. Kết quả
1-cause của nhóm 2 đã là phía "lạc quan" của sự mong manh theo seed;
không có biến thể nào cho ≥2 cause. Đổi seed để tìm ≥2 cause sẽ là
săn số (fishing), không phải sửa lỗi — không làm.

## 7. Hệ quả cho bản không-lọc (giữ lại, kèm giải thích)

Bản `fidelity_nofilter.json` bỏ lọc valid và chạy trên toàn bộ hàng.
Vì tín hiệu cục bộ rỗng, vòng gộp ITE của nó rơi vào nhánh fallback
(trung bình không trọng số) ở **1.004/1.133** hàng test (A) và
**1.135/1.280** hàng test (B) — tức ~89% dự đoán "BART" không dùng
trọng số nhân quả nào. Các số BART không-lọc (A: acc 25,15% / F1w
27,62%; B: acc 30,86% / F1w 32,57%) vì vậy không được đọc như hiệu năng
của mô hình nhân quả; chúng chủ yếu phản ánh một hồi quy BART duy nhất
dồn quanh trung bình. LM/DT trong cùng bản cũng chỉ ngang baseline đa
số (xem §9.3 của báo cáo MicroRTS). **Giữ bản không-lọc làm vết kiểm
toán, không trình bày nó như fidelity đúng protocol.**

## 8. Những gì đã thử và chưa thử

Đã thử (bằng chứng ở trên + `fidelity_diagnosis.json`): kiểm tra cây
HTE từng cột; 3 cách scaling; đối chứng `worker1`; kiểm tra ngưỡng
`>1` khớp notebook; kiểm tra mô hình unpickle (`predict`/`ITE` chạy
thật); học lại với seed khác và biến thể gym; đo độ khớp BART và probe
cây quyết định.

Chưa thử và không thử trong phạm vi này: thu thêm dữ liệu (~500 episode
như quy mô tên file tác giả gợi ý, ước ~5,5 giờ rollout) — đó là hướng
duy nhất có cơ sở để tăng lực thống kê của kiểm định HTE, nhưng vượt
giới hạn 2–3 giờ của bước này và không đảm bảo cho ≥2 cause; mã hóa
action khác (paper không công bố) — sẽ là một thí nghiệm mới, không
phải sửa lỗi.

## 9. Kết luận

Tập valid rỗng là **kết quả thật của dữ liệu và của máy HTE của tác
giả**, đã được giải thích tới cấp cơ chế (1 cause toàn cục → tối đa 1
flag cục bộ → ngưỡng >1 không đạt được) và đã được kiểm tra là bền vững
trước scaling, ngưỡng, và biến thể protocol (các biến thể còn cho ít
cause hơn). **Không có sửa chữa nào được áp dụng cho fidelity.** Cập
nhật báo cáo: §9.5 của `muc3/microRTS-thu-nghiem.md`.
