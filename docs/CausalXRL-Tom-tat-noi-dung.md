# Tóm tắt bài báo CausalXRL — nội dung và những gì tác giả đã làm

**Bài báo:** CausalXRL: Explainable Reinforcement Learning through Causal Graph Reasoning
**Tác giả:** Yanming Zhang, Eric Papenhausen, Klaus Mueller — Department of Computer Science, Stony Brook University, USA
**Xuất bản:** arXiv:2610.42028v1, 30/10/2026; Proceedings of the 43rd ICML, Seoul, PMLR 306, 2026. 17 trang.
**Code:** https://github.com/yanmluk/causal-xrl
**Bản dịch toàn văn tiếng Việt:** `causalxrl-ban-dich-tieng-viet/causalxrl-ban-dich-tieng-viet.docx` (21 trang Word)
**PDF gốc:** `CausalXRL-2610.42028.pdf`

---

## 1. Bài báo nói về cái gì? (1 đoạn)

Học tăng cường (Reinforcement Learning, RL) train được agent chơi game, lái xe, làm robot rất giỏi, nhưng agent là “hộp đen”: người ta không biết vì sao nó chọn hành động đó, nên không tin tưởng, không debug được, nhất là trong lĩnh vực quan trọng như y tế.

CausalXRL là một framework giải thích hành vi của agent RL bằng **đồ thị nhân quả (causal graph)**. Thay vì chỉ tô màu xem agent “nhìn” vào đâu (saliency map), nó học xem biến nào gây ra biến nào — ví dụ “số lính ranged tăng lên làm giảm số lính heavy của địch” — rồi trình bày thành đồ thị có nhiều tầng: đồ thị toàn cục (chiến lược chung) và mô hình local (tình huống cụ thể, có thể suy luận phản thực: “nếu lúc đó làm khác thì sao?”).

## 2. Vấn đề tác giả đặt ra

1. RL / Deep RL mạnh nhưng không giải thích được.
2. Các phương pháp giải thích RL (XRL) trước đây chủ yếu là post-hoc, chia thành model-level, reward-level, state-level, task-level; chúng nói được “agent chú ý vào đâu” nhưng ít nói được “vì sao”.
3. Cách dùng mô hình nhân quả trước đây (Action Influence Model — AIM, Madumal et al. 2020) có 2 hạn chế:
   - Cần chuyên gia định trước cấu trúc nhân quả, và gắn cạnh nhân quả trực tiếp vào action — không dùng được trong game chiến thuật thời gian thực (RTS) nhiều unit, vì quan hệ lúc đó trở thành tất định (ví dụ lính tấn công thì đương nhiên phá nhà địch), không còn thông tin gì thêm.
   - Yu et al. (2023) cải tiến bằng conditional independence test + attention, dùng được cho action liên tục, nhưng vẫn giữ cách gắn cạnh vào action.

CausalXRL khắc phục bằng cách: **không gắn cạnh vào action**, mà học quan hệ nhân quả giữa các **biểu diễn có thể hiểu được (interpretable representations)** — ví dụ số lượng từng loại unit, tài nguyên, vị trí — ở tầng chiến lược, rồi mới dùng các biểu diễn đó để dự đoán hành động.

## 3. Tác giả đã làm gì? — phương pháp, 3 bước

### Bước 1: Trích xuất biểu diễn có thể hiểu được
- Từ dữ liệu thô của môi trường (pixel, sensor), trích ra các đặc trưng mà con người hiểu được.
- Trong MicroRTS: quan sát là tensor 16×16×27, trích ra 14 biểu diễn chiến lược (số unit mỗi loại của 2 phe, tài nguyên, v.v.).
- Trong CartPole / LunarLander: dùng luôn các biến trạng thái gốc (vị trí xe, góc cột, vận tốc, v.v.) — 4 và 8 biểu diễn.

### Bước 2: Học đồ thị nhân quả toàn cục (global causal graph)
- Chạy agent RL đã train sẵn 200 episode, lưu toàn bộ transition (state t, action t, state t+1) vào replay buffer.
- Với mỗi biến đích ở t+1, dùng mô hình BART (Bayesian Additive Regression Trees, qua R package `dbarts`) để ước lượng hiệu ứng nhân quả trung bình (ATE) và hiệu ứng dị biệt (HTE / CATE) của từng biến ở t lên nó.
- Một biến được coi là nguyên nhân nếu khoảng tin cậy 95% của ATE không chứa 0, hoặc có ít nhất một cụm (gom bằng cây quyết định) có hiệu ứng đáng kể.
- Kết quả là một đồ thị có hướng (DAG): node là các biểu diễn, cạnh là quan hệ nhân quả. Thuật toán 1 trong paper.
- Điểm mạnh tác giả nhấn mạnh: cách này nhanh hơn các thuật toán học cấu trúc dựa trên conditional independence test (PC, FCI), vì chỉ cần fit BART cho từng biến đích thay vì test độc lập có điều kiện cho từng cặp.

### Bước 3: Mô hình nhân quả local + giải thích phản thực
- **Local causal model:** với một trạng thái cụ thể đang xét, dùng các model BART đã học để xác định trong tình huống đó, biến nào đang thực sự gây ra biến nào — cho phép suy luận phản thực (counterfactual): “nếu tăng biến X lên thì biến Y sẽ thay đổi bao nhiêu?”. Thuật toán 2 trong paper.
- **Dự đoán hành vi:** dùng các model local để dự đoán hành động tiếp theo của agent, có trọng số theo độ lớn |ITE| (individual treatment effect) của từng nguyên nhân.
- **Counterfactual bằng DiCE:** sinh các trạng thái giả định “nếu state khác đi thì agent sẽ làm khác không”, để người dùng kiểm tra tính nhất quán của lời giải thích.
- Tác giả còn làm một giao diện web tương tác cho MicroRTS: bảng game 16×16, thanh trượt thời gian, đồ thị nhân quả local hiển thị kèm theo, để chuyên gia tự khám phá.

## 4. Tác giả đã đánh giá như thế nào? — 2 phần

### 4.1. Đánh giá định lượng (Chương 5, Table 1)
Chạy trên 4 môi trường: CartPole (4 biểu diễn), LunarLander (8), PongDuel (12, multi-agent), MicroRTS-Lite (14). Mỗi môi trường chạy agent đã train 200 episode, lấy dữ liệu làm hold-out.

Hai tiêu chí:
- **Behavior prediction (tính trung thực):** model local có dự đoán đúng hành động tiếp theo của agent không? So với 2 baseline: Linear Model (LM) và Decision Tree (DT).
- **Structural learning efficiency:** học đồ thị toàn cục mất bao lâu? So với PC và FCI (CIT-based).

**Kết quả Table 1 (Accuracy % / F1 % / thời gian giây):**

| Môi trường | #Rep | LM Acc | DT Acc | **CausalXRL Acc** | LM F1 | DT F1 | **CausalXRL F1** | PC (s) | FCI (s) | **CausalXRL (s)** |
|---|---|---|---|---|---|---|---|---|---|---|
| CartPole | 4 | 88,5 | 94,1 | **91,4** | 90,0 | 94,6 | **91,6** | 1,7 | 0,8 | **0,1** |
| LunarLander | 8 | 81,3 | 85,1 | **88,7** | 80,2 | 84,7 | **86,6** | 12,6 | 5,1 | **0,3** |
| PongDuel | 12 | 81,5 | 88,3 | **90,2** | 81,1 | 88,4 | **90,7** | 23,2 | 8,7 | **0,5** |
| MicroRTS-Lite | 14 | 75,3 | 80,3 | **86,0** | 74,5 | 82,2 | **85,9** | 20,9 | 7,4 | **0,5** |

Đọc kết quả đúng như tác giả viết:
- Môi trường đơn giản (CartPole): CausalXRL **thua Decision Tree** (91,4% so với 94,1%) — tác giả thừa nhận, chỉ nói là “competitive”.
- Môi trường phức tạp hơn (PongDuel, MicroRTS-Lite): CausalXRL thắng rõ (90,2% so với 88,3%; 86,0% so với 80,3%).
- Về tốc độ học cấu trúc: CausalXRL nhanh hơn PC/FCI rất nhiều, càng nhiều biểu diễn càng chênh lệch (MicroRTS-Lite: 0,5s so với 20,9s / 7,4s).
- Tác giả **không** so với AIM làm baseline, với lý do mục tiêu giải thích khác nhau: AIM gắn cạnh vào action đơn lẻ, còn CausalXRL học quan hệ giữa các biểu diễn chiến lược.

### 4.2. Đánh giá với chuyên gia (Chương 6, nghiên cứu định tính)
- **8 chuyên gia** (E1–E8), mỗi người >7 năm kinh nghiệm, trung bình >1.000 giờ chơi các game RTS (StarCraft II, Age of Empires II, Age of Mythology). Được duyệt IRB số 1020950, trả $8/giờ, làm qua Zoom, think-aloud.
- Quy trình: học luật MicroRTS + kiến thức nhân quả cơ bản → làm quiz (phải đúng hết mới được tiếp tục) → xem video gameplay, tự viết ra các quan hệ nhân quả mình quan sát được → so với đồ thị toàn cục do CausalXRL học → làm bài kiểm tra dự đoán hành vi dạng trắc nghiệm, chia ngẫu nhiên 2 nhóm: nhóm đối chứng chỉ xem SHAP, nhóm thử nghiệm được xem model nhân quả local.
- **Kết quả chính:** ở bài dự đoán hành vi, nhóm có CausalXRL trả lời đúng **8/8**, nhóm chỉ có SHAP đúng **4/8**. Các chuyên gia đánh giá đồ thị toàn cục và local đều hữu ích (điểm trung bình khoảng 4,5/5), và cho rằng công cụ giúp họ hiểu chiến lược cấp cao của agent, tăng tin tưởng trong bối cảnh human–autonomy teaming.
- Đây là nghiên cứu **khám phá, định tính, mẫu nhỏ (n=8)** — tác giả không khẳng định đây là chứng minh thống kê.

## 5. Kết quả chính trong 5 dòng
1. Học được đồ thị nhân quả giữa các biểu diễn chiến lược, không cần chuyên gia định trước cấu trúc.
2. Dự đoán hành vi agent khá trung thực, đặc biệt tốt ở môi trường phức tạp (PongDuel 90,2%, MicroRTS-Lite 86,0%).
3. Học cấu trúc nhanh hơn PC/FCI hàng chục lần ở môi trường nhiều biểu diễn.
4. Chuyên gia RTS thấy công cụ hữu ích để đọc chiến lược và tin tưởng agent hơn (8/8 so với 4/8 ở bài dự đoán).
5. Hạn chế tác giả tự nhận: cần biểu diễn có thể hiểu được trích sẵn; hiệu quả phụ thuộc chất lượng biểu diễn; nghiên cứu chuyên gia chỉ là khám phá, mẫu nhỏ; chưa có human-in-the-loop / interactive alignment — đó là hướng phát triển tương lai.

## 6. Những gì tác giả KHÔNG làm (đừng nhầm)
- Không train agent RL mới. Agent dùng để thu dữ liệu là agent đã train sẵn / heuristic có sẵn; repo chỉ học đồ thị nhân quả và giải thích cho policy có sẵn.
- Không chứng minh đồ thị học được là đồ thị nhân quả “đúng” theo nghĩa triết học — chỉ chứng minh nó dự đoán hành vi tốt và học nhanh, và chuyên gia thấy nó hợp lý.
- Không công bố dữ liệu MicroRTS đầy đủ — README repo ghi phải liên hệ tác giả để lấy full artifact.
- Không có code train, không có checkpoint agent; code chính là: thu thập rollout → học DBN bằng BART → local model → DiCE counterfactual → dashboard Flask cho MicroRTS.

## 7. Nếu chỉ có 3 phút để trình bày
- **Câu hỏi:** Làm sao giải thích vì sao agent RL làm vậy, ở tầng chiến lược mà con người hiểu được?
- **Ý tưởng:** Đừng giải thích bằng pixel hay action đơn lẻ; hãy học đồ thị nhân quả giữa các biểu diễn chiến lược bằng BART (ATE/HTE), rồi dùng đồ thị đó để dự đoán và giải thích phản thực.
- **Bằng chứng:** Table 1 — thắng DT/LM ở môi trường phức tạp, học cấu trúc nhanh hơn PC/FCI hàng chục lần; nghiên cứu 8 chuyên gia: 8/8 so với 4/8.
- **Điểm yếu phải nói:** CartPole thua Decision Tree; mẫu chuyên gia chỉ có 8 người; phụ thuộc vào biểu diễn trích sẵn và vào R/dbarts khó cài.
