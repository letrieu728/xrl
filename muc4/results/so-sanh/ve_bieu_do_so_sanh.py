#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vẽ 3 biểu đồ so sánh số liệu tác giả CausalXRL vs số liệu của nhóm.
Nguồn số liệu (đọc từ nguồn gốc, không bịa):
- Tác giả: arXiv:2610.42028v1, Table 1 (tr. 8), §5.2 (F1 weighted khi
  multiclass), Appendix D + Hình 8 (sensitivity). Số cạnh Hình 8
  (21/16/13) là số đếm từ hình gốc, khớp với số cạnh mất ghi trong
  text Appendix D (90%->95% mất 5 cạnh, 95%->99% mất thêm 3 cạnh).
- Nhóm Mục 3: muc3/EXPERIMENT_REPORT.md (đã sửa 12/11/2026, commit
  5406c94) + muc3/results/baselines_recomputed.json + nhóm 1
  (commit 2237c0f): muc3/results/lunarlander/bart_fidelity_weighted.json
  (weighted-F1 81,52% khôi phục đúng từ checkpoint điểm từng action,
  không chạy lại BART). Seed 99.
- Nhóm GES (nhóm 3, commit d105410): muc3/results/*/ges.json
  (causal-learn 0.1.4.8, local_score_BIC, 3 lần lặp, tập cạnh giống hệt;
  thời gian là median wall-clock). PC/FCI cùng phiên GES ghi trong
  ges.json từng môi trường.
- Nhóm Mục 4: muc4/results/ablation-bart/ci_sensitivity.md (BART thật,
  CartPole 200 episode, seed 0; đã kiểm chứng lại 13/11/2026, số
  không đổi so với bản vẽ trước).
Chạy: python3 ve_bieu_do_so_sanh.py [thư-mục-đích]
Mặc định ghi 3 file PNG vào cùng thư mục chứa script này.
Ảnh PNG không push lên GitHub qua connector (chỉ nhận UTF-8) — xem
SO-SANH-TAC-GIA-VS-NHOM.md để biết link Drive chứa ảnh.
Cập nhật 13/11/2026 (HEAD 7a3c64b): biểu đồ 1 dùng weighted-F1 LunarLander
81,52% (không còn macro-F1 cũ); biểu đồ 2 thêm cột GES đủ 4 môi trường
và thêm nhóm MicroRTS; biểu đồ 3 giữ nguyên số (đã kiểm chứng lại).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

C_AUTHOR = "#1f77b4"   # xanh dương: tác giả
C_GROUP = "#ff7f0e"    # cam: nhóm
C_GROUP2 = "#2ca02c"   # xanh lá: nhóm (phương pháp thứ hai)

ENVS = ["CartPole", "LunarLander", "PongDuel"]

# ---------------------------------------------------------------- chart 1
# BART fidelity: Accuracy + F1, tác giả vs nhóm.
# Tác giả: Table 1, F1 weighted. Nhóm: CartPole F1 nhị phân (so được,
# chênh -0,10 điểm %); LunarLander weighted-F1 81,52% khôi phục đúng từ
# checkpoint (chênh -5,08 điểm %, so được); PongDuel là PROXY hồi quy
# làm tròn (cả Acc lẫn F1 đều KHÔNG so được với paper).
author_acc = [91.4, 88.7, 90.2]
author_f1 = [91.6, 86.6, 90.7]          # weighted-F1
group_acc = [91.62, 82.80, 81.73]
group_f1 = [91.50, 81.52, 19.51]        # CartPole: F1 nhị phân;
# LunarLander: weighted-F1; PongDuel: macro-F1 của PROXY — KHÔNG so được.

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
x = np.arange(len(ENVS))
w = 0.36
for ax, a_vals, g_vals, title in [
    (axes[0], author_acc, group_acc, "Accuracy (%) — BART / CausalXRL"),
    (axes[1], author_f1, group_f1, "F1 (%) — BART / CausalXRL"),
]:
    b1 = ax.bar(x - w / 2, a_vals, w, color=C_AUTHOR, label="Tác giả (Table 1)")
    b2 = ax.bar(x + w / 2, g_vals, w, color=C_GROUP, label="Nhóm (Mục 3)")
    # Gạch chéo cột PongDuel của nhóm ở CẢ 2 panel: PROXY, không so được.
    b2[2].set_hatch("xxx")
    b2[2].set_edgecolor("black")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.2f}".rstrip("0").rstrip("."),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ENVS)
    ax.set_ylim(0, 108)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.3)
axes[1].text(0.5, -0.30,
             "Cột gạch chéo KHÔNG so được với tác giả: PongDuel của nhóm là PROXY\n"
             "hồi quy số làm tròn 0–8 (Acc 81,73%, macro-F1 19,51%), không phải\n"
             "fidelity phân loại. LunarLander nhóm dùng weighted-F1 81,52% (khôi\n"
             "phục đúng từ checkpoint, chênh −5,08 điểm % so với paper, so được).",
             transform=axes[1].transAxes, ha="center", va="top", fontsize=9,
             color="#b00000")
fig.suptitle("So sánh fidelity BART: tác giả vs nhóm (Mục 3, seed 99)",
             fontsize=13, fontweight="bold")
from matplotlib.patches import Patch
fig.legend([Patch(facecolor=C_AUTHOR), Patch(facecolor=C_GROUP)],
           ["Tác giả (Table 1)", "Nhóm (Mục 3)"],
           loc="upper center", bbox_to_anchor=(0.5, 0.93),
           ncol=2, fontsize=10, frameon=False)
fig.tight_layout(rect=[0, 0.02, 1, 0.88])
fig.savefig(os.path.join(OUT, "bieu-do-1-fidelity-bart.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- chart 2
# Thời gian học cấu trúc (giây, thang log), đủ 4 môi trường, thêm GES.
# CAVEAT QUAN TRỌNG: thời gian của tác giả đo bằng timer mô phỏng trong
# notebook eval_time_* (dữ liệu ngẫu nhiên + CITestFast), KHÔNG phải
# wall-clock BART thật; thời gian của nhóm là wall-clock thật. Hai loại
# thời gian này KHÔNG so trực tiếp được — biểu đồ chỉ cho thấy chúng
# ngược hướng nhau, và ghi chú này bắt buộc phải đi kèm biểu đồ.
# GES của nhóm: causal-learn 0.1.4.8, local_score_BIC, median 3 lần lặp.
# PC/FCI của nhóm ở 3 env gym lấy từ metrics.json của Mục 3; PC/FCI của
# MicroRTS (0,165/0,520 s) đo cùng phiên GES, biến thể states-only, khớp
# với cột GES MicroRTS 0,563 s (biến thể states+action: 0,685 s).
# BART MicroRTS 498,3 s là 13 target trên 100 episode (nhóm 2), không
# phải protocol Table 1 của paper.
ENVS4 = ["CartPole", "LunarLander", "PongDuel", "MicroRTS-Lite"]
author_pc = [1.7, 12.6, 23.2, 20.9]
author_fci = [0.8, 5.1, 8.7, 7.4]
author_ours = [0.1, 0.3, 0.5, 0.5]
group_pc = [0.076, 0.197, 0.997, 0.165]
group_fci = [0.061, 0.799, 0.957, 0.520]
group_ges = [0.018, 0.090, 0.346, 0.563]
group_bart = [242.19, 1086.40, 35531.20, 498.30]

fig, ax = plt.subplots(figsize=(13.5, 5.8))
x = np.arange(len(ENVS4))
w = 0.11
series = [
    ("Tác giả: PC", author_pc, "#4e79a7", None),
    ("Tác giả: FCI", author_fci, "#59a14f", None),
    ("Tác giả: CausalXRL", author_ours, "#1f77b4", None),
    ("Nhóm: PC", group_pc, "#a0cbe8", "///"),
    ("Nhóm: FCI", group_fci, "#8cd17d", "///"),
    ("Nhóm: GES", group_ges, "#b07aa1", "///"),
    ("Nhóm: BART (wall-clock thật)", group_bart, "#ff7f0e", "///"),
]
for i, (label, vals, color, hatch) in enumerate(series):
    bars = ax.bar(x + (i - 3) * w, vals, w, color=color, label=label,
                  hatch=hatch, edgecolor="black" if hatch else None,
                  linewidth=0.5 if hatch else 0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.25,
                f"{v:g}", ha="center", va="bottom", fontsize=7.5, rotation=90)
ax.set_yscale("log")
ax.set_ylim(0.008, 3e5)  # chừa chỗ cho nhãn 35531.2, tránh đè tiêu đề
ax.set_xticks(x)
ax.set_xticklabels(ENVS4)
ax.set_ylabel("Thời gian (giây, thang log)")
ax.set_title("Thời gian học cấu trúc: tác giả vs nhóm — HAI CÁCH ĐO KHÁC NHAU, KHÔNG SO TRỰC TIẾP ĐƯỢC",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3, which="both")
ax.legend(fontsize=8, ncol=4, loc="upper left")
ax.text(0.5, -0.24,
        "CAVEAT: thời gian của tác giả (Table 1) là timer mô phỏng trong notebook eval_time_* của repo,\n"
        "không phải wall-clock BART thật. Thời gian của nhóm là wall-clock thật (BART PongDuel 35.531 s ≈ 9,9 giờ).\n"
        "GES của nhóm: causal-learn 0.1.4.8, BIC, median 3 lần lặp (MicroRTS: biến thể states-only 0,563 s;\n"
        "states+action 0,685 s). BART MicroRTS 498,3 s là 13 target / 100 episode, không phải protocol Table 1.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#b00000")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bieu-do-2-thoi-gian-hoc-cau-truc.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- chart 3
# Sensitivity: số cạnh theo ngưỡng CI 90/95/99%. Số KHÔNG ĐỔI so với bản
# vẽ trước — đã kiểm chứng lại 13/11/2026 từ ci_sensitivity.md.
# Tác giả: MicroRTS (Hình 8), đếm từ hình: 21/16/13 (mất 5 rồi 3 cạnh).
# Nhóm: CartPole BART thật, 200 episode: hte 13/13/11, ate 17/15/14.
# KHÁC MÔI TRƯỜNG nên chỉ so XU HƯỚNG (cạnh giảm khi ngưỡng chặt hơn),
# không so số tuyệt đối.
levels = ["90%", "95%", "99%"]
author_edges = [21, 16, 13]   # MicroRTS, đếm từ Hình 8
group_hte = [13, 13, 11]      # CartPole, BART thật
group_ate = [17, 15, 14]      # CartPole, BART thật

fig, ax = plt.subplots(figsize=(9.5, 5.4))
x = np.arange(len(levels))
w = 0.26
for offset, vals, color, label, hatch in [
    (-w, author_edges, C_AUTHOR, "Tác giả: MicroRTS (Hình 8)", None),
    (0, group_hte, C_GROUP, "Nhóm: CartPole hte (BART thật)", "///"),
    (w, group_ate, C_GROUP2, "Nhóm: CartPole ate (BART thật)", "///"),
]:
    bars = ax.bar(x + offset, vals, w, color=color, label=label, hatch=hatch,
                  edgecolor="black" if hatch else None, linewidth=0.5 if hatch else 0)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([f"Ngưỡng CI {lv}" for lv in levels])
ax.set_ylabel("Số cạnh của đồ thị toàn cục")
ax.set_ylim(0, 25)
ax.set_title("Độ nhạy theo ngưỡng CI: số cạnh giảm khi ngưỡng chặt hơn",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)
ax.text(0.5, -0.22,
        "CHỈ SO XU HƯỚNG: tác giả đo trên MicroRTS, nhóm đo trên CartPole —\n"
        "môi trường khác nhau nên số tuyệt đối không so được. Tác giả: 90%→95%\n"
        "mất 5 cạnh, 95%→99% mất thêm 3 cạnh (Appendix D). Nhóm hte: 13→13→11;\n"
        "ate: 17→15→14 (cùng model đã fit, tính lại từ 500 ITE draws).",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#b00000")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bieu-do-3-do-nhay-ci.png"), dpi=200)
plt.close(fig)

print("Đã ghi 3 biểu đồ vào:", OUT)
for name in ["bieu-do-1-fidelity-bart.png",
             "bieu-do-2-thoi-gian-hoc-cau-truc.png",
             "bieu-do-3-do-nhay-ci.png"]:
    p = os.path.join(OUT, name)
    print(name, os.path.getsize(p), "byte")
