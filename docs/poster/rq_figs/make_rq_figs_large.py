"""포스터 연구문제 칸 그림, 글자 키운 버전 (rq1_large, rq2_large)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
plt.rcParams["font.family"] = "Noto Sans CJK KR"
NAVY = "#26386f"; BLUE = "#3f6cb5"; LBLUE = "#dfe8f6"; ORG = "#e07b39"; LORG = "#fbe3d3"
INK = "#1d1d1f"; GRAY = "#6b7180"; LG = "#eef0f4"
OUT = "docs/poster/rq_figs/"

def tri(ax, x0, y0, k, s):
    for i in range(k):
        for j in range(i, k):
            ax.add_patch(Rectangle((x0 + j * s, y0 - (i + 1) * s), s, s, fc=LBLUE, ec="white", lw=1.2))

def box(ax, x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none"))

# ---------- RQ1 ----------
fig, ax = plt.subplots(figsize=(8, 4)); ax.set_xlim(0, 8); ax.set_ylim(-0.2, 4); ax.axis("off")
ax.text(0.1, 3.75, "덧셈표의 크기 (그림은 축소)", fontsize=15, color=GRAY)
tri(ax, 0.3, 3.4, 6, 0.27)
tri(ax, 2.55, 3.4, 12, 0.135)
for cx, a, b in [(1.1, "숫자 1~23", "칸 276개"), (3.36, "숫자 1~101", "칸 5,151개")]:
    ax.text(cx, 1.2, a, ha="center", fontsize=19, color=INK, weight="bold")
    ax.text(cx, 0.7, b, ha="center", fontsize=18, color=BLUE, weight="bold")
    ax.text(cx, 0.1, "필요한 데이터 수 ?", ha="center", fontsize=14, color=NAVY, weight="bold")
steps = [("① 식 세우기", "칸 수 → 데이터 수"), ("② 미리 예측", "실험 전에 기록"), ("③ 실험으로 확인", "예측과 비교")]
for k, (t, s) in enumerate(steps):
    y = 3.65 - k * 1.3
    box(ax, 5.15, y - 1.0, 2.8, 1.0, LORG if k == 1 else LG)
    ax.text(6.55, y - 0.38, t, ha="center", fontsize=18, weight="bold", color=ORG if k == 1 else NAVY)
    ax.text(6.55, y - 0.8, s, ha="center", fontsize=14.5, color=INK)
    if k < 2:
        ax.annotate("", xy=(6.55, y - 1.3), xytext=(6.55, y - 1.02), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
fig.savefig(OUT + "rq1_large.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT + "rq1_large.svg", bbox_inches="tight")

# ---------- RQ2 ----------
fig, ax = plt.subplots(figsize=(8, 3.4)); ax.set_xlim(0, 8); ax.set_ylim(0.55, 4); ax.axis("off")
ax.text(0.1, 3.75, "학습 데이터 하나", fontsize=15, color=GRAY)
bx = 0.72
for i, v in enumerate([3, 17, 8, 42, 25]):
    box(ax, 0.1 + i * bx, 2.75, 0.62, 0.62, LG)
    ax.text(0.41 + i * bx, 3.06, str(v), ha="center", va="center", fontsize=20, weight="bold", color=INK)
ax.text(1.88, 2.45, "이웃한 두 수의 쌍 5개\n(3,17) (17,8) (8,42)\n(42,25) (25,3)", ha="center", va="top",
        fontsize=15, color=BLUE, linespacing=1.45)
ax.text(1.88, 1.05, "→ 덧셈표의 칸 5개를 본다", ha="center", va="top", fontsize=16, color=INK, weight="bold")
ax.annotate("", xy=(4.2, 2.1), xytext=(3.75, 2.1), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.5))
ax.text(6.0, 3.75, "덧셈표 예시 (칸의 수 = 본 횟수)", ha="center", fontsize=15, color=GRAY)
rng = np.random.default_rng(3); k = 7; s = 0.4; x0 = 4.6; y0 = 3.5
cols = ["#f4f5f8", "#dfe8f6", "#a9c0e6", "#6f95d2", BLUE]
for i in range(k):
    for j in range(i, k):
        c = int(rng.choice([0, 1, 1, 1, 2, 2, 3, 4]))
        ax.add_patch(Rectangle((x0 + j * s, y0 - (i + 1) * s), s, s, fc=cols[c], ec="white", lw=1.2))
        if c:
            ax.text(x0 + j * s + s / 2, y0 - (i + 0.5) * s, str(c), ha="center", va="center", fontsize=14,
                    color="white" if c >= 3 else NAVY, weight="bold")
fig.savefig(OUT + "rq2_large.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT + "rq2_large.svg", bbox_inches="tight")
