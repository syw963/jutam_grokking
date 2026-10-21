"""포스터 동기 칸: 과제(수열 변환 규칙) 그림."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "Noto Sans CJK KR"
NAVY = "#26386f"; BLUE = "#3f6cb5"; LBLUE = "#dfe8f6"; ORG = "#e07b39"; LORG = "#fbe3d3"
INK = "#1d1d1f"; GRAY = "#6b7180"; LG = "#eef0f4"

def box(ax, x, y, w, h, fc, r=0.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}", fc=fc, ec="none"))

fig, ax = plt.subplots(figsize=(8, 3.3)); ax.set_xlim(0, 8); ax.set_ylim(0, 3.3); ax.axis("off")
xs = [3, 17, 8, 42, 25]
outs = [f"{xs[i]}+{xs[(i + 1) % 5]}" for i in range(5)]
X0, W, GAP = 1.25, 1.0, 0.18
cx = [X0 + i * (W + GAP) + W / 2 for i in range(5)]
# 입력
ax.text(0.05, 2.73, "입력", fontsize=16, color=NAVY, weight="bold", va="center")
for c, v in zip(cx, xs):
    box(ax, c - W / 2, 2.45, W, 0.56, LG)
    ax.text(c, 2.73, str(v), ha="center", va="center", fontsize=20, weight="bold", color=INK)
# 출력
ax.text(0.05, 0.98, "출력", fontsize=16, color=ORG, weight="bold", va="center")
for c, o in zip(cx, outs):
    box(ax, c - W / 2, 0.7, W, 0.56, LORG)
    ax.text(c, 0.98, o, ha="center", va="center", fontsize=15, weight="bold", color=ORG)
# 가운데: 모델
box(ax, X0, 1.55, 5 * W + 4 * GAP, 0.62, NAVY, r=0.12)
ax.text(X0 + (5 * W + 4 * GAP) / 2, 1.86, "AI (작은 Transformer)",
        ha="center", va="center", fontsize=16, color="white", weight="bold")
for c in cx:
    ax.annotate("", xy=(c, 2.2), xytext=(c, 2.43), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
    ax.annotate("", xy=(c, 1.28), xytext=(c, 1.53), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
# 아래 설명
ax.text(0.05, 0.28, "각 숫자 + 바로 다음 숫자 (마지막은 첫 숫자와).  합이 50을 넘으면 1~50으로 되돌린다.",
        fontsize=12.5, color=GRAY, va="center")
fig.savefig("docs/poster/rq_figs/task.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("docs/poster/rq_figs/task.svg", bbox_inches="tight")
