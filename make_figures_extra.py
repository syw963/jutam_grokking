"""추가 그림 2장 (10월 22일): 값은 docs/실험_기록.md의 E30~E37 표에서 옮김.

F9  법칙의 일반성: 모델 크기·구조·연산을 바꿔도 같은 꼴 (E31·E32·E33·E35)
F10 외운 오답과 규제 세기: wd가 약하면 공존, 강하면 고침, 너무 강하면 규칙도 잃음 (E30·E34·E36·E37·E41)

    python make_figures_extra.py --out docs/figures
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import C, INK, INK2, GRID, SURF, setup_style

N_ = np.array([31, 41, 50, 61, 71])
C5 = N_ * (N_ + 1) / 2
BASE = [298, 554, 618, 797, 1047]          # E32 기본 d128·2층
BIG = [277, 490, 574, 774, 1003]           # E32 큰 d256·4층
GPT = [984, 1406, 1855, 2398, 3384]        # E31
MUL_N = np.array([31, 41, 47, 53, 61, 71])  # E33·E35 곱셈, n 소수
MUL = [240, 397, 400, 545, 752, 975]
MUL50 = (1275, 325)                         # E35 곱셈 n=50 (합성수)


def f9(out):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    xs = np.logspace(np.log10(400), np.log10(3000), 50)
    ax.plot(xs, 2.81 * xs ** 0.764, "--", color=INK2, lw=1.2, zorder=1)
    ax.text(3050, 2.81 * 3000 ** 0.764, "기본 법칙\n2.81·C^0.764", color=INK2, fontsize=8.5, ha="left", va="center")
    series = [("GPT (구조 다름) · 지수 0.73", C5, GPT, C[1], "s"),
              ("큰 모델 (파라미터 8배) · 0.75", C5, BIG, C[2], "^"),
              ("기본 모델 · 0.72", C5, BASE, C[0], "o"),
              ("곱셈 a×b (n 소수) · 0.85", MUL_N * (MUL_N + 1) / 2, MUL, C[3], "D")]
    for lab, x, y, col, mk in series:
        ax.plot(x, y, "-", color=col, lw=2, marker=mk, ms=8, mec=SURF, mew=2, zorder=3, label="_")
        ax.plot([], [], "-", color=col, marker=mk, ms=7, label=lab)
    ax.legend(loc="upper left", fontsize=9, handlelength=2.2)
    ax.plot(*MUL50, marker="D", ms=9, mfc=SURF, mec=C[3], mew=2, ls="none", zorder=4)
    ax.annotate("곱셈 n=50 (합성수): 같은 크기의 소수 표보다 쉬움", MUL50, xytext=(10, -14),
                textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(400, 5000); ax.set_ylim(180, 4500)
    ax.set_xticks([500, 1000, 2000]); ax.set_xticklabels(["500", "1,000", "2,000"])
    ax.set_yticks([200, 500, 1000, 2000, 4000]); ax.set_yticklabels(["200", "500", "1,000", "2,000", "4,000"])
    ax.minorticks_off()
    ax.set_xlabel("연산 표의 칸 수 C (로그)")
    ax.set_ylabel("일반화에 필요한 데이터 수 N0 (로그)")
    ax.set_title("모델과 연산을 바꿔도 같은 꼴: 기울기(지수)는 비슷하고 높이(상수)만 다르다", loc="left")
    ax.spines["left"].set_visible(False)
    fig.savefig(Path(out, "F9_law_generality.png"))
    plt.close(fig)


def f10(out):
    """(가) 외운 오답 비율 vs wd, (나) 두 수 과제의 처음 보는 문제 정답률 vs wd. 값은 docs/실험_기록.md E30·E34·E36·E37·E41."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    series = [("숫자 5개 · 흩어진 오답", [1, 2, 3, 5, 10], [100, 100, 100, 11, 0], C[0], "o", (8, -14)),
              ("숫자 5개 · 같은 칸에 일관된 오답", [1, 2, 3, 5, 10], [100, 100, 100, 100, 5.5], C[1], "s", (8, 10)),
              ("두 수 과제 · 정답 전체 오답", [1, 2, 3, 4, 5, 7, 10], [100, 100, 100, 100, 100, 21, 0.5], C[2], "^", (8, -4))]
    for lab, x, y, col, mk, off in series:
        ax.plot(x, y, "-", color=col, lw=2, marker=mk, ms=8, mec=SURF, mew=2, zorder=3, label=lab)
    ax.plot([15], [4.4], marker="^", ms=8, mfc=SURF, mec=C[2], mew=2, ls="none", zorder=3)
    ax.annotate("wd 15: 오답을 외우지 않지만\n규칙도 잃음 (나)", (15, 4.4), xytext=(4, 110), textcoords="offset points",
                ha="center", fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate("5개 중 4개 고침", (7, 21), xytext=(9, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.axvspan(0.9, 3.4, color=GRID, alpha=0.45, zorder=0, lw=0)
    ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.06), frameon=True, facecolor=SURF, edgecolor="none")
    ax.text(1.75, 60, "공존\n규칙을 깨치고도\n외운 오답을 그대로 냄", ha="center", va="center", fontsize=9, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(0.9, 30); ax.set_ylim(-5, 108)
    ax.set_xticks([1, 2, 3, 4, 5, 7, 10, 15]); ax.set_xticklabels(["1", "2", "3", "4", "5", "7", "10", "15"]); ax.minorticks_off()
    ax.set_xlabel("weight decay (규제 세기, 로그)")
    ax.set_ylabel("학습 끝에 외운 오답을 내는 비율 (%)")
    ax.set_title("(가) 외운 오답은 규제가 약하면 남고, 강하면 고쳐진다", loc="left", fontsize=10.5)
    ax = axes[1]
    wd = [1, 2, 3, 4, 5, 7, 10, 15]
    noisy = [97.6, 97.4, 81.8, 36.8, 60.8, 97.4, 97.3, 28.0]
    clean = [100, 100, 100, 100, 100, 100, 100, 84.0]
    ax.plot(wd, clean, "-", color=INK2, lw=1.6, marker="o", ms=6, mec=SURF, mew=1.5, zorder=2, label="오답 없음 (대조군)")
    ax.plot(wd, noisy, "-", color=C[2], lw=2, marker="^", ms=8, mec=SURF, mew=2, zorder=3, label="오답 예제 10%")
    ax.axvspan(3.4, 5.9, color=C[1], alpha=0.10, zorder=0, lw=0)
    ax.text(4.5, 12, "오답은 지키고\n규칙이 무너짐", ha="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(0.9, 18); ax.set_ylim(0, 105)
    ax.set_xticks(wd); ax.set_xticklabels([str(w) for w in wd]); ax.minorticks_off()
    ax.set_xlabel("weight decay (규제 세기, 로그)")
    ax.set_ylabel("처음 보는 문제 정답률, 학습 끝 (%)")
    ax.set_title("(나) 두 수 과제: 규칙은 지켜지는가", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left", frameon=False)
    for a in axes:
        a.spines["left"].set_visible(False)
    fig.text(0.01, -0.04, "숫자 5개: 데이터 2,048개, 오답 10%, 10만~20만 step (E30·E34·E37). 두 수 과제: 1,100개, 오답 예제 10%, 20만 step, seed 4~5개 평균 (E34·E36·E41).\n"
             "숫자 5개는 처음 보는 문제 정답률이 모든 점에서 93~100%. 데이터 768개는 wd 10에서 아무것도 배우지 못함.",
             fontsize=7.8, color=INK2, va="top")
    fig.savefig(Path(out, "F10_noise_vs_wd.png"), bbox_inches="tight")
    plt.close(fig)


def f11(out):
    """E39: 흩어진 오답은 2층 attention(문맥)을 거쳐 외워지고, 같은 칸 오답은 표처럼 저장된다. 값은 docs/실험_기록.md E39."""
    WR, WR2, OK = C[1], "#f4b79c", C[2]   # 외운 오답(진한/연한 주황), 원래 정답·처음 보는 문제(초록)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    labels = ["흩어진 오답\n(예제마다 한 칸)", "같은 칸 오답\n(칸마다 일관)", "오답 없음\n(대조군)"]
    x = np.arange(3)
    ax = axes[0]
    mem_on, mem_off, test_off = [0.88, 1.00], [0.00, 1.00], [1.00, 1.00, 1.00]
    ax.bar(x[:2] - 0.27, mem_on, 0.26, color=WR2, label="외운 오답: 그대로")
    ax.bar(x[:2], mem_off, 0.26, color=WR, label="외운 오답: 2층 attention 끔")
    ax.bar(x + 0.27, test_off, 0.26, color=OK, label="처음 보는 문제 정답률: 2층 attention 끔")
    ax.text(x[0], 0.03, "0", ha="center", fontsize=10, color=WR, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 1.05)
    ax.set_title("(가) 2층 attention을 끄면", loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    ax = axes[1]
    keep, back = [0.40, 1.00], [0.60, 0.00]
    ax.bar(x[:2] - 0.14, keep, 0.26, color=WR, label="외운 오답 유지")
    ax.bar(x[:2] + 0.14, back, 0.26, color=OK, label="원래 정답으로 돌아감")
    ax.text(x[1] + 0.14, 0.03, "0", ha="center", fontsize=10, color=OK, fontweight="bold")
    ax.set_xticks(x[:2]); ax.set_xticklabels(labels[:2], fontsize=9); ax.set_ylim(0, 1.05); ax.set_xlim(-0.6, 1.6)
    ax.set_title("(나) 규칙과 무관한 숫자 하나만 바꾸면", loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    for a in axes:
        a.spines["left"].set_visible(False)
    fig.text(0.01, -0.23, "N 2,048, 오답 10%. (가) wd 1, seed 4개 평균. (나) seed 0, 흩어진 오답 wd 3·같은 칸 오답 wd 1,\n오답 자리마다 규칙과 무관한 숫자 3자리를 하나씩 바꿈 (E39).",
             fontsize=7.8, color=INK2, va="top")
    fig.savefig(Path(out, "F11_noise_mechanism.png"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    setup_style()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f9(a.out); f10(a.out); f11(a.out)
    print("저장:", a.out)
