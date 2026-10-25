"""보고서용 그림 1~9 (흑백, 제목 없음, 한국어 라벨). 번호는 Google Docs 보고서 기준(10월 25일 4.2절과 4.3절 순서를 바꾸며 그림 4~8 번호를 새로 매김: 옛 8 → 4, 옛 4~7 → 5~8. 같은 날 분량 조정으로 배선 12종 그림과 연산별 배선 그림을 보고서에서 빼고 그 뒤 번호를 당김: 옛 7~11 → 5~9, 뺀 두 그림은 제외/ 폴더에 저장).

    python make_report_figures.py --res <결과 tgz를 푼 폴더> --out docs/figures/report

<결과 폴더> 아래에 필요한 것:
    E06_boundary_results/runs_boundary, E07_commute_results/runs_commute,
    E12_prediction_results/runs_e12_*, E11d_dense_internal/e11d_internal/internal.csv, E30_noise/runs_e30_*
그 밖의 값(경계, 오답 유지율 등)은 docs/실험_기록.md의 표에서 옮겼다.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

K, G1, G2, G3 = "#000000", "#555555", "#8c8c8c", "#c8c8c8"   # 검정과 회색 세 단계
W1, W2 = 6.3, 3.9   # 그림 폭(inch): 두 패널, 한 패널


def setup():
    for name in ("Noto Sans CJK KR", "NanumGothic", "AppleGothic", "Malgun Gothic"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": K, "axes.labelcolor": K, "xtick.color": K, "ytick.color": K,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
        "lines.markersize": 5, "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "legend.frameon": False, "mathtext.fontset": "dejavusans",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def comma(ax, axis="x"):
    f = FuncFormatter(lambda v, p: f"{v:,.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def log_ticks(ax, axis, ticks):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_formatter(NullFormatter())
    a.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))


def panel(ax, s):
    ax.text(-0.13, 1.03, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


# 흑백 계열: (색, 선, 마커, 채움)
STY = [(K, "-", "o", True), (K, "--", "s", False), (G2, "-", "^", True), (G1, ":", "D", False)]


def sty(i, **extra):
    c, ls, mk, fill = STY[i]
    d = dict(color=c, linestyle=ls, marker=mk, markerfacecolor=c if fill else "white", markeredgecolor=c, markeredgewidth=1)
    d.update(extra)
    return d


def load_runs(roots):
    rows = []
    for root in roots:
        for d in glob.glob(f"{root}/*"):
            d = Path(d)
            try:
                s = json.loads((d / "summary.json").read_text())
                c = json.loads((d / "config.json").read_text())
            except (FileNotFoundError, NotADirectoryError):
                continue
            tf, tg = s.get("t_fit"), s.get("t_gen")
            rows.append(dict(run=d.name, rule=c["rule"], spec=c["rule_spec"], train=c["data_meta"]["train_sets"],
                             grokked=tg is not None, delay=(tg - tf) if tg is not None and tf is not None else np.nan))
    return pd.DataFrame(rows)


def agg(df, keys):
    return df.groupby(keys, as_index=False).agg(runs=("grokked", "size"), grokked=("grokked", "sum"),
                                                 delay=("delay", "median")).assign(rate=lambda t: t.grokked / t.runs)


# ---------------------------------------------------------------- 그림 1
def fig01(res, out):
    rows = pd.read_csv(Path(res, "E06_boundary_results/runs_boundary/w5_next_m0_c768_wd1_s0/metrics.csv"))
    med = lambda v: v.rolling(5, center=True, min_periods=1).median() * 100
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.6))
    ax.plot(rows.step, med(rows.train_exact), color=K, ls="--", label="학습 정확도")
    ax.plot(rows.step, med(rows.test_exact), color=K, ls="-", label="평가 정확도")
    for x in (300, 24500):
        ax.axvline(x, color=G2, lw=0.7, ls=":")
    ax.annotate("", xy=(24500, 50), xytext=(300, 50), arrowprops=dict(arrowstyle="<->", color=K, lw=0.8))
    ax.text(12400, 53, r"지연 $\Delta t$", ha="center", fontsize=8.5)
    ax.text(700, 88, r"$t_{fit}$", fontsize=8.5)
    ax.text(24800, 10, r"$t_{gen}$", fontsize=8.5)
    ax.set_xlim(0, 30000); ax.set_ylim(-3, 103); comma(ax)
    ax.set_xlabel("학습 step"); ax.set_ylabel("정확도 (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.05, 0.12))
    save(fig, out, "그림01")


# ---------------------------------------------------------------- 그림 2
def fig02(out):
    """(가) 과제의 구조, (나) 표 4의 네 가지 배선 구조의 참조 그래프."""
    fig = plt.figure(figsize=(W1, 4.0))
    ax = fig.add_axes([0, 0.40, 1, 0.60])
    ax.set_xlim(0, 10.6); ax.set_ylim(0.2, 4.2); ax.axis("off")
    ax.text(0.0, 4.15, "(가)", fontsize=10, fontweight="bold", va="top")
    xs = np.arange(5) * 1.15 + 1.2
    inp, sym = [3, 17, 8, 42, 25], [f"$x_{i}$" for i in range(1, 6)]
    box = dict(boxstyle="square,pad=0.25", fc="white", ec=K, lw=0.8)
    ax.text(0.05, 3.4, "입력", va="center", fontsize=9, fontweight="bold")
    ax.text(0.05, 0.7, "출력", va="center", fontsize=9, fontweight="bold")
    for i, x in enumerate(xs):
        ax.text(x, 3.4, f"{inp[i]}", ha="center", va="center", fontsize=10, bbox=box)
        ax.text(x, 3.95, sym[i], ha="center", va="center", fontsize=8, color=G1)
        j = (i + 1) % 5
        ax.text(x, 0.7, f"T({inp[i]}, {inp[j]})", ha="center", va="center", fontsize=8.5, bbox=box)
        ax.annotate("", xy=(x, 1.0), xytext=(x, 3.1), arrowprops=dict(arrowstyle="-|>", color=K, lw=0.7, shrinkA=0, shrinkB=0))
        if i < 4:   # 참조 위치(다음 칸)에서 오는 선
            ax.annotate("", xy=(x + 0.12, 1.0), xytext=(xs[j] - 0.12, 3.1),
                        arrowprops=dict(arrowstyle="-|>", color=G2, lw=0.7, ls="--", shrinkA=0, shrinkB=0))
    # 연산 표
    x0, y0, s, n = 7.3, 0.55, 0.26, 10
    for a in range(n):
        for b in range(n):
            fc = G3 if (a, b) in ((0, 3), (3, 1), (1, 6), (6, 8), (8, 0)) else "white"
            ax.add_patch(Rectangle((x0 + b * s, y0 + (n - 1 - a) * s), s, s, fc=fc, ec=G2, lw=0.4))
    ax.text(x0 + n * s / 2, y0 + n * s + 0.25, "연산 표 T(a, b)", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(x0 + n * s / 2, y0 - 0.25, "b", ha="center", fontsize=8.5, style="italic")
    ax.text(x0 - 0.25, y0 + n * s / 2, "a", va="center", fontsize=8.5, style="italic")
    ax.text(x0 + n * s + 0.15, y0 + n * s / 2, "한 예제가\n표의 칸 5개를\n제공", va="center", fontsize=7.5, color=G1)

    # (나) 참조 그래프: 위치 1~5를 오각형 꼭짓점에 둔다
    bx = fig.add_axes([0, 0, 1, 0.42])
    bx.set_xlim(0, 10.6); bx.set_ylim(-1.12, 1.45); bx.axis("off"); bx.set_aspect("equal")
    bx.text(0.0, 1.43, "(나)", fontsize=10, fontweight="bold", va="top")
    ang = np.pi / 2 - 2 * np.pi * np.arange(5) / 5
    r = 0.68

    def graph(cx, edges, doubled, title, note):
        P = np.c_[cx + r * np.cos(ang), 0.55 + r * np.sin(ang)]
        for a, b in edges:
            pa, pb = P[a - 1], P[b - 1]
            if (a, b) in doubled:   # 같은 선이 두 번 그어짐
                d = pb - pa; nrm = np.array([-d[1], d[0]]) / np.hypot(*d) * 0.045
                for sgn in (1, -1):
                    bx.plot([pa[0] + sgn * nrm[0], pb[0] + sgn * nrm[0]], [pa[1] + sgn * nrm[1], pb[1] + sgn * nrm[1]], color=K, lw=1.0)
            else:
                bx.plot([pa[0], pb[0]], [pa[1], pb[1]], color=K, lw=1.0)
        for k, (x, y) in enumerate(P, 1):
            bx.add_patch(plt.Circle((x, y), 0.13, fc="white", ec=K, lw=0.9, zorder=3))
            bx.text(x, y, str(k), ha="center", va="center", fontsize=8, zorder=4)
        bx.text(cx, -0.38, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        bx.text(cx, -0.80, note, ha="center", va="center", fontsize=7.5, color=G1, linespacing=1.35)

    # 표 4의 네 가지 배선 구조. 2-순환이 있는 두 그래프는 선이 엇갈리지 않는 실제 배선(w2_m4, w2_m15)을 예로 그렸다.
    # a+b에서 서로 다른 칸 수 = 5 − (2-순환 수)
    graph(1.35, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)], set(), "5-순환 배선",
          "기본 규칙, 연결\n서로 다른 칸 5개")
    graph(3.97, [(1, 5), (1, 4), (2, 4), (3, 4)], {(1, 5)}, "2-순환 포함 배선",
          "1–5 선이 두 번, 연결\n서로 다른 칸 4개")
    graph(6.59, [(1, 2), (2, 3), (3, 1), (4, 5)], {(4, 5)}, "3+2 순환 배선",
          "4–5 선이 두 번, 비연결\n서로 다른 칸 4개")
    graph(9.21, [(1, 5), (2, 5), (3, 4)], {(1, 5), (3, 4)}, "2-순환 2개 배선",
          "1–5, 3–4 선이 두 번, 비연결\n서로 다른 칸 3개")
    save(fig, out, "그림02")


# ---------------------------------------------------------------- 그림 3
BOUND = [("a+b", 276, 211), ("a+b", 496, 329), ("a+b", 1275, 623), ("a+b", 1891, 880), ("a+b", 2556, 1180),
         ("a+b", 5151, 2063), ("a+3b", 961, 533), ("a+3b", 1681, 827), ("a+3b", 2500, 993)]  # 로그 보간 (10월 25일, 표 3)


def fig03(out):
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    Cs = np.array([b[1] for b in BOUND]); Ns = np.array([b[2] for b in BOUND])
    xs = np.logspace(np.log10(220), np.log10(6500), 50)
    ax.plot(xs, 2.82 * xs ** 0.763, color=K, lw=1, label=r"$N_0 = 2.82 \times C^{0.763}$")
    for i, op in enumerate(("a+b", "a+3b")):
        sel = [b for b in BOUND if b[0] == op]
        ax.plot([b[1] for b in sel], [b[2] for b in sel], ls="none", **{k: v for k, v in sty(i).items() if k != "linestyle"}, label=op)
    ax.set_xscale("log"); ax.set_yscale("log")
    log_ticks(ax, "x", [300, 1000, 3000]); log_ticks(ax, "y", [200, 500, 1000, 2000])
    ax.set_xlabel("연산 표의 칸 수 C"); ax.set_ylabel(r"일반화 경계 $N_0$ (학습 예제 수)")
    ax.legend(loc="upper left")
    save(fig, out, "그림03")


# ---------------------------------------------------------------- 그림 5 (10월 25일 전 그림 4)
def fig04(res, out):
    from src.rules import Rule, analyze, ref_graph
    df = load_runs([Path(res, "E06_boundary_results/runs_boundary")])
    df = df[df.rule.str.startswith("w")].copy()
    specs = df[["rule", "spec"]].drop_duplicates().values
    eq = {r: analyze(sp, 50)["eq_size"] for r, sp in specs}
    pairs = {r: ref_graph(Rule(sp, 50))["edges_unordered"] for r, sp in specs}
    df["eq"], df["pairs"] = df.rule.map(eq), df.rule.map(pairs)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.7), sharey=True)
    for a, key, lab, order in ((ax[0], "eq", "등변군 크기 {}", [6, 5, 2, 1]), (ax[1], "pairs", "서로 다른 칸 {}개", [5, 4, 3])):
        t = agg(df, [key, "train"])
        for i, v in enumerate(order):
            s = t[t[key] == v].sort_values("train")
            a.plot(s.train, s.delay, label=lab.format(v), **sty(i))
        a.set_yscale("log"); a.set_xlabel("학습 예제 수 N"); comma(a)
        a.set_xticks([768, 896, 1024, 1152, 1280]); a.tick_params(axis="x", labelsize=7.5); a.legend(loc="lower left")
    log_ticks(ax[0], "y", [100, 1000, 10000])
    ax[0].set_ylabel("지연 중앙값 (step)")
    panel(ax[0], "(가)"); panel(ax[1], "(나)")
    save(fig, out / "제외", "배선12종")


# ---------------------------------------------------------------- 그림 6 (10월 25일 전 그림 5)
def fig05(res, out):
    df = pd.concat([load_runs([Path(res, "E06_boundary_results/runs_boundary")]),
                    load_runs([Path(res, "E07_commute_results/runs_commute")])])
    df = df[df.rule.str.match(r"w\d")].copy()
    df["op"] = np.where(df.rule.str.endswith("_n3"), "a+3b", "a+b")
    df["fam"] = df.rule.str.extract(r"^(w\d)")[0]
    keep = df[df.op == "a+3b"].rule.str.replace("_n3", "").unique()
    df = df[df.rule.str.replace("_n3", "").isin(keep)]
    fams = ["w5", "w6", "w2", "w1"]
    names = ["5-순환\n(등변군 5)", "3+2 순환\n(등변군 6)", "2-순환 포함\n(등변군 2)", "대칭 없음\n(등변군 1)"]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.7), sharey=True)
    for a, N, p in zip(ax, (1152, 1280), ("(가)", "(나)")):
        t = agg(df[df.train == N], ["fam", "op"])
        x = np.arange(4)
        for j, (op, fc, hatch) in enumerate((("a+b", K, None), ("a+3b", "white", None))):
            s = t[t.op == op].set_index("fam").reindex(fams)
            a.bar(x + (j - 0.5) * 0.36, s.delay, width=0.34, color=fc, edgecolor=K, linewidth=0.8, label=op)
        a.set_xticks(x, names, fontsize=7.5); a.set_yscale("log")
        a.text(0.02, 0.97, f"N = {N:,}", transform=a.transAxes, va="top", fontsize=8.5)
        panel(a, p)
    log_ticks(ax[0], "y", [100, 1000, 10000])
    ax[0].set_ylabel("지연 중앙값 (step)")
    ax[1].legend(loc="upper right")
    save(fig, out / "제외", "연산별_배선")


# ---------------------------------------------------------------- 그림 7 (10월 25일 전 그림 6)
def fig06(res, out):
    from analyze_obs import PRED_A, PRED_B, from_runs
    def tab(pattern):
        dirs = [p for r in glob.glob(str(Path(res, pattern))) for p in glob.glob(f"{r}/*") if Path(p, "summary.json").exists()]
        return from_runs(dirs)
    old = pd.concat([tab("E06_boundary_results/runs_boundary"), tab("E07_commute_results/runs_commute")])
    new = tab("E12_prediction_results/runs_e12_*")
    groups = [("a+b", 5, "a+b, 서로 다른 칸 5개"), ("a+b", 4, "a+b, 4개"), ("a+b", 3, "a+b, 3개"), ("a+3b", 5, "a+3b, 5개")]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.8), gridspec_kw=dict(wspace=0.38))
    for i, (op, p, lab) in enumerate(groups):
        c, ls, mk, _ = STY[i]
        for t, fill in ((old, True), (new, False)):
            s = t[(t.op == op) & (t.pairs == p)].groupby("obs_per_cell", as_index=False).agg(
                runs=("runs", "sum"), grokked=("grokked", "sum"), delay=("delay_median", "median"))
            if s.empty:
                continue
            s["rate"] = s.grokked / s.runs
            kw = dict(color=c, marker=mk, ls="none", markerfacecolor=c if fill else "white", markeredgecolor=c, markeredgewidth=1)
            ax[0].plot(s.obs_per_cell, s.rate, **kw)
            g = s[s.rate >= 0.5]
            ax[1].plot(g.obs_per_cell, g.delay, **kw)
        ax[1].plot([], [], color=c, marker=mk, ls="none", label=lab)
    xs = np.linspace(1.8, 5.1, 50)
    ax[1].plot(xs, 10 ** (PRED_A - PRED_B * xs), color=K, lw=1, label="회귀식(사전 예측선)")
    ax[1].plot([], [], marker="o", ls="none", color=K, label="회귀식 도출 조건")
    ax[1].plot([], [], marker="o", ls="none", color=K, markerfacecolor="white", label="사전 예측 검증 조건")
    ax[0].set_ylabel("일반화 비율"); ax[1].set_ylabel("지연 중앙값 (step)")
    ax[1].set_yscale("log"); log_ticks(ax[1], "y", [100, 1000, 10000])
    for a in ax:
        a.set_xlabel("칸당 관찰 수")
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    panel(ax[0], "(가)"); panel(ax[1], "(나)")
    save(fig, out, "그림05")


# ---------------------------------------------------------------- 그림 8 (10월 25일 전 그림 7)
COV = {  # N = 768. (관찰 비율, 지연 중앙값, 일반화 수/10)
    "서로 다른 칸 5개 배선": [(0.944, 6900, 10), (0.923, 5300, 10), (0.882, 3700, 10), (0.862, 3450, 10),
                         (0.842, 2950, 10), (0.822, 2700, 10), (0.802, 2600, 10), (0.782, 2500, 10)],
    "서로 다른 칸 4개 배선": [(0.898, 21000, 5), (0.884, 15500, 5), (0.851, 9700, 9), (0.831, 7000, 10),
                         (0.811, 5350, 10), (0.791, 4800, 10), (0.772, 4050, 10), (0.742, 3450, 10)]}


def fig07(out):
    fig, ax = plt.subplots(figsize=(W2, 2.8))
    for i, (lab, pts) in enumerate(COV.items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=lab, **sty(i))
        for x, y, g in pts:
            if g < 10:
                ax.annotate(f"{g}/10", (x, y), textcoords="offset points", xytext=(7, 2), ha="left", fontsize=7)
    ax.set_yscale("log"); log_ticks(ax, "y", [2500, 5000, 10000, 20000])
    ax.set_xlabel("관찰 비율"); ax.set_ylabel("지연 중앙값 (step)")
    ax.legend(loc="upper left")
    save(fig, out, "그림06")


# ---------------------------------------------------------------- 그림 4 (10월 25일 전 그림 8)
N_ = np.array([31, 41, 50, 61, 71]); C5 = N_ * (N_ + 1) / 2
MUL_N = np.array([31, 41, 47, 53, 61, 71])
WD_N = np.array([31, 50, 71]); CW = WD_N * (WD_N + 1) / 2


def fig08(out):
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.9), gridspec_kw=dict(wspace=0.38))
    xs = np.logspace(np.log10(400), np.log10(2900), 50)
    ax[0].plot(xs, 2.82 * xs ** 0.763, color=G3, lw=2.5, zorder=0)
    series = [("기본 모델", C5, [298, 554, 618, 797, 1047]), ("확장 모델", C5, [277, 490, 574, 774, 1003]),
              ("GPT형 디코더", C5, [984, 1406, 1855, 2398, 3384]), ("곱셈 a×b (n 소수)", MUL_N * (MUL_N + 1) / 2, [240, 397, 400, 545, 752, 975])]
    for i, (lab, x, y) in enumerate(series):
        ax[0].plot(x, y, label=lab, **sty(i))
    ax[0].plot(1275, 325, marker="x", ls="none", color=K, ms=6, label="곱셈 n = 50 (합성수)")
    # (나) Weight Decay에 따른 경계 비율(같은 n의 wd 1 대비). E32 기본·E38·E40, docs/analysis/e40_wd_law.csv
    WD = [0.3, 1, 2, 3, 5, 7, 10, 15]
    RAT = {31: [0.931, 1, 1.024, 1.024, 0.965, 0.931, 1.388, 2.171],
           50: [0.964, 1, 0.964, 1.025, 0.964, 0.964, 1.615, 2.766],
           71: [1.202, 1, 1.024, 1.129, 0.931, 1.202, 1.739, 2.775]}
    for i, (n, y) in enumerate(RAT.items()):
        kw = sty(i)
        ax[1].plot(WD[:-1], y[:-1], label=f"n = {n}", **kw)
        x15 = 15 * (1 + 0.05 * (i - 1))
        ax[1].plot([10, x15], y[-2:], color=kw["color"], ls=":", marker="none")   # wd 15: 일시적 일반화의 경계
        ax[1].plot(x15, y[-1], ls="none", marker=kw["marker"], color=kw["color"], markerfacecolor="white", markeredgewidth=1)
    ax[1].plot(0.3, RAT[71][0], ls="none", marker=sty(2)["marker"], color=sty(2)["color"], markerfacecolor="white", markeredgewidth=1)
    ax[1].axhline(1, color=G3, lw=2.5, zorder=0)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    log_ticks(ax[0], "x", [500, 1000, 2000]); log_ticks(ax[0], "y", [200, 500, 1000, 2000, 5000])
    ax[0].set_xlabel("연산 표의 칸 수 C"); panel(ax[0], "(가)")
    ax[1].set_xscale("log"); ax[1].set_xticks(WD); ax[1].set_xticklabels(["0.3", "1", "2", "3", "5", "7", "10", "15"])
    ax[1].minorticks_off(); ax[1].set_ylim(0.8, 3.0); ax[1].set_xlabel("Weight Decay")
    ax[1].set_ylabel("경계 비율 (Weight Decay 1 대비)"); panel(ax[1], "(나)")
    ax[1].yaxis.set_tick_params(labelleft=True)
    ax[0].set_ylim(200, 12000)
    ax[0].legend(loc="upper left", ncol=2, fontsize=7, columnspacing=0.8, handlelength=2.2)
    ax[1].legend(loc="upper left", fontsize=7, handlelength=2.2)
    ax[0].set_ylabel(r"일반화 경계 $N_0$")
    save(fig, out, "그림04")


# ---------------------------------------------------------------- 그림 9
def fig09(res, out, example="w6_p1_m0_c1024_wd1_s1"):
    d = pd.read_csv(Path(res, "E11d_dense_internal/e11d_internal/internal.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.8), gridspec_kw=dict(width_ratios=[1.4, 1], wspace=0.42))
    r = d[(d.run == example) & (d.step >= 50)].sort_values("step")
    for k, ls, col, lab in (("train_exact", "--", K, "학습 정확도"), ("acc_seen", "-", K, "관찰 칸 정확도"),
                            ("acc_unseen", "-", G2, "미관찰 칸 정확도")):
        ax[0].plot(r.step, r[k], ls=ls, color=col, label=lab)
    ax[0].set_xscale("log"); log_ticks(ax[0], "x", [100, 1000, 10000])
    ax[0].set_xlabel("학습 step"); ax[0].set_ylabel("정확도"); ax[0].legend(loc="center left", bbox_to_anchor=(0.27, 0.36), fontsize=7.5)
    rows = []
    for run, g in d.groupby("run"):
        s50, u50 = g[g.acc_seen >= 0.5].step.min(), g[g.acc_unseen >= 0.5].step.min()
        if not np.isnan(s50) and not np.isnan(u50):
            rows.append((s50, u50, "_n3" in run))
    s = pd.DataFrame(rows, columns=["s50", "u50", "n3"])
    for flag, fill, lab in ((False, K, "a+b"), (True, "white", "a+3b")):
        q = s[s.n3 == flag]
        ax[1].plot(q.s50, q.u50, "o", ls="none", color=K, markerfacecolor=fill, label=lab)
    lim = [200, s[["s50", "u50"]].max().max() * 1.3]
    ax[1].plot(lim, lim, color=G2, lw=0.8, ls="--", label="y = x")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    log_ticks(ax[1], "x", [300, 1000, 3000, 10000]); log_ticks(ax[1], "y", [1000, 3000, 10000])
    ax[1].set_xlabel("관찰 칸 정확도 50% 도달 step"); ax[1].set_ylabel("미관찰 칸 정확도 50% 도달 step")
    ax[1].legend(loc="upper left", fontsize=7.5)
    panel(ax[0], "(가)"); panel(ax[1], "(나)")
    save(fig, out, "그림07")


# ---------------------------------------------------------------- 그림 10
def fig10(res, out, wd=1, frac="0.1"):
    base = Path(res, "E30_noise")
    def curves(mode, N, col):
        fs = glob.glob(str(base / f"runs_e30_{mode}{frac}" / f"w5_next_m0_c{N}_wd{wd}_s*" / "noise.csv"))
        return pd.concat([pd.read_csv(f) for f in fs]).groupby("step")[col].mean()
    def clean(N):
        fs = glob.glob(str(base / "runs_e30_none" / f"w5_next_m0_c{N}_wd{wd}_s*" / "metrics.csv"))
        return pd.concat([pd.read_csv(f) for f in fs]).groupby("step").test_exact.mean()
    fig, ax = plt.subplots(2, 3, figsize=(W1, 4.2), sharex=True, sharey=True)
    cols = [("mem", "오답 유지율"), ("test_clean_rows_exact", "평가 정확도"), ("test_spread", "전파율")]
    for r, N in enumerate((768, 2048)):
        for c, (col, ylab) in enumerate(cols):
            a = ax[r, c]
            m = curves("scatter", N, col); a.plot(m.index, m.values, color=K, ls="-", label="흩어진 오답")
            m = curves("cell", N, col); a.plot(m.index, m.values, color=K, ls="--", label="같은 칸 오답")
            if col == "test_clean_rows_exact":
                m = clean(N); a.plot(m.index, m.values, color=G2, ls=":", label="오답 없음")
            a.set_xscale("symlog", linthresh=1000); a.set_ylim(-0.03, 1.05)
            a.set_xticks([0, 1000, 10000, 100000]); a.set_xticklabels(["0", "$10^3$", "$10^4$", "$10^5$"])
            if r == 0:
                a.set_title(ylab, fontsize=9)
            if r == 1:
                a.set_xlabel("학습 step")
        ax[r, 0].set_ylabel(f"N = {N:,}")
    h1, l1 = ax[0, 1].get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.17)
    save(fig, out, "그림08")


# ---------------------------------------------------------------- 그림 11
def fig11(out):
    """(가) 오답 유지율 vs wd (E30·E34·E36·E37·E41), (나) 두 수 과제의 평가 정확도 vs wd (E34·E36·E41). 값은 docs/analysis/e41_m2_wd.csv."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.8), gridspec_kw=dict(wspace=0.38))
    # 수열 과제: E40 N 2,048, 6만 step (wd 1·3·5·7·10). 두 수 과제: E34·E36·E41, 20만 step.
    # wd 15는 규칙도 학습하지 못한 점이라 빈 기호로 따로 둔다(수열 1.3·4.4%, 두 수 4.0%).
    series = [("수열 과제, 흩어진 오답", [1, 3, 5, 7, 10], [100, 100, 8.4, 0, 0], 1.3),
              ("수열 과제, 같은 칸 오답", [1, 3, 5, 7, 10], [100, 100, 100, 15, 4.4], 4.4),
              ("두 수 과제", [1, 2, 3, 4, 5, 7, 10], [100, 100, 100, 100, 100, 21, 0.5], 4.0)]
    for i, (lab, x, y, y15) in enumerate(series):
        ax[0].plot(x, y, label=lab, **sty(i))
        kw = sty(i); kw.update(markerfacecolor="white", linestyle="none")
        ax[0].plot([15 * (1 + 0.06 * (i - 1))], [y15], **kw)
    ax[0].annotate("규칙 상실", (15, 6), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax[0].set_ylabel("오답 유지율 (%)"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7)
    wd = [1, 2, 3, 4, 5, 7, 10, 15]
    ax[1].plot(wd, [100, 100, 100, 100, 100, 100, 100, 84], label="오답 없음", **sty(1))
    # 원자료 metrics.csv 끝 평가 정확도 평균 (10월 25일 재계산)
    ax[1].plot(wd, [97.4, 97.4, 81.6, 36.8, 61.0, 97.4, 97.2, 28.0], label="오답 예제 10%", **sty(2))
    ax[1].set_ylabel("평가 정확도 (%)"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7)
    for a, p in zip(ax, ("(가)", "(나)")):
        a.set_xscale("log"); a.set_xticks(wd); a.set_xticklabels([str(w) for w in wd]); a.minorticks_off()
        a.set_xlim(0.85, 18); a.set_ylim(-5, 108); a.set_xlabel("Weight Decay"); panel(a, p)
    save(fig, out, "그림09")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--out", default="docs/figures/report")
    ap.add_argument("--only", nargs="+", type=int)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setup()
    jobs = {1: lambda: fig01(a.res, out), 2: lambda: fig02(out), 3: lambda: fig03(out), 4: lambda: fig04(a.res, out),
            5: lambda: fig05(a.res, out), 6: lambda: fig06(a.res, out), 7: lambda: fig07(out), 8: lambda: fig08(out),
            9: lambda: fig09(a.res, out), 10: lambda: fig10(a.res, out), 11: lambda: fig11(out)}
    for k in (a.only or jobs):
        jobs[k]()
    print("저장:", out)


if __name__ == "__main__":
    main()
