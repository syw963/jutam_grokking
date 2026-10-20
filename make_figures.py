"""보고서용 그림 5장. 결과 tgz를 푼 폴더들을 넣으면 docs/figures/에 PNG를 만든다.

    python make_figures.py --e6 runs_boundary --e7 runs_commute --e8 runs_e8 --e8-data data_e8 \
        --e12 "runs_e12_*" --e12b "runs_e12b_*" --e12b-data data_e12 --e11d e11d_internal/internal.csv --out docs/figures

F1 대칭 vs 정보량 (E6)        F2 교환법칙 실험 (E6 vs E7)     F3 칸당 관찰 수 한 곡선 (E6·E7·E12)
F4 coverage 직접 조작 (E8·E12B) F5 학습 단계 (E11 촘촘한 체크포인트)
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

from analyze_obs import PRED_A, PRED_B, from_runs
from src.rules import Rule, analyze

# 참조 팔레트 (dataviz 기본, 라이트): 순서 고정
C_ = C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def setup_style():
    for name in ("Noto Sans CJK KR", "NanumGothic", "AppleGothic", "Malgun Gothic"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
        "legend.fontsize": 8.5, "savefig.dpi": 160, "savefig.bbox": "tight"})


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
            rows.append(dict(run=d.name, rule=c["rule"], spec=c["rule_spec"], data=Path(c["data"]).name,
                             train=c["data_meta"]["train_sets"], seed=c["seed"], grokked=tg is not None,
                             delay=(tg - tf) if tg is not None and tf is not None else np.nan))
    return pd.DataFrame(rows)


def agg(df, keys):
    return df.groupby(keys, as_index=False).agg(runs=("grokked", "size"), grokked=("grokked", "sum"),
                                                 delay=("delay", "median")).assign(rate=lambda t: t.grokked / t.runs)


def fig1(e6, out):
    """E6: 등변군 크기로 묶기 vs 예제당 서로 다른 쌍 수로 묶기."""
    df = load_runs([e6])
    df = df[df.rule.str.startswith("w")]
    info = {r: analyze(sp, 50) for r, sp in df[["rule", "spec"]].drop_duplicates().values}
    df["eq"] = df.rule.map(lambda r: info[r]["eq_size"])
    from src.rules import ref_graph
    df["pairs"] = df.rule.map(lambda r: ref_graph(Rule(df[df.rule == r].spec.iloc[0], 50))["edges_unordered"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, key, lab, order in ((ax[0], "eq", "등변군 크기 {}", [6, 5, 2, 1]), (ax[1], "pairs", "예제당 서로 다른 쌍 {}개", [5, 4, 3])):
        t = agg(df, [key, "train"])
        for i, v in enumerate(order):
            s = t[t[key] == v].sort_values("train")
            a.plot(s.train, s.delay, "o-", color=C[i], label=lab.format(v))
        a.set_yscale("log"); a.set_xlabel("train 예제 수")
        a.legend()
    ax[0].set_ylabel("지연 중앙값 (step, 로그)")
    ax[0].set_title("① 대칭(등변군 크기)으로 묶으면: 순서가 뒤섞임")
    ax[1].set_title("② 예제당 서로 다른 쌍 수로 묶으면: 깔끔히 정렬")
    fig.suptitle("E6 · 같은 덧셈(a+b)을 배우는 배선 12개, seed 5", color=INK2, fontsize=9, y=0.995)
    fig.savefig(out / "F1_symmetry_vs_pairs.png"); plt.close(fig)


def fig2(e6, e7, out):
    """E6(a+b) vs E7(a+3b): 배선 가족별 지연."""
    df = pd.concat([load_runs([e6]), load_runs([e7])])
    df = df[df.rule.str.match(r"w\d")]
    df["op"] = np.where(df.rule.str.endswith("_n3"), "a+3b", "a+b")
    df["fam"] = df.rule.str.extract(r"^(w\d)")[0]
    df = df[df.rule.str.replace("_n3", "").isin(df[df.op == "a+3b"].rule.str.replace("_n3", "").unique())]
    names = {"w5": "5-순환\n(등변군 5)", "w6": "3+2 순환\n(등변군 6)", "w2": "2-순환 포함\n(등변군 2)", "w1": "대칭 없음\n(등변군 1)"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, N in zip(ax, (1152, 1280)):
        t = agg(df[df.train == N], ["fam", "op"])
        x = np.arange(4)
        for j, (op, col) in enumerate((("a+b", C[0]), ("a+3b", C[1]))):
            s = t[t.op == op].set_index("fam").reindex(["w5", "w6", "w2", "w1"])
            a.bar(x + (j - 0.5) * 0.38, s.delay, width=0.36, color=col, label=f"T = {op}", edgecolor=SURF, linewidth=2)
            for xi, d in zip(x, s.delay):
                a.text(xi + (j - 0.5) * 0.38, d * 1.08, f"{d:,.0f}", ha="center", fontsize=7.5, color=INK2)
        a.set_xticks(x, [names[f] for f in ["w5", "w6", "w2", "w1"]], fontsize=8.5)
        a.set_yscale("log"); a.set_title(f"train {N:,}개"); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("지연 중앙값 (step, 로그)")
    for a in ax:
        a.set_ylim(top=a.get_ylim()[1] * 1.6)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.04))
    fig.suptitle("E7 · 교환법칙을 없애면(a+3b) 배선 간 차이가 사라진다 (등변군은 그대로)", color=INK, fontsize=11,
                 fontweight="bold", x=0.02, ha="left", y=1.03)
    fig.savefig(out / "F2_commutativity.png"); plt.close(fig)


def fig3(e6, e7, e12_roots, out):
    """칸당 관찰 수 하나로 E6·E7·E12를 한 곡선에."""
    def tab(roots):
        dirs = [p for r in roots for p in glob.glob(f"{r}/*") if Path(p, "summary.json").exists()]
        return from_runs(dirs)
    old, new = tab([e6, e7]), tab(e12_roots)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    groups = [("a+b", 5, "a+b · 예제당 쌍 5"), ("a+b", 4, "a+b · 예제당 쌍 4"), ("a+b", 3, "a+b · 예제당 쌍 3"), ("a+3b", 5, "a+3b · 예제당 쌍 5")]
    for i, (op, p, lab) in enumerate(groups):
        for t, mk, fill, suffix in ((old, "o", True, " (E6·E7)"), (new, "s", False, " (E12)")):
            s = t[(t.op == op) & (t.pairs == p)].groupby("obs_per_cell", as_index=False).agg(
                runs=("runs", "sum"), grokked=("grokked", "sum"), delay=("delay_median", "median"))
            if s.empty:
                continue
            s["rate"] = s.grokked / s.runs
            kw = dict(color=C[i], marker=mk, linestyle="-" if fill else ":", markerfacecolor=C[i] if fill else SURF,
                      markeredgewidth=1.6, label=lab + suffix)
            ax[0].plot(s.obs_per_cell, s.rate, **kw)
            g = s[s.rate >= 0.5]
            ax[1].plot(g.obs_per_cell, g.delay, **kw)
    xs = np.linspace(1.8, 5.1, 50)
    ax[1].plot(xs, 10 ** (PRED_A - PRED_B * xs), "--", color=INK2, lw=1.2, label="E12 사전 예측선 (E6·E7 적합)")
    ax[0].set_ylabel("일반화한 run 비율"); ax[1].set_ylabel("지연 중앙값 (step, 로그)"); ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("칸당 평균 관찰 수 = train 수 × 예제당 서로 다른 쌍 / 연산 표 칸 수")
    ax[0].set_title("일반화 경계: 약 2.2~2.5회"); ax[1].set_title("지연: 관찰 수 0.4회마다 약 절반")
    ax[1].legend(fontsize=7, loc="upper right"); ax[0].legend(fontsize=7, loc="lower right")
    fig.savefig(out / "F3_obs_per_cell.png"); plt.close(fig)


def fig4(e8, e8_data, e12b, e12b_data, out):
    """coverage 직접 조작: 같은 train 768개에서 coverage를 낮출수록(같은 칸 반복) 빨라진다."""
    df = pd.concat([load_runs([e8]), load_runs(e12b)])
    cov = {}
    for d in list(glob.glob(f"{e8_data}/*")) + list(glob.glob(f"{e12b_data}/*")):
        m = json.loads(Path(d, "meta.json").read_text())
        cov[Path(d).name] = m["curated"]["coverage"]
    df = df[df.train == 768]
    df["coverage"] = df.data.map(cov)
    t = agg(df, ["rule", "coverage"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (r, lab) in enumerate((("w5_next", "w5_next (5-순환)"), ("w1_m0", "w1_m0 (대칭 없음)"))):
        s = t[t.rule == r].sort_values("coverage")
        ax.plot(s.coverage, s.delay, "o-", color=C[i], label=lab)
        for _, row in s.iterrows():
            ax.annotate(f"{int(row.grokked)}/{int(row.runs)}", (row.coverage, row.delay), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=INK2)
    ax.set_yscale("log"); ax.set_xlabel("연산 표 coverage (train에 나온 칸 비율)"); ax.set_ylabel("지연 중앙값 (step, 로그)")
    ax.set_title("E8 · train 768개 고정: coverage를 낮출수록(같은 칸 반복) 빠르다")
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, "숫자: 일반화한 run / 전체", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    fig.savefig(out / "F4_coverage.png"); plt.close(fig)


def fig5(internal_csv, out, example="w6_p1_m0_c1024_wd1_s1"):
    d = pd.read_csv(internal_csv)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    r = d[d.run == example].sort_values("step")
    r = r[r.step >= 50]
    for k, col, lab in (("train_exact", C[3], "train 완전 일치"), ("acc_seen", C[0], "test · train에 나온 칸"),
                        ("acc_unseen", C[1], "test · train에 안 나온 칸")):
        ax[0].plot(r.step, r[k], "-", color=col, label=lab)
    ax[0].plot(r.step, (r.emb_top5 - 0.2) / 0.8, ":", color=C[2], label="임베딩 푸리에 집중도 (0=무작위)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("step (로그)"); ax[0].set_ylabel("정확도")
    ax[0].set_title("한 run의 학습 과정 (w6_p1, train 1,024개)"); ax[0].legend(fontsize=7.5, loc="center left")
    rows = []
    for run, g in d.groupby("run"):
        g = g.sort_values("step")
        s50, u50 = g[g.acc_seen >= 0.5].step.min(), g[g.acc_unseen >= 0.5].step.min()
        if not np.isnan(s50) and not np.isnan(u50):
            rows.append((s50, u50, "_n3" in run))
    s = pd.DataFrame(rows, columns=["s50", "u50", "n3"])
    for flag, col, lab in ((False, C[0], "a+b"), (True, C[1], "a+3b")):
        q = s[s.n3 == flag]
        ax[1].scatter(q.s50, q.u50, s=40, color=col, edgecolor=SURF, linewidth=1.5, label=lab, zorder=3)
    lim = [150, s[["s50", "u50"]].max().max() * 1.4]
    ax[1].plot(lim, lim, "--", color=INK2, lw=1, label="같은 시점 (y = x)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("나온 칸 50% 도달 step"); ax[1].set_ylabel("안 나온 칸 50% 도달 step")
    ax[1].set_title(f"{len(s)}개 run 모두 대각선 위: 나온 칸이 먼저"); ax[1].legend(fontsize=7.5)
    fig.savefig(out / "F5_stages.png"); plt.close(fig)



def fig6(e16_roots, e12_roots, e6_root, out):
    """수열 길이 m: 같은 규칙(이웃 덧셈)을 m=2·3·5로. 관찰 수 기준과 예제 수 기준."""
    df = load_runs(list(e16_roots) + list(e12_roots) + [e6_root])
    df = df[df.rule.isin(["p_next", "w5_next"])].copy()
    ms = {}
    for r in list(e16_roots) + list(e12_roots) + [e6_root]:
        for d in glob.glob(f"{r}/*/config.json"):
            c = json.loads(Path(d).read_text())
            ms[Path(d).parent.name] = c["m"]
    df["m"] = df.run.map(ms)
    df["pairs"] = np.where(df.m == 2, 1, df.m)
    df["obs"] = df.train * df.pairs / 1275
    t = agg(df, ["m", "train"]).merge(df[["m", "train", "obs"]].drop_duplicates(), on=["m", "train"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, m in enumerate([2, 3, 5]):
        s = t[t.m == m].sort_values("train")
        lab = f"m={m}" + (" (두 수 → 합, 기존 grokking 과제)" if m == 2 else "")
        ax[0].plot(s.obs, s.rate, "o-", color=C[i], label=lab)
        ax[1].plot(s.train, s.rate, "o-", color=C[i], label=lab)
    ax[0].set_xlabel("칸당 평균 관찰 수"); ax[1].set_xlabel("train 예제 수")
    ax[0].set_ylabel("일반화한 run 비율")
    ax[0].set_title("관찰 수 기준: 문턱이 m마다 다르다"); ax[1].set_title("예제 수 기준: 약 600~1,000개 사이로 모인다")
    ax[0].legend(fontsize=8)
    fig.savefig(out / "F6_seq_length.png"); plt.close(fig)


def fig7(e17_internal_csv, out, N=1100, seed=1):
    """무작위 연산표 vs a+3b: train에 나온 칸 / 안 나온 칸 정확도."""
    d = pd.read_csv(e17_internal_csv)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, kind, title in ((ax[0], "table", "규칙 없는 무작위 표"), (ax[1], "n3", "a+3b (같은 2,500칸)")):
        r = d[(d.train == N) & (d.seed == seed) & (d.rule.str.contains("table") == (kind == "table"))].sort_values("step")
        r = r[r.step >= 50]
        a.plot(r.step, r.train_exact, "-", color=C[3], label="train 완전 일치")
        a.plot(r.step, r.acc_seen, "-", color=C[0], label="test · train에 나온 칸")
        a.plot(r.step, r.acc_unseen, "-", color=C[1], label="test · train에 안 나온 칸")
        a.set_xscale("log"); a.set_xlabel("step (로그)"); a.set_title(f"{title}, train {N:,}개")
    ax[0].set_ylabel("정확도"); ax[0].legend(fontsize=8, loc="center right")
    fig.savefig(out / "F7_random_table.png"); plt.close(fig)


# 일반화 50% 경계(보간)와 표 칸 수 C. 출처 실험의 일반화 비율 표에서 선형 보간 (docs/결과_정리.md 표와 같음)
BOUNDARIES = [("a+b", 23, 276, 211, "E24"), ("a+b", 31, 496, 329, "E13"), ("a+b", 50, 1275, 625, "E12"),
              ("a+b", 61, 1891, 882, "E21"), ("a+b", 71, 2556, 1183, "E13"), ("a+b", 101, 5151, 2067, "E24"),
              ("a+3b", 31, 961, 534, "E13"), ("a+3b", 41, 1681, 829, "E21"), ("a+3b", 50, 2500, 995, "E7")]


def fig8(out, extra=None):
    """필요한 예제 수 법칙: 50% 경계 N0 vs 연산 표 칸 수 C (로그-로그)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    C = np.array([b[2] for b in BOUNDARIES]); N = np.array([b[3] for b in BOUNDARIES])
    k, a = np.polyfit(np.log10(C), np.log10(N), 1)
    xs = np.logspace(np.log10(C.min() * 0.8), np.log10((extra or {}).get("xmax", C.max() * 1.3)), 50)
    ax.plot(xs, 10 ** a * xs ** k, "--", color=INK2, lw=1.2, label=f"N0 = {10**a:.2f} × C^{k:.2f}")
    for i, op in enumerate(("a+b", "a+3b")):
        sel = [b for b in BOUNDARIES if b[0] == op]
        ax.plot([b[2] for b in sel], [b[3] for b in sel], "o", color=C_[i], label=f"{op} (n = " + ", ".join(str(b[1]) for b in sel) + ")")
    for lab, pts in (extra or {}).get("series", {}).items():
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], fmt="s", color=C_[2], capsize=3, label=lab)
        ax.plot(xs, 2 * 10 ** a * xs ** k, ":", color=C_[2], lw=1.2, label="법칙 × 2 (합성 비용)")
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([300, 1000, 3000])); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(["300", "1,000", "3,000"])
    ax.yaxis.set_major_locator(FixedLocator([200, 500, 1000, 2000, 5000])); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels(["200", "500", "1,000", "2,000", "5,000"])
    ax.set_xlabel("배워야 할 연산 표의 칸 수 C"); ax.set_ylabel("일반화 50% 경계 train 예제 수")
    ax.set_title("필요한 예제 수는 표 크기의 약 0.76제곱, 합성 규칙은 약 2배")
    ax.legend(fontsize=7.5)
    fig.savefig(out / "F8_examples_law.png"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e6"); ap.add_argument("--e7"); ap.add_argument("--e8"); ap.add_argument("--e8-data")
    ap.add_argument("--e12", nargs="+"); ap.add_argument("--e12b", nargs="+"); ap.add_argument("--e12b-data")
    ap.add_argument("--e11d"); ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--e16", nargs="+"); ap.add_argument("--e17-internal")
    ap.add_argument("--only", nargs="+", help="그릴 그림 번호만 (예: 6 7)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setup_style()
    want = set(a.only or "12345678")
    if "1" in want: fig1(a.e6, out)
    if "2" in want: fig2(a.e6, a.e7, out)
    if "3" in want: fig3(a.e6, a.e7, a.e12, out)
    if "4" in want: fig4(a.e8, a.e8_data, a.e12b, a.e12b_data, out)
    if "5" in want: fig5(a.e11d, out)
    if "6" in want: fig6(a.e16, a.e12, a.e6, out)
    if "7" in want: fig7(a.e17_internal, out)
    if "8" in want:
        # 덧셈 두 번(E26 실패 / E29 성공 사이, 기하평균과 구간). C = n × n(n+1)/2
        comp = [(3610, (2400 * 3600) ** 0.5, 2400, 3600), (6348, (4000 * 6000) ** 0.5, 4000, 6000)]
        fig8(out, dict(series={"덧셈 두 번 (합성, n = 19, 23)": comp}, xmax=8000))
    print("저장:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
