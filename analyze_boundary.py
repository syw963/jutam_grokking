"""경계 실험(E6, E7) 결과 정리.

예:
    python analyze_boundary.py runs_boundary --out boundary_report
    python analyze_boundary.py runs_boundary runs_commute --out compare_report   # 여러 실험을 합쳐서
출력:
    table.csv          규칙 × train 크기: 일반화 비율, 지연 중앙값, coverage
    by_symmetry.csv    등변군 크기 × train 크기
    by_graph.csv       참조 그래프(서로 다른 쌍 수, 연결 여부) × 연산 × train 크기
    by_symmetry.png, by_graph.png   일반화 비율과 지연 중앙값 그래프
    화면: 두 묶음 기준의 요약 표, w5_next vs w5_p0 검증

지연 = t_gen − t_fit. 미도달 run은 일반화 비율에는 실패로, 지연 중앙값에는 빠진다.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from src.rules import Rule, ref_graph  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#8a6d3b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"
SYM_ORDER = ["eq6", "eq5", "eq2", "eq1", "first"]
SYM_LABEL = {"eq6": ("등변군 6", "equivariant group 6"), "eq5": ("등변군 5 (Z5)", "group 5 (Z5)"),
             "eq2": ("등변군 2", "group 2"), "eq1": ("등변군 1 (없음)", "group 1 (none)"),
             "first": ("첫 칸 (S4)", "first cell (S4)")}
KOREAN = False


def lab(ko, en):
    return ko if KOREAN else en


def load(dirs):
    out = []
    for d in dirs:
        for p in glob.glob(f"{d}/*/summary.json"):
            s = json.loads(Path(p).read_text())
            s["_unfinished"] = not s.get("stop_reason")
            out.append(s)
    return out


def sym_class(name):
    for key, prefix in [("eq6", "w6_"), ("eq5", "w5_"), ("eq2", "w2_"), ("eq1", "w1_"), ("first", "l1_first")]:
        if name.startswith(prefix):
            return key
    return None


_graph_cache = {}


def graph_class(s):
    """(연산, 서로 다른 쌍 수, 연결 여부) 문자열. 교환법칙 있는 연산은 무순서 쌍, 없으면 순서쌍을 센다."""
    name = s["rule_name"]
    if name not in _graph_cache:
        rule = Rule(s["rule"], 50)
        g = ref_graph(rule)
        op = rule.steps[0].p("op") if rule.steps and rule.steps[0].code == "REF" else None
        if g is None or g["edges_unordered"] is None or name.startswith("l1_first"):
            _graph_cache[name] = None
        else:
            k = g["edges_unordered"] if op == "add" else g["edges_ordered"]
            conn = lab("연결", "conn.") if g["connected"] else lab("끊김", "split")
            opname = "a+b" if op == "add" else "a+3b"
            _graph_cache[name] = f"{opname} · {lab('쌍', 'pairs')} {k} · {conn}"
    return _graph_cache[name]


def summarize(runs, key):
    groups = defaultdict(list)
    for s in runs:
        k = key(s)
        if k[0] is not None:
            groups[k].append(s)
    rows = []
    for k, rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        done = [r for r in rs if r.get("t_gen") is not None]
        delays = [r["t_gen"] - r["t_fit"] for r in done if r.get("t_fit") is not None]
        covs = [r["op_coverage"] for r in rs if r.get("op_coverage") is not None]
        rows.append(dict(name=k[0], train=k[1], runs=len(rs), grokked=len(done),
                         unfinished=sum(1 for r in rs if r["_unfinished"]),
                         grok_rate=round(len(done) / len(rs), 3),
                         delay_median=statistics.median(delays) if delays else None,
                         delay_min=min(delays) if delays else None,
                         delay_max=max(delays) if delays else None,
                         coverage=round(statistics.mean(covs), 3) if covs else None))
    return rows


def write_csv(rows, path):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def setup_font():
    global KOREAN
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR",
                "Noto Sans CJK JP"):
        if fam in names:
            plt.rcParams["font.family"] = fam
            KOREAN = True
            break
    plt.rcParams["axes.unicode_minus"] = False


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(lab("train 값 집합 수", "train sets"), color=MUTED, fontsize=9)


def plot(rows, order, labels, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i, key in enumerate(order[: len(COLORS)]):
        rs = [r for r in rows if r["name"] == key]
        if not rs:
            continue
        xs = [r["train"] for r in rs]
        ls = "--" if "a+3b" in str(key) else "-"
        a1.plot(xs, [r["grok_rate"] for r in rs], color=COLORS[i], lw=2, ls=ls, marker="o", ms=6, label=labels[key])
        pts = [(r["train"], r["delay_median"]) for r in rs if r["delay_median"]]
        if pts:
            a2.plot(*zip(*pts), color=COLORS[i], lw=2, ls=ls, marker="o", ms=6, label=labels[key])
    style(a1, lab("일반화한 run 비율", "fraction of runs that generalized"))
    a1.set_ylim(-0.03, 1.03)
    style(a2, lab("지연 중앙값 (t_gen − t_fit, step)", "median delay (t_gen - t_fit, steps)"))
    a2.set_yscale("log")
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_table(rows, order, labels, sizes):
    print("".ljust(30) + "".join(f"{c:>15}" for c in sizes))
    for key in order:
        cells = []
        for c in sizes:
            r = next((r for r in rows if r["name"] == key and r["train"] == c), None)
            cells.append("-" if r is None else f"{r['grokked']}/{r['runs']} {r['delay_median'] or '-'}")
        if any(x != "-" for x in cells):
            print(str(labels[key]).ljust(30) + "".join(f"{x:>15}" for x in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dirs", nargs="*", default=["runs_boundary"])
    ap.add_argument("--out", default="boundary_report")
    a = ap.parse_args()
    setup_font()
    runs = load(a.runs_dirs)
    if not runs:
        raise SystemExit(f"{a.runs_dirs}에 summary.json이 없습니다")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rule_rows = summarize(runs, lambda s: (s["rule_name"], s["train_sets"]))
    sym_rows = summarize(runs, lambda s: (sym_class(s["rule_name"]) and
                                          sym_class(s["rule_name"]) + ("·a+3b" if s["rule_name"].endswith("_n3") else ""),
                                          s["train_sets"]))
    graph_rows = summarize(runs, lambda s: (graph_class(s), s["train_sets"]))
    write_csv(rule_rows, out / "table.csv")
    write_csv(sym_rows, out / "by_symmetry.csv")
    write_csv(graph_rows, out / "by_graph.csv")

    sym_order = [k + suf for suf in ("", "·a+3b") for k in SYM_ORDER]
    sym_labels = {k + suf: lab(*SYM_LABEL[k]) + (" · a+3b" if suf else "") for suf in ("", "·a+3b") for k in SYM_ORDER}
    graph_order = sorted({r["name"] for r in graph_rows}, key=lambda g: (g.startswith("a+3b"), g), reverse=False)
    graph_order = sorted(graph_order, key=lambda g: (g.startswith("a+3b"), -int(g.split()[3]), g))
    graph_labels = {g: g for g in graph_order}
    plot(sym_rows, sym_order, sym_labels, out / "by_symmetry.png")
    plot(graph_rows, graph_order, graph_labels, out / "by_graph.png")

    sizes = sorted({r["train"] for r in rule_rows})
    print(f"run {len(runs)}개 (진행 중 {sum(1 for s in runs if s['_unfinished'])}개) → {out}/")
    print("\n[등변군 크기별: 일반화 run / 지연 중앙값]")
    print_table(sym_rows, sym_order, sym_labels, sizes)
    print("\n[참조 그래프별: 연산 · 한 예제가 주는 서로 다른 쌍 수 · 연결 여부]")
    print_table(graph_rows, graph_order, graph_labels, sizes)
    print("\n[검증] w5_next vs w5_p0 (위치만 다른 같은 규칙 → 비슷해야 함)")
    for c in sizes:
        x = next((r for r in rule_rows if r["name"] == "w5_next" and r["train"] == c), None)
        y = next((r for r in rule_rows if r["name"] == "w5_p0" and r["train"] == c), None)
        if x and y:
            print(f"  tr{c}: next {x['grokked']}/{x['runs']} 지연 {x['delay_median']}   "
                  f"p0 {y['grokked']}/{y['runs']} 지연 {y['delay_median']}")


if __name__ == "__main__":
    main()
