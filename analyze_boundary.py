"""경계 정밀 측정(E6) 결과 정리.

예:
    python analyze_boundary.py runs_boundary --out boundary_report
출력:
    boundary_report/table.csv      규칙 × train 크기: 일반화 비율, 지연 중앙값, coverage
    boundary_report/classes.csv    대칭 크기(등변군) × train 크기로 묶은 표
    boundary_report/boundary.png   그래프 2개: 일반화 비율, 지연 (대칭 크기별)
    화면에 요약과 검증 결과(w5_next vs w5_p0)를 출력한다.

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

# 범주 색 (고정 순서)
CLASS_STYLE = [  # (이름, 규칙 이름 접두사, 색)
    ("등변군 6", "w6_", "#2a78d6"),
    ("등변군 5 (Z5)", "w5_", "#eb6834"),
    ("등변군 2", "w2_", "#1baf7a"),
    ("등변군 1 (없음)", "w1_", "#eda100"),
    ("첫 칸 (S4, coverage 낮음)", "l1_first", "#e87ba4"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"


def load(runs_dir):
    out = []
    for p in glob.glob(f"{runs_dir}/*/summary.json"):
        s = json.loads(Path(p).read_text())
        if not s.get("stop_reason"):
            s["_unfinished"] = True
        out.append(s)
    return out


def cls_of(name):
    for label, prefix, _ in CLASS_STYLE:
        if name.startswith(prefix):
            return label
    return None


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
                         unfinished=sum(1 for r in rs if r.get("_unfinished")),
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
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if fam in names:
            plt.rcParams["font.family"] = fam
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
    ax.set_xlabel("train 값 집합 수", color=MUTED, fontsize=9)


def plot(class_rows, path):
    setup_font()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, _, color in CLASS_STYLE:
        rs = [r for r in class_rows if r["name"] == label]
        if not rs:
            continue
        xs = [r["train"] for r in rs]
        a1.plot(xs, [r["grok_rate"] for r in rs], color=color, lw=2, marker="o", ms=6, label=label)
        pts = [(r["train"], r["delay_median"]) for r in rs if r["delay_median"] is not None]
        if pts:
            a2.plot(*zip(*pts), color=color, lw=2, marker="o", ms=6, label=label)
    style(a1, "일반화한 run 비율")
    a1.set_ylim(-0.03, 1.03)
    style(a2, "지연 중앙값 (t_gen − t_fit, step)")
    a2.set_yscale("log")
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", nargs="?", default="runs_boundary")
    ap.add_argument("--out", default="boundary_report")
    a = ap.parse_args()
    runs = load(a.runs_dir)
    if not runs:
        raise SystemExit(f"{a.runs_dir}에 summary.json이 없습니다")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rule_rows = summarize(runs, lambda s: (s["rule_name"], s["train_sets"]))
    class_rows = summarize(runs, lambda s: (cls_of(s["rule_name"]), s["train_sets"]))
    write_csv(rule_rows, out / "table.csv")
    write_csv(class_rows, out / "classes.csv")
    plot(class_rows, out / "boundary.png")

    unfinished = sum(1 for s in runs if s.get("_unfinished"))
    print(f"run {len(runs)}개 (진행 중 {unfinished}개) → {out}/")
    sizes = sorted({r["train"] for r in class_rows})
    print("\n[대칭 크기별 일반화 비율 / 지연 중앙값]")
    print("".ljust(26) + "".join(f"{c:>16}" for c in sizes))
    for label, _, _ in CLASS_STYLE:
        cells = []
        for c in sizes:
            r = next((r for r in class_rows if r["name"] == label and r["train"] == c), None)
            cells.append("-" if r is None else f"{r['grokked']}/{r['runs']} {r['delay_median'] or '-'}")
        print(label.ljust(26) + "".join(f"{x:>16}" for x in cells))

    print("\n[검증] w5_next vs w5_p0 (위치만 다른 같은 규칙 → 비슷해야 함)")
    for c in sizes:
        a_ = next((r for r in rule_rows if r["name"] == "w5_next" and r["train"] == c), None)
        b_ = next((r for r in rule_rows if r["name"] == "w5_p0" and r["train"] == c), None)
        if a_ and b_:
            print(f"  tr{c}: next {a_['grokked']}/{a_['runs']} 지연 {a_['delay_median']}   "
                  f"p0 {b_['grokked']}/{b_['runs']} 지연 {b_['delay_median']}")


if __name__ == "__main__":
    main()
