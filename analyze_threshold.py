"""문턱 민감도: 일반화·암기 기준(test 95%, train 99%)을 바꿔도 결론이 같은가.

A. 지연(연구문제 2): 칸당 관찰 수와 log 지연의 상관을 기준 조합마다 다시 계산한다(E6·E7·E12 run 폴더).
B. 경계와 지수(연구문제 1·일반성): 일반화 기준을 0.90/0.95/0.99로 바꿔 경계와 지수를 다시 적합한다(E31·E32·E33·E35 run 폴더).
주의: 대부분의 run은 test 95% 도달 1,000 step 뒤 학습을 멈췄다. 0.99 기준은 그 1,000 step 안에 도달해야 하므로 보수적이다.

    python analyze_threshold.py --delay "runs_boundary" "runs_commute" "runs_e12_*" --law "runs_e32_*" "runs_e31_*" --out sens_report
"""
import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_law import boundary, fit


def run_dirs(patterns):
    for pat in patterns:
        for root in glob.glob(pat):
            for d in glob.glob(f"{root}/*"):
                d = Path(d)
                if (d / "metrics.csv").exists() and (d / "config.json").exists():
                    yield d


def curves(d):
    rows = list(csv.DictReader((d / "metrics.csv").open()))
    st = np.array([int(r["step"]) for r in rows])
    tr = np.array([float(r["train_exact"] or "nan") for r in rows])
    te = np.array([float(r["test_exact"] or "nan") for r in rows])
    return st, tr, te


def first(st, x, thr):
    i = np.nonzero(x >= thr)[0]
    return int(st[i[0]]) if len(i) else None


def part_a(patterns, out):
    from src.rules import Rule, ref_graph
    runs = []
    for d in run_dirs(patterns):
        c = json.loads((d / "config.json").read_text())
        rule = Rule(c["rule_spec"], c["n"])
        if len(rule.steps) != 1 or rule.steps[0].code != "REF":
            continue
        g = ref_graph(rule, c["m"]) or {}
        if g.get("edges_unordered") is None:
            continue
        comm = rule.steps[0].p("op") == "add"
        pairs = g["edges_unordered"] if comm else g["edges_ordered"]
        k = c.get("train_mask", 0) or 0
        if k:
            continue
        n = c["n"]
        cells = n * (n + 1) / 2 if comm else n * n
        N = c["data_meta"]["train_sets"]
        st, tr, te = curves(d)
        runs.append(dict(key=(c["rule_spec"], n, N, c.get("wd", 1.0), c.get("d", 128)), obs=N * pairs / cells,
                         st=st, tr=tr, te=te))
    lines = [f"A. 지연 (run {len(runs)}개, 조건 = 규칙·n·N·wd·모델 크기, 일반화율 0.9 이상 조건만)",
             "fit기준 gen기준 | 조건 수 | log10 지연과 관찰 수 상관 | 기울기(관찰 수 1회당 log10 지연) | 관찰 수 몇 회마다 지연 절반"]
    for ft in (0.95, 0.99):
        for gt in (0.90, 0.95, 0.99):
            by = defaultdict(list)
            for r in runs:
                tf, tg = first(r["st"], r["tr"], ft), first(r["st"], r["te"], gt)
                by[r["key"]].append((r["obs"], tg, (tg - tf) if (tg is not None and tf is not None) else None))
            xs, ys = [], []
            for key, v in by.items():
                ok = [x for x in v if x[1] is not None]
                if len(ok) / len(v) < 0.9:
                    continue
                dl = np.median([max(x[2], 50) for x in ok])
                xs.append(v[0][0]); ys.append(math.log10(dl))
            if len(xs) > 3:
                r = np.corrcoef(xs, ys)[0, 1]; b = np.polyfit(xs, ys, 1)[0]
                lines.append(f"{ft:.2f} {gt:.2f} | {len(xs)} | {r:+.2f} | {b:+.2f} | {math.log10(2) / -b:.2f}")
    reach = [first(r["st"], r["te"], 0.99) is not None for r in runs if first(r["st"], r["te"], 0.95) is not None]
    lines.append(f"(test 95%에 도달한 run 중 99%에도 도달한 비율: {np.mean(reach):.2f})")
    return lines


def part_b(patterns):
    rows = []
    for d in run_dirs(patterns):
        c = json.loads((d / "config.json").read_text())
        spec, n = c["rule_spec"], c["n"]
        op = "a*b" if "op=mul" in spec else "a+3b" if "op=add3" in spec else "a+b"
        C = n * (n + 1) / 2 if op in ("a+b", "a*b") else n * n
        mdl = c.get("model", "m0")
        if c.get("d", 128) != 128 or c.get("layers", 2) != 2:
            mdl += f"-d{c['d']}L{c['layers']}"
        st, tr, te = curves(d)
        rows.append(dict(model=mdl, op=op, n=n, C=C, N=c["data_meta"]["train_sets"], mx=np.nanmax(te)))
    lines = ["B. 경계와 지수 (일반화 = 학습 중 test 정확도가 기준 이상에 한 번이라도 도달)",
             "모델·연산 | 기준 | 경계 (n 순) | 지수"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["op"])].append(r)
    for (mdl, op), rs in sorted(groups.items()):
        for thr in (0.90, 0.95, 0.99):
            cells = defaultdict(lambda: defaultdict(list))
            for r in rs:
                if op == "a*b" and r["n"] == 50:
                    continue  # 합성수는 따로 (구조가 다름)
                cells[(r["n"], r["C"])][r["N"]].append(r["mx"] >= thr)
            pts = []
            for (n, C), cl in sorted(cells.items()):
                b = boundary(cl)
                pts.append((n, C, b))
            good = [(C, b) for _, C, b in pts if b]
            ex = f"{fit(good)[1]:.2f}" if len(good) >= 2 else "-"
            lines.append(f"{mdl} {op} | {thr:.2f} | " + " ".join(f"n{n}:{(round(b) if b else '-')}" for n, _, b in pts) + f" | {ex}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", nargs="*", default=[])
    ap.add_argument("--law", nargs="*", default=[])
    ap.add_argument("--out", default="sens_report")
    a = ap.parse_args()
    lines = []
    if a.delay:
        lines += part_a(a.delay, a.out) + [""]
    if a.law:
        lines += part_b(a.law)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "threshold_sensitivity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
