"""필요한 예제 수 법칙 N0 = a × C^b 를 run 폴더에서 바로 적합한다 (E31 등).

경계 N0: 같은 (n, 연산, 모델) 조건에서 일반화 비율이 50%가 되는 train 예제 수. 이웃 두 점 사이를 log N으로 보간.
지수 b의 불확실성: 조건 칸마다 run을 복원추출해 경계와 지수를 다시 계산(부트스트랩).

    python analyze_law.py "runs_e31_*" --out e31_report
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(patterns):
    rows = []
    for pat in patterns:
        for root in glob.glob(pat):
            for d in glob.glob(f"{root}/*"):
                d = Path(d)
                try:
                    s = json.loads((d / "summary.json").read_text())
                    c = json.loads((d / "config.json").read_text())
                except (FileNotFoundError, NotADirectoryError):
                    continue
                if not s.get("stop_reason"):
                    continue
                n = c["n"]
                op = "a+b" if "op=add)" in c["rule_spec"] or c["rule_spec"].endswith("(op=add)") else "other"
                if "add3" in c["rule_spec"]:
                    op = "a+3b"
                C = n * (n + 1) / 2 if op == "a+b" else n * n
                rows.append(dict(n=n, op=op, model=c.get("model", "m0"), C=C, N=c["data_meta"]["train_sets"],
                                 g=s.get("t_gen") is not None, seed=c["seed"]))
    return rows


def boundary(cells):
    """cells: {N: [bool, ...]} → 일반화 비율 50% 지점 (log 보간). 없으면 None."""
    Ns = sorted(cells)
    r = [np.mean(cells[N]) for N in Ns]
    for i in range(len(Ns) - 1):
        if r[i] < 0.5 <= r[i + 1]:
            t = (0.5 - r[i]) / (r[i + 1] - r[i])
            return math.exp(math.log(Ns[i]) + t * (math.log(Ns[i + 1]) - math.log(Ns[i])))
    if r and r[0] >= 0.5:
        return None
    return None


def fit(points):
    x = np.log([p[0] for p in points]); y = np.log([p[1] for p in points])
    b, a = np.polyfit(x, y, 1)
    return math.exp(a), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="law_report")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rows = load(a.inputs)
    if not rows:
        raise SystemExit("끝난 run이 없습니다")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["model"], r["op"], r["n"], r["C"])][r["N"]].append(r["g"])
    lines = ["model,op,n,C,N,runs,grokked,rate"]
    bnd = {}
    for key in sorted(groups):
        cells = groups[key]
        for N in sorted(cells):
            lines.append(f"{key[0]},{key[1]},{key[2]},{key[3]:.0f},{N},{len(cells[N])},{sum(cells[N])},{np.mean(cells[N]):.2f}")
        bnd[key] = boundary(cells)
    Path(a.out, "grid.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n경계 N0 (50% 보간):")
    for key, v in bnd.items():
        print(f"  {key[0]} {key[1]} n={key[2]} C={key[3]:.0f}: {v if v is None else round(v)}")
    by_model = defaultdict(list)
    for key, v in bnd.items():
        if v:
            by_model[(key[0], key[1])].append((key[3], v, key))
    rng = np.random.default_rng(0)
    res = {}
    for mk, pts in by_model.items():
        if len(pts) < 2:
            continue
        A, B = fit([(p[0], p[1]) for p in pts])
        boots = []
        for _ in range(a.boot):
            bp = []
            for C, _, key in pts:
                cells = {N: list(rng.choice(v, len(v))) for N, v in groups[key].items()}
                v = boundary(cells)
                if v:
                    bp.append((C, v))
            if len(bp) >= 2:
                boots.append(fit(bp)[1])
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (math.nan, math.nan))
        res[f"{mk[0]} {mk[1]}"] = dict(a=A, b=B, b_lo=lo, b_hi=hi, points=[(p[0], p[1]) for p in pts])
        print(f"\n{mk[0]} {mk[1]}: N0 = {A:.2f} × C^{B:.3f}  (지수 95% 구간 {lo:.2f}~{hi:.2f}, 부트스트랩 {len(boots)}회)"
              f"  [m0 법칙: 2.81 × C^0.764]")
    Path(a.out, "law_fit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.logspace(2.3, 4, 50)
    ax.plot(xs, 2.81 * xs ** 0.764, "k--", lw=1, label="m0 law 2.81·C^0.764")
    for name, r in res.items():
        p = np.array(r["points"])
        l = ax.plot(p[:, 0], p[:, 1], "o", label=f"{name}: {r['a']:.1f}·C^{r['b']:.2f}")
        ax.plot(xs, r["a"] * xs ** r["b"], "-", color=l[0].get_color(), lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("table cells C"); ax.set_ylabel("boundary N0")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(a.out, "law_fit.png"), dpi=130)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
