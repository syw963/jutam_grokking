"""E40 weight decay의 권장 구간: 경계(필요한 데이터) 손해와 외운 오답 교정을 wd 하나의 축에 놓는다.

A. 경계: run 폴더(--law)를 wd·n별로 묶어 일반화 50% 경계 N0를 구하고(analyze_law.boundary), 같은 n의 wd 1 경계 대비 배수와
   wd별 지수(N0 = a·C^b, 부트스트랩 구간)를 낸다. 최대 step에 잘린 run(일반화 못 하고 한도까지 감)이 경계 근처에 많으면 표시.
B. 오답: 오답 run(--noise, noise.csv)을 (N, wd, 방식)으로 묶어
   학습 성공(오답 없는 test 행 정확도 ≥ 0.95, 마지막), 외운 오답 최대·마지막, 교정 시점(최대 이후 처음 0.1 아래로 내려간 step).
   오답 없는 대조 run은 test 정확도로 학습 성공만.
→ wd_law.csv, wd_noise.csv, wd_summary.txt, wd_sweet.png

    python analyze_wd_sweet.py --law "runs_e32_base_*" "runs_e38_*" "runs_e40_law_*" --noise "runs_e40_noise_*" --out e40_report
"""
from __future__ import annotations

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
                if (d / "config.json").exists() and (d / "summary.json").exists():
                    yield d


def law_part(patterns, boots=1000):
    cells = defaultdict(lambda: defaultdict(list))   # (wd, n, C) -> N -> [bool]
    cens = defaultdict(int)
    for d in run_dirs(patterns):
        c = json.loads((d / "config.json").read_text())
        s = json.loads((d / "summary.json").read_text())
        if not s.get("stop_reason") or c.get("noise_frac") or c.get("model", "m0") != "m0":
            continue
        if c.get("d", 128) != 128 or c.get("layers", 2) != 2 or c.get("rule") != "w5_next":
            continue
        n = c["n"]
        C = n * (n + 1) / 2
        wd = float(c.get("wd", 1.0))
        g = s.get("t_gen") is not None
        cells[(wd, n, C)][c["data_meta"]["train_sets"]].append(g)
        if not g and s.get("stop_reason") == "max_steps":
            cens[(wd, n, c["data_meta"]["train_sets"])] += 1
    bnd = {k: boundary(v) for k, v in cells.items()}
    rows = []
    base = {n: b for (wd, n, C), b in bnd.items() if wd == 1.0 and b}
    rng = np.random.default_rng(0)
    by_wd = defaultdict(list)
    for (wd, n, C), b in sorted(bnd.items()):
        Ns = sorted(cells[(wd, n, C)])
        near = [N for N in Ns if b and 0.6 * b <= N <= 1.6 * b]
        cen = sum(cens.get((wd, n, N), 0) for N in near)
        rows.append(dict(wd=wd, n=n, C=int(C), boundary=round(b) if b else "", ratio_vs_wd1=round(b / base[n], 3) if b and n in base else "",
                         grid=" ".join(f"{N}:{sum(cells[(wd, n, C)][N])}/{len(cells[(wd, n, C)][N])}" for N in Ns),
                         censored_near=cen))
        if b:
            by_wd[wd].append((C, b, (wd, n, C)))
    fits = {}
    for wd, pts in sorted(by_wd.items()):
        if len(pts) < 2:
            continue
        A, B = fit([(p[0], p[1]) for p in pts])
        bs = []
        for _ in range(boots):
            bp = []
            for C, _, key in pts:
                v = boundary({N: list(rng.choice(x, len(x))) for N, x in cells[key].items()})
                if v:
                    bp.append((C, v))
            if len(bp) >= 2:
                bs.append(fit(bp)[1])
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (math.nan, math.nan)
        fits[wd] = dict(a=A, b=B, lo=lo, hi=hi, k=len(pts))
    return rows, fits


def noise_part(patterns, low=0.1):
    groups = defaultdict(list)
    for d in run_dirs(patterns):
        c = json.loads((d / "config.json").read_text())
        N, wd = c["data_meta"]["train_sets"], float(c.get("wd", 1.0))
        mode = c.get("noise_mode") if c.get("noise_frac") else "none"
        r = dict(seed=c["seed"])
        if mode == "none":
            rows = list(csv.DictReader((d / "metrics.csv").open()))
            te = [float(x["test_exact"]) for x in rows if x.get("test_exact") not in (None, "")]
            r["learned"] = bool(te) and te[-1] >= 0.95
        else:
            rows = list(csv.DictReader((d / "noise.csv").open()))
            st = np.array([int(x["step"]) for x in rows])
            mem = np.array([float(x["mem"]) for x in rows])
            te = np.array([float(x["test_clean_rows_exact"]) for x in rows])
            r["learned"] = bool(len(te)) and te[-1] >= 0.95
            r["mem_end"] = float(mem[-1]) if len(mem) else math.nan
            ipk = int(np.argmax(mem)) if len(mem) else 0
            r["mem_peak"], r["peak_step"] = (float(mem[ipk]), int(st[ipk])) if len(mem) else (math.nan, None)
            # 교정 = 최대 이후 low 아래로 내려가 학습 끝까지 유지된 경우만 (10월 25일: 한 번 내려갔다 다시 오른 경우 제외)
            above = np.nonzero(mem[ipk:] >= low)[0]
            last = ipk + int(above[-1]) if len(above) else ipk
            r["t_correct"] = int(st[last + 1]) if r["mem_peak"] >= low and last + 1 < len(mem) else None
            if "test_spread" in rows[0]:
                sp = [float(x["test_spread"]) for x in rows if x.get("test_spread") not in (None, "")]
                r["spread_end"] = sp[-1] if sp else math.nan
        groups[(N, wd, mode)].append(r)
    out = []
    for (N, wd, mode), rs in sorted(groups.items()):
        o = dict(N=N, wd=wd, mode=mode, runs=len(rs), learned=f"{sum(r['learned'] for r in rs)}/{len(rs)}")
        if mode != "none":
            L = [r for r in rs if r["learned"]] or rs
            o["mem_end"] = round(float(np.mean([r["mem_end"] for r in L])), 3)
            o["mem_peak"] = round(float(np.mean([r["mem_peak"] for r in L])), 3)
            tc = [r["t_correct"] for r in L if r["t_correct"] is not None]
            o["corrected"] = f"{len(tc)}/{len(L)}"
            o["t_correct_med"] = int(np.median(tc)) if tc else ""
            if mode == "cell":
                o["spread_end"] = round(float(np.nanmean([r.get("spread_end", math.nan) for r in L])), 3)
        out.append(o)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def plot(law_rows, noise_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in sorted({r["n"] for r in law_rows}):
        R = sorted([r for r in law_rows if r["n"] == n and r["ratio_vs_wd1"] != ""], key=lambda r: r["wd"])
        ax[0].plot([r["wd"] for r in R], [r["ratio_vs_wd1"] for r in R], "o-", label=f"n={n}")
    ax[0].axhline(1, color="gray", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("weight decay"); ax[0].set_ylabel("boundary / boundary at wd 1")
    ax[0].set_title("data cost of weight decay"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for (mode, ls) in (("scatter", "-"), ("cell", "--")):
        for N in sorted({r["N"] for r in noise_rows if r["mode"] == mode}):
            R = sorted([r for r in noise_rows if r["mode"] == mode and r["N"] == N], key=lambda r: r["wd"])
            ax[1].plot([r["wd"] for r in R], [r["mem_end"] for r in R], ls, marker="o", label=f"{mode} N={N}")
            for r in R:
                k, t = map(int, r["learned"].split("/"))
                if k < t:
                    ax[1].annotate(f"learn {k}/{t}", (r["wd"], r["mem_end"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("weight decay"); ax[1].set_ylabel("memorized wrong labels at end")
    ax[1].set_title("noise correction"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3); ax[1].set_ylim(-0.03, 1.05)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", nargs="*", default=[])
    ap.add_argument("--noise", nargs="*", default=[])
    ap.add_argument("--out", default="wd_report")
    ap.add_argument("--boots", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    law_rows, fits = law_part(a.law, a.boots) if a.law else ([], {})
    noise_rows = noise_part(a.noise) if a.noise else []
    write_csv(out / "wd_law.csv", law_rows); write_csv(out / "wd_noise.csv", noise_rows)
    L = ["A. 경계 (wd · n: 경계, wd 1 대비, 격자 일반화 수, 경계 근처에서 한도까지 실패한 run 수)"]
    for r in law_rows:
        L.append(f"  wd {r['wd']:g} · n{r['n']}: {r['boundary']}  ×{r['ratio_vs_wd1']}  [{r['grid']}]" + (f"  한도 실패 {r['censored_near']}" if r["censored_near"] else ""))
    L.append("지수 (N0 = a·C^b): " + ", ".join(f"wd {wd:g}: b={f['b']:.2f} ({f['lo']:.2f}~{f['hi']:.2f}), a={f['a']:.2f}" for wd, f in fits.items()))
    L.append("\nB. 오답 (N · wd · 방식: 학습 성공, 외운 오답 마지막 / 최대, 교정 run, 교정 step 중앙값, 번짐)")
    for r in noise_rows:
        L.append(f"  N {r['N']} · wd {r['wd']:g} · {r['mode']}: 학습 {r['learned']}" +
                 (f", 오답 {r['mem_end']} / 최대 {r['mem_peak']}, 교정 {r['corrected']} @ {r['t_correct_med']}" if r["mode"] != "none" else "") +
                 (f", 번짐 {r['spread_end']}" if r["mode"] == "cell" else ""))
    s = "\n".join(L)
    (out / "wd_summary.txt").write_text(s + "\n")
    print(s)
    try:
        plot(law_rows, noise_rows, out / "wd_sweet.png")
    except Exception as e:
        print("그림 실패:", e)


if __name__ == "__main__":
    main()
