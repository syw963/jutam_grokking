"""E30 오답 실험 정리: run별 noise.csv를 조건별로 모은다.

조건 = (train 수 N, wd, 오답 방식, 오답 비율). 시점 두 개에서 seed 평균을 낸다.
  일반화 시점: 오답 칸이 없는 test 행 정확도(test_clean_rows_exact)가 처음 0.95 이상인 step
  마지막 시점: 학습 끝
지표:
  mem     오답이 달린 train 자리에서 외운 오답을 그대로 내는 비율
  fixed   같은 자리에서 원래 정답을 내는 비율 (오답을 "고침")
  true_rank2  원래 정답이 2순위인 비율,  true_prob 원래 정답 확률 평균
  test_spread 오답 칸이 나오는 test 자리에서 그 오답을 내는 비율 (오답이 처음 보는 문제로 번짐)

    python analyze_noise.py "runs_e30_*" --out e30_report
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["mem", "fixed", "true_rank2", "true_rank_med", "true_prob", "clean_rows_exact", "test_spread",
        "test_true_on_cells", "test_clean_rows_exact"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="e30_report")
    a = ap.parse_args()
    rows, curves = [], []
    for pat in a.inputs:
        for root in glob.glob(pat):
            for d in glob.glob(f"{root}/*"):
                d = Path(d)
                if not (d / "config.json").exists() or not (d / "metrics.csv").exists():
                    continue
                c = json.loads((d / "config.json").read_text())
                if (d / "noise.csv").exists():
                    nz = json.loads((d / "noise.json").read_text())
                    t = pd.read_csv(d / "noise.csv")
                else:  # 대조군(오답 없음): test 정확도만
                    nz = dict(mode="none", frac=0.0, tokens=0, cells=0)
                    m = pd.read_csv(d / "metrics.csv")
                    t = pd.DataFrame(dict(step=m.step, test_clean_rows_exact=m.test_exact))
                    for k in COLS:
                        if k not in t:
                            t[k] = np.nan
                if t.empty:
                    continue
                key = dict(N=c["data_meta"]["train_sets"], wd=c["wd"], mode=nz["mode"], frac=nz["frac"], seed=c["seed"])
                g = t[t.test_clean_rows_exact >= 0.95]
                t_gen = int(g.step.iloc[0]) if len(g) else None
                at_gen = g.iloc[0] if len(g) else None
                last = t.iloc[-1]
                r = dict(key, t_gen=t_gen, last_step=int(last.step), noisy_tokens=nz["tokens"], noisy_cells=nz["cells"])
                for k in COLS:
                    r[f"{k}@gen"] = at_gen[k] if at_gen is not None else np.nan
                    r[f"{k}@end"] = last[k]
                rows.append(r)
                tt = t.assign(**key)
                curves.append(tt)
    if not rows:
        raise SystemExit("noise.csv가 있는 run이 없습니다")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows).sort_values(["N", "wd", "mode", "frac", "seed"])
    R.to_csv(f"{a.out}/runs.csv", index=False)
    keys = ["N", "wd", "mode", "frac"]
    agg = {"seed": "count", "t_gen": lambda s: f"{s.notna().sum()}/{len(s)} med {s.median():.0f}" if s.notna().any() else f"0/{len(s)}"}
    for k in ["mem", "fixed", "true_rank2", "true_prob", "test_spread", "test_clean_rows_exact"]:
        agg[f"{k}@gen"] = "mean"; agg[f"{k}@end"] = "mean"
    G = R.groupby(keys).agg(agg).reset_index().rename(columns={"seed": "runs"})
    G.to_csv(f"{a.out}/grouped.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(G.round(3).to_string(index=False))

    C = pd.concat(curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combos = C[keys[:2]].drop_duplicates().sort_values(keys[:2]).values.tolist()
    fig, axes = plt.subplots(len(combos), 3, figsize=(15, 3.6 * len(combos)), squeeze=False)
    for row, (N, wd) in zip(axes, combos):
        s = C[(C.N == N) & (C.wd == wd)]
        for (mode, frac), q in s.groupby(["mode", "frac"]):
            if frac == 0:
                m = q.groupby("step").test_clean_rows_exact.mean()
                row[2].plot(m.index, m.values, "k-", lw=1, label="no noise")
                continue
            ls = "-" if mode == "scatter" else "--"
            lab = f"{mode} {frac:g}"
            for ax, col in zip(row, ["mem", "test_spread", "test_clean_rows_exact"]):
                m = q.groupby("step")[col].mean()
                ax.plot(m.index, m.values, ls, label=lab)
        for ax, title in zip(row, ["memorized wrong answer (train)", "wrong answer on unseen (test, same cell)",
                                   "test exact (rows without noisy cell)"]):
            ax.set_xscale("symlog", linthresh=1000); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3)
            ax.set_title(f"N={N} wd={wd:g} · {title}", fontsize=9); ax.set_xlabel("step")
        row[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.out}/curves.png", dpi=110)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
