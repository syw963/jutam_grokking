"""칸당 평균 관찰 수로 E6(a+b)·E7(a+3b)를 한 곡선에 올린다.

칸당 평균 관찰 수 = train 개수 × 한 예제가 주는 서로 다른 쌍 수 / 연산 표 칸 수
  (a+b: 무순서 칸 1,275개, a+3b: 순서쌍 칸 2,500개)

예:
    python analyze_boundary.py runs_boundary runs_commute --out merged
    python analyze_obs.py merged/by_graph.csv --out merged/obs_curve.png
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("by_graph_csv")
    ap.add_argument("--out", default="obs_curve.png")
    a = ap.parse_args()
    t = pd.read_csv(a.by_graph_csv)
    t["op"] = np.where(t.name.str.startswith("a+3b"), "a+3b", "a+b")
    t["pairs"] = t.name.str.extract(r"쌍 (\d)").astype(int)
    t["obs_per_cell"] = t.train * t.pairs / np.where(t.op == "a+b", 1275, 2500)
    t = t.sort_values("obs_per_cell")
    print(t[["name", "train", "obs_per_cell", "coverage", "grok_rate", "delay_median"]].round(2).to_string(index=False))
    g = t[t.grok_rate == 1]
    print(f"\n완전 일반화 칸의 log 지연과 상관: 칸당 관찰 수 "
          f"{np.corrcoef(np.log(g.obs_per_cell), np.log(g.delay_median + 50))[0, 1]:.2f}, "
          f"coverage {np.corrcoef(g.coverage, np.log(g.delay_median + 50))[0, 1]:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in t.groupby("name"):
        op, p = s.op.iloc[0], s.pairs.iloc[0]
        mk = "o" if op == "a+b" else "s"
        lab = f"{op}, {p} pairs/example" + (", connected" if "연결" in name else ", split" if "끊김" in name else "")
        ax[0].plot(s.obs_per_cell, s.grok_rate, mk + "-", label=lab, alpha=.8)
        ax[1].plot(s.obs_per_cell, s.delay_median, mk + "-", label=lab, alpha=.8)
    ax[0].set_ylabel("fraction of runs generalized")
    ax[1].set_ylabel("delay (median steps)"); ax[1].set_yscale("log")
    for x in ax:
        x.set_xlabel("observations per table cell"); x.grid(alpha=.3); x.legend(fontsize=7)
    xs = np.linspace(t.obs_per_cell.min(), t.obs_per_cell.max(), 50)
    ax[1].plot(xs, 10 ** (5.78 - 0.754 * xs), "k--", lw=1, label="E12 prediction (fit to E6+E7)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
