"""칸당 평균 관찰 수로 여러 실험을 한 곡선에 올린다.

칸당 평균 관찰 수 = train 개수 × 한 예제가 주는 서로 다른 쌍 수 / 연산 표 칸 수
  a+b(교환): 서로 다른 무순서 쌍 수 / n(n+1)/2칸,  a+3b 등: 서로 다른 순서쌍 수 / n²칸

두 가지 입력:
    python analyze_obs.py merged/by_graph.csv --out obs.png        # analyze_boundary의 by_graph.csv (n=50 가정)
    python analyze_obs.py runs_e12_w5 runs_e13_* --out e13_report   # run 폴더들 (n, wd, 모델 크기를 run마다 읽음)
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRED_A, PRED_B = 5.78, 0.754   # E12 사전 예측: 지연 ≈ 10^(A − B × 관찰 수)


def from_csv(path):
    t = pd.read_csv(path)
    t["op"] = np.where(t.name.str.startswith("a+3b"), "a+3b", "a+b")
    t["pairs"] = t.name.str.extract(r"쌍 (\d)").astype(int)
    t["obs_per_cell"] = t.train * t.pairs / np.where(t.op == "a+b", 1275, 2500)
    t["label"] = t.name
    return t


def from_runs(dirs):
    from src.rules import Rule, ref_graph
    rows = []
    for d in dirs:
        d = Path(d)
        try:
            s = json.loads((d / "summary.json").read_text())
            c = json.loads((d / "config.json").read_text())
        except FileNotFoundError:
            continue
        if not s.get("stop_reason") and s.get("t_gen") is None and s.get("last_step", 0) < c.get("steps", 0):
            continue  # 아직 진행 중
        n = c["n"]
        rule = Rule(c["rule_spec"], n)
        g = ref_graph(rule, c["m"]) or {}
        if g.get("edges_unordered") is None:
            continue
        op = rule.steps[0].p("op")
        comm = op == "add"
        pairs = g["edges_unordered"] if comm else g["edges_ordered"]
        k = c.get("train_mask", 0) or 0
        if k:  # 출력 칸 k개만 학습: 5쌍 규칙(w5 계열)에서는 예제당 서로 다른 쌍도 k개
            pairs = min(pairs, k)
        cells = n * (n + 1) / 2 if comm else n * n
        N = c["data_meta"]["train_sets"]
        variant = f"d{c['d']}L{c['layers']}" + (f" wd{c['wd']:g}" if c["wd"] != 1.0 else "") + (f" k{k}" if k else "")
        tf, tg = s.get("t_fit"), s.get("t_gen")
        rows.append(dict(n=n, op="a+b" if comm else "a+3b", pairs=pairs, connected=g.get("connected"),
                         variant=variant, rule=c["rule"], train=N, obs_per_cell=N * pairs / cells,
                         grokked=tg is not None, delay=(tg - tf) if tg is not None and tf is not None else np.nan))
    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("끝난 run이 없습니다")
    r["label"] = (r.op + " · 쌍 " + r.pairs.astype(str) + " · " + np.where(r.connected, "연결", "끊김")
                  + " · n" + r.n.astype(str) + " · " + r.variant)
    t = r.groupby(["label", "n", "op", "pairs", "variant", "train"], as_index=False).agg(
        obs_per_cell=("obs_per_cell", "first"), runs=("grokked", "size"), grokked=("grokked", "sum"),
        delay_median=("delay", "median"))
    t["grok_rate"] = t.grokked / t.runs
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="by_graph.csv 하나, 또는 run 폴더 루트들(glob 가능)")
    ap.add_argument("--out", default="obs_report")
    a = ap.parse_args()
    if len(a.inputs) == 1 and a.inputs[0].endswith(".csv"):
        t = from_csv(a.inputs[0])
        png = a.out if a.out.endswith(".png") else a.out + ".png"
    else:
        dirs = [p for pat in a.inputs for root in glob.glob(pat) for p in glob.glob(f"{root}/*")
                if Path(p, "summary.json").exists()]
        t = from_runs(dirs)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        t.to_csv(f"{a.out}/obs_table.csv", index=False)
        png = f"{a.out}/obs_curve.png"
    t = t.sort_values(["label", "obs_per_cell"])
    cols = [c for c in ["label", "train", "obs_per_cell", "runs", "grok_rate", "delay_median"] if c in t]
    print(t[cols].round(2).to_string(index=False))
    g = t[t.grok_rate >= 0.9]
    if len(g) > 2:
        pred = PRED_A - PRED_B * g.obs_per_cell
        err = np.log10(g.delay_median.clip(lower=50)) - pred
        print(f"\n일반화율 0.9 이상 칸 {len(g)}개: log10 지연과 관찰 수 상관 "
              f"{np.corrcoef(g.obs_per_cell, np.log10(g.delay_median.clip(lower=50)))[0, 1]:.2f}, "
              f"사전 예측 대비 오차 log10 평균 {err.mean():+.2f} (배수 {10 ** err.mean():.2f}), 표준편차 {err.std():.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for lab, s in t.groupby("label"):
        mk = "o" if s.op.iloc[0] == "a+b" else "s"
        name = (lab.replace("쌍", "pairs").replace("연결", "conn").replace("끊김", "split"))
        ax[0].plot(s.obs_per_cell, s.grok_rate, mk + "-", label=name, alpha=.8)
        ax[1].plot(s.obs_per_cell, s.delay_median, mk + "-", label=name, alpha=.8)
    xs = np.linspace(t.obs_per_cell.min(), t.obs_per_cell.max(), 50)
    ax[1].plot(xs, 10 ** (PRED_A - PRED_B * xs), "k--", lw=1, label="prediction (fit to E6+E7)")
    ax[0].set_ylabel("fraction of runs generalized")
    ax[1].set_ylabel("delay (median steps)"); ax[1].set_yscale("log")
    for x in ax:
        x.set_xlabel("observations per table cell"); x.grid(alpha=.3); x.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(png, dpi=120)
    print("저장:", png)


if __name__ == "__main__":
    main()
