"""칸당 관찰 수, 관찰된 칸당 관찰 수, 관찰 비율이 지연을 얼마나 설명하는지 비교 (E6, E7, E8, E12, E18).

    python analyze_obs_seen.py --res <결과 tgz를 푼 폴더> --data <무작위 분할 데이터 폴더> --out docs/analysis/obs_seen.txt

- 칸당 관찰 수 = N × 예제당 서로 다른 칸 수 / C
- 관찰 비율 = 학습 집합에 한 번 이상 나온 칸 / C (src.rules.op_coverage와 같은 정의: a+b는 무순서 쌍, a+3b는 순서쌍)
- 관찰된 칸당 관찰 수 = 칸당 관찰 수 / 관찰 비율

무작위 분할은 python -m src.data --train-count N --test-count 20000 --seed 0 으로 재생성해 관찰 비율을 다시 잰다.
E6(10월 19일) 학습 때 summary.json에 기록된 op_coverage는 a+b에서도 순서쌍 / n²(옛 코드)이라 쓰지 않는다.
관찰 비율을 고른 분할(E8, E12B, E18)은 결과에 들어 있는 data/*/meta.json의 curated.coverage를 쓴다.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.rules import Rule, op_coverage, ref_graph


def load(res: Path, data: Path) -> pd.DataFrame:
    cache, rows = {}, []
    for d in glob.glob(str(res / "*/runs_*/*")):
        d = Path(d)
        try:
            s = json.loads((d / "summary.json").read_text())
            c = json.loads((d / "config.json").read_text())
        except (FileNotFoundError, NotADirectoryError):
            continue
        if not c["rule"].startswith("w") or c.get("wd", 1.0) != 1.0 or c["d"] != 128:
            continue
        n = c["n"]
        rule = Rule(c["rule_spec"], n)
        g = ref_graph(rule, c["m"])
        comm = rule.steps[0].p("op") in ("add", "mul")
        pairs = g["edges_unordered"] if comm else g["edges_ordered"]
        C = n * (n + 1) / 2 if comm else n * n
        N = c["data_meta"]["train_sets"]
        dn = Path(c["data"]).name
        meta_p = list(res.glob(f"*/data/{dn}/meta.json"))
        meta = json.loads(meta_p[0].read_text()) if meta_p else None
        curated = bool(meta and "curated" in meta)
        if curated:
            if meta["curated"]["rule"] != c["rule_spec"]:
                continue
            cov = meta["curated"]["coverage"]
        else:
            if not (data / dn).exists():
                continue
            key = (c["rule_spec"], dn)
            if key not in cache:
                cache[key] = op_coverage(rule, np.load(data / dn / "train.npy").tolist())
            cov = cache[key]
        tf, tg = s.get("t_fit"), s.get("t_gen")
        rows.append(dict(exp=d.parent.parent.name.split("_")[0], rule=c["rule"], op="a+b" if comm else "a+3b",
                         data=dn, curated=curated, N=N, opc=N * pairs / C, cov=cov, grok=tg is not None,
                         delay=(tg - tf) if tg is not None and tf is not None else np.nan))
    r = pd.DataFrame(rows)
    r["opc_seen"] = r["opc"] / r["cov"]
    return r.groupby(["exp", "rule", "op", "data", "curated", "N"], as_index=False).agg(
        runs=("grok", "size"), rate=("grok", "mean"), delay=("delay", "median"),
        opc=("opc", "first"), cov=("cov", "first"), opc_seen=("opc_seen", "first"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = load(Path(a.res), Path(a.data))
    lines = [f"조건 {len(t)}개 (실험별 {t.groupby('exp').size().to_dict()}), 학습 {int(t.runs.sum())}회", ""]
    g = t[t.rate >= 0.9].copy()
    g["y"] = np.log10(g.delay.clip(lower=50))
    cur = g.curated | g.exp.isin(["E08", "E18"])

    def corr(sub, name):
        c = {k: np.corrcoef(sub[k], sub.y)[0, 1] for k in ("opc", "opc_seen", "cov")}
        lines.append(f"[{name}] 조건 {len(sub)}개(일반화율 0.9 이상), log10 지연과의 상관: "
                     f"칸당 관찰 수 {c['opc']:+.2f}, 관찰된 칸당 관찰 수 {c['opc_seen']:+.2f}, 관찰 비율 {c['cov']:+.2f}")

    corr(g[g.exp.isin(["E06", "E07"])], "E6+E7")
    corr(g[~cur], "무작위 분할 전체(E6, E7, E12A)")
    corr(g, "관찰 비율 조절 포함 전체")
    lines.append("")
    tr, te = g[~cur], g[cur]
    lines.append(f"무작위 분할 {len(tr)}개 조건으로 식을 세우고 관찰 비율 조절 실험 {len(te)}개 조건의 지연을 예측:")
    for k, name in (("opc", "칸당 관찰 수"), ("opc_seen", "관찰된 칸당 관찰 수")):
        b, a0 = np.polyfit(tr[k], tr.y, 1)
        e = te.y - (a0 + b * te[k])
        lines.append(f"  {name}: log10 지연 = {a0:.2f} {b:+.3f} x (지연 절반 {np.log10(2) / -b:.2f}회) → "
                     f"실측/예측 평균 {10 ** e.mean():.2f}배, 로그 표준편차 {e.std():.2f}")
    lines.append("")
    for rule in ("w5_next", "w1_m0"):
        s = te[(te.rule == rule) & (te.N == 768)]
        b, _ = np.polyfit(s.opc_seen, s.y, 1)
        lines.append(f"관찰 비율 실험 {rule} N=768 ({len(s)}개 조건, 칸당 관찰 수 {s.opc.iloc[0]:.2f}로 고정): "
                     f"관찰된 칸당 관찰 수와 상관 {np.corrcoef(s.opc_seen, s.y)[0, 1]:+.2f}, 지연 절반 {np.log10(2) / -b:.2f}회")
        lines.append(s.sort_values("cov")[["cov", "opc_seen", "rate", "delay"]].round(3).to_string(index=False))
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
