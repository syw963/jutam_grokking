"""E11 내부 분석: --ckpt-log로 저장한 체크포인트(ckpt_XXXXXXXX.pt)를 step 순서로 읽어 모델 안을 본다.

체크포인트마다 재는 것:
  1. 정확도: train/test 완전 일치, test 토큰 정확도를
     "train에 나온 연산 표 칸"(seen)과 "train에 한 번도 안 나온 칸"(unseen)으로 나눠서.
     unseen 칸을 맞히기 시작하는 시점 = 연산(덧셈 mod 50)을 규칙으로 배운 시점.
     칸 = (자기 값, 참조값). a+b는 무순서 쌍, a+3b 등은 순서쌍.
  2. 임베딩 푸리에 구조: 입력 임베딩(50 × d)과 출력층(50 × d)을 값 축으로 이산 푸리에 변환.
     상위 5개 주파수가 차지하는 비율(top5). 무작위면 약 0.2, 몇 개 주파수로 모이면(원형 표현) 1에 가까워진다.
  3. attention: 출력 칸 i의 query가 자기 칸 i와 참조 칸 ref(i)에 주는 attention 비율 (층별).

예:
    python analyze_internal.py "runs_e11/*" --out e11_internal
    → e11_internal/internal.csv (run × step), summary.csv (run별 요약), <run>.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

from src.data import compute_targets, load_split
from src.model import build_model
from src.rules import Rule, ref_pairs


def cell_keys(rule, xs):
    """각 행·각 출력 칸의 연산 표 칸 key. a+b(교환)는 무순서."""
    st = rule.steps[0]
    comm = st.p("op") == "add"
    out = []
    for row in xs.tolist():
        ps = ref_pairs(st, row, rule.n)
        out.append([tuple(sorted(p)) if comm else tuple(p) for p in ps])
    return out


def ref_positions(rule, m):
    st = rule.steps[0]
    ps = ref_pairs(st, list(range(1, m + 1)), rule.n)
    return [b - 1 for _, b in ps]


def fourier_top(w, k=5):
    """w: (n, d). 값 축 푸리에 power에서 0이 아닌 주파수 상위 k개 비율과 상위 주파수."""
    w = w - w.mean(0, keepdims=True)
    f = np.fft.rfft(w, axis=0)                     # (n//2+1, d)
    p = (np.abs(f) ** 2).sum(1)[1:]                # 주파수 1..n/2
    order = np.argsort(p)[::-1]
    return float(p[order[:k]].sum() / p.sum()), [int(i + 1) for i in order[:k]]


@torch.no_grad()
def analyze_run(run_dir: Path, device, n_test=4096):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpts = sorted(run_dir.glob("ckpt_[0-9]*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not ckpts:
        return []
    rule = Rule(cfg["rule_spec"], cfg["n"])
    if len(rule.steps) != 1 or rule.steps[0].code != "REF":
        print(f"건너뜀 (REF 한 단계 아님): {run_dir.name}")
        return []
    n, m = cfg["n"], cfg["m"]
    split = load_split(cfg["data"])
    xtr, xte = split["train"], split["test"][:n_test]
    ytr, yte = compute_targets(rule, xtr), compute_targets(rule, xte)
    seen = {k for row in cell_keys(rule, xtr) for k in row}
    te_keys = cell_keys(rule, xte)
    seen_mask = torch.tensor([[k in seen for k in row] for row in te_keys], device=device)
    refs = ref_positions(rule, m)
    to_t = lambda a: torch.as_tensor(a.astype(np.int64) - 1, device=device)
    Xtr, Ytr, Xte, Yte = to_t(xtr), to_t(ytr), to_t(xte), to_t(yte)
    tie = None if cfg.get("tie", "auto") == "auto" else cfg["tie"] == "yes"
    model = build_model(cfg["model"], n, m, cfg["d"], cfg["layers"], cfg["heads"], tie, cfg["init_std"]).to(device)
    rows = []
    for cp in ckpts:
        sd = torch.load(cp, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        model.eval()
        step = int(sd["step"])
        r = dict(run=run_dir.name, rule=cfg["rule"], train=cfg["data_meta"]["train_sets"], seed=cfg["seed"], step=step)
        r["train_exact"] = float((model.predict(Xtr) == Ytr).all(-1).float().mean())
        pte = model.predict(Xte)
        ok = pte == Yte
        r["test_exact"] = float(ok.all(-1).float().mean())
        r["acc_seen"] = float(ok[seen_mask].float().mean()) if seen_mask.any() else float("nan")
        r["acc_unseen"] = float(ok[~seen_mask].float().mean()) if (~seen_mask).any() else float("nan")
        r["unseen_frac"] = float((~seen_mask).float().mean())
        if hasattr(model, "tok_emb") and cfg["model"] != "gpt":
            r["emb_top5"], fe = fourier_top(model.tok_emb.weight.float().cpu().numpy())
            r["head_top5"], fh = fourier_top(model.head.weight.float().cpu().numpy())
            r["emb_freqs"], r["head_freqs"] = " ".join(map(str, fe)), " ".join(map(str, fh))
            # attention: query 행(m..2m-1)이 입력 칸에 주는 비율
            model.hidden(Xte[:1024], record_attn=True)
            for li, b in enumerate(model.blocks):
                a = b.attn.last_attn.float().mean(1)            # (B, T, T) 머리 평균
                q = a[:, m:, :m]                                 # 출력 query → 입력 칸
                self_m = torch.stack([q[:, i, i] for i in range(m)], 1).mean()
                ref_m = torch.stack([q[:, i, refs[i]] for i in range(m) if refs[i] != i], 1).mean() \
                    if any(refs[i] != i for i in range(m)) else torch.tensor(float("nan"))
                r[f"L{li}_self"], r[f"L{li}_ref"] = float(self_m), float(ref_m)
                r[f"L{li}_input"] = float(q.sum(-1).mean())      # query가 입력 칸 전체에 주는 비율
        rows.append(r)
    return rows


def plot_run(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [r["step"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in ("train_exact", "test_exact", "acc_seen", "acc_unseen"):
        ax[0].plot(s, [r[k] for r in rows], marker=".", label=k)
    ax[0].set_title("accuracy (seen / unseen table cells)")
    for k in ("emb_top5", "head_top5"):
        if k in rows[0]:
            ax[1].plot(s, [r[k] for r in rows], marker=".", label=k)
    ax[1].set_title("Fourier top-5 power fraction")
    for k in ("L0_self", "L0_ref", "L1_self", "L1_ref"):
        if k in rows[0]:
            ax[2].plot(s, [r[k] for r in rows], marker=".", label=k)
    ax[2].set_title("attention: output query -> self / ref input")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(rows[0]["run"])
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def summarize(rows):
    """run별: 일반화 직전·직후 비교."""
    out = []
    by = {}
    for r in rows:
        by.setdefault(r["run"], []).append(r)
    for run, rs in by.items():
        rs.sort(key=lambda r: r["step"])
        last = rs[-1]
        gen = next((r["step"] for r in rs if r["test_exact"] >= 0.95), None)
        unseen50 = next((r["step"] for r in rs if r["acc_unseen"] >= 0.5), None)
        four = next((r["step"] for r in rs if r.get("emb_top5", 0) >= 0.5), None)
        out.append(dict(run=run, rule=last["rule"], train=last["train"], seed=last["seed"],
                        unseen_frac=round(last["unseen_frac"], 3), ckpt_gen=gen, ckpt_unseen50=unseen50,
                        ckpt_emb_fourier50=four, final_acc_seen=round(last["acc_seen"], 3),
                        final_acc_unseen=round(last["acc_unseen"], 3),
                        final_emb_top5=round(last.get("emb_top5", float("nan")), 3),
                        final_emb_freqs=last.get("emb_freqs", ""),
                        final_L_ref=round(max(last.get("L0_ref", 0), last.get("L1_ref", 0)), 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run 폴더 (glob 가능)")
    ap.add_argument("--out", default="internal_report")
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if a.device == "auto" else a.device))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dirs = sorted({Path(p) for g in a.runs for p in glob.glob(g) if (Path(p) / "config.json").exists()})
    allrows = []
    for d in dirs:
        rows = analyze_run(d, device, a.n_test)
        if not rows:
            continue
        allrows += rows
        try:
            plot_run(rows, out / f"{d.name}.png")
        except Exception as e:  # matplotlib 없음 등
            print("그림 실패:", e)
        last = rows[-1]
        print(f"{d.name:34s} 체크포인트 {len(rows):2d}개  최종 test {last['test_exact']:.3f}  "
              f"seen {last['acc_seen']:.3f}  unseen {last['acc_unseen']:.3f}  emb top5 {last.get('emb_top5', float('nan')):.2f}")
    if not allrows:
        raise SystemExit("체크포인트가 있는 run이 없습니다 (--ckpt-log로 학습한 run 폴더를 주세요)")
    keys = list(dict.fromkeys(k for r in allrows for k in r))
    with open(out / "internal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(allrows)
    summ = summarize(allrows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summ[0])); w.writeheader(); w.writerows(summ)
    print(f"\n저장: {out}/internal.csv, summary.csv, run별 그림")
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
