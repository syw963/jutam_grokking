"""E39 오답 실험 내부 분석: 외운 오답은 모델 안 어디에 있는가.

--ckpt-log로 저장한 체크포인트를 step 순서로 읽고, 오답이 섞인 train 자리(noisy)와 처음 보는 문제(test)에서 잰다.

1. 층별 읽기 (logit lens): 잔차 흐름을 다섯 지점(임베딩, 0층 attention 뒤, 0층 MLP 뒤, 1층 attention 뒤, 1층 MLP 뒤)에서
   최종 LayerNorm + 출력층으로 바로 읽는다. 오답 자리에서 각 지점이 외운 오답을 1위로 내는 비율, 원래 정답을 1위로 내는 비율,
   (오답 logit − 정답 logit) 평균. test에서는 지점별 토큰 정확도.
   → "앞쪽은 규칙(정답)을 계산하고 뒤쪽 어딘가에서 오답으로 덮어쓰는가"를 본다.
2. 부품 끄기 (평균 대체 ablation): attention 0·1, MLP 0·1의 출력을 train 평균으로 바꾸고
   외운 오답 비율 / 원래 정답 비율 / test 정확도 / 오답 없는 train 자리 정확도를 다시 잰다.
3. 오답 담당 뉴런: 1층 MLP의 뉴런마다 오답 자리에서 (오답 − 정답) logit에 직접 더하는 양을 계산해(직접 기여, 최종 LN 배율 고정)
   상위 k개(8·16·32·64)를 평균으로 바꾼다. 같은 수의 무작위 뉴런을 끈 것과 비교.
   → 소수의 뉴런만 꺼도 외운 오답이 사라지고 test 정확도는 그대로면, 암기는 규칙과 따로 저장되어 있다.

예:
    python analyze_noise_internal.py "runs_e39_*/*" --out e39_report
    → e39_report/lens.csv, ablate.csv, neurons.csv, summary.txt, run별 그림
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
from src.noise import make_noise
from src.rules import Rule

POINTS = ["emb", "L0_attn", "L0_mlp", "L1_attn", "L1_mlp"]
COMPS = ["attn0", "mlp0", "attn1", "mlp1"]


def forward(model, x, ablate=None, means=None, neuron_mask=None, neuron_mean=None, record=False):
    """ParallelModel(m0) 직접 전개. ablate: 평균으로 바꿀 부품 이름 집합.
    neuron_mask: (4d,) bool, 1층 MLP에서 평균으로 바꿀 뉴런. record=True면 지점별 잔차와 1층 MLP 활성값을 돌려준다."""
    m = model.m
    h = model.tok_emb(x) + model.pos_emb.weight
    h = torch.cat([h, model.queries.expand(x.size(0), -1, -1)], dim=1)
    pts, acts = {"emb": h}, {}
    for li, b in enumerate(model.blocks):
        a = b.attn(b.ln1(h), False)
        if ablate and f"attn{li}" in ablate:
            a = means[f"attn{li}"].expand_as(a)
        h = h + a
        pts[f"L{li}_attn"] = h
        z = b.ln2(h)
        g = b.mlp[1](b.mlp[0](z))                      # (B, T, 4d) GELU 뒤 활성값
        if li == len(model.blocks) - 1:
            acts["g"] = g
            if neuron_mask is not None:
                g = torch.where(neuron_mask, neuron_mean.expand_as(g), g)
        o = b.mlp[2](g)
        if ablate and f"mlp{li}" in ablate:
            o = means[f"mlp{li}"].expand_as(o)
        h = h + o
        pts[f"L{li}_mlp"] = h
    logits = model.head(model.ln_f(h[:, m:]))
    if record:
        return logits, pts, acts
    return logits


@torch.no_grad()
def component_means(model, x):
    """부품 출력의 train 평균 (자리별). ablation 기준값."""
    m = model.m
    h = model.tok_emb(x) + model.pos_emb.weight
    h = torch.cat([h, model.queries.expand(x.size(0), -1, -1)], dim=1)
    out = {}
    for li, b in enumerate(model.blocks):
        a = b.attn(b.ln1(h), False)
        out[f"attn{li}"] = a.mean(0, keepdim=True)
        h = h + a
        g = b.mlp[1](b.mlp[0](b.ln2(h)))
        o = b.mlp[2](g)
        out[f"mlp{li}"] = o.mean(0, keepdim=True)
        out[f"g{li}"] = g.mean(0, keepdim=True)
        h = h + o
    return out


def lens_logits(model, h):
    return model.head(model.ln_f(h[:, model.m:]))


def noisy_stats(logits, rows, pos, y_true, y_noisy):
    """오답 자리에서 (외운 오답 1위 비율, 정답 1위 비율, 오답−정답 logit 평균, 정답 순위 중앙값)."""
    lg = logits[rows, pos]                                   # (K, n)
    pred = lg.argmax(-1)
    t, w = y_true[rows, pos], y_noisy[rows, pos]
    diff = lg.gather(1, w[:, None])[:, 0] - lg.gather(1, t[:, None])[:, 0]
    rank_t = (lg > lg.gather(1, t[:, None])).sum(1) + 1
    return (float((pred == w).float().mean()), float((pred == t).float().mean()),
            float(diff.mean()), float(rank_t.float().median()))


@torch.no_grad()
def analyze_run(run_dir: Path, device, n_test=4096, n_clean=4096, every=1, seed=0):
    cfg = json.loads((run_dir / "config.json").read_text())
    if not cfg.get("noise_frac"):
        return None
    ckpts = sorted(run_dir.glob("ckpt_[0-9]*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not ckpts:
        return None
    ckpts = ckpts[::every] if every > 1 else ckpts
    rule = Rule(cfg["rule_spec"], cfg["n"])
    n, m = cfg["n"], cfg["m"]
    split = load_split(cfg["data"])
    xtr = split["train"]
    ytr = compute_targets(rule, xtr).astype(np.int64) - 1
    nz = make_noise(rule, xtr, ytr, cfg["noise_frac"], cfg["noise_mode"], cfg["noise_seed"])
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
    Xtr, Ytr, Ynz = T(xtr - 1), T(ytr), T(nz["y_noisy"])
    mask = torch.as_tensor(nz["mask"], device=device)
    rows, pos = mask.nonzero(as_tuple=True)
    clean_rows = (~mask.any(1)).nonzero()[:, 0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    clean_rows = clean_rows[torch.randperm(len(clean_rows), generator=g)[:n_clean].to(device)]
    xte = split["test"][:n_test]
    Xte = T(xte - 1)
    Yte = T(compute_targets(rule, xte).astype(np.int64) - 1)
    tie = None if cfg.get("tie", "auto") == "auto" else cfg["tie"] == "yes"
    model = build_model(cfg["model"], n, m, cfg["d"], cfg["layers"], cfg["heads"], tie, cfg["init_std"]).to(device).eval()
    base = dict(run=run_dir.name, mode=cfg["noise_mode"], frac=cfg["noise_frac"], wd=cfg["wd"], seed=cfg["seed"],
                train=cfg["data_meta"]["train_sets"], noisy_tokens=int(mask.sum()))
    lens_rows, abl_rows, neu_rows = [], [], []

    def measure(**kw):
        lt = forward(model, Xtr, **kw)
        mem, fix, diff, rk = noisy_stats(lt, rows, pos, Ytr, Ynz)
        clean = float((lt[clean_rows].argmax(-1) == Ytr[clean_rows]).float().mean())
        te = float((forward(model, Xte, **kw).argmax(-1) == Yte).float().mean())
        return dict(mem=mem, fixed=fix, diff=diff, true_rank=rk, clean_acc=clean, test_acc=te)

    for cp in ckpts:
        sd = torch.load(cp, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        step = int(sd["step"])
        # 1. 층별 읽기
        lt, pts, acts = forward(model, Xtr, record=True)
        _, pts_te, _ = forward(model, Xte, record=True)
        for p in POINTS:
            mem, fix, diff, rk = noisy_stats(lens_logits(model, pts[p]), rows, pos, Ytr, Ynz)
            te = float((lens_logits(model, pts_te[p]).argmax(-1) == Yte).float().mean())
            lens_rows.append(dict(base, step=step, point=p, mem=mem, fixed=fix, diff=diff, true_rank=rk, test_acc=te))
        # 2. 부품 끄기
        means = component_means(model, Xtr)
        abl_rows.append(dict(base, step=step, ablate="none", **measure()))
        for c in COMPS:
            abl_rows.append(dict(base, step=step, ablate=c, **measure(ablate={c}, means=means)))
        # 3. 오답 담당 뉴런 (1층 MLP, 직접 기여)
        L = len(model.blocks) - 1
        gq = acts["g"][:, m:]                                     # (N, m, 4d)
        gmean = means[f"g{L}"]                                    # (1, T, 4d)
        hq = pts[f"L{L}_mlp"][:, m:]
        sigma = hq.var(-1, unbiased=False).add(model.ln_f.eps).sqrt()   # (N, m) 최종 LN 배율
        Wout = model.blocks[L].mlp[2].weight                      # (d, 4d)
        Wh = model.head.weight                                    # (n, d)
        dvec = (Wh[Ynz[rows, pos]] - Wh[Ytr[rows, pos]]) * model.ln_f.weight   # (K, d)
        per = (dvec @ Wout)                                       # (K, 4d): 뉴런 1단위가 (오답−정답)에 주는 양
        dg = gq[rows, pos] - gmean[0, m:][pos]                    # 평균 대비 활성 차이
        score = (dg * per / sigma[rows, pos][:, None]).mean(0)    # (4d,)
        order = torch.argsort(score, descending=True)
        width = score.numel()
        neu_rows.append(dict(base, step=step, k=0, kind="none", top_score=float(score.max()),
                             score_share_top16=float(score[order[:16]].clamp(min=0).sum() / score.clamp(min=0).sum().clamp(min=1e-9)),
                             **measure()))
        for k in (8, 16, 32, 64):
            nm = torch.zeros(width, dtype=torch.bool, device=device)
            nm[order[:k]] = True
            r1 = measure(neuron_mask=nm, neuron_mean=gmean)
            nr = torch.zeros(width, dtype=torch.bool, device=device)
            nr[torch.randperm(width, generator=g)[:k].to(device)] = True
            r2 = measure(neuron_mask=nr, neuron_mean=gmean)
            neu_rows.append(dict(base, step=step, k=k, kind="top", top_score=float(score.max()), score_share_top16=float("nan"), **r1))
            neu_rows.append(dict(base, step=step, k=k, kind="random", top_score=float(score.max()), score_share_top16=float("nan"), **r2))
    return lens_rows, abl_rows, neu_rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def plot_run(run, lens, abl, neu, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    steps = sorted({r["step"] for r in abl})
    base = {r["step"]: r for r in abl if r["ablate"] == "none"}
    ax[0].plot(steps, [base[s]["mem"] for s in steps], label="memorized wrong (output)")
    ax[0].plot(steps, [base[s]["fixed"] for s in steps], label="original answer (output)")
    ax[0].plot(steps, [base[s]["test_acc"] for s in steps], label="test token acc")
    ax[0].set_title("outputs over training"); ax[0].set_xscale("symlog", linthresh=100)
    last = max(steps)
    L = [r for r in lens if r["step"] == last]
    xs = range(len(POINTS))
    ax[1].plot(xs, [next(r for r in L if r["point"] == p)["mem"] for p in POINTS], marker="o", label="top-1 = memorized wrong")
    ax[1].plot(xs, [next(r for r in L if r["point"] == p)["fixed"] for p in POINTS], marker="o", label="top-1 = original answer")
    ax[1].plot(xs, [next(r for r in L if r["point"] == p)["test_acc"] for p in POINTS], marker="o", label="test acc")
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(POINTS, rotation=20); ax[1].set_title(f"logit lens at noisy positions (step {last})")
    A = [r for r in abl if r["step"] == last]
    names = ["none"] + COMPS
    xb = np.arange(len(names))
    ax[2].bar(xb - 0.2, [next(r for r in A if r["ablate"] == c)["mem"] for c in names], 0.4, label="memorized wrong")
    ax[2].bar(xb + 0.2, [next(r for r in A if r["ablate"] == c)["test_acc"] for c in names], 0.4, label="test acc")
    ax[2].set_xticks(xb); ax[2].set_xticklabels(names); ax[2].set_title("mean-ablate a component")
    N = [r for r in neu if r["step"] == last and r["k"] > 0]
    ks = sorted({r["k"] for r in N})
    for kind, ls in (("top", "-"), ("random", ":")):
        ax[3].plot(ks, [next(r for r in N if r["k"] == k and r["kind"] == kind)["mem"] for k in ks], ls, marker="o", label=f"mem ({kind} neurons)")
        ax[3].plot(ks, [next(r for r in N if r["k"] == k and r["kind"] == kind)["test_acc"] for k in ks], ls, marker="s", label=f"test ({kind})")
    ax[3].set_xscale("log", base=2); ax[3].set_title("ablate k MLP1 neurons (of 512)")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=.3); a.set_ylim(-0.02, 1.02)
    fig.suptitle(run)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def summarize(lens, abl, neu):
    lines = []
    runs = sorted({r["run"] for r in abl})
    for run in runs:
        A = [r for r in abl if r["run"] == run]
        last = max(r["step"] for r in A)
        b = next(r for r in A if r["step"] == last and r["ablate"] == "none")
        lines.append(f"\n## {run}  (mode {b['mode']}, wd {b['wd']:g}, seed {b['seed']}, 오답 토큰 {b['noisy_tokens']}, 마지막 step {last})")
        lines.append(f"출력: 외운 오답 {b['mem']:.3f}, 원래 정답 {b['fixed']:.3f}, 정답 순위 중앙값 {b['true_rank']:.0f}, "
                     f"오답 없는 train {b['clean_acc']:.3f}, test {b['test_acc']:.3f}")
        L = {r["point"]: r for r in lens if r["run"] == run and r["step"] == last}
        lines.append("층별 읽기 (지점: 외운 오답 1위 / 원래 정답 1위 / 오답−정답 logit / test 정확도):")
        for p in POINTS:
            r = L[p]
            lines.append(f"  {p:8s} {r['mem']:.3f} / {r['fixed']:.3f} / {r['diff']:+.2f} / {r['test_acc']:.3f}")
        lines.append("부품 끄기 (외운 오답 / 원래 정답 / 오답 없는 train / test):")
        for c in COMPS:
            r = next(x for x in A if x["step"] == last and x["ablate"] == c)
            lines.append(f"  {c:6s} {r['mem']:.3f} / {r['fixed']:.3f} / {r['clean_acc']:.3f} / {r['test_acc']:.3f}")
        N = [r for r in neu if r["run"] == run and r["step"] == last]
        lines.append("1층 MLP 뉴런 k개 끄기 (오답 담당 상위 vs 무작위: 외운 오답 / test):")
        for k in (8, 16, 32, 64):
            t = next(r for r in N if r["k"] == k and r["kind"] == "top")
            z = next(r for r in N if r["k"] == k and r["kind"] == "random")
            lines.append(f"  k={k:3d}  상위 {t['mem']:.3f} / {t['test_acc']:.3f}   무작위 {z['mem']:.3f} / {z['test_acc']:.3f}")
        # 시간 변화: 마지막 지점 직전(L1_attn)이 이미 오답을 내는가
        steps = sorted({r["step"] for r in lens if r["run"] == run})
        tr = [(s, next(r for r in lens if r["run"] == run and r["step"] == s and r["point"] == "L1_attn"),
               next(r for r in lens if r["run"] == run and r["step"] == s and r["point"] == "L1_mlp")) for s in steps]
        pick = [t for t in tr if t[0] in steps[:: max(1, len(steps) // 8)]] + [tr[-1]]
        lines.append("시간 변화 (step: 1층 MLP 전 외운 오답/원래 정답 → 출력 외운 오답/원래 정답):")
        for s, a, z in pick:
            lines.append(f"  {s:7d}: {a['mem']:.2f}/{a['fixed']:.2f} → {z['mem']:.2f}/{z['fixed']:.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="noise_internal_report")
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--every", type=int, default=1, help="체크포인트 k개 중 하나만")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if a.device == "auto" else a.device))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dirs = sorted({Path(p) for g in a.runs for p in glob.glob(g) if (Path(p) / "config.json").exists()})
    L, A, N = [], [], []
    for d in dirs:
        r = analyze_run(d, device, a.n_test, every=a.every)
        if not r:
            continue
        L += r[0]; A += r[1]; N += r[2]
        try:
            plot_run(d.name, r[0], r[1], r[2], out / f"{d.name}.png")
        except Exception as e:
            print("그림 실패:", e)
        print("완료:", d.name, "체크포인트", len({x['step'] for x in r[1]}))
    if not A:
        raise SystemExit("오답 run + 체크포인트가 없습니다")
    write_csv(out / "lens.csv", L); write_csv(out / "ablate.csv", A); write_csv(out / "neurons.csv", N)
    s = summarize(L, A, N)
    (out / "summary.txt").write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
