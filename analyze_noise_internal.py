"""E39 오답 실험 내부 분석: 외운 오답은 모델 안 어디에 있는가.

--ckpt-log로 저장한 체크포인트를 step 순서로 읽는다. 오답이 섞인 train 자리(noisy)를 무작위로 반씩 A·B로 나눈다
(A로 뉴런을 고르고 B에서 효과를 잰다: 고른 자리에서 재면 효과가 부풀려지므로).

1. 층별 읽기 (logit lens): 잔차 흐름을 다섯 지점(임베딩, 0층 attention 뒤, 0층 MLP 뒤, 1층 attention 뒤, 1층 MLP 뒤)에서
   최종 LayerNorm + 출력층으로 바로 읽는다. 오답 자리에서 지점별 외운 오답 1위 비율, 원래 정답 1위 비율, (오답 − 정답) logit,
   test 토큰 정확도. 오답 없는 대조 run은 test 정확도만.
2. 부품 끄기 (평균 대체): attention 0·1, MLP 0·1 출력을 train 평균으로 바꿔 외운 오답 / 원래 정답 / 오답 없는 train / test를 다시 잰다.
3. 오답 담당 뉴런 (두 MLP 층의 뉴런 1,024개 전체): 기울기 × (활성 − 평균) 기여도로 A 자리의 (오답 − 정답) logit을 키우는 뉴런을 고르고
   상위 k개(4·8·16·32·64·128)를 평균으로 바꾼다. B 자리(고르지 않은 오답)에서 외운 오답, 오답 없는 train, test를 잰다.
   대조: 같은 수의 무작위 뉴런, 그리고 test 정답 logit 기여 상위 뉴런("규칙 담당")을 끈 경우.
4. 오답 담당 뉴런의 크기: 상위 32개 뉴런의 가중치 노름(입력·출력 벡터) / 전체 뉴런 노름 중앙값.
   weight decay가 오답을 고칠 때 이 뉴런들이 줄어드는지 본다.
5. cell 오답: 오답 칸이 나오는 test 자리에서 오답이 번진 비율과, 오답 담당 뉴런을 끄면 원래 정답으로 돌아오는 비율.

예:
    python analyze_noise_internal.py "runs_e39_*/*" --out e39_report/internal
    python analyze_noise_internal.py --merge e39_report/internal_* --out e39_report/internal   (나눠 돌린 결과 합치기)
→ lens.csv, ablate.csv, neurons.csv, sizes.csv, summary.txt(조건별 seed 평균), run별 그림
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data import compute_targets, load_split
from src.model import build_model
from src.noise import make_noise, test_cells
from src.rules import Rule

POINTS = ["emb", "L0_attn", "L0_mlp", "L1_attn", "L1_mlp"]
COMPS = ["attn0", "mlp0", "attn1", "mlp1"]
KS = (4, 8, 16, 32, 64, 128)


def forward(model, x, ablate=None, means=None, nmask=None, record=False, grad_g=False):
    """ParallelModel(m0) 직접 전개.
    ablate: 평균으로 바꿀 부품 이름 집합. nmask: {층: (4d,) bool} 평균으로 바꿀 MLP 뉴런.
    record=True면 (logits, 지점별 잔차, 층별 MLP 활성값) 반환. grad_g=True면 활성값에 기울기를 남긴다."""
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
        g = b.mlp[1](b.mlp[0](b.ln2(h)))                 # (B, T, 4d) GELU 뒤
        if grad_g:
            g.retain_grad()
        acts[li] = g
        if nmask is not None and li in nmask and nmask[li].any():
            g = torch.where(nmask[li], means[f"g{li}"].expand_as(g), g)
        o = b.mlp[2](g)
        if ablate and f"mlp{li}" in ablate:
            o = means[f"mlp{li}"].expand_as(o)
        h = h + o
        pts[f"L{li}_mlp"] = h
    logits = model.head(model.ln_f(h[:, m:]))
    return (logits, pts, acts) if record else logits


@torch.no_grad()
def component_means(model, x):
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
    if len(rows) == 0:
        return dict(mem=float("nan"), fixed=float("nan"), diff=float("nan"), true_rank=float("nan"))
    lg = logits[rows, pos]
    pred = lg.argmax(-1)
    t, w = y_true[rows, pos], y_noisy[rows, pos]
    lt, lw = lg.gather(1, t[:, None])[:, 0], lg.gather(1, w[:, None])[:, 0]
    rank_t = (lg > lt[:, None]).sum(1) + 1
    return dict(mem=float((pred == w).float().mean()), fixed=float((pred == t).float().mean()),
                diff=float((lw - lt).mean()), true_rank=float(rank_t.float().median()))


def attribution(model, x, means, rows, pos, target_hi, target_lo):
    """뉴런별 기여도: 선택 자리에서 (logit[target_hi] − logit[target_lo]) 평균에 대한 Σ 기울기 × (활성 − 평균)."""
    with torch.enable_grad():
        logits, _, acts = forward(model, x, record=True, grad_g=True)
        lg = logits[rows, pos]
        metric = (lg.gather(1, target_hi[:, None]) - lg.gather(1, target_lo[:, None])).mean()
        metric.backward()
    m = model.m
    scores = []
    for li in sorted(acts):
        g = acts[li]
        s = (g.grad * (g - means[f"g{li}"])).detach()[:, m:]       # (N, m, 4d)
        scores.append(s.sum((0, 1)))
    model.zero_grad(set_to_none=True)
    return torch.cat(scores)                                          # (layers × 4d,)


def split_mask(idx, width, n_layers, device):
    out = {}
    for li in range(n_layers):
        mk = torch.zeros(width, dtype=torch.bool, device=device)
        sel = idx[(idx >= li * width) & (idx < (li + 1) * width)] - li * width
        mk[sel] = True
        out[li] = mk
    return out


@torch.no_grad()
def analyze_run(run_dir: Path, device, n_test=4096, n_clean=4096, every=1, seed=0):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpts = sorted(run_dir.glob("ckpt_[0-9]*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not ckpts or cfg.get("model") != "m0":
        return None
    if every > 1:
        ckpts = ckpts[::every] + ([ckpts[-1]] if (len(ckpts) - 1) % every else [])
    rule = Rule(cfg["rule_spec"], cfg["n"])
    n, m = cfg["n"], cfg["m"]
    split = load_split(cfg["data"])
    xtr = split["train"]
    ytr = compute_targets(rule, xtr).astype(np.int64) - 1
    noisy = bool(cfg.get("noise_frac"))
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
    Xtr, Ytr = T(xtr - 1), T(ytr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if noisy:
        nz = make_noise(rule, xtr, ytr, cfg["noise_frac"], cfg["noise_mode"], cfg["noise_seed"])
        Ynz = T(nz["y_noisy"])
        mask = torch.as_tensor(nz["mask"], device=device)
        rows, pos = mask.nonzero(as_tuple=True)
        perm = torch.randperm(len(rows), generator=gen).to(device)
        half = len(rows) // 2
        A, B = perm[:half], perm[half:]
        rA, pA, rB, pB = rows[A], pos[A], rows[B], pos[B]
        clean_rows = (~mask.any(1)).nonzero()[:, 0]
    else:
        Ynz, mask = Ytr, torch.zeros_like(Ytr, dtype=torch.bool)
        rows = pos = rA = pA = rB = pB = torch.zeros(0, dtype=torch.long, device=device)
        clean_rows = torch.arange(len(Ytr), device=device)
    clean_rows = clean_rows[torch.randperm(len(clean_rows), generator=gen)[:n_clean].to(device)]
    xte = split["test"][:n_test]
    Xte = T(xte - 1)
    Yte = T(compute_targets(rule, xte).astype(np.int64) - 1)
    if noisy and cfg["noise_mode"] == "cell":
        te_mask, te_wrong = test_cells(rule, xte, nz["cell_wrong"])
        te_r, te_p = torch.as_tensor(te_mask, device=device).nonzero(as_tuple=True)
        Wte = T(te_wrong)
    else:
        te_r = te_p = torch.zeros(0, dtype=torch.long, device=device)
        Wte = Yte
    te_plain = torch.ones_like(Yte, dtype=torch.bool)
    te_plain[te_r, te_p] = False
    tie = None if cfg.get("tie", "auto") == "auto" else cfg["tie"] == "yes"
    model = build_model(cfg["model"], n, m, cfg["d"], cfg["layers"], cfg["heads"], tie, cfg["init_std"]).to(device).eval()
    base = dict(run=f"{run_dir.parent.name}/{run_dir.name}", mode=cfg.get("noise_mode") if noisy else "none", frac=cfg.get("noise_frac", 0.0),
                wd=cfg["wd"], seed=cfg["seed"], train=cfg["data_meta"]["train_sets"], noisy_tokens=int(mask.sum()))
    nl, width = len(model.blocks), model.blocks[0].mlp[0].out_features
    lens_rows, abl_rows, neu_rows, size_rows = [], [], [], []

    def measure(**kw):
        lt = forward(model, Xtr, **kw)
        r = {f"{k}_B": v for k, v in noisy_stats(lt, rB, pB, Ytr, Ynz).items()}
        r.update({f"{k}_all": v for k, v in noisy_stats(lt, rows, pos, Ytr, Ynz).items() if k in ("mem", "fixed")})
        r["clean_acc"] = float((lt[clean_rows].argmax(-1) == Ytr[clean_rows]).float().mean())
        pe = forward(model, Xte, **kw).argmax(-1)
        r["test_acc"] = float((pe == Yte)[te_plain].float().mean())
        if len(te_r):
            r["spill"] = float((pe[te_r, te_p] == Wte[te_r, te_p]).float().mean())
            r["spill_true"] = float((pe[te_r, te_p] == Yte[te_r, te_p]).float().mean())
        return r

    for cp in ckpts:
        sd = torch.load(cp, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        step = int(sd["step"])
        means = component_means(model, Xtr)
        # 1. 층별 읽기
        _, pts, _ = forward(model, Xtr, record=True)
        _, pts_te, _ = forward(model, Xte, record=True)
        for p in POINTS:
            r = noisy_stats(lens_logits(model, pts[p]), rows, pos, Ytr, Ynz)
            r["test_acc"] = float((lens_logits(model, pts_te[p]).argmax(-1) == Yte)[te_plain].float().mean())
            lens_rows.append(dict(base, step=step, point=p, **r))
        # 2. 부품 끄기
        abl_rows.append(dict(base, step=step, ablate="none", **measure()))
        for c in COMPS:
            abl_rows.append(dict(base, step=step, ablate=c, **measure(ablate={c}, means=means)))
        if not noisy:
            continue
        # 3. 오답 담당 뉴런 (A로 고르고 B에서 잰다) vs 무작위 vs 규칙 담당
        sc_mem = attribution(model, Xtr, means, rA, pA, Ynz[rA, pA], Ytr[rA, pA])
        lt_te = forward(model, Xte)
        runner = lt_te.scatter(-1, Yte[..., None], float("-inf")).argmax(-1)    # test에서 정답 다음 후보
        sub = torch.randperm(Xte.size(0) * m, generator=gen)[:4096].to(device)
        tr_, tp_ = sub // m, sub % m
        sc_rule = attribution(model, Xte, means, tr_, tp_, Yte[tr_, tp_], runner[tr_, tp_])
        o_mem = torch.argsort(sc_mem, descending=True)
        o_rule = torch.argsort(sc_rule, descending=True)
        pos_mem = sc_mem.clamp(min=0)
        neu_rows.append(dict(base, step=step, k=0, kind="none",
                             mem_top32_share=float(pos_mem[o_mem[:32]].sum() / pos_mem.sum().clamp(min=1e-9)),
                             mem_top32_in_L1=float((o_mem[:32] >= width).float().mean()),
                             overlap_mem_rule_top32=len(set(o_mem[:32].tolist()) & set(o_rule[:32].tolist())),
                             **measure()))
        for k in KS:
            for kind, idx in (("mem_top", o_mem[:k]), ("random", torch.randperm(nl * width, generator=gen)[:k].to(device)),
                              ("rule_top", o_rule[:k])):
                neu_rows.append(dict(base, step=step, k=k, kind=kind, **measure(nmask=split_mask(idx, width, nl, device), means=means)))
        # 4. 오답 담당 뉴런 크기
        top = o_mem[:32]
        norms = []
        for li, b in enumerate(model.blocks):
            w_in = b.mlp[0].weight.norm(dim=1)             # (4d,)
            w_out = b.mlp[2].weight.norm(dim=0)            # (4d,)
            norms.append(w_in * w_out)
        norms = torch.cat(norms)
        size_rows.append(dict(base, step=step, top32_norm_ratio=float(norms[top].median() / norms.median()),
                              rule32_norm_ratio=float(norms[o_rule[:32]].median() / norms.median()),
                              total_mlp_norm=float(sum(b.mlp[0].weight.norm() ** 2 + b.mlp[2].weight.norm() ** 2
                                                       for b in model.blocks).sqrt())))
    return lens_rows, abl_rows, neu_rows, size_rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not Path(path).exists():
        return []
    out = []
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            if k in ("run", "mode", "point", "ablate", "kind"):
                continue
            if v == "":
                r[k] = float("nan")
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
        out.append(r)
    return out


def plot_run(run, lens, abl, neu, size, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(25, 4.2))
    steps = sorted({r["step"] for r in abl})
    base = {r["step"]: r for r in abl if r["ablate"] == "none"}
    ax[0].plot(steps, [base[s]["mem_all"] for s in steps], label="memorized wrong")
    ax[0].plot(steps, [base[s]["fixed_all"] for s in steps], label="original answer")
    ax[0].plot(steps, [base[s]["test_acc"] for s in steps], label="test acc")
    if "spill" in base[steps[-1]]:
        ax[0].plot(steps, [base[s].get("spill", np.nan) for s in steps], label="spill to test (cell)")
    ax[0].set_xscale("symlog", linthresh=100); ax[0].set_title("outputs")
    for p, c in zip(POINTS, ["C0", "C1", "C2", "C3", "C4"]):
        ax[1].plot(steps, [next((r["mem"] for r in lens if r["step"] == s and r["point"] == p), np.nan) for s in steps], c, label=f"{p} wrong")
        ax[1].plot(steps, [next((r["fixed"] for r in lens if r["step"] == s and r["point"] == p), np.nan) for s in steps], c + "--")
    ax[1].set_xscale("symlog", linthresh=100); ax[1].set_title("logit lens at noisy pos (solid wrong, dashed original)")
    last = max(steps)
    A = [r for r in abl if r["step"] == last]
    names = ["none"] + COMPS
    xb = np.arange(len(names))
    ax[2].bar(xb - 0.2, [next(r for r in A if r["ablate"] == c)["mem_all"] for c in names], 0.4, label="memorized wrong")
    ax[2].bar(xb + 0.2, [next(r for r in A if r["ablate"] == c)["test_acc"] for c in names], 0.4, label="test acc")
    ax[2].set_xticks(xb); ax[2].set_xticklabels(names); ax[2].set_title(f"mean-ablate component (step {last:g})")
    N = [r for r in neu if r["step"] == last and r["k"] > 0]
    if N:
        for kind, ls in (("mem_top", "-"), ("random", ":"), ("rule_top", "--")):
            ks = sorted({r["k"] for r in N if r["kind"] == kind})
            ax[3].plot(ks, [next(r for r in N if r["k"] == k and r["kind"] == kind)["mem_B"] for k in ks], ls, color="C3", marker="o", label=f"held-out wrong ({kind})")
            ax[3].plot(ks, [next(r for r in N if r["k"] == k and r["kind"] == kind)["test_acc"] for k in ks], ls, color="C0", marker="s", label=f"test ({kind})")
        ax[3].set_xscale("log", base=2); ax[3].set_title("ablate k of 1,024 MLP neurons")
    if size:
        ss = sorted({r["step"] for r in size})
        ax[4].plot(ss, [next(r for r in size if r["step"] == s)["top32_norm_ratio"] for s in ss], label="memorization top-32 / median")
        ax[4].plot(ss, [next(r for r in size if r["step"] == s)["rule32_norm_ratio"] for s in ss], label="rule top-32 / median")
        ax[4].set_xscale("symlog", linthresh=100); ax[4].set_title("neuron weight size")
    for a in ax:
        a.legend(fontsize=6); a.grid(alpha=.3)
    for a in ax[:4]:
        a.set_ylim(-0.02, 1.02)
    fig.suptitle(run)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def summarize(lens, abl, neu, size):
    """조건(N, 방식, wd)별 seed 평균. 마지막 체크포인트 기준 + 오답 최대 시점."""
    lines = []
    cond = defaultdict(set)
    for r in abl:
        cond[(int(r["train"]), r["mode"], float(r["wd"]))].add(r["run"])
    f = lambda xs: (f"{np.nanmean(xs):.2f}" + (f"±{np.nanstd(xs):.2f}" if len(xs) > 1 else "")) if xs else "-"
    for key in sorted(cond):
        runs = sorted(cond[key])
        last = {run: max(r["step"] for r in abl if r["run"] == run) for run in runs}
        g = lambda rows, run, **kw: next((r for r in rows if r["run"] == run and r["step"] == last[run]
                                          and all(r.get(k) == v for k, v in kw.items())), None)
        N, mode, wd = key
        lines.append(f"\n## N {N} · {mode} · wd {wd:g}  (seed {len(runs)}개, 마지막 step {int(max(last.values()))})")
        b = [g(abl, run, ablate="none") for run in runs]
        lines.append(f"출력: 외운 오답 {f([x['mem_all'] for x in b])}, 원래 정답 {f([x['fixed_all'] for x in b])}, "
                     f"오답 없는 train {f([x['clean_acc'] for x in b])}, test {f([x['test_acc'] for x in b])}"
                     + (f", 번짐 {f([x['spill'] for x in b])}" if mode == "cell" else ""))
        # 오답이 가장 많이 외워졌던 시점
        if mode != "none":
            peaks = []
            for run in runs:
                rs = [r for r in abl if r["run"] == run and r["ablate"] == "none"]
                pk = max(rs, key=lambda r: r["mem_all"])
                peaks.append((pk["step"], pk["mem_all"]))
            lines.append("오답 최대 (step, 비율): " + ", ".join(f"{int(s)}:{v:.2f}" for s, v in peaks))
        lines.append("층별 읽기, 마지막 step (외운 오답 1위 / 원래 정답 1위 / 오답−정답 logit / test):")
        for p in POINTS:
            L = [g(lens, run, point=p) for run in runs]
            lines.append(f"  {p:8s} {f([x['mem'] for x in L])} / {f([x['fixed'] for x in L])} / {f([x['diff'] for x in L])} / {f([x['test_acc'] for x in L])}")
        lines.append("부품 끄기 (외운 오답 / 원래 정답 / 오답 없는 train / test):")
        for c in COMPS:
            L = [g(abl, run, ablate=c) for run in runs]
            lines.append(f"  {c:6s} {f([x['mem_all'] for x in L])} / {f([x['fixed_all'] for x in L])} / {f([x['clean_acc'] for x in L])} / {f([x['test_acc'] for x in L])}")
        if mode == "none":
            continue
        z = [g(neu, run, k=0.0, kind="none") for run in runs]
        z = [x for x in z if x]
        if z:
            lines.append(f"오답 담당 상위 32개: 기여 비중 {f([x['mem_top32_share'] for x in z])}, 1층 비율 {f([x['mem_top32_in_L1'] for x in z])}, "
                         f"규칙 담당 상위 32개와 겹침 {f([x['overlap_mem_rule_top32'] for x in z])}개")
        lines.append("뉴런 k개 끄기 (고르지 않은 오답 자리의 외운 오답 / 오답 없는 train / test" + (" / 번짐" if mode == "cell" else "") + "):")
        for k in KS:
            parts = []
            for kind in ("mem_top", "random", "rule_top"):
                L = [g(neu, run, k=float(k), kind=kind) for run in runs]
                L = [x for x in L if x]
                if not L:
                    continue
                s = f"{f([x['mem_B'] for x in L])}/{f([x['clean_acc'] for x in L])}/{f([x['test_acc'] for x in L])}"
                if mode == "cell":
                    s += f"/{f([x.get('spill', np.nan) for x in L])}"
                parts.append(f"{kind} {s}")
            lines.append(f"  k={k:3d}  " + "   ".join(parts))
        S = [g(size, run) for run in runs]
        S = [x for x in S if x]
        if S:
            lines.append(f"뉴런 크기 (마지막): 오답 담당 상위 32 / 중앙값 {f([x['top32_norm_ratio'] for x in S])}, 규칙 담당 {f([x['rule32_norm_ratio'] for x in S])}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", default="noise_internal_report")
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--every", type=int, default=1, help="체크포인트 k개 중 하나만 (마지막은 항상)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--merge", nargs="*", help="나눠 돌린 결과 폴더들을 합쳐 요약만 다시 만든다")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = ("lens", "ablate", "neurons", "sizes")
    if a.merge:
        dirs = sorted({p for g in a.merge for p in glob.glob(g)})
        R = [sum((read_csv(Path(d) / f"{nm}.csv") for d in dirs), []) for nm in names]
    else:
        device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                              ("cpu" if a.device == "auto" else a.device))
        dirs = sorted({Path(p) for g in a.runs for p in glob.glob(g) if (Path(p) / "config.json").exists()})
        R = [[], [], [], []]
        for d in dirs:
            r = analyze_run(d, device, a.n_test, every=a.every)
            if not r:
                continue
            for acc, x in zip(R, r):
                acc += x
            if not a.no_plot:
                try:
                    plot_run(f"{d.parent.name}/{d.name}", *r, out / f"{d.parent.name}__{d.name}.png")
                except Exception as e:
                    print("그림 실패:", e)
            print("완료:", d.name, "체크포인트", len({x['step'] for x in r[1]}), flush=True)
    if not R[1]:
        raise SystemExit("체크포인트가 있는 run이 없습니다")
    for nm, rows in zip(names, R):
        write_csv(out / f"{nm}.csv", rows)
    s = summarize(*R)
    (out / "summary.txt").write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
