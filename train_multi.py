"""여러 run을 한 프로세스에서 같이 학습한다 (torch.func.vmap).

같은 데이터·같은 모델 구조인 run들을 묶는다. 묶음 안에서는 규칙, seed, weight decay가 달라도 된다.
각 run은 train.py로 따로 돌린 것과 같은 계산을 한다:
  - 초기화: run마다 torch.manual_seed(seed) 후 모델 생성 (train.py와 같은 초기값)
  - loss: run별 loss의 합을 미분하므로 각 run의 gradient는 혼자 학습할 때와 같다
  - AdamW: 원소별 연산이라 묶어도 run별로 독립. weight decay는 run별 값으로 직접 적용
    (torch AdamW와 같은 decoupled 방식: p ← p·(1 − lr·wd) 후 Adam 갱신)
다른 점: 평가용 train/test 부분집합을 묶음 전체가 공유한다 (--n-eval -1이면 전부 평가해서 차이 없음).
끝난 run(--stop-after-gen)은 묶음에서 빼서 남은 run만 계속 학습한다.

출력은 train.py와 같다: runs/<run 이름>/ config.json, rule_info.json, metrics.csv, summary.json
(+ --ckpt-log이면 ckpt_XXXXXXXX.pt). 이어 학습용 묶음 체크포인트는 runs/_groups/<묶음>/ckpt_last.pt.

예:
    python train_multi.py --data data/c2048 --model m0 --members members.json --runs-dir runs
    members.json = [{"rule": "sort", "seed": 0, "wd": 1.0}, {"rule": "l1_sum", "seed": 0}, ...]
보통은 sweep.py --multi가 members.json을 만들어 이 스크립트를 부른다.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

from src.data import compute_targets, load_split
from src.metrics import SymmetryProbe
from src.model import KINDS, build_model
from src.rules import Rule, analyze
from train import CSV_FIELDS, log_grid, make_autocast, pick_device, rule_name


def member_name(rule, model, data, wd, seed):
    return f"{rule_name(rule)}_{model}_{Path(data).name}_wd{wd:g}_s{seed}"


def group_id(names):
    return "g" + hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest()[:10]


class MemberView:
    """묶음 안 run 하나를 보통 모델처럼 쓰게 하는 얇은 래퍼 (대칭 위반 점수용)."""

    def __init__(self, base, params):
        self.base, self.params = base, params

    def logits(self, x, y=None):
        return functional_call(self.base, self.params, (x, y))

    def predict(self, x):
        return functional_call(self.base, self.params, (x,), {"predict": True})

    def eval(self):
        pass

    def train(self):
        pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="여러 run 묶음 학습")
    ap.add_argument("--data")
    ap.add_argument("--members", help="members.json 경로")
    ap.add_argument("--model", choices=KINDS, default="m0")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tie", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0, help="members.json에 wd가 없을 때 기본값")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=-1, help="-1: full batch, 양수: run별 복원추출")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=4096)
    ap.add_argument("--rows-budget", type=int, default=65536,
                    help="한 번에 계산할 (run 수 × 행 수) 상한. 넘으면 나눠 계산하고 gradient를 모은다 (결과 같음)")
    ap.add_argument("--sym-probe", type=int, default=512)
    ap.add_argument("--sym-sigmas", type=int, default=8)
    ap.add_argument("--fit-thr", type=float, default=0.99)
    ap.add_argument("--gen-thr", type=float, default=0.95)
    ap.add_argument("--sym-thr", type=float, default=0.05)
    ap.add_argument("--stop-after-gen", type=int, default=-1)
    ap.add_argument("--budget-mult", type=float, default=0.0)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--ckpt-log", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--resume", help="runs/_groups/<묶음>/ckpt_last.pt")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        keep = {"steps", "resume", "device", "quiet", "rows_budget", "stop_after_gen", "budget_mult",
                "ckpt_every"}
        for k, v in resume["config"].items():
            if k not in keep:
                setattr(args, k, v)
        members = resume["members"]
    else:
        if not args.data or not args.members:
            sys.exit("--data와 --members가 필요합니다")
        members = json.loads(Path(args.members).read_text())
        if isinstance(members, dict):  # sweep.py --multi 형식
            members = members["members"]
        for mb in members:
            mb.setdefault("wd", args.wd)
            mb["seed"] = int(mb["seed"])
            mb["wd"] = float(mb["wd"])
            mb["name"] = member_name(mb["rule"], args.model, args.data, mb["wd"], mb["seed"])

    split = load_split(args.data)
    meta = split["meta"]
    n, m = meta["n"], meta["m"]
    device = pick_device(args.device)
    autocast = make_autocast(device, args.dtype)
    runs_dir = Path(args.runs_dir)
    gid = group_id([mb["name"] for mb in members])
    gdir = runs_dir / "_groups" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    K = len(members)

    # ---- run별 준비 -------------------------------------------------------
    x_tr = torch.as_tensor(split["train"].astype(np.int64) - 1, device=device)
    x_te = torch.as_tensor(split["test"].astype(np.int64) - 1, device=device)
    g = torch.Generator().manual_seed(1)
    pick = lambda N, k: (torch.arange(N) if k < 0 or k >= N else torch.randperm(N, generator=g)[:k]).to(device)
    i_tr, i_te = pick(len(x_tr), args.n_eval), pick(len(x_te), args.n_eval)
    i_probe = pick(len(x_te), args.sym_probe)

    rules, infos, Y_tr, Y_te, probes, run_dirs = [], [], [], [], [], []
    for mb in members:
        rule = Rule(mb["rule"], n)
        rd = runs_dir / mb["name"]
        if not resume and rd.exists() and (rd / "metrics.csv").exists():
            # 같은 묶음이 체크포인트 저장 전에 끊긴 경우만 처음부터 다시 쓴다
            prev = json.loads((rd / "config.json").read_text()).get("group") if (rd / "config.json").exists() else None
            if prev != gid:
                sys.exit(f"이미 있는 run입니다: {rd}")
        rd.mkdir(parents=True, exist_ok=True)
        ip = rd / "rule_info.json"
        info = json.loads(ip.read_text()) if ip.exists() else analyze(rule.spec, n, m)
        ip.write_text(json.dumps(info, ensure_ascii=False))
        ytr = torch.as_tensor(compute_targets(rule, split["train"]).astype(np.int64) - 1, device=device)
        yte = torch.as_tensor(compute_targets(rule, split["test"]).astype(np.int64) - 1, device=device)
        group = info["inv_group"] if info["sym_kind"] == "invariant" else info["eq_group"]
        probes.append(SymmetryProbe(info["sym_kind"], group, x_te[i_probe], yte[i_probe],
                                    args.sym_sigmas, mb["seed"]))
        rules.append(rule); infos.append(info); Y_tr.append(ytr); Y_te.append(yte); run_dirs.append(rd)
        cfg = {k: v for k, v in vars(args).items()}
        cfg.update(rule=mb["rule"], seed=mb["seed"], wd=mb["wd"], rule_spec=rule.spec, n=n, m=m,
                   run_dir=str(rd), device=str(device), data_meta=meta, group=gid, trainer="train_multi")
        (rd / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        if not resume:
            with (rd / "metrics.csv").open("w", newline="") as f:
                csv.DictWriter(f, CSV_FIELDS).writeheader()
    Y_tr, Y_te = torch.stack(Y_tr), torch.stack(Y_te)  # (K, N, m)

    # ---- 모델 묶기 ---------------------------------------------------------
    tie = None if args.tie == "auto" else args.tie == "yes"
    models = []
    for mb in members:
        torch.manual_seed(mb["seed"])
        models.append(build_model(args.model, n, m, args.d, args.layers, args.heads, tie, args.init_std).to(device))
    base = copy.deepcopy(models[0])
    params, _ = stack_module_state(models)  # dict name -> (K, ...) leaf tensor
    del models
    n_params = sum(v[0].numel() for v in params.values())
    wd = torch.tensor([mb["wd"] for mb in members], device=device)

    def make_opt(ps):
        kw = dict(lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0)
        if device.type == "cuda":
            kw["fused"] = True
        return torch.optim.AdamW(list(ps.values()), **kw)

    opt = make_opt(params)
    active = list(range(K))  # 아직 학습 중인 run (params의 0번째 축 순서)
    step, elapsed = 0, 0.0
    states = [dict(t_fit=None, t_gen=None, t_sym=None, done=False, stop_reason=None) for _ in members]
    if resume:
        active, step, elapsed, states = resume["active"], resume["step"], resume["elapsed"], resume["states"]
        params = {k: v.to(device).requires_grad_() for k, v in resume["params"].items()}
        opt = make_opt(params)
        opt.load_state_dict(resume["opt"])
        for i, mb in enumerate(members):  # metrics.csv를 체크포인트 step까지로 자르기
            p = run_dirs[i] / "metrics.csv"
            rows = [r for r in csv.DictReader(p.open()) if int(r["step"]) <= step] if p.exists() else []
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)

    lr_at = lambda s: args.lr * min(1.0, (s + 1) / max(1, args.warmup))
    batch_gen = torch.Generator().manual_seed(12345)
    grid = log_grid(args.steps)

    def f_logits(p, x, y):
        return functional_call(base, p, (x, y))

    def f_pred(p, x):
        return functional_call(base, p, (x,), {"predict": True})

    def save_group_ckpt():
        torch.save(dict(params={k: v.detach().cpu() for k, v in params.items()}, opt=opt.state_dict(),
                        active=active, step=step, elapsed=elapsed, states=states, members=members,
                        config=vars(args)), gdir / "ckpt_last.pt")

    def save_member_ckpt(slot, i):
        mdl = copy.deepcopy(base)
        with torch.no_grad():
            for name, p in mdl.named_parameters():
                p.copy_(params[name][slot])
        torch.save(dict(model=mdl.state_dict(), step=step, config=json.loads((run_dirs[i] / "config.json").read_text())),
                    run_dirs[i] / f"ckpt_{step:08d}.pt")

    def write_summary(i):
        st, info, mb = states[i], infos[i], members[i]
        t_fit, t_gen = st["t_fit"], st["t_gen"]
        summ = dict(rule=rules[i].spec, rule_name=rule_name(mb["rule"]), model=args.model, seed=mb["seed"],
                    data=str(args.data), train_sets=meta["train_sets"], wd=mb["wd"],
                    sym_kind=info["sym_kind"], inv_size=info["inv_size"], eq_size=info["eq_size"],
                    dep_mean=info["dep_mean"], effective_depth=info["effective_depth"],
                    nominal_depth=info["nominal_depth"], t_fit=t_fit, t_gen=t_gen, t_sym=st["t_sym"],
                    delay=(t_gen - t_fit) if (t_fit is not None and t_gen is not None) else None,
                    ratio=(t_gen / t_fit) if (t_fit and t_gen is not None) else None,
                    censored=t_gen is None, last_step=step, stop_reason=st["stop_reason"],
                    fit_thr=args.fit_thr, gen_thr=args.gen_thr, group=gid)
        (run_dirs[i] / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))

    @torch.no_grad()
    def eval_split(x, Y, idx):
        """활성 run 전체를 한꺼번에 평가. 반환: 지표 이름 -> (A,) 텐서."""
        A = len(active)
        Ya = Y[active][:, idx]
        xs = x[idx]
        loss = torch.zeros(A, device=device)
        tok = torch.zeros(A, device=device); ex = torch.zeros(A, device=device)
        tok_tf = torch.zeros(A, device=device); ex_tf = torch.zeros(A, device=device)
        chunk = max(64, args.rows_budget // A)
        for s in range(0, len(idx), chunk):
            xb, yb = xs[s:s + chunk], Ya[:, s:s + chunk]
            with autocast():
                lg = vmap(f_logits, in_dims=(0, None, 0))(params, xb, yb).float()
                pr = vmap(f_pred, in_dims=(0, None))(params, xb)
            loss += F.cross_entropy(lg.permute(0, 3, 1, 2), yb, reduction="none").sum((1, 2))
            am = lg.argmax(-1)
            tok_tf += (am == yb).float().sum((1, 2)); ex_tf += (am == yb).all(-1).float().sum(1)
            tok += (pr == yb).float().sum((1, 2)); ex += (pr == yb).all(-1).float().sum(1)
        N = len(idx)
        return dict(loss=loss / (N * m), tok=tok / (N * m), exact=ex / N, tok_tf=tok_tf / (N * m), exact_tf=ex_tf / N)

    def do_eval():
        a, b = eval_split(x_tr, Y_tr, i_tr), eval_split(x_te, Y_te, i_te)
        for slot, i in enumerate(active):
            p_i = {k: v[slot].detach() for k, v in params.items()}
            s = probes[i](MemberView(base, p_i), autocast)
            pn = math.sqrt(sum(v.float().pow(2).sum().item() for v in p_i.values()))
            en = p_i["tok_emb.weight"].float().norm().item()
            row = dict(step=step, lr=lr_at(step), elapsed=round(elapsed, 2),
                       train_loss=a["loss"][slot].item(), test_loss=b["loss"][slot].item(),
                       train_tok=a["tok"][slot].item(), test_tok=b["tok"][slot].item(),
                       train_exact=a["exact"][slot].item(), test_exact=b["exact"][slot].item(),
                       train_exact_tf=a["exact_tf"][slot].item(), test_exact_tf=b["exact_tf"][slot].item(),
                       param_norm=pn, emb_norm=en,
                       sym_viol_tok=s["viol_tok"], sym_viol_seq=s["viol_seq"], sym_viol_tv=s["viol_tv"])
            with (run_dirs[i] / "metrics.csv").open("a", newline="") as f:
                csv.DictWriter(f, CSV_FIELDS).writerow(
                    {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
            st = states[i]
            if st["t_fit"] is None and row["train_exact"] >= args.fit_thr:
                st["t_fit"] = step
            if st["t_gen"] is None and row["test_exact"] >= args.gen_thr:
                st["t_gen"] = step
            if st["t_sym"] is None and not math.isnan(s["viol_tok"]) and s["viol_tok"] <= args.sym_thr:
                st["t_sym"] = step
            if args.stop_after_gen >= 0 and st["t_gen"] is not None and step >= st["t_gen"] + args.stop_after_gen:
                st["done"], st["stop_reason"] = True, "after_gen"
            if args.budget_mult > 0 and st["t_fit"] is not None and st["t_gen"] is None \
                    and step >= args.budget_mult * max(1, st["t_fit"]):
                st["done"], st["stop_reason"] = True, "budget"
            write_summary(i)
        if not args.quiet:
            ex_tr = " ".join(f"{v:.2f}" for v in a["exact"].tolist())
            ex_te = " ".join(f"{v:.2f}" for v in b["exact"].tolist())
            print(f"step {step:>7d}  active {len(active)}/{K}  train [{ex_tr}]  test [{ex_te}]  [{elapsed:.0f}s]",
                  flush=True)

    def compact():
        """끝난 run을 묶음에서 뺀다 (params와 AdamW 상태를 잘라 새로 만든다)."""
        nonlocal params, opt, active
        keep = [slot for slot, i in enumerate(active) if not states[i]["done"]]
        if len(keep) == len(active):
            return
        if not keep:
            active = []
            return
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)
        old_state = {name: opt.state.get(p, {}) for name, p in params.items()}
        params = {k: v.detach()[keep_t].clone().requires_grad_() for k, v in params.items()}
        opt = make_opt(params)
        for name, p in params.items():
            st = old_state[name]
            if st:
                opt.state[p] = {"step": st["step"].clone(),
                                "exp_avg": st["exp_avg"][keep_t].clone(),
                                "exp_avg_sq": st["exp_avg_sq"][keep_t].clone()}
        active = [active[s] for s in keep]

    if step == 0:
        do_eval()
    N = len(x_tr)
    t0 = time.time()
    while step < args.steps and active:
        lr = lr_at(step)
        for gr in opt.param_groups:
            gr["lr"] = lr
        Ya = Y_tr[active] if len(active) < K else Y_tr
        opt.zero_grad(set_to_none=True)
        # run별 평균 loss의 합을 미분 → 각 run의 gradient는 혼자 학습할 때와 같다.
        # 행이 많으면 나눠서 backward하고 gradient를 모은다 (합이 같으므로 결과 동일).
        if args.batch_size < 0:
            B, shared = N, True
        else:
            idx = torch.randint(0, N, (len(active), args.batch_size), generator=batch_gen).to(device)
            B, shared = args.batch_size, False
        chunk = max(64, args.rows_budget // len(active))
        for s in range(0, B, chunk):
            with autocast():
                if shared:
                    yb = Ya[:, s:s + chunk]
                    lg = vmap(f_logits, in_dims=(0, None, 0))(params, x_tr[s:s + chunk], yb)
                else:
                    ib = idx[:, s:s + chunk]
                    yb = torch.gather(Ya, 1, ib.unsqueeze(-1).expand(-1, -1, m))
                    lg = vmap(f_logits, in_dims=(0, 0, 0))(params, x_tr[ib], yb)
            ce = F.cross_entropy(lg.float().permute(0, 3, 1, 2), yb, reduction="none")
            (ce.sum((1, 2)) / (B * m)).sum().backward()
        with torch.no_grad():  # decoupled weight decay (run별 wd)
            w = wd[active]
            for v in params.values():
                v.mul_((1 - lr * w).view(-1, *([1] * (v.dim() - 1))).to(v.dtype))
        opt.step()
        step += 1
        if step % args.eval_every == 0 or step in grid or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - t0
            do_eval()
            if args.ckpt_log and step in grid:
                for slot, i in enumerate(active):
                    save_member_ckpt(slot, i)
            compact()
            t0 = time.time()
        if step % args.ckpt_every == 0:
            save_group_ckpt()
    elapsed += time.time() - t0
    for i in active:
        states[i]["stop_reason"] = states[i]["stop_reason"] or "max_steps"
        write_summary(i)
    save_group_ckpt()
    if not args.quiet:
        print(f"끝: step {step}, {elapsed:.0f}s, run {K}개")
    return [json.loads((rd / "summary.json").read_text()) for rd in run_dirs]


if __name__ == "__main__":
    main()
