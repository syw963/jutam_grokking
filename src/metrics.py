"""평가 지표."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, x, y, batch=4096, autocast=None):
    """loss, teacher-forced 정확도, 자유 예측 정확도를 계산한다.

    tok_acc / exact      : 자유 예측(predict) 기준. 판정에 쓰는 값.
    tok_acc_tf / exact_tf: teacher forcing 기준 (병렬 모델은 자유 예측과 같음).
    """
    model.eval()
    tot_loss, tok_tf, ex_tf, tok, ex, cnt = 0.0, 0, 0, 0, 0, 0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        with autocast():
            lg = model.logits(xb, yb)
            pred = model.predict(xb)
        lg = lg.float()
        tot_loss += F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1), reduction="sum").item()
        am = lg.argmax(-1)
        tok_tf += (am == yb).sum().item()
        ex_tf += (am == yb).all(-1).sum().item()
        tok += (pred == yb).sum().item()
        ex += (pred == yb).all(-1).sum().item()
        cnt += len(xb)
    m = y.size(1)
    model.train()
    return dict(loss=tot_loss / (cnt * m), tok_acc=tok / (cnt * m), exact=ex / cnt,
                tok_acc_tf=tok_tf / (cnt * m), exact_tf=ex_tf / cnt)


class SymmetryProbe:
    """규칙의 대칭 관계에서 모델 출력이 벗어난 정도.

    invariant  규칙: model(σx) 와 model(x) 비교
    equivariant 규칙: model(σx) 와 σ·model(x) 비교
    σ는 규칙의 대칭군에서 항등원을 뺀 원소들 중에서 고른다.

    viol_tok : 어긋난 출력 칸의 비율 (자유 예측)
    viol_seq : 수열 전체가 어긋난 비율
    viol_tv  : 대응 칸의 출력 분포 사이 total variation 평균 (teacher-forced logits)
    """

    def __init__(self, kind, group, x, y, n_sigma=8, seed=0):
        self.kind = kind
        group = [tuple(s) for s in group if list(s) != sorted(s)]
        self.active = kind in ("invariant", "equivariant") and len(group) > 0
        if not self.active:
            return
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(group), generator=g)[:n_sigma].tolist()
        self.sigmas = [torch.tensor(group[i], dtype=torch.long) for i in idx]
        self.x, self.y = x, y

    @torch.no_grad()
    def __call__(self, model, autocast):
        if not self.active:
            return dict(viol_tok=math.nan, viol_seq=math.nan, viol_tv=math.nan)
        model.eval()
        dev = self.x.device
        with autocast():
            base_pred = model.predict(self.x)
            base_p = model.logits(self.x, self.y).float().softmax(-1)
        vt, vs, tv = 0.0, 0.0, 0.0
        for s in self.sigmas:
            s = s.to(dev)
            xs = self.x[:, s]
            ys = self.y if self.kind == "invariant" else self.y[:, s]
            with autocast():
                pred = model.predict(xs)
                p = model.logits(xs, ys).float().softmax(-1)
            if self.kind == "invariant":
                ref_pred, ref_p = base_pred, base_p
            else:
                ref_pred, ref_p = base_pred[:, s], base_p[:, s]
            diff = pred != ref_pred
            vt += diff.float().mean().item()
            vs += diff.any(-1).float().mean().item()
            tv += 0.5 * (p - ref_p).abs().sum(-1).mean().item()
        k = len(self.sigmas)
        model.train()
        return dict(viol_tok=vt / k, viol_seq=vs / k, viol_tv=tv / k)


@torch.no_grad()
def norms(model):
    total = math.sqrt(sum(p.detach().float().pow(2).sum().item() for p in model.parameters()))
    emb = model.tok_emb.weight.detach().float().norm().item()
    return dict(param_norm=total, emb_norm=emb)
