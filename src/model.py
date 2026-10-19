"""모델.

모든 모델은 같은 인터페이스를 가진다. 토큰은 값-1 (0..n-1).
    logits(x, y)  -> (B, m, n)   학습용 logits (GPT는 teacher forcing, 병렬 모델은 y를 쓰지 않음)
    predict(x)    -> (B, m)      자유 예측 (GPT는 greedy 생성)

kind:
    m0     병렬 예측. 입력 위치 임베딩 O + 출력 질의 칸 m개. 대칭 없음.
    minv   병렬 예측. 입력 위치 임베딩 X + 출력 질의 칸. 입력 순서에 정확히 불변.
    meq    병렬 예측. 입력 위치 임베딩 X, 입력 i번째 자리에서 y_i 예측. 정확히 등변.
    m0eq   병렬 예측. 입력 위치 임베딩 O, 입력 자리에서 바로 예측 (meq의 대조군).
    gpt    decoder-only GPT. BOS x1..xm SEP y1..ym, 출력 쪽에만 loss (RJ와 같은 꼴).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("m0", "minv", "meq", "m0eq", "gpt")


class Attention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.last_attn = None  # record_attn=True일 때 (B, H, T, T)

    def forward(self, x, causal: bool, record_attn: bool = False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        # 길이가 짧아(최대 11) 직접 계산해도 빠르고, vmap(묶음 학습)에서도 그대로 동작한다.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(-1)
        if record_attn:
            self.last_attn = att.detach()
        y = att @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, n_head, mlp_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d, bias=False), nn.GELU(),
                                 nn.Linear(mlp_mult * d, d, bias=False))

    def forward(self, x, causal=False, record_attn=False):
        x = x + self.attn(self.ln1(x), causal, record_attn)
        return x + self.mlp(self.ln2(x))


def _init(module, std):
    for p_name, p in module.named_parameters():
        if p.dim() >= 2 or "emb" in p_name or "queries" in p_name:
            nn.init.normal_(p, 0.0, std)
    for mod in module.modules():
        if isinstance(mod, nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)


class ParallelModel(nn.Module):
    def __init__(self, n, m, d=128, n_layer=2, n_head=4, input_pos=True, mode="query",
                 tie=False, init_std=0.02):
        super().__init__()
        assert mode in ("query", "aligned")
        self.n, self.m, self.mode = n, m, mode
        self.tok_emb = nn.Embedding(n, d)
        self.pos_emb = nn.Embedding(m, d) if input_pos else None
        self.queries = nn.Parameter(torch.zeros(m, d)) if mode == "query" else None
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, n, bias=False)
        _init(self, init_std)
        if tie:
            self.head.weight = self.tok_emb.weight

    def hidden(self, x, record_attn=False):
        h = self.tok_emb(x)
        if self.pos_emb is not None:
            h = h + self.pos_emb.weight
        if self.queries is not None:
            h = torch.cat([h, self.queries.expand(x.size(0), -1, -1)], dim=1)
        hs = [h]
        for b in self.blocks:
            h = b(h, causal=False, record_attn=record_attn)
            hs.append(h)
        if self.queries is not None:
            h = h[:, self.m:]
        return self.ln_f(h), hs

    def logits(self, x, y=None):
        return self.head(self.hidden(x)[0])

    def _predict(self, x):
        return self.logits(x).argmax(-1)

    @torch.no_grad()
    def predict(self, x):
        return self._predict(x)

    def forward(self, x, y=None, predict=False):
        """torch.func.functional_call용. predict=True면 자유 예측."""
        return self._predict(x) if predict else self.logits(x, y)


class GPTModel(nn.Module):
    """BOS x1..xm SEP y1..ym. 출력 logits는 값 토큰 n개로 제한한다."""

    def __init__(self, n, m, d=128, n_layer=2, n_head=4, tie=True, init_std=0.02):
        super().__init__()
        self.n, self.m = n, m
        self.bos, self.sep = n, n + 1
        self.tok_emb = nn.Embedding(n + 2, d)
        self.pos_emb = nn.Embedding(2 * m + 2, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, n + 2, bias=False)
        _init(self, init_std)
        if tie:
            self.head.weight = self.tok_emb.weight

    def _forward(self, seq, record_attn=False):
        T = seq.size(1)
        h = self.tok_emb(seq) + self.pos_emb.weight[:T]
        for b in self.blocks:
            h = b(h, causal=True, record_attn=record_attn)
        return self.head(self.ln_f(h))[..., : self.n]

    def _prefix(self, x):
        B = x.size(0)
        bos = torch.full((B, 1), self.bos, dtype=x.dtype, device=x.device)
        sep = torch.full((B, 1), self.sep, dtype=x.dtype, device=x.device)
        return torch.cat([bos, x, sep], dim=1)  # 길이 m+2

    def logits(self, x, y):
        seq = torch.cat([self._prefix(x), y[:, :-1]], dim=1)  # 길이 2m+1
        return self._forward(seq)[:, self.m + 1:]  # SEP 자리부터 m개

    def _predict(self, x):
        seq = self._prefix(x)
        for _ in range(self.m):
            nxt = self._forward(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, self.m + 2:]

    @torch.no_grad()
    def predict(self, x):
        return self._predict(x)

    def forward(self, x, y=None, predict=False):
        """torch.func.functional_call용. predict=True면 greedy 생성."""
        return self._predict(x) if predict else self.logits(x, y)


def build_model(kind, n, m, d=128, n_layer=2, n_head=4, tie=None, init_std=0.02):
    if kind not in KINDS:
        raise ValueError(f"kind는 {KINDS} 중 하나")
    if kind == "gpt":
        return GPTModel(n, m, d, n_layer, n_head, tie=True if tie is None else tie, init_std=init_std)
    cfg = dict(m0=(True, "query"), minv=(False, "query"), meq=(False, "aligned"), m0eq=(True, "aligned"))
    pos, mode = cfg[kind]
    return ParallelModel(n, m, d, n_layer, n_head, input_pos=pos, mode=mode,
                         tie=False if tie is None else tie, init_std=init_std)


def model_symmetry(kind: str) -> str | None:
    """모델 구조가 보장하는 대칭."""
    return {"minv": "invariant", "meq": "equivariant"}.get(kind)
