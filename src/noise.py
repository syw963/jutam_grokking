"""학습 데이터에 오답 섞기 (E30).

두 방식 (오답 토큰 수는 같게 맞춘다: round(frac × N)개):
  scatter: train 예제 중 frac 비율을 골라, 예제마다 출력 한 칸을 무작위 오답으로 바꾼다.
           같은 연산 표 칸이 다른 예제에서는 정답으로 나온다.
  cell:    연산 표 칸을 무작위로 골라, 그 칸이 train에 나오는 모든 자리를 같은 오답으로 바꾼다.
           (표 자체가 몇 칸 바뀐 것과 같다.) 오답 토큰 수가 scatter와 같아질 때까지 칸을 더한다.
오답 값은 정답이 아닌 값 중 균등. 같은 (데이터, 규칙, seed)면 언제나 같은 오답이 나온다.

칸 단위 정보가 필요하므로 단일 REF 단계 규칙만 지원한다 (w5_next 등).
값은 모두 0-based (0..n-1, train_multi 내부 표현).
"""
from __future__ import annotations

import numpy as np

from src.rules import Rule, ref_pairs


def cell_keys(rule: Rule, xs: np.ndarray) -> np.ndarray:
    """(N, m) 입력(1..n) → (N, m) 칸 번호. 교환 연산(add)은 무순서 쌍으로 센다."""
    if len(rule.steps) != 1 or rule.steps[0].code != "REF":
        raise ValueError(f"오답 섞기는 단일 REF 규칙만 지원합니다: {rule.spec}")
    st, n = rule.steps[0], rule.n
    comm = st.p("op") == "add"
    out = np.empty(xs.shape, dtype=np.int64)
    for r, row in enumerate(xs.tolist()):
        for i, (a, b) in enumerate(ref_pairs(st, row, n)):
            if comm and a > b:
                a, b = b, a
            out[r, i] = (a - 1) * n + (b - 1)
    return out


def make_noise(rule: Rule, x_tr: np.ndarray, y_tr: np.ndarray, frac: float, mode: str, seed: int = 777) -> dict:
    """y_tr: (N, m) 0-based 정답. 반환: y_noisy, mask(오답 자리), cell_wrong(칸 번호 → 오답 값), info."""
    N, m = y_tr.shape
    n = rule.n
    rng = np.random.default_rng(seed)
    keys = cell_keys(rule, x_tr)
    y = y_tr.copy()
    mask = np.zeros((N, m), dtype=bool)
    cell_wrong: dict[int, int] = {}
    target = int(round(frac * N))
    if target == 0:
        pass
    elif mode == "scatter":
        rows = rng.choice(N, target, replace=False)
        pos = rng.integers(0, m, target)
        wrong = (y_tr[rows, pos] + rng.integers(1, n, target)) % n
        y[rows, pos] = wrong
        mask[rows, pos] = True
        for k, w in zip(keys[rows, pos].tolist(), wrong.tolist()):
            cell_wrong.setdefault(k, w)  # 같은 칸에 오답이 둘 이상이면 처음 것
    elif mode == "cell":
        uniq, counts = np.unique(keys, return_counts=True)
        order = rng.permutation(len(uniq))
        got = 0
        for j in order:
            if got >= target:
                break
            k = int(uniq[j])
            sel = keys == k
            true_v = int(y_tr[sel][0])
            w = (true_v + int(rng.integers(1, n))) % n
            y[sel] = w
            mask |= sel
            cell_wrong[k] = w
            got += int(counts[j])
    else:
        raise ValueError(f"알 수 없는 오답 방식: {mode}")
    info = dict(mode=mode, frac=frac, seed=seed, tokens=int(mask.sum()), rows=int(mask.any(1).sum()),
                cells=len(cell_wrong), train_rows=N, total_cells=int(len(np.unique(keys))))
    return dict(y_noisy=y, mask=mask, cell_wrong=cell_wrong, keys=keys, info=info)


def test_cells(rule: Rule, x_te: np.ndarray, cell_wrong: dict) -> tuple:
    """test에서 오답 칸이 나오는 자리. 반환: mask (N, m), wrong (N, m) (-1이면 해당 없음)."""
    keys = cell_keys(rule, x_te)
    wrong = np.full(keys.shape, -1, dtype=np.int64)
    for k, w in cell_wrong.items():
        wrong[keys == k] = w
    return wrong >= 0, wrong
