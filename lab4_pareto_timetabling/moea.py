"""Многокритериальные эволюционные алгоритмы для расписания (собственная реализация).

* nsga2            — NSGA-II с доминированием с ограничениями (Деб);
* nsga2 + memetic  — то же + локальный поиск по перемещениям занятий для части потомков;
* weighted_sum     — однокритериальный ГА на взвешенной свёртке для набора весов.
Все варианты используют одни и те же операторы вариации и одинаковый бюджет вычислений.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# =============================================================== сортировка и разнообразие
def fast_nondominated_sort(F, CV):
    feas = CV == 0
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    D = (le & lt) & feas[:, None] & feas[None, :]
    D |= feas[:, None] & ~feas[None, :]
    D |= (~feas[:, None] & ~feas[None, :]) & (CV[:, None] < CV[None, :])
    count = D.sum(axis=0)
    fronts, cur = [], np.flatnonzero(count == 0)
    while cur.size:
        fronts.append(cur)
        count = count - D[cur].sum(axis=0)
        count[cur] = -1
        cur = np.flatnonzero(count == 0)
    return fronts


def crowding_distance(F):
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for k in range(m):
        o = np.argsort(F[:, k], kind="stable")
        span = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def rank_crowd(F, CV):
    rank, crowd = np.empty(len(F), int), np.empty(len(F))
    fronts = fast_nondominated_sort(F, CV)
    for r, fr in enumerate(fronts):
        rank[fr], crowd[fr] = r, crowding_distance(F[fr])
    return rank, crowd, fronts


def nondominated(F):
    if len(F) == 0:
        return np.zeros(0, bool)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return ~np.any(le & lt, axis=0)


def hv2d(P, ref):
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, by = 0.0, ref[1]
    for x, y in P:
        if y < by:
            hv += (ref[0] - x) * (by - y)
            by = y
    return hv


def hypervolume(F, ref):
    F = np.asarray(F, float)
    if len(F) == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 2])]
    hv = 0.0
    for i in range(len(F)):
        z = F[i + 1, 2] if i + 1 < len(F) else ref[2]
        if z > F[i, 2]:
            hv += hv2d(F[: i + 1, :2], ref[:2]) * (z - F[i, 2])
    return hv


def igd(F, ref_front):
    """Средняя дистанция от точек эталонного фронта до ближайшей найденной точки."""
    if len(F) == 0:
        return np.inf
    return float(np.mean(np.min(np.linalg.norm(ref_front[:, None] - F[None], axis=2), axis=1)))


# =============================================================== вариация
def uniform_crossover(P1, P2, pc, rng):
    m = rng.random(P1.shape) < 0.5
    c1, c2 = np.where(m, P1, P2), np.where(m, P2, P1)
    no = rng.random(len(P1)) >= pc
    c1[no], c2[no] = P1[no], P2[no]
    return c1, c2


def mutate(X, n_slots, pm, p_swap, rng):
    X = X.copy()
    mask = rng.random(X.shape) < pm                              # перенос занятия в случайный слот
    X[mask] = rng.integers(0, n_slots, mask.sum())
    sw = np.flatnonzero(rng.random(len(X)) < p_swap)             # обмен слотами двух занятий
    if sw.size:
        a = rng.integers(0, X.shape[1], sw.size)
        b = rng.integers(0, X.shape[1], sw.size)
        X[sw, a], X[sw, b] = X[sw, b], X[sw, a]
    return X


def better(fa, ca, fb, cb):
    """a не хуже b по правилам Деба и строго лучше хотя бы в одном."""
    if ca != cb:
        return ca < cb
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def local_search(x, f, c, evaluate, n_slots, steps, rng, conflicts_of=None, scale=None):
    """Меметический шаг (в духе MOGLS): выбирается случайный вектор весов w, затем steps попыток
    перенести одно занятие (у недопустимого решения — преимущественно конфликтное) в другой слот
    или обменять слоты двух занятий. Ход принимается, если нарушение не растёт и
    (нарушение уменьшилось или взвешенная сумма w·f/scale не ухудшилась)."""
    w = rng.dirichlet(np.ones(len(f)))
    sc = np.ones(len(f)) if scale is None else scale
    used = 0
    for _ in range(steps):
        y = x.copy()
        if c > 0 and conflicts_of is not None and rng.random() < 0.8:
            bad = conflicts_of(x)
            e = int(rng.choice(bad)) if len(bad) else int(rng.integers(len(x)))
            y[e] = rng.integers(0, n_slots)
        elif rng.random() < 0.5:
            y[int(rng.integers(len(x)))] = rng.integers(0, n_slots)
        else:
            i, j = rng.integers(0, len(x), 2)
            y[i], y[j] = y[j], y[i]
        fy, cy = evaluate(y[None])
        used += 1
        if cy[0] < c or (cy[0] == c and (fy[0] / sc) @ w <= (f / sc) @ w):
            x, f, c = y, fy[0], cy[0]
    return x, f, c, used


# =============================================================== алгоритмы
@dataclass
class Config:
    pop_size: int = 100
    budget: int = 30_000
    p_crossover: float = 0.9
    pm: float = -1.0               # < 0 -> 1/E
    p_swap: float = 0.3
    ls_share: float = 0.1          # меметика: доля потомков, проходящих локальный поиск
    ls_steps: int = 10
    ls_only_infeasible: bool = False   # True — локальный поиск только как ремонт недопустимых потомков
    ws_weights_H: int = 3          # взвешенная сумма: решётка весов с шагом 1/H


@dataclass
class Result:
    seed: int
    X: np.ndarray
    F: np.ndarray
    CV: np.ndarray
    evaluations: int
    seconds: float
    hist: list = field(default_factory=list)       # (вычисления, доля допустимых, F недоминируемых)


def nsga2(evaluate, E, n_slots, cfg: Config, seed: int, memetic: bool = False, conflicts_of=None,
          snapshot_every: int = 5000, scale=None) -> Result:
    rng = np.random.default_rng(seed)
    N = cfg.pop_size
    pm = cfg.pm if cfg.pm >= 0 else 1.0 / E
    t0 = time.perf_counter()
    X = rng.integers(0, n_slots, (N, E))
    F, CV = evaluate(X)
    fe = N
    rank, crowd, _ = rank_crowd(F, CV)
    hist, next_snap = [], 0

    def snap():
        ok = (rank == 0) & (CV == 0)
        hist.append((fe, float(np.mean(CV == 0)), F[ok].copy()))

    while fe < cfg.budget:
        if fe >= next_snap:
            snap()
            next_snap += snapshot_every
        a, b = rng.integers(0, N, (2, N))
        win = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b])), a, b)
        c1, c2 = uniform_crossover(X[win[: N // 2]], X[win[N // 2:]], cfg.p_crossover, rng)
        Q = mutate(np.vstack([c1, c2]), n_slots, pm, cfg.p_swap, rng)
        FQ, CVQ = evaluate(Q)
        fe += len(Q)
        if memetic:
            cand = np.flatnonzero(rng.random(len(Q)) < cfg.ls_share)
            if cfg.ls_only_infeasible:
                cand = np.flatnonzero(CVQ > 0)
            for i in cand:
                if fe >= cfg.budget:
                    break
                steps = min(cfg.ls_steps, cfg.budget - fe)
                Q[i], FQ[i], CVQ[i], used = local_search(Q[i], FQ[i], CVQ[i], evaluate, n_slots, steps, rng,
                                                         conflicts_of, scale)
                fe += used
        RX, RF, RC = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([CV, CVQ])
        _, _, fronts = rank_crowd(RF, RC)
        chosen = []
        for fr in fronts:
            if len(chosen) + len(fr) <= N:
                chosen += fr.tolist()
            else:
                cd = crowding_distance(RF[fr])
                chosen += fr[np.argsort(-cd, kind="stable")[: N - len(chosen)]].tolist()
                break
        idx = np.array(chosen)
        X, F, CV = RX[idx], RF[idx], RC[idx]
        rank, crowd, _ = rank_crowd(F, CV)
    snap()
    return Result(seed, X, F, CV, fe, time.perf_counter() - t0, hist)


def simplex_weights(m, H):
    import itertools
    return np.array([np.array(c) / H for c in itertools.product(range(H + 1), repeat=m) if sum(c) == H])


def weighted_sum(evaluate, E, n_slots, cfg: Config, seed: int, scale: np.ndarray, pop_size: int = 40) -> Result:
    """Для каждого вектора весов — ГА с (μ+λ)-отбором по ключу (CV, Σ w_k f_k / scale_k)."""
    rng = np.random.default_rng(seed)
    W = simplex_weights(3, cfg.ws_weights_H)
    per = cfg.budget // len(W)
    pm = cfg.pm if cfg.pm >= 0 else 1.0 / E
    t0 = time.perf_counter()
    outX, outF, outC, fe = [], [], [], 0
    for w in W:
        X = rng.integers(0, n_slots, (pop_size, E))
        F, CV = evaluate(X)
        used = pop_size

        def key(F, CV):
            return CV * 1e6 + (F / scale) @ (w + 1e-3)     # малая добавка исключает слабо доминируемые точки

        k = key(F, CV)
        while used + pop_size <= per:
            a, b = rng.integers(0, pop_size, (2, pop_size))
            win = np.where(k[a] < k[b], a, b)
            c1, c2 = uniform_crossover(X[win[: pop_size // 2]], X[win[pop_size // 2:]], cfg.p_crossover, rng)
            Q = mutate(np.vstack([c1, c2]), n_slots, pm, cfg.p_swap, rng)
            FQ, CVQ = evaluate(Q)
            used += pop_size
            RX, RF, RC = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([CV, CVQ])
            rk = key(RF, RC)
            o = np.argsort(rk, kind="stable")[:pop_size]
            X, F, CV, k = RX[o], RF[o], RC[o], rk[o]
        fe += used
        outX.append(X[0])
        outF.append(F[0])
        outC.append(CV[0])
    return Result(seed, np.array(outX), np.array(outF), np.array(outC), fe, time.perf_counter() - t0)
