"""Собственная реализация NSGA-II с ограничениями (Deb et al., 2002) и
взвешенной свёртки для сравнения. Готовые реализации NSGA-II не используются.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- доминирование
def constrained_dominates(fa, ca, fb, cb) -> bool:
    """Правила Деба: допустимое лучше недопустимого; из двух недопустимых лучше то,
    у которого меньше нарушение; из двух допустимых — доминирование по Парето."""
    if ca == 0 and cb > 0:
        return True
    if ca > 0 and cb == 0:
        return False
    if ca > 0 and cb > 0:
        return ca < cb
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def fast_nondominated_sort(F: np.ndarray, CV: np.ndarray) -> list[np.ndarray]:
    """Быстрая недоминируемая сортировка, O(M N^2). Возвращает список фронтов."""
    feas = CV == 0
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt                                            # dom[i, j]: i доминирует j по Парето
    both = feas[:, None] & feas[None, :]
    D = np.where(both, dom, False)
    D |= feas[:, None] & ~feas[None, :]                       # допустимое доминирует недопустимое
    inf = ~feas[:, None] & ~feas[None, :]
    D |= inf & (CV[:, None] < CV[None, :])                    # меньше нарушение — лучше
    count = D.sum(axis=0)                                     # сколько особей доминирует j
    fronts, current = [], np.flatnonzero(count == 0)
    while current.size:
        fronts.append(current)
        count = count - D[current].sum(axis=0)
        count[current] = -1
        current = np.flatnonzero(count == 0)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k], kind="stable")
        span = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def rank_and_crowding(F, CV, rng=None):
    """rng задан -> абляция: вместо расстояния скопления случайные значения."""
    rank = np.empty(len(F), dtype=int)
    crowd = np.empty(len(F))
    fronts = fast_nondominated_sort(F, CV)
    for r, fr in enumerate(fronts):
        rank[fr] = r
        crowd[fr] = crowding_distance(F[fr]) if rng is None else rng.random(len(fr))
    return rank, crowd, fronts


# ----------------------------------------------------------------- вариация
def sbx(p1, p2, lo, hi, eta, pc, rng):
    m, d = p1.shape
    u = rng.random((m, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    swap_gene = rng.random((m, d)) < 0.5                     # покоординатный SBX
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    c1, c2 = np.where(swap_gene, c2, c1), np.where(swap_gene, c1, c2)
    no = rng.random(m) >= pc
    c1[no], c2[no] = p1[no], p2[no]
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def polynomial_mutation(x, lo, hi, eta, pm, rng):
    x = x.copy()
    mask = rng.random(x.shape) < pm
    u = rng.random(x.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    x = np.where(mask, x + delta * (hi - lo), x)
    return np.clip(x, lo, hi)


@dataclass
class MOConfig:
    pop_size: int = 100
    generations: int = 250              # бюджет = pop_size * (generations + 1) вычислений
    p_crossover: float = 0.9
    eta_c: float = 15.0
    eta_m: float = 20.0
    p_mutation: float = -1.0            # < 0 -> 1/d
    init: str = "uniform"               # uniform | mixed (половина популяции — разреженные профили)
    use_crowding: bool = True           # False — абляция: внутри фронта выбор случайный


@dataclass
class MOResult:
    seed: int
    X: np.ndarray
    F: np.ndarray
    CV: np.ndarray
    evaluations: int
    seconds: float
    hv_hist: list = field(default_factory=list)


def initial(n, d, lo, hi, cfg, rng):
    X = rng.uniform(lo, hi, (n, d))
    if cfg.init == "mixed":
        # половина особей «разреженная»: заряд только в части слотов — иначе равномерная
        # инициализация почти не порождает решений с низким пиком и низкой стоимостью
        k = n // 2
        X[:k] *= rng.random((k, d)) < rng.uniform(0.1, 0.6, (k, 1))
    return X


def nsga2(evaluate, lo, hi, cfg: MOConfig, seed: int, hv_fn=None, hv_every: int = 10,
          extra_mutation=None) -> MOResult:
    """extra_mutation(Q, rng) -> Q — необязательный проблемно-ориентированный оператор."""
    rng = np.random.default_rng(seed)
    d, N = len(lo), cfg.pop_size
    pm = cfg.p_mutation if cfg.p_mutation >= 0 else 1.0 / d
    t0 = time.perf_counter()
    X = initial(N, d, lo, hi, cfg, rng)
    F, CV = evaluate(X)
    fe = N
    abl = None if cfg.use_crowding else rng
    rank, crowd, _ = rank_and_crowding(F, CV, abl)
    hv_hist = []
    for g in range(cfg.generations):
        # бинарный турнир по (ранг, −crowding)
        a, b = rng.integers(0, N, (2, N))
        better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
        parents = np.where(better, a, b)
        p1, p2 = X[parents[: N // 2]], X[parents[N // 2:]]
        c1, c2 = sbx(p1, p2, lo, hi, cfg.eta_c, cfg.p_crossover, rng)
        Q = polynomial_mutation(np.vstack([c1, c2]), lo, hi, cfg.eta_m, pm, rng)
        if extra_mutation is not None:
            Q = extra_mutation(Q, rng)
        FQ, CVQ = evaluate(Q)
        fe += len(Q)
        # (μ + λ)-отбор: объединение родителей и потомков, заполнение по фронтам
        RX, RF, RCV = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([CV, CVQ])
        _, _, fronts = rank_and_crowding(RF, RCV)
        chosen = []
        for fr in fronts:
            if len(chosen) + len(fr) <= N:
                chosen.extend(fr.tolist())
            else:
                cd = crowding_distance(RF[fr]) if cfg.use_crowding else rng.random(len(fr))
                chosen.extend(fr[np.argsort(-cd, kind="stable")[: N - len(chosen)]].tolist())
                break
        idx = np.array(chosen)
        X, F, CV = RX[idx], RF[idx], RCV[idx]
        rank, crowd, _ = rank_and_crowding(F, CV, abl)
        if hv_fn is not None and (g % hv_every == 0 or g == cfg.generations - 1):
            hv_hist.append((fe, hv_fn(F[(rank == 0) & (CV == 0)])))
    return MOResult(seed, X, F, CV, fe, time.perf_counter() - t0, hv_hist)


def weighted_sum_ga(evaluate, lo, hi, weights, scale, cfg: MOConfig, seed: int,
                    cv_penalty: float = 1e3, extra_mutation=None) -> MOResult:
    """Для каждого вектора весов w — однокритериальный ГА (те же SBX и полиномиальная
    мутация, турнир, элитизм) на свёртке Σ w_k f_k / scale_k. Бюджет делится поровну."""
    rng = np.random.default_rng(seed)
    d, N = len(lo), cfg.pop_size
    pm = cfg.p_mutation if cfg.p_mutation >= 0 else 1.0 / d
    total = cfg.pop_size * (cfg.generations + 1)
    per_w = total // len(weights)
    t0 = time.perf_counter()
    outX, outF, outCV, fe = [], [], [], 0
    lo_f, span = scale
    for w in weights:
        X = initial(N, d, lo, hi, cfg, rng)
        F, CV = evaluate(X)
        used = N

        def fit(F, CV):
            return ((F - lo_f) / span) @ w + cv_penalty * CV

        f = fit(F, CV)
        while used + N <= per_w:
            a, b = rng.integers(0, N, (2, N))
            parents = np.where(f[a] < f[b], a, b)
            c1, c2 = sbx(X[parents[: N // 2]], X[parents[N // 2:]], lo, hi, cfg.eta_c, cfg.p_crossover, rng)
            Q = polynomial_mutation(np.vstack([c1, c2]), lo, hi, cfg.eta_m, pm, rng)
            if extra_mutation is not None:
                Q = extra_mutation(Q, rng)
            FQ, CVQ = evaluate(Q)
            used += N
            RX, RF, RCV = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([CV, CVQ])
            rf = fit(RF, RCV)
            o = np.argsort(rf)[:N]                            # (μ+λ) — элитизм
            X, F, CV, f = RX[o], RF[o], RCV[o], rf[o]
        fe += used
        i = int(np.argmin(f))
        outX.append(X[i])
        outF.append(F[i])
        outCV.append(CV[i])
    return MOResult(seed, np.array(outX), np.array(outF), np.array(outCV), fe, time.perf_counter() - t0)


# ----------------------------------------------------------------- показатели качества
def nondominated(F: np.ndarray) -> np.ndarray:
    """Маска недоминируемых точек (минимизация)."""
    if len(F) == 0:
        return np.zeros(0, dtype=bool)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return ~np.any(le & lt, axis=0)


def hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    """Точный гиперобъём для 2 или 3 критериев (метод срезов по последней координате)."""
    F = np.asarray(F, dtype=float)
    F = F[np.all(F < ref, axis=1)] if len(F) else F
    if len(F) == 0:
        return 0.0
    if F.shape[1] == 2:
        return hv2d(F, ref)
    F = F[np.argsort(F[:, 2])]
    hv = 0.0
    for i in range(len(F)):
        z_next = F[i + 1, 2] if i + 1 < len(F) else ref[2]
        if z_next > F[i, 2]:
            hv += hv2d(F[: i + 1, :2], ref[:2]) * (z_next - F[i, 2])
    return hv
