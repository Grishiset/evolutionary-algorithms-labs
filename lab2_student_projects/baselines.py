"""Базовые методы: жадная конструктивная эвристика, случайный допустимый поиск
и точное решение MILP-решателем (только как эталон качества, не как часть ГА)."""
from __future__ import annotations

import time

import numpy as np

from problem import Instance


def greedy(inst: Instance, seed: int) -> np.ndarray:
    """Студенты в случайном порядке получают самый предпочтительный проект со свободным
    местом и без конфликтов. Затем общий ремонт добирает команды и навыки."""
    rng = np.random.default_rng(seed)
    a = np.full(inst.n, -1)
    counts = np.zeros(inst.P, dtype=int)
    for i in rng.permutation(inst.n):
        order = np.argsort(inst.cost[i] + 1e-3 * rng.random(inst.P))   # предпочтения, затем случайно
        for j in order:
            if counts[j] < inst.cap[j] and not any(a[p] == j for p in inst.partners[i]):
                a[i] = j
                counts[j] += 1
                break
    return inst.repair(a)


def random_feasible_search(inst: Instance, seed: int, budget: int, penalty: float = 1000.0):
    """Случайное назначение + ремонт; лучшее из budget вычислений."""
    rng = np.random.default_rng(seed)
    best, best_f, hist = None, np.inf, []
    for e in range(1, budget + 1):
        a = inst.repair(rng.integers(0, inst.P, inst.n))
        f = inst.objective(a)[0] + penalty * inst.total_violation(a)[0]
        if f < best_f:
            best, best_f = a, f
        if e % 100 == 0:
            hist.append((e, best_f))
    return best, best_f, np.array(hist)


def milp_optimum(inst: Instance, time_limit: float = 60.0):
    """Точный оптимум через scipy.optimize.milp (HiGHS).

    Переменные x_ij ∈ {0,1}. Ограничения:
      sum_j x_ij = 1;  m_j <= sum_i x_ij <= c_j;
      sum_{i: s ∈ skills_i} x_ij >= 1 для s ∈ req_j;
      x_uj + x_vj <= 1 для конфликтующих (u, v).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n, P = inst.n, inst.P
    idx = lambda i, j: i * P + j                    # noqa: E731
    rows, lb, ub = [], [], []

    def add(coefs, lo, hi):
        rows.append(coefs)
        lb.append(lo)
        ub.append(hi)

    for i in range(n):
        add({idx(i, j): 1 for j in range(P)}, 1, 1)
    for j in range(P):
        add({idx(i, j): 1 for i in range(n)}, inst.min_size[j], inst.cap[j])
        for s in inst.req[j]:
            add({idx(i, j): 1 for i in range(n) if inst.skill[i, s]}, 1, np.inf)
        for u, v in inst.conflicts:
            add({idx(u, j): 1, idx(v, j): 1}, 0, 1)
    A = lil_matrix((len(rows), n * P))
    for r, coefs in enumerate(rows):
        for c, v in coefs.items():
            A[r, c] = v
    t0 = time.perf_counter()
    res = milp(inst.cost.ravel(), constraints=LinearConstraint(A.tocsr(), lb, ub),
               integrality=np.ones(n * P), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    x = np.round(res.x).reshape(n, P)
    return x.argmax(axis=1), float(res.fun), res.status, time.perf_counter() - t0
