"""Базовые методы для сравнения при одинаковом бюджете вычислений f."""
from __future__ import annotations

import time

import numpy as np

from ga import RunResult
from zakharov import DIM, bounds, zakharov


def random_search(seed: int, budget: int, dim: int = DIM, batch: int = 50, f=zakharov) -> RunResult:
    """Равномерная случайная выборка из D; запоминается лучшая точка."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds(dim)
    t0 = time.perf_counter()
    best_f, best_x, fe = np.inf, None, 0
    h_fe, h_best = [], []
    while fe < budget:
        n = min(batch, budget - fe)
        x = rng.uniform(lo, hi, size=(n, dim))
        y = f(x)
        fe += n
        i = int(np.argmin(y))
        if y[i] < best_f:
            best_f, best_x = float(y[i]), x[i].copy()
        h_fe.append(fe)
        h_best.append(best_f)
    return RunResult(seed, best_x, best_f, fe, len(h_fe), time.perf_counter() - t0,
                     np.array(h_fe), np.array(h_best), np.zeros(len(h_fe)))


def fixed_step_local_search(seed: int, budget: int, step: float = 0.05, dim: int = DIM,
                            f=zakharov) -> RunResult:
    """Простейший локальный поиск (1+1) с фиксированным гауссовским шагом.

    step задаётся в долях ширины области. Нужен только для иллюстрации того,
    почему фиксированный масштаб шага плохо подходит для функции Захарова.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds(dim)
    w = hi - lo
    t0 = time.perf_counter()
    x = rng.uniform(lo, hi)
    fx = float(f(x)[0])
    fe = 1
    h_fe, h_best = [fe], [fx]
    while fe < budget:
        y = np.clip(x + rng.normal(0, step, dim) * w, lo, hi)
        fy = float(f(y)[0])
        fe += 1
        if fy < fx:
            x, fx = y, fy
        if fe % 50 == 0:
            h_fe.append(fe)
            h_best.append(fx)
    return RunResult(seed, x, fx, fe, fe, time.perf_counter() - t0,
                     np.array(h_fe), np.array(h_best), np.zeros(len(h_fe)))
