"""Функция Захарова (вариант V = 8 лабораторной работы № 1).

    f(x) = sum x_i^2 + (sum 0.5 i x_i)^2 + (sum 0.5 i x_i)^4,  i = 1..d
    D = [-5, 10]^d,  d = 5 + (V mod 6) = 7
    Глобальный минимум: f(0, ..., 0) = 0.
"""
from __future__ import annotations

import numpy as np

VARIANT = 8
DIM = 5 + (VARIANT % 6)          # = 7
LOWER, UPPER = -5.0, 10.0
F_OPT = 0.0


def zakharov(x: np.ndarray) -> np.ndarray:
    """Векторизованное вычисление: x имеет форму (d,) или (n, d)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    i = np.arange(1, x.shape[1] + 1)
    s1 = np.sum(x * x, axis=1)
    s2 = np.sum(0.5 * i * x, axis=1)
    return s1 + s2 ** 2 + s2 ** 4          # форма (n,)


def bounds(dim: int = DIM) -> tuple[np.ndarray, np.ndarray]:
    return np.full(dim, LOWER), np.full(dim, UPPER)


def hessian_at_optimum(dim: int = DIM) -> np.ndarray:
    """Гессиан в точке x* = 0: H = 2 I + 2 c c^T, где c_i = 0.5 i."""
    c = 0.5 * np.arange(1, dim + 1)
    return 2.0 * np.eye(dim) + 2.0 * np.outer(c, c)
