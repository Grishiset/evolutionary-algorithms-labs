"""Модель зарядки электромобиля (ЛР № 3, трек A, вариант 8).

Электромобиль подключён к домашней сети с 18:00 до 08:00: T = 28 слотов по Δt = 0.5 ч.
Генотип: p ∈ [0, P_max]^T — мощность зарядки в каждом слоте, кВт.
Декодирование (фенотип):
  1) мощность обрезается так, чтобы суммарная нагрузка дома b_t + p_t не превышала
     лимит подключения G_max (ремонт, ограничение H1);
  2) заряд аккумулятора считается по шагам; в слоте, где батарея заполнится, мощность
     уменьшается до необходимой. Энергия, которую батарея не примет, не покупается.
Критерии (все минимизируются):
  f1 = Σ c_t p_t Δt                      — стоимость электроэнергии, руб.;
  f2 = max_t (b_t + p_t)                — пиковое потребление дома, кВт;
  f3 = −SoC_final                       — максимизация итогового заряда, %.
Жёсткое ограничение H2: SoC_final >= SOC_MIN (50 %). Нарушение = max(0, SOC_MIN − SoC_final);
учитывается в доминировании с ограничениями (правила Деба).
"""
from __future__ import annotations

import numpy as np

T = 28                      # 18:00 … 08:00, шаг 30 минут
DT = 0.5                    # ч
P_MAX = 11.0                # кВт, трёхфазная домашняя зарядка
G_MAX = 15.0                # кВт, лимит подключения дома
CAPACITY = 60.0             # кВт·ч
SOC0 = 20.0                 # %, заряд при подключении
SOC_MIN = 50.0              # %, минимально допустимый заряд к отъезду
EFF = 0.92                  # КПД зарядки

HOURS = (18 + DT * np.arange(T)) % 24


def tariff() -> np.ndarray:
    """Условный трёхзонный тариф, руб/кВт·ч: пик 18–21 ч, полупик 21–23 и 07–08 ч, ночь 23–07 ч.

    Значения синтетические (порядок величин типичен для зонных тарифов), задаются здесь.
    """
    c = np.empty(T)
    for k, h in enumerate(HOURS):
        if 18 <= h < 21:
            c[k] = 9.80
        elif 21 <= h < 23 or 7 <= h < 8:
            c[k] = 6.90
        else:
            c[k] = 3.20
    return c


def base_load() -> np.ndarray:
    """Нагрузка дома без электромобиля, кВт: вечерний пик (плита, бойлер), ночной минимум."""
    h = np.where(HOURS < 12, HOURS + 24, HOURS)          # непрерывная ось 18…32 ч
    return 0.6 + 3.4 * np.exp(-((h - 19.5) / 1.5) ** 2) + 1.2 * np.exp(-((h - 31.0) / 0.8) ** 2)


C = tariff()
B = base_load()


def decode(P: np.ndarray):
    """P: (m, T) генотипы -> фактическая мощность (m, T) и SoC после каждого слота (m, T)."""
    P = np.clip(np.atleast_2d(P), 0.0, P_MAX)
    P = np.minimum(P, G_MAX - B)                          # ремонт по лимиту подключения
    actual = np.empty_like(P)
    soc = np.full(P.shape[0], SOC0)
    traj = np.empty_like(P)
    for t in range(T):
        room = (100.0 - soc) / 100.0 * CAPACITY / EFF / DT   # макс. мощность, которую примет батарея
        actual[:, t] = np.minimum(P[:, t], room)
        soc = soc + actual[:, t] * DT * EFF / CAPACITY * 100.0
        traj[:, t] = soc
    return actual, traj


def evaluate(P: np.ndarray):
    """Возвращает F (m, 3) — критерии для минимизации, CV (m,) — нарушение H2, actual, traj."""
    actual, traj = decode(P)
    f1 = (actual * C).sum(axis=1) * DT
    f2 = (B + actual).max(axis=1)
    soc_final = traj[:, -1]
    F = np.column_stack([f1, f2, -soc_final])
    cv = np.maximum(0.0, SOC_MIN - soc_final)
    return F, cv, actual, traj


def lp_reference() -> dict:
    """Точные ориентиры для крайних точек фронта (линейное программирование, scipy/HiGHS).

    Переменные p_t ∈ [0, min(P_MAX, G_MAX - b_t)], энергия в батарею Σ p_t Δt η = E.
    Используются только для оценки качества найденного фронта, не в алгоритме.
    """
    from scipy.optimize import linprog

    ub = np.minimum(P_MAX, G_MAX - B)
    out = {}
    for soc in (SOC_MIN, 80.0, 100.0):
        E = (soc - SOC0) / 100 * CAPACITY
        # минимальная стоимость
        r = linprog(C * DT, A_eq=[np.full(T, DT * EFF)], b_eq=[E], bounds=list(zip(np.zeros(T), ub)))
        out[f"min_cost@{soc:g}%"] = float(r.fun)
        # минимальный пик: переменные (p, z), min z, b_t + p_t <= z
        c = np.r_[np.zeros(T), 1.0]
        A_ub = np.hstack([np.eye(T), -np.ones((T, 1))])
        r = linprog(c, A_ub=A_ub, b_ub=-B, A_eq=[np.r_[np.full(T, DT * EFF), 0]], b_eq=[E],
                    bounds=list(zip(np.zeros(T), ub)) + [(0, None)])
        out[f"min_peak@{soc:g}%"] = float(r.fun)
    return out


def shift_mutation(X: np.ndarray, rng: np.random.Generator, prob: float = 0.3) -> np.ndarray:
    """Проблемно-ориентированная мутация «перенос энергии» (сохраняет купленную энергию).

    С вероятностью prob особь получает один перенос части мощности:
      * «срез пика»: из слота с максимальной суммарной нагрузкой b_t + p_t в слот с минимальной;
      * «в дешёвые часы»: из случайного заряжающего слота с максимальным тарифом в самый
        недогруженный слот с минимальным тарифом.
    """
    X = X.copy()
    rows = np.flatnonzero(rng.random(len(X)) < prob)
    for r in rows:
        p = np.minimum(np.clip(X[r], 0, P_MAX), G_MAX - B)
        load = B + p
        if rng.random() < 0.5:
            i, j = int(np.argmax(load)), int(np.argmin(load))
            delta = rng.random() * (load[i] - load[j]) / 2
        else:
            charging = np.flatnonzero(p > 1e-9)
            if charging.size == 0:
                continue
            i = int(charging[np.argmax(C[charging] + 1e-6 * rng.random(charging.size))])
            cheap = np.flatnonzero(C == C.min())
            j = int(cheap[np.argmin(load[cheap])])
            delta = rng.random() * p[i]
        delta = min(delta, P_MAX - p[j], G_MAX - B[j] - p[j])
        X[r, i] = p[i] - delta
        X[r, j] = p[j] + delta
    return X
