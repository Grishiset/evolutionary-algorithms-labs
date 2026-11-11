"""Вещественный генетический алгоритм для непрерывной минимизации.

Все эволюционные операторы реализованы вручную, numpy используется только
для векторной арифметики и генератора случайных чисел.

Запуск одного прогона:
    python ga.py --config config.json --seed 1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from zakharov import DIM, bounds, zakharov


@dataclass
class GAConfig:
    pop_size: int = 50              # размер популяции
    budget: int = 20_000            # лимит вычислений целевой функции (критерий остановки)
    selection: str = "tournament"   # tournament | rank | roulette
    tournament_k: int = 3
    rank_pressure: float = 1.7      # s в линейном ранжировании, 1 < s <= 2
    crossover: str = "sbx"          # sbx | blx | arith
    p_crossover: float = 0.9
    sbx_eta: float = 15.0
    blx_alpha: float = 0.5
    p_mutation: float = -1.0        # вероятность мутации гена; <0 означает 1/d
    sigma0: float = 0.1             # начальный масштаб мутации (доля ширины области)
    adaptive_sigma: bool = True     # адаптация масштаба по правилу 1/5 успеха
    sigma_min: float = 1e-12
    sigma_max: float = 0.5
    adapt_factor: float = 0.9
    elite: int = 2                  # число элитных особей
    boundary: str = "reflect"       # reflect | clip
    dim: int = DIM

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> "GAConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        data.update(overrides)
        return cls(**data)


@dataclass
class RunResult:
    seed: int
    best_x: np.ndarray
    best_f: float
    evaluations: int
    generations: int
    seconds: float
    hist_fe: np.ndarray = field(repr=False)      # число вычислений f после каждого поколения
    hist_best: np.ndarray = field(repr=False)    # лучшее найденное значение к этому моменту
    hist_sigma: np.ndarray = field(repr=False)   # масштаб мутации


# ---------------------------------------------------------------- селекция
def select(fit: np.ndarray, n: int, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    """Возвращает индексы n родителей (минимизация)."""
    N = fit.size
    if cfg.selection == "tournament":
        cand = rng.integers(0, N, size=(n, cfg.tournament_k))
        return cand[np.arange(n), np.argmin(fit[cand], axis=1)]
    if cfg.selection == "rank":
        # линейное ранжирование: лучший получает s/N, худший (2-s)/N
        order = np.argsort(fit)
        ranks = np.empty(N, dtype=int)
        ranks[order] = np.arange(N)                     # 0 — лучший
        s = cfg.rank_pressure
        p = (2 - s) / N + 2 * (s - 1) * (N - 1 - ranks) / (N * (N - 1))
        return rng.choice(N, size=n, p=p)
    if cfg.selection == "roulette":
        # масштабирование «окном»: g = f_max - f + eps, чтобы перейти к максимизации
        g = fit.max() - fit + 1e-12 * (1 + abs(fit.max()))
        return rng.choice(N, size=n, p=g / g.sum())
    raise ValueError(f"unknown selection {cfg.selection}")


# ---------------------------------------------------------------- кроссовер
def crossover(p1: np.ndarray, p2: np.ndarray, cfg: GAConfig, rng: np.random.Generator):
    m, d = p1.shape
    if cfg.crossover == "sbx":
        u = rng.random((m, d))
        e = 1.0 / (cfg.sbx_eta + 1.0)
        beta = np.where(u <= 0.5, (2 * u) ** e, (1 / (2 * (1 - u))) ** e)
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    elif cfg.crossover == "blx":
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        span = hi - lo
        a = cfg.blx_alpha
        c1 = rng.uniform(lo - a * span, hi + a * span)
        c2 = rng.uniform(lo - a * span, hi + a * span)
    elif cfg.crossover == "arith":
        lam = rng.random((m, 1))
        c1 = lam * p1 + (1 - lam) * p2
        c2 = (1 - lam) * p1 + lam * p2
    else:
        raise ValueError(f"unknown crossover {cfg.crossover}")
    # пары, к которым кроссовер не применяется, копируются без изменений
    keep = rng.random(m) >= cfg.p_crossover
    c1[keep], c2[keep] = p1[keep], p2[keep]
    return c1, c2


# ---------------------------------------------------------------- мутация и границы
def mutate(x: np.ndarray, sigma: float, width: np.ndarray, pm: float, rng: np.random.Generator):
    mask = rng.random(x.shape) < pm
    return x + mask * rng.normal(0.0, 1.0, x.shape) * sigma * width


def repair_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, mode: str) -> np.ndarray:
    if mode == "reflect":
        w = hi - lo
        y = np.mod(x - lo, 2 * w)                    # «зеркальное» отражение от границ
        x = lo + np.where(y > w, 2 * w - y, y)
    return np.clip(x, lo, hi)


# ---------------------------------------------------------------- основной цикл
def run_ga(cfg: GAConfig, seed: int, f: Callable[[np.ndarray], np.ndarray] = zakharov) -> RunResult:
    rng = np.random.default_rng(seed)
    lo, hi = bounds(cfg.dim)
    width = hi - lo
    N, d = cfg.pop_size, cfg.dim
    pm = cfg.p_mutation if cfg.p_mutation >= 0 else 1.0 / d
    sigma = cfg.sigma0
    t0 = time.perf_counter()

    pop = rng.uniform(lo, hi, size=(N, d))            # инициализация в допустимой области
    fit = f(pop)
    fe = N
    best_i = int(np.argmin(fit))
    best_x, best_f = pop[best_i].copy(), float(fit[best_i])
    h_fe, h_best, h_sigma = [fe], [best_f], [sigma]
    gen = 0

    while fe < cfg.budget:
        n_off = min(N - cfg.elite, cfg.budget - fe)
        n_pairs = (n_off + 1) // 2
        i1 = select(fit, n_pairs, cfg, rng)
        i2 = select(fit, n_pairs, cfg, rng)
        c1, c2 = crossover(pop[i1], pop[i2], cfg, rng)
        kids = np.vstack([c1, c2])[:n_off]
        ref = np.concatenate([fit[i1], fit[i2]])[:n_off]   # «свой» родитель каждого потомка
        kids = mutate(kids, sigma, width, pm, rng)
        kids = repair_bounds(kids, lo, hi, cfg.boundary)
        kfit = f(kids)
        fe += n_off

        if cfg.adaptive_sigma:                        # правило 1/5 успеха
            success = float(np.mean(kfit < ref))
            sigma = sigma / cfg.adapt_factor if success > 0.2 else sigma * cfg.adapt_factor
            sigma = float(np.clip(sigma, cfg.sigma_min, cfg.sigma_max))

        # элитизм: N - n_off лучших старых особей переходят без изменений
        keep = np.argsort(fit)[: N - n_off]
        pop = np.vstack([pop[keep], kids])
        fit = np.concatenate([fit[keep], kfit])

        gen += 1
        i = int(np.argmin(fit))
        if fit[i] < best_f:
            best_f, best_x = float(fit[i]), pop[i].copy()
        h_fe.append(fe)
        h_best.append(best_f)
        h_sigma.append(sigma)

    return RunResult(seed, best_x, best_f, fe, gen, time.perf_counter() - t0,
                     np.array(h_fe), np.array(h_best), np.array(h_sigma))


def main() -> None:
    ap = argparse.ArgumentParser(description="ГА для функции Захарова (вариант 8)")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()
    over = {"budget": args.budget} if args.budget else {}
    cfg = GAConfig.from_json(args.config, **over)
    r = run_ga(cfg, args.seed)
    print(json.dumps({"config": asdict(cfg), "seed": r.seed, "best_f": r.best_f,
                      "best_x": np.round(r.best_x, 8).tolist(), "evaluations": r.evaluations,
                      "generations": r.generations, "seconds": round(r.seconds, 3)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
