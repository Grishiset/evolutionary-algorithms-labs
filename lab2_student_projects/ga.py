"""ГА для назначения студентов на проекты (вектор назначений).

    python ga.py --seed 1                       # параметры из config.json
    python ga.py --seed 1 --constraint penalty --penalty 5
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from problem import DEFAULT_INSTANCE, Instance

BIG_PENALTY = 1000.0      # штраф за остаточные нарушения после ремонта (в режиме repair)


@dataclass
class GAConfig:
    pop_size: int = 60
    budget: int = 12_000              # лимит вычислений фитнес-функции
    tournament_k: int = 3
    crossover: str = "uniform"        # uniform | team
    p_crossover: float = 0.9
    mutation: str = "mixed"           # reset | swap | guided | mixed
    mutation_lambda: float = 0.5      # число ходов мутации = 1 + Poisson(lambda)
    constraint: str = "repair"        # repair | penalty
    penalty: float = 50.0             # λ — штраф за одно нарушение (режим penalty)
    init: str = "random"              # random | preference
    elite: int = 2

    @classmethod
    def from_json(cls, path, **over):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d.update({k: v for k, v in over.items() if v is not None})
        return cls(**d)


@dataclass
class RunResult:
    seed: int
    best: np.ndarray                  # лучшее допустимое решение (или лучшее по фитнесу, если допустимых нет)
    best_cost: float                  # его стоимость (без штрафов)
    feasible: bool
    evaluations: int
    seconds: float
    hist_fe: np.ndarray = field(repr=False)
    hist_best: np.ndarray = field(repr=False)      # лучшая допустимая стоимость к моменту (inf, если нет)
    hist_feas_share: np.ndarray = field(repr=False)  # доля допустимых особей в популяции


# ------------------------------------------------------------------ операторы
def crossover(p1, p2, inst: Instance, kind: str, rng):
    if kind == "uniform":
        m = rng.random(p1.size) < 0.5
        return np.where(m, p1, p2), np.where(m, p2, p1)
    if kind == "team":
        # командный кроссовер: ребёнок наследует целиком команды части проектов от одного
        # родителя, остальные студенты сохраняют проект другого родителя, если тот не занят
        # унаследованными командами; иначе — лучший по предпочтениям свободный проект.
        kids = []
        for a, b in ((p1, p2), (p2, p1)):
            S = rng.random(inst.P) < 0.5
            child = np.where(S[a], a, -1)
            counts = np.bincount(child[child >= 0], minlength=inst.P)
            for i in np.flatnonzero(child < 0):
                if not S[b[i]]:
                    child[i] = b[i]
                else:
                    free = [j for j in np.argsort(inst.cost[i]) if not S[j]] or [b[i]]
                    child[i] = free[0]
                counts[child[i]] += 1
            kids.append(child)
        return kids[0], kids[1]
    raise ValueError(kind)


def mutate(a, inst: Instance, kind: str, lam: float, rng):
    a = a.copy()
    for _ in range(1 + rng.poisson(lam)):
        k = kind if kind != "mixed" else ("swap" if rng.random() < 0.5 else "guided")
        i = rng.integers(inst.n)
        if k == "reset":                    # случайный новый проект
            a[i] = rng.integers(inst.P)
        elif k == "swap":                   # обмен проектами двух студентов: размеры команд не меняются
            j = rng.integers(inst.n)
            a[i], a[j] = a[j], a[i]
        elif k == "guided":                 # перевод в один из трёх любимых проектов студента
            top = np.argsort(inst.cost[i] + 1e-6 * rng.random(inst.P))[:3]
            a[i] = rng.choice(top)
        else:
            raise ValueError(kind)
    return a


class Evaluator:
    """Считает фитнес и число вычислений. В режиме repair ремонт ламарковский:
    исправленный генотип записывается обратно в особь."""

    def __init__(self, inst: Instance, cfg: GAConfig):
        self.inst, self.cfg, self.fe = inst, cfg, 0

    def __call__(self, A: np.ndarray):
        if self.cfg.constraint == "repair":
            A = np.array([self.inst.repair(a) for a in A])
        cost = self.inst.objective(A)
        viol = self.inst.total_violation(A)
        lam = BIG_PENALTY if self.cfg.constraint == "repair" else self.cfg.penalty
        self.fe += len(A)
        return A, cost + lam * viol, cost, viol


def init_population(inst, cfg, rng):
    if cfg.init == "random":
        return rng.integers(0, inst.P, size=(cfg.pop_size, inst.n))
    # каждому студенту — случайный проект из его списка с вероятностью ~ 1/место
    P = np.zeros((cfg.pop_size, inst.n), dtype=int)
    for i in range(inst.n):
        w = np.where(inst.rank[i] > 0, 1.0 / np.maximum(inst.rank[i], 1), 0.0)
        P[:, i] = rng.choice(inst.P, size=cfg.pop_size, p=w / w.sum())
    return P


def run_ga(inst: Instance, cfg: GAConfig, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    ev = Evaluator(inst, cfg)
    t0 = time.perf_counter()
    pop, fit, cost, viol = ev(init_population(inst, cfg, rng))
    best, best_cost = None, np.inf

    def update_best():
        nonlocal best, best_cost
        feas = np.flatnonzero(viol == 0)
        if feas.size:
            i = feas[np.argmin(cost[feas])]
            if cost[i] < best_cost:
                best, best_cost = pop[i].copy(), float(cost[i])

    update_best()
    h_fe, h_best, h_feas = [ev.fe], [best_cost], [float(np.mean(viol == 0))]
    N = cfg.pop_size
    while ev.fe < cfg.budget:
        n_off = min(N - cfg.elite, cfg.budget - ev.fe)
        kids = []
        while len(kids) < n_off:
            c = rng.integers(0, N, size=(2, cfg.tournament_k))
            i1, i2 = (row[np.argmin(fit[row])] for row in c)
            if rng.random() < cfg.p_crossover:
                c1, c2 = crossover(pop[i1], pop[i2], inst, cfg.crossover, rng)
            else:
                c1, c2 = pop[i1].copy(), pop[i2].copy()
            kids += [mutate(c1, inst, cfg.mutation, cfg.mutation_lambda, rng),
                     mutate(c2, inst, cfg.mutation, cfg.mutation_lambda, rng)]
        K, kf, kc, kv = ev(np.array(kids[:n_off]))
        keep = np.argsort(fit)[: N - n_off]                      # элитизм
        pop = np.vstack([pop[keep], K])
        fit = np.concatenate([fit[keep], kf])
        cost = np.concatenate([cost[keep], kc])
        viol = np.concatenate([viol[keep], kv])
        update_best()
        h_fe.append(ev.fe)
        h_best.append(best_cost)
        h_feas.append(float(np.mean(viol == 0)))

    feasible = best is not None
    if not feasible:                                             # допустимых не нашли
        i = int(np.argmin(fit))
        best, best_cost = pop[i].copy(), float(cost[i])
    return RunResult(seed, best, best_cost, feasible, ev.fe, time.perf_counter() - t0,
                     np.array(h_fe), np.array(h_best), np.array(h_feas))


def main():
    ap = argparse.ArgumentParser(description="ГА: назначение студентов на проекты (вариант 8)")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--instance", default=str(DEFAULT_INSTANCE))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--constraint", choices=["repair", "penalty"])
    ap.add_argument("--penalty", type=float)
    ap.add_argument("--mutation", choices=["reset", "swap", "guided", "mixed"])
    ap.add_argument("--crossover", choices=["uniform", "team"])
    ap.add_argument("--budget", type=int)
    ap.add_argument("--show", action="store_true", help="напечатать составы команд")
    a = ap.parse_args()
    inst = Instance.load(a.instance)
    cfg = GAConfig.from_json(a.config, constraint=a.constraint, penalty=a.penalty,
                             mutation=a.mutation, crossover=a.crossover, budget=a.budget)
    r = run_ga(inst, cfg, a.seed)
    print(json.dumps({"config": asdict(cfg), "seed": a.seed, "cost": r.best_cost,
                      "feasible": r.feasible, "evaluations": r.evaluations,
                      "seconds": round(r.seconds, 2), "ranks": inst.rank_histogram(r.best),
                      "assignment": r.best.tolist()}, ensure_ascii=False))
    if a.show:
        print(inst.describe(r.best))


if __name__ == "__main__":
    main()
