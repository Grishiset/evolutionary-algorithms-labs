import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import greedy, milp_optimum  # noqa: E402
from ga import GAConfig, crossover, mutate, run_ga  # noqa: E402
from generate_data import generate  # noqa: E402
from problem import Instance  # noqa: E402

INST = Instance.load()
CFG = GAConfig.from_json(ROOT / "config.json")


def test_instance_matches_generator_and_minimum_scale():
    import json
    assert json.loads((ROOT / "data" / "instance.json").read_text(encoding="utf-8")) == \
        json.loads(json.dumps(generate(2026, 60, 10), ensure_ascii=False))
    assert INST.n >= 20 and INST.P >= 5 and len(INST.conflicts) >= 10
    assert INST.cap.sum() >= INST.n >= INST.min_size.sum()


def test_violation_counting():
    a = np.zeros(INST.n, dtype=int)                 # все в проекте 0
    v = {k: int(x[0]) for k, x in INST.violations(a).items()}
    assert v["capacity"] == INST.n - INST.cap[0]
    assert v["min_size"] == INST.min_size[1:].sum()
    assert v["conflicts"] == len(INST.conflicts)


def test_repair_makes_random_vectors_feasible_mostly():
    rng = np.random.default_rng(0)
    ok = [INST.total_violation(INST.repair(rng.integers(0, INST.P, INST.n)))[0] == 0 for _ in range(200)]
    assert np.mean(ok) > 0.95


def test_swap_mutation_preserves_team_sizes():
    rng = np.random.default_rng(1)
    a = greedy(INST, 0)
    b = mutate(a, INST, "swap", 2.0, rng)
    assert np.array_equal(np.bincount(a, minlength=INST.P), np.bincount(b, minlength=INST.P))


def test_crossover_outputs_valid_project_ids():
    rng = np.random.default_rng(2)
    a, b = greedy(INST, 0), greedy(INST, 1)
    for kind in ("uniform", "team"):
        c1, c2 = crossover(a, b, INST, kind, rng)
        assert c1.min() >= 0 and c1.max() < INST.P and c2.shape == a.shape


def test_ga_reproducible_and_not_better_than_optimum():
    cfg = replace(CFG, budget=1200)
    r1, r2 = run_ga(INST, cfg, 5), run_ga(INST, cfg, 5)
    assert r1.best_cost == r2.best_cost and np.array_equal(r1.best, r2.best)
    _, opt, status, _ = milp_optimum(INST)
    assert status == 0 and r1.feasible and r1.best_cost >= opt
    assert INST.total_violation(r1.best)[0] == 0
