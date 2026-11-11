import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ga import GAConfig, crossover, repair_bounds, run_ga, select  # noqa: E402
from zakharov import DIM, bounds, zakharov  # noqa: E402

CFG = GAConfig.from_json(Path(__file__).resolve().parents[1] / "config.json")


def test_variant_dimension():
    assert DIM == 7


def test_zakharov_values():
    assert zakharov(np.zeros(7))[0] == 0.0
    x = np.ones(3)          # s1 = 3, s2 = 0.5*(1+2+3) = 3 -> 3 + 9 + 81
    assert np.isclose(zakharov(x)[0], 93.0)
    assert zakharov(np.random.default_rng(0).uniform(-5, 10, (20, 7))).shape == (20,)


def test_reproducible_with_seed():
    cfg = replace(CFG, budget=2000)
    a, b = run_ga(cfg, 7), run_ga(cfg, 7)
    assert a.best_f == b.best_f and np.array_equal(a.best_x, b.best_x)
    assert run_ga(cfg, 8).best_f != a.best_f


def test_budget_and_bounds_respected():
    for sel in ("tournament", "rank", "roulette"):
        for cx in ("sbx", "blx", "arith"):
            r = run_ga(replace(CFG, budget=1500, selection=sel, crossover=cx), 3)
            assert r.evaluations == 1500
            lo, hi = bounds()
            assert np.all(r.best_x >= lo) and np.all(r.best_x <= hi)
            assert np.all(np.diff(r.hist_best) <= 0)          # элитизм: монотонность


def test_repair_bounds():
    lo, hi = np.zeros(2), np.ones(2)
    x = np.array([[1.2, -0.3], [3.5, 0.5]])
    y = repair_bounds(x, lo, hi, "reflect")
    assert np.allclose(y, [[0.8, 0.3], [0.5, 0.5]])
    assert np.all((y >= lo) & (y <= hi))


def test_operators_shapes_and_selection_pressure():
    rng = np.random.default_rng(0)
    p1, p2 = rng.random((10, 7)), rng.random((10, 7))
    c1, c2 = crossover(p1, p2, CFG, rng)
    assert c1.shape == c2.shape == (10, 7)
    fit = np.arange(50, dtype=float)
    idx = select(fit, 5000, CFG, rng)
    assert idx.mean() < 24.5          # турнир предпочитает лучшие (меньшие) значения
