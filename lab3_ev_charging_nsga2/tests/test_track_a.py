import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model  # noqa: E402
from nsga2 import (MOConfig, crowding_distance, fast_nondominated_sort, hypervolume,  # noqa: E402
                   nondominated, nsga2)


def brute_rank(F, CV):
    n = len(F)
    rank = np.full(n, -1)
    left, r = set(range(n)), 0
    from nsga2 import constrained_dominates
    while left:
        front = [i for i in left if not any(constrained_dominates(F[j], CV[j], F[i], CV[i]) for j in left if j != i)]
        for i in front:
            rank[i] = r
        left -= set(front)
        r += 1
    return rank


def test_nondominated_sort_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        F = rng.integers(0, 5, (30, 3)).astype(float)
        CV = np.where(rng.random(30) < 0.3, rng.integers(1, 4, 30), 0).astype(float)
        fronts = fast_nondominated_sort(F, CV)
        rank = np.empty(30, int)
        for r, fr in enumerate(fronts):
            rank[fr] = r
        assert np.array_equal(rank, brute_rank(F, CV))


def test_crowding_boundary_infinite():
    F = np.array([[0, 3], [1, 2], [2, 1], [3, 0]], float)
    d = crowding_distance(F)
    assert np.isinf(d[0]) and np.isinf(d[3]) and np.allclose(d[1:3], [4 / 3, 4 / 3])


def test_hypervolume_known_values():
    assert np.isclose(hypervolume(np.array([[0.5, 0.5]]), np.array([1, 1])), 0.25)
    assert np.isclose(hypervolume(np.array([[0, 0, 0]]), np.ones(3)), 1.0)
    P = np.array([[0.2, 0.6, 0.4], [0.6, 0.2, 0.4], [0.4, 0.4, 0.1]])
    rng = np.random.default_rng(1)
    S = rng.random((200_000, 3))
    mc = np.mean(np.any(np.all(P[None] <= S[:, None], axis=2), axis=1))
    assert abs(hypervolume(P, np.ones(3)) - mc) < 0.01


def test_decoder_respects_physics():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, model.P_MAX, (50, model.T))
    F, cv, actual, traj = model.evaluate(X)
    assert np.all(model.B + actual <= model.G_MAX + 1e-9)
    assert np.all(traj <= 100 + 1e-9) and np.all(np.diff(traj, axis=1) >= -1e-12)
    assert np.allclose(F[:, 0], (actual * model.C).sum(1) * model.DT)


def test_lp_reference_is_a_lower_bound():
    lp = model.lp_reference()
    p = np.where(model.C == model.C.min(), model.P_MAX, 0.0)        # заряд только ночью на полной мощности
    F, _, _, _ = model.evaluate(p[None])
    assert -F[0, 2] >= 99.9 and F[0, 0] >= lp["min_cost@100%"] - 1e-6


def test_nsga2_reproducible_and_feasible_front():
    lo, hi = np.zeros(model.T), np.full(model.T, model.P_MAX)
    ev = lambda X: model.evaluate(X)[:2]            # noqa: E731
    cfg = replace(MOConfig(), pop_size=40, generations=30)
    a = nsga2(ev, lo, hi, cfg, 3, extra_mutation=model.shift_mutation)
    b = nsga2(ev, lo, hi, cfg, 3, extra_mutation=model.shift_mutation)
    assert np.array_equal(a.F, b.F)
    assert np.any(a.CV == 0) and nondominated(a.F[a.CV == 0]).any()
