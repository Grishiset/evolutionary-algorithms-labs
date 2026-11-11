import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_data import generate  # noqa: E402
from moea import Config, fast_nondominated_sort, hypervolume, nondominated, nsga2, weighted_sum  # noqa: E402
from timetable import N_PER, S, Instance  # noqa: E402

I = Instance()


def brute(x):
    """Наивный подсчёт нарушений и критериев по определениям — для сверки с векторизованной оценкой."""
    d = I.raw
    ev = d["events"]
    h1 = sum(1 for a in ev for b in ev if a["id"] < b["id"] and set(a["groups"]) & set(b["groups"])
             and x[a["id"]] == x[b["id"]])
    h2 = sum(1 for a in ev for b in ev if a["id"] < b["id"] and a["teacher"] == b["teacher"] and x[a["id"]] == x[b["id"]])
    h3 = sum(1 for e in ev if x[e["id"]] in d["teachers"][e["teacher"]]["unavailable"])
    rooms = I.assign_rooms(x)
    h4 = int((rooms < 0).sum())

    def windows(slots):
        w = 0
        for day in range(5):
            ps = sorted(s % N_PER for s in slots if s // N_PER == day)
            if ps:
                w += ps[-1] - ps[0] + 1 - len(set(ps))
        return w
    f1 = sum(windows([x[e["id"]] for e in ev if g in e["groups"]]) for g in range(I.G))
    f2 = sum(e["size"] for e in ev if x[e["id"]] % N_PER in (0, N_PER - 1)) / 100
    f3 = sum(1 for e in ev if x[e["id"]] // N_PER not in d["teachers"][e["teacher"]]["preferred_days"]) + \
        sum(windows([x[e["id"]] for e in ev if e["teacher"] == t]) for t in range(I.T))
    return [f1, f2, f3], h1 + h2 + h3 + h4


def test_data_reproducible():
    assert json.loads((ROOT / "data" / "instance.json").read_text(encoding="utf-8")) == \
        json.loads(json.dumps(generate(2026), ensure_ascii=False))
    assert I.E == 80


def test_vectorized_evaluation_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.integers(0, S, (25, I.E))
    F, CV, _ = I.evaluate(X)
    for x, f, c in zip(X, F, CV):
        bf, bc = brute(x)
        assert np.allclose(f, bf) and c == bc


def test_room_assignment_respects_capacity_and_type():
    rng = np.random.default_rng(1)
    rooms = I.raw["rooms"]
    for x in rng.integers(0, S, (20, I.E)):
        r = I.assign_rooms(x)
        used = set()
        for e, ri in enumerate(r):
            if ri >= 0:
                assert rooms[ri]["capacity"] >= I.size[e] and rooms[ri]["lab"] == I.is_lab[e]
                assert (x[e], ri) not in used
                used.add((x[e], ri))


def test_sorting_and_hv():
    F = np.array([[1, 2, 3], [2, 1, 3], [2, 2, 4], [0, 0, 9]], float)
    fr = fast_nondominated_sort(F, np.zeros(4))
    assert set(fr[0]) == {0, 1, 3} and set(fr[1]) == {2}
    assert np.isclose(hypervolume(np.array([[0, 0, 0]]), np.ones(3)), 1.0)
    assert nondominated(F).tolist() == [True, True, False, True]


def test_algorithms_reproducible_and_feasible():
    ev = lambda X: I.evaluate(X)[:2]     # noqa: E731
    cfg = replace(Config(), budget=3000, pop_size=40)
    a = nsga2(ev, I.E, S, cfg, 5)
    b = nsga2(ev, I.E, S, cfg, 5)
    assert np.array_equal(a.F, b.F)
    m = nsga2(ev, I.E, S, cfg, 5, memetic=True, conflicts_of=I.conflicting_events, scale=np.array([10, 2, 40.]))
    assert m.evaluations <= cfg.budget + cfg.pop_size
    w = weighted_sum(ev, I.E, S, cfg, 5, np.array([10, 2, 40.]))
    assert w.evaluations <= cfg.budget and len(w.F) == 10


def test_front_export_is_feasible_if_present():
    p = ROOT / "results" / "front.json"
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    X = np.array([s["slots"] for s in data["solutions"]])
    F, CV, _ = I.evaluate(X)
    assert np.all(CV == 0)
    assert np.allclose(F, [s["objectives"] for s in data["solutions"]])
    assert nondominated(F).all()
