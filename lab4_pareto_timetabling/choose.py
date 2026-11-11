"""Консольный выбор расписания из фронта Парето (дополнение к pareto_viewer.html).

    python choose.py                                   # интерактивный диалог
    python choose.py --max 6 2.0 40 --weights 2 1 1 --group 0      # без диалога
    python choose.py --id 12 --teacher 4               # показать конкретное решение
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from timetable import Instance

HERE = Path(__file__).parent


def recommend(F: np.ndarray, max_v, weights):
    lo, hi = F.min(0), F.max(0)
    ok = np.all(F <= np.asarray(max_v) + 1e-9, axis=1)
    if not ok.any():
        return None, ok
    Fn = (F - lo) / np.where(hi > lo, hi - lo, 1)
    score = np.where(ok, Fn @ np.asarray(weights, float), np.inf)
    return int(np.argmin(score)), ok


def ask(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return s or default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", default=str(HERE / "results" / "front.json"))
    ap.add_argument("--max", nargs=3, type=float, help="пороги «не больше» для трёх критериев")
    ap.add_argument("--weights", nargs=3, type=float, help="важность критериев")
    ap.add_argument("--id", type=int, help="показать решение с этим номером")
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--teacher", type=int)
    a = ap.parse_args()
    data = json.loads(Path(a.front).read_text(encoding="utf-8"))
    names = data["objective_names"]
    F = np.array([s["objectives"] for s in data["solutions"]])
    inst = Instance()
    print(f"Фронт: {len(F)} расписаний ({data['algorithm']}).")
    for k, n in enumerate(names):
        print(f"  {n}: от {F[:, k].min():g} до {F[:, k].max():g}")

    if a.id is not None:
        i = a.id
    else:
        interactive = a.max is None and a.weights is None
        max_v = a.max or [F[:, k].max() for k in range(3)]
        w = a.weights or [1, 1, 1]
        if interactive:
            max_v = [float(ask(f"{n}: не больше", f"{F[:, k].max():g}")) for k, n in enumerate(names)]
            w = [float(ask(f"Важность «{n}» (0–5)", "1")) for n in names]
        i, ok = recommend(F, max_v, w)
        print(f"Проходят ограничения: {int(ok.sum())} из {len(F)}")
        if i is None:
            print("Ни одно расписание не удовлетворяет ограничениям — ослабьте пороги.")
            return
    x = np.array(data["solutions"][i]["slots"])
    print(f"\nВыбрано расписание #{i}: " + "; ".join(f"{n} = {v:g}" for n, v in zip(names, F[i])))
    print()
    print(inst.group_grid(x, a.group))
    if a.teacher is not None:
        t = a.teacher
        print(f"\nЗанятия: {inst.raw['teachers'][t]['name']}")
        for r in sorted(inst.timetable(x), key=lambda r: (r["day"], r["period"])):
            if r["teacher"] == t:
                print(f"  {inst.raw['days'][r['day']]} {inst.raw['periods'][r['period']]}  {r['title']} ({r['kind']}), "
                      f"{', '.join(inst.raw['groups'][g]['name'] for g in r['groups'])}")


if __name__ == "__main__":
    main()
