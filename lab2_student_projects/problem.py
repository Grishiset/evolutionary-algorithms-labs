"""Модель задачи «назначение студентов на проекты» (вариант 8, ЛР № 2).

Особь — вектор назначений a ∈ {0..P-1}^n: a[i] — номер проекта студента i.

Жёсткие ограничения:
  (H1) |{i: a_i = j}| <= capacity_j   — вместимость команды;
  (H2) |{i: a_i = j}| >= min_size_j   — каждая команда укомплектована;
  (H3) для каждого требуемого навыка s проекта j в команде есть носитель s;
  (H4) конфликтующие пары (u, v) не попадают в одну команду.
Цель: минимизировать суммарную «неудовлетворённость» sum_i cost[i, a_i],
где cost зависит от места проекта в списке предпочтений студента.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_INSTANCE = Path(__file__).with_name("data") / "instance.json"


@dataclass
class Instance:
    n: int
    P: int
    cost: np.ndarray            # (n, P)
    rank: np.ndarray            # (n, P): 1..5 или 0, если проекта нет в списке
    cap: np.ndarray             # (P,)
    min_size: np.ndarray        # (P,)
    skill: np.ndarray           # (n, S) bool
    req: list[list[int]]        # требуемые навыки каждого проекта
    conflicts: np.ndarray       # (K, 2)
    partners: list[set[int]]    # конфликтующие партнёры каждого студента
    raw: dict

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INSTANCE) -> "Instance":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        n, P, S = len(d["students"]), len(d["projects"]), len(d["skills"])
        cost = np.full((n, P), float(d["unranked_cost"]))
        rank = np.zeros((n, P), dtype=int)
        skill = np.zeros((n, S), dtype=bool)
        for st in d["students"]:
            for k, j in enumerate(st["prefs"]):
                cost[st["id"], j] = d["rank_cost"][k]
                rank[st["id"], j] = k + 1
            skill[st["id"], st["skills"]] = True
        conf = np.array(d["conflicts"], dtype=int).reshape(-1, 2)
        partners = [set() for _ in range(n)]
        for u, v in conf:
            partners[u].add(int(v))
            partners[v].add(int(u))
        return cls(n, P, cost, rank,
                   np.array([p["capacity"] for p in d["projects"]]),
                   np.array([p["min_size"] for p in d["projects"]]),
                   skill, [p["required_skills"] for p in d["projects"]], conf, partners, d)

    # ------------------------------------------------------------ оценка
    def objective(self, A: np.ndarray) -> np.ndarray:
        A = np.atleast_2d(A)
        return self.cost[np.arange(self.n), A].sum(axis=1)

    def violations(self, A: np.ndarray) -> dict[str, np.ndarray]:
        """Число нарушений каждого вида для каждой особи популяции A (m, n)."""
        A = np.atleast_2d(A)
        counts = (A[:, :, None] == np.arange(self.P)).sum(axis=1)          # (m, P)
        over = np.maximum(0, counts - self.cap).sum(axis=1)
        under = np.maximum(0, self.min_size - counts).sum(axis=1)
        missing = np.zeros(A.shape[0], dtype=int)
        for j, skills in enumerate(self.req):
            for s in skills:
                holders = self.skill[:, s]
                missing += ~np.any(A[:, holders] == j, axis=1)
        conflict = (A[:, self.conflicts[:, 0]] == A[:, self.conflicts[:, 1]]).sum(axis=1)
        return {"capacity": over, "min_size": under, "skills": missing, "conflicts": conflict}

    def total_violation(self, A: np.ndarray) -> np.ndarray:
        return sum(self.violations(A).values())

    # ------------------------------------------------------------ ремонт
    def repair(self, a: np.ndarray, max_iter: int = 200) -> np.ndarray:
        """Жадное восстановление допустимости с минимальным ростом стоимости.

        Порядок: конфликты → переполнение → недобор → навыки. Каждый шаг выбирает
        перемещение (или обмен двух студентов) с наименьшим приростом cost
        среди тех, что не создают новых нарушений. Если ни одного такого хода нет,
        ремонт останавливается: остаток нарушений учитывается штрафом.
        """
        a = np.array(a, dtype=int).copy()
        members = [set(np.flatnonzero(a == j).tolist()) for j in range(self.P)]
        C, cap, mn, req, sk, prt = self.cost, self.cap, self.min_size, self.req, self.skill, self.partners

        def holders(j, s):
            return sum(1 for i in members[j] if sk[i, s])

        def needed(i, j):           # i — единственный носитель требуемого навыка в j
            return any(sk[i, s] and holders(j, s) == 1 for s in req[j])

        def ok_join(i, j, ignore=None):
            return not any((p in members[j]) and p != ignore for p in prt[i])

        def move(i, k):
            members[a[i]].discard(i)
            members[k].add(i)
            a[i] = k

        for _ in range(max_iter):
            changed = False
            # (H4) конфликты
            for u, v in self.conflicts:
                if a[u] != a[v]:
                    continue
                j, best = a[u], None
                for i in (u, v):
                    if needed(i, j):
                        continue
                    for k in range(self.P):
                        if k != j and len(members[k]) < cap[k] and ok_join(i, k):
                            d = C[i, k] - C[i, j]
                            if best is None or d < best[0]:
                                best = (d, i, k)
                if best:
                    move(best[1], best[2])
                    changed = True
            # (H1) переполнение
            for j in range(self.P):
                while len(members[j]) > cap[j]:
                    best = None
                    for i in members[j]:
                        if needed(i, j):
                            continue
                        for k in range(self.P):
                            if k != j and len(members[k]) < cap[k] and ok_join(i, k):
                                d = C[i, k] - C[i, j]
                                if best is None or d < best[0]:
                                    best = (d, i, k)
                    if not best:
                        break
                    move(best[1], best[2])
                    changed = True
            # (H2) недобор
            for j in range(self.P):
                while len(members[j]) < mn[j]:
                    best = None
                    for k in range(self.P):
                        if k == j or len(members[k]) <= mn[k]:
                            continue
                        for i in members[k]:
                            if not needed(i, k) and ok_join(i, j):
                                d = C[i, j] - C[i, k]
                                if best is None or d < best[0]:
                                    best = (d, i, j)
                    if not best:
                        break
                    move(best[1], best[2])
                    changed = True
            # (H3) навыки: перемещение носителя или обмен с участником команды j
            for j in range(self.P):
                for s in req[j]:
                    if holders(j, s) > 0:
                        continue
                    best = None
                    for k in range(self.P):
                        if k == j:
                            continue
                        for i in members[k]:
                            if not sk[i, s] or needed(i, k):
                                continue
                            if len(members[k]) > mn[k] and len(members[j]) < cap[j] and ok_join(i, j):
                                d = C[i, j] - C[i, k]
                                if best is None or d < best[0]:
                                    best = (d, i, None, j, k)
                            for t in members[j]:
                                if needed(t, j) or not ok_join(i, j, ignore=t) or not ok_join(t, k, ignore=i):
                                    continue
                                d = C[i, j] - C[i, k] + C[t, k] - C[t, j]
                                if best is None or d < best[0]:
                                    best = (d, i, t, j, k)
                    if best:
                        _, i, t, j2, k = best
                        move(i, j2)
                        if t is not None:
                            move(t, k)
                        changed = True
            if not changed:
                break
        return a

    # ------------------------------------------------------------ вывод
    def describe(self, a: np.ndarray) -> str:
        a = np.asarray(a)
        lines = []
        names = self.raw["skills"]
        for j, p in enumerate(self.raw["projects"]):
            mem = np.flatnonzero(a == j)
            req = ", ".join(names[s] for s in p["required_skills"])
            lines.append(f"### {j}. {p['name']} — {len(mem)}/{p['capacity']} (мин. {p['min_size']}), "
                         f"нужны навыки: {req}\n")
            lines.append("| Студент | Место проекта в списке | Навыки |\n|---|---|---|")
            for i in mem:
                r = self.rank[i, j]
                sks = ", ".join(names[s] for s in np.flatnonzero(self.skill[i]))
                lines.append(f"| {self.raw['students'][i]['name']} | {r if r else 'вне списка'} | {sks} |")
            lines.append("")
        return "\n".join(lines)

    def rank_histogram(self, a: np.ndarray) -> dict[str, int]:
        r = self.rank[np.arange(self.n), np.asarray(a)]
        h = {f"{k}-й выбор": int(np.sum(r == k)) for k in range(1, 6)}
        h["вне списка"] = int(np.sum(r == 0))
        return h
