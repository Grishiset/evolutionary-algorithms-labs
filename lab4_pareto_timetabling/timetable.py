"""Модель многокритериального расписания (ЛР № 4, вариант 8).

Генотип: x ∈ {0..29}^E — слот (день × пара) каждого занятия.
Фенотип: расписание «занятие → (слот, аудитория)». Аудитории назначает декодер:
в каждом слоте занятия по убыванию размера получают наибольшие свободные аудитории
подходящего типа (для «пороговой» структуры вместимостей это оптимально:
назначение существует тогда и только тогда, когда k-е по размеру занятие помещается
в k-ю по вместимости аудиторию для всех k).

Жёсткие ограничения (CV — суммарное число нарушений):
  H1 у группы не более одного занятия в слоте;
  H2 у преподавателя не более одного занятия в слоте;
  H3 преподаватель не ставится в недоступный слот;
  H4 каждому занятию нашлась аудитория нужного типа и вместимости.
Критерии (минимизация):
  f1 «окна» студентов: Σ по группам и дням числа пустых пар между первой и последней;
  f2 неудобное время: число студенто-занятий на 1-й (09:00) и 6-й (17:20) парах, в сотнях студентов;
  f3 неудобство преподавателей: занятия в непредпочитаемые дни + окна преподавателей.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).with_name("data") / "instance.json"
N_DAYS, N_PER = 5, 6
S = N_DAYS * N_PER
OBJ_NAMES = ["Окна студентов", "Неудобное время, сотни студ.-занятий", "Неудобство преподавателей"]
OBJ_SHORT = ["windows", "bad_time", "teacher"]


def _windows(occ: np.ndarray) -> np.ndarray:
    """occ: (..., D, P) bool -> число «окон» по последней оси, суммированное по дням."""
    cnt = occ.sum(-1)
    first = np.argmax(occ, axis=-1)
    last = occ.shape[-1] - 1 - np.argmax(occ[..., ::-1], axis=-1)
    w = np.where(cnt > 0, last - first + 1 - cnt, 0)
    return w.sum(-1)


class Instance:
    def __init__(self, path: str | Path = DEFAULT):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.raw = d
        ev = d["events"]
        self.E, self.G, self.T = len(ev), len(d["groups"]), len(d["teachers"])
        self.size = np.array([e["size"] for e in ev])
        self.teacher = np.array([e["teacher"] for e in ev])
        self.is_lab = np.array([e["lab"] for e in ev])
        self.G_of = np.zeros((self.E, self.G), dtype=bool)
        for e in ev:
            self.G_of[e["id"], e["groups"]] = True
        pairs_g, pairs_t = [], []
        for i in range(self.E):
            for j in range(i + 1, self.E):
                if np.any(self.G_of[i] & self.G_of[j]):
                    pairs_g.append((i, j))
                if self.teacher[i] == self.teacher[j]:
                    pairs_t.append((i, j))
        self.pairs_g = np.array(pairs_g)
        self.pairs_t = np.array(pairs_t)
        self.unavail = np.zeros((self.T, S), dtype=bool)
        self.pref_day = np.zeros((self.T, N_DAYS), dtype=bool)
        for t in d["teachers"]:
            self.unavail[t["id"], t["unavailable"]] = True
            self.pref_day[t["id"], t["preferred_days"]] = True
        rooms = d["rooms"]
        self.room_cap = {lab: np.array(sorted([r["capacity"] for r in rooms if r["lab"] == lab], reverse=True))
                         for lab in (False, True)}
        # порядок занятий по убыванию размера — для проверки вместимости
        self.order = {lab: np.flatnonzero(self.is_lab == lab)[np.argsort(-self.size[self.is_lab == lab],
                                                                          kind="stable")]
                      for lab in (False, True)}

    # ------------------------------------------------------------- оценка популяции
    def evaluate(self, X: np.ndarray):
        X = np.atleast_2d(X)
        m = X.shape[0]
        # H1, H2 — совпадающие слоты в парах конфликтующих занятий
        h1 = (X[:, self.pairs_g[:, 0]] == X[:, self.pairs_g[:, 1]]).sum(1)
        h2 = (X[:, self.pairs_t[:, 0]] == X[:, self.pairs_t[:, 1]]).sum(1)
        h3 = self.unavail[self.teacher[None, :], X].sum(1)
        # H4 — ранг занятия среди занятий своего типа в том же слоте (по убыванию размера)
        h4 = np.zeros(m, dtype=int)
        for lab in (False, True):
            o = self.order[lab]
            caps = self.room_cap[lab]
            onehot = X[:, o][:, :, None] == np.arange(S)              # (m, n, S)
            rank = (np.cumsum(onehot, axis=1) - 1)[onehot].reshape(m, len(o))
            fits = np.where(rank < len(caps), caps[np.minimum(rank, len(caps) - 1)] >= self.size[o], False)
            h4 += (~fits).sum(1)
        cv = h1 + h2 + h3 + h4
        # критерии
        day, per = X // N_PER, X % N_PER
        occ_g = np.zeros((m, self.G, N_DAYS, N_PER), dtype=bool)
        for g in range(self.G):
            ev = np.flatnonzero(self.G_of[:, g])
            occ_g[np.arange(m)[:, None], g, day[:, ev], per[:, ev]] = True
        f1 = _windows(occ_g).sum(1)
        bad = (per == 0) | (per == N_PER - 1)
        f2 = (bad * self.size).sum(1) / 100.0
        occ_t = np.zeros((m, self.T, N_DAYS, N_PER), dtype=bool)
        for t in range(self.T):
            ev = np.flatnonzero(self.teacher == t)
            occ_t[np.arange(m)[:, None], t, day[:, ev], per[:, ev]] = True
        not_pref = (~self.pref_day[self.teacher[None, :], day]).sum(1)
        f3 = not_pref + _windows(occ_t).sum(1)
        F = np.column_stack([f1, f2, f3]).astype(float)
        return F, cv.astype(float), np.column_stack([h1, h2, h3, h4])

    # ------------------------------------------------------------- фенотип
    def assign_rooms(self, x: np.ndarray) -> np.ndarray:
        """Аудитории для фенотипа. В каждом слоте занятия по убыванию размера получают
        наименьшую свободную подходящую аудиторию (best-fit): семинар не займёт поточную.
        Для вложенных вместимостей best-fit размещает всех тогда и только тогда, когда
        выполнено условие из evaluate (H4 = 0). Если в слоте кто-то не поместился,
        используется схема «k-е по размеру занятие — в k-ю по вместимости аудиторию»,
        чтобы число неразмещённых совпадало с H4."""
        rooms = self.raw["rooms"]
        out = np.full(self.E, -1)
        for s in range(S):
            for lab in (False, True):
                ev = [e for e in self.order[lab] if x[e] == s]
                pool = sorted([r for r in rooms if r["lab"] == lab], key=lambda r: r["capacity"])
                taken, best = set(), {}
                for e in ev:
                    fit = [r for r in pool if r["id"] not in taken and r["capacity"] >= self.size[e]]
                    if not fit:
                        break
                    best[e] = fit[0]["id"]
                    taken.add(fit[0]["id"])
                if len(best) == len(ev):
                    for e, r in best.items():
                        out[e] = r
                    continue
                for e, r in zip(ev, pool[::-1]):
                    if r["capacity"] >= self.size[e]:
                        out[e] = r["id"]
        return out

    def timetable(self, x: np.ndarray) -> list[dict]:
        rooms = self.assign_rooms(x)
        d = self.raw
        res = []
        for e in d["events"]:
            s = int(x[e["id"]])
            res.append({"event": e["id"], "day": s // N_PER, "period": s % N_PER,
                        "room": int(rooms[e["id"]]), "title": e["title"], "kind": e["kind"],
                        "groups": e["groups"], "teacher": e["teacher"]})
        return res

    def group_grid(self, x: np.ndarray, g: int) -> str:
        """Текстовая сетка расписания группы g (для отчёта и CLI)."""
        d = self.raw
        tt = [r for r in self.timetable(x) if g in r["groups"]]
        cell = {(r["day"], r["period"]): r for r in tt}
        head = "| Пара | " + " | ".join(d["days"]) + " |"
        lines = [f"**{d['groups'][g]['name']}**\n", head, "|---" * (N_DAYS + 1) + "|"]
        for p in range(N_PER):
            row = []
            for day in range(N_DAYS):
                r = cell.get((day, p))
                row.append("" if r is None else
                           f"{r['title']} ({r['kind'][:3]}.), {d['rooms'][r['room']]['name'] if r['room'] >= 0 else '—'}")
            lines.append(f"| {d['periods'][p]} | " + " | ".join(row) + " |")
        return "\n".join(lines)

    def conflicting_events(self, x: np.ndarray) -> np.ndarray:
        """Номера занятий, участвующих хотя бы в одном нарушении H1–H3 (для меметического поиска)."""
        bad = np.zeros(self.E, dtype=bool)
        for P in (self.pairs_g, self.pairs_t):
            hit = x[P[:, 0]] == x[P[:, 1]]
            bad[P[hit, 0]] = True
            bad[P[hit, 1]] = True
        bad |= self.unavail[self.teacher, x]
        return np.flatnonzero(bad)
