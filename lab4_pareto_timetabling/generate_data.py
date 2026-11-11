"""Синтетический экземпляр задачи составления недельного расписания занятий (ЛР № 4, вариант 8).

    python generate_data.py --seed 2026 --out data/instance.json

Процедура генерации (все случайности — из numpy.random.default_rng(seed)):
* 5 дней × 6 пар = 30 слотов; 8 учебных групп по 20–28 студентов, два потока по 4 группы;
* на поток 4 лекционных курса по 2 лекции в неделю; лектор курса читает его обоим потокам;
* у каждой группы 6 семинаров (4 курса + программирование + английский; 6 семинаристов,
  распределение случайное) и 2 лабораторные (2 преподавателя лабораторных);
  итого 16 занятий у группы в неделю, 80 занятий всего;
* аудитории: 2 поточные (120 и 110 мест), 1 на 60, 4 семинарские (30–35), 2 лаборатории (30);
* у каждого преподавателя 3–6 недоступных слотов и 2 предпочитаемых дня из 5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт"]
PERIODS = ["09:00", "10:40", "12:20", "14:00", "15:40", "17:20"]
COURSES = ["Матанализ", "Алгоритмы", "Физика", "Теория вероятностей"]
SEMINARS = COURSES + ["Программирование", "Английский"]


def generate(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = [{"id": g, "name": f"Б26-{50 + g}", "size": int(rng.integers(20, 29)), "stream": g // 4}
              for g in range(8)]
    teachers = []
    for t in range(4):
        teachers.append({"id": t, "name": f"Лектор {COURSES[t]}", "role": "lecturer"})
    for t in range(6):
        teachers.append({"id": 4 + t, "name": f"Семинарист {t + 1}", "role": "seminar"})
    for t in range(2):
        teachers.append({"id": 10 + t, "name": f"Лаборант {t + 1}", "role": "lab"})
    for t in teachers:
        k = int(rng.integers(3, 7))
        t["unavailable"] = sorted(rng.choice(30, size=k, replace=False).tolist())
        t["preferred_days"] = sorted(rng.choice(5, size=2, replace=False).tolist())

    events = []

    def add(kind, title, grps, teacher, lab=False):
        size = sum(groups[g]["size"] for g in grps)
        events.append({"id": len(events), "kind": kind, "title": title, "groups": grps,
                       "teacher": teacher, "size": size, "lab": lab})

    for s in range(2):
        stream = [g["id"] for g in groups if g["stream"] == s]
        for c in range(4):
            for k in range(2):
                add("лекция", COURSES[c], stream, c)
    for g in range(8):
        for c in range(6):
            add("семинар", SEMINARS[c], [g], 4 + int(rng.integers(0, 6)))
        for k in range(2):
            add("лабораторная", ["Физпрактикум", "Практикум по электронике"][k], [g], 10 + k, lab=True)

    rooms = [{"id": 0, "name": "Ауд. 100 (поточная)", "capacity": 120, "lab": False},
             {"id": 1, "name": "Ауд. 200 (поточная)", "capacity": 110, "lab": False},
             {"id": 2, "name": "Ауд. 301", "capacity": 60, "lab": False},
             {"id": 3, "name": "Ауд. 302", "capacity": 35, "lab": False},
             {"id": 4, "name": "Ауд. 303", "capacity": 32, "lab": False},
             {"id": 5, "name": "Ауд. 304", "capacity": 30, "lab": False},
             {"id": 6, "name": "Ауд. 305", "capacity": 30, "lab": False},
             {"id": 7, "name": "Лаб. 1", "capacity": 30, "lab": True},
             {"id": 8, "name": "Лаб. 2", "capacity": 30, "lab": True}]
    return {"seed": seed, "days": DAYS, "periods": PERIODS, "groups": groups, "teachers": teachers,
            "rooms": rooms, "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).with_name("data") / "instance.json"))
    a = ap.parse_args()
    inst = generate(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(inst, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.out}: {len(inst['events'])} занятий, {len(inst['groups'])} групп, "
          f"{len(inst['teachers'])} преподавателей, {len(inst['rooms'])} аудиторий")


if __name__ == "__main__":
    main()
