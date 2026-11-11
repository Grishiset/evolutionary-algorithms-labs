"""Генерация синтетического экземпляра задачи «назначение студентов на проекты».

    python generate_data.py --seed 2026 --students 60 --projects 10 --out data/instance.json

Процедура:
1. Каждому проекту назначается «популярность» w_j ~ LogNormal(0, 0.6). Популярные
   проекты создают конкуренцию за места.
2. Каждый студент ранжирует 5 различных проектов. Выбор без возвращения
   с вероятностями, пропорциональными w_j (модель Плакетта–Льюса).
3. Навыки: у каждого студента 1–2 навыка из {ML, backend, frontend, data},
   вероятности 0.3/0.3/0.25/0.15.
4. Проекту требуется 1–2 различных навыка: в команде должен быть хотя бы один
   их носитель.
5. Вместимость команды c_j ∈ {5, 6, 7}, минимальный размер m_j = 3.
6. Конфликтующие пары: 12 случайных пар студентов, которых нельзя ставить в одну команду.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SKILLS = ["ML", "backend", "frontend", "data"]
PROJECT_NAMES = ["Чат-бот поддержки", "Рекомендательная система", "Мобильное приложение кампуса",
                 "Мониторинг серверов", "Анализ успеваемости", "Компьютерное зрение для лабораторий",
                 "Веб-портал олимпиад", "Хранилище научных данных", "Голосовой ассистент",
                 "Игра для обучения"]


def generate(seed: int, n_students: int, n_projects: int, n_ranked: int = 5,
             n_conflicts: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    popularity = rng.lognormal(0.0, 0.6, n_projects)
    p = popularity / popularity.sum()
    prefs = [rng.choice(n_projects, size=n_ranked, replace=False, p=p).tolist()
             for _ in range(n_students)]
    skills = []
    for _ in range(n_students):
        k = 1 + int(rng.random() < 0.4)
        skills.append(sorted(rng.choice(len(SKILLS), size=k, replace=False,
                                        p=[0.3, 0.3, 0.25, 0.15]).tolist()))
    req = [sorted(rng.choice(len(SKILLS), size=1 + int(rng.random() < 0.4), replace=False).tolist())
           for _ in range(n_projects)]
    cap = rng.choice([5, 6, 7], size=n_projects, p=[0.25, 0.5, 0.25]).tolist()
    while sum(cap) < n_students + 4:                     # гарантируем запас мест
        cap[int(rng.integers(n_projects))] += 1
    conflicts = set()
    while len(conflicts) < n_conflicts:
        a, b = sorted(rng.choice(n_students, size=2, replace=False).tolist())
        conflicts.add((a, b))
    return {
        "seed": seed,
        "skills": SKILLS,
        "students": [{"id": i, "name": f"Студент {i + 1:02d}", "prefs": prefs[i],
                      "skills": skills[i]} for i in range(n_students)],
        "projects": [{"id": j, "name": PROJECT_NAMES[j % len(PROJECT_NAMES)],
                      "capacity": int(cap[j]), "min_size": 3, "required_skills": req[j],
                      "popularity": round(float(popularity[j]), 3)} for j in range(n_projects)],
        "conflicts": [list(c) for c in sorted(conflicts)],
        # штраф за попадание на k-й по предпочтению проект (k = 1..5) и вне списка
        "rank_cost": [0, 1, 2, 4, 7],
        "unranked_cost": 15,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--students", type=int, default=60)
    ap.add_argument("--projects", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).with_name("data") / "instance.json"))
    a = ap.parse_args()
    inst = generate(a.seed, a.students, a.projects)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(inst, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {a.out}: {a.students} students, {a.projects} projects, "
          f"capacity {sum(p['capacity'] for p in inst['projects'])}")


if __name__ == "__main__":
    main()
