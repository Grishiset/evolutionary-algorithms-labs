# Лабораторная работа № 2 — назначение студентов на проекты (вариант 8)

Генетический алгоритм для дискретной задачи с ограничениями. Особь — вектор назначений «студент → проект».
Учитываются предпочтения студентов, вместимость и минимальный размер команд, обязательные навыки и конфликтующие пары.

Отчёт: [REPORT.md](REPORT.md). Лучшее решение: [results/best_solution.md](results/best_solution.md).
Примеры допустимых и недопустимых решений: [results/examples.md](results/examples.md).

## Запуск

```bash
python -m pip install -r ../requirements.txt

# данные уже лежат в data/instance.json; пересоздать их (результат совпадёт байт в байт):
python generate_data.py --seed 2026 --students 60 --projects 10

# один прогон ГА с параметрами из config.json; --show печатает составы команд
python ga.py --seed 1 --show
python ga.py --seed 1 --constraint penalty --penalty 10
python ga.py --seed 1 --mutation guided

# все эксперименты: 30 запусков на конфигурацию (seed 1..30), около 5 мин на 2 ядрах
python run_experiments.py --workers 2

python -m pytest -q tests
```

## Файлы

| Файл | Назначение |
|---|---|
| `generate_data.py` | генератор синтетического экземпляра (процедура описана в docstring, seed 2026) |
| `data/instance.json` | 60 студентов, 10 проектов, 12 конфликтующих пар, требования к навыкам |
| `problem.py` | модель: стоимость, подсчёт нарушений (векторизованный), процедура ремонта, вывод решения |
| `ga.py` | ГА: турнир, равномерный и командный кроссовер, мутации reset/swap/guided/mixed, ремонт или штраф, элитизм |
| `baselines.py` | жадная эвристика, случайный допустимый поиск, точный оптимум через MILP (только эталон) |
| `run_experiments.py` | все эксперименты, таблицы, графики, примеры решений |
| `config.json` | базовые параметры ГА |
| `results/` | `runs.csv` (каждый запуск с вектором назначений), `summary.csv/.md`, `experiments_config.json`, графики |
