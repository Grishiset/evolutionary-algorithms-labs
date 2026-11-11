"""Все эксперименты ЛР № 2 (вариант 8).

    python run_experiments.py                 # 30 запусков, seed 1..30, ~5 мин на 2 ядрах
    python run_experiments.py --runs 20 --seed0 101 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from baselines import greedy, milp_optimum, random_feasible_search
from ga import GAConfig, crossover, run_ga
from problem import Instance

HERE = Path(__file__).parent


def _job(args):
    cfg_dict, seed = args
    inst = Instance.load()
    return run_ga(inst, GAConfig(**cfg_dict), seed)


def _rs_job(args):
    seed, budget = args
    inst = Instance.load()
    t0 = time.perf_counter()
    a, f, hist = random_feasible_search(inst, seed, budget)
    return (seed, a, float(inst.objective(a)[0]), int(inst.total_violation(a)[0]), hist,
            time.perf_counter() - t0)


def viol_str(inst, a):
    v = {k: int(x[0]) for k, x in inst.violations(a).items()}
    names = {"capacity": "переполнение команд", "min_size": "недобор до минимума",
             "skills": "нет носителя навыка", "conflicts": "конфликтующие пары вместе"}
    return ", ".join(f"{names[k]}: {x}" for k, x in v.items()) + f" — **всего {sum(v.values())}**"


def main():
    warnings.filterwarnings("ignore", message="All-NaN slice")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.runs))
    inst = Instance.load()
    base = GAConfig.from_json(HERE / "config.json")

    opt_a, opt_cost, status, opt_t = milp_optimum(inst)
    assert status == 0 and inst.total_violation(opt_a)[0] == 0

    configs = {
        "constraints": [("ремонт (base)", {}),
                        ("штраф λ=2", {"constraint": "penalty", "penalty": 2.0}),
                        ("штраф λ=10", {"constraint": "penalty", "penalty": 10.0}),
                        ("штраф λ=50", {"constraint": "penalty", "penalty": 50.0}),
                        ("штраф λ=10, бюджет ×5 (≈равное время)",
                         {"constraint": "penalty", "penalty": 10.0, "budget": 5 * base.budget})],
        "mutation": [("mixed: swap+guided (base)", {}), ("reset", {"mutation": "reset"}),
                     ("swap", {"mutation": "swap"}), ("guided", {"mutation": "guided"})],
        "crossover": [("uniform (base)", {}), ("team", {"crossover": "team"})],
    }
    cache, results = {}, {}
    with Pool(args.workers) as pool:
        for group, items in configs.items():
            for label, over in items:
                cfg = replace(base, **over)
                key = json.dumps(asdict(cfg), sort_keys=True)
                if key not in cache:
                    cache[key] = pool.map(_job, [(asdict(cfg), s) for s in seeds])
                results[(group, label)] = (cfg, cache[key])
        rs = pool.map(_rs_job, [(s, base.budget) for s in seeds])
    greedy_sol, greedy_t = [], []
    for s in seeds:
        t0 = time.perf_counter()
        greedy_sol.append(greedy(inst, s))
        greedy_t.append(time.perf_counter() - t0)

    base_costs = np.array([r.best_cost for r in results[("constraints", "ремонт (base)")][1]])
    rows, summary, cfg_dump = [], [], {"seeds": seeds, "milp_optimum": opt_cost}

    def add(group, label, costs, feas, evals, times, sols, extra=None):
        costs = np.array(costs, dtype=float)
        fc = costs[np.array(feas)]
        p = np.nan
        if not np.array_equal(costs, base_costs) and len(fc) == len(costs):
            p = mannwhitneyu(costs, base_costs, alternative="two-sided").pvalue
        s = {"group": group, "config": label, "runs": len(costs),
             "feasible_rate": float(np.mean(feas)),
             "best": fc.min() if len(fc) else np.nan, "mean": fc.mean() if len(fc) else np.nan,
             "median": np.median(fc) if len(fc) else np.nan,
             "std": fc.std(ddof=1) if len(fc) > 1 else np.nan, "worst": fc.max() if len(fc) else np.nan,
             "optimum_hit_rate": float(np.mean(costs[np.array(feas)] == opt_cost)) if len(fc) else 0.0,
             "gap_median_%": 100 * (np.median(fc) - opt_cost) / opt_cost if len(fc) else np.nan,
             "evals": int(np.mean(evals)), "mean_time_s": float(np.mean(times)), "p_vs_base": p}
        if extra:
            s.update(extra)
        summary.append(s)
        for seed, c, f, sol in zip(seeds, costs, feas, sols):
            rows.append({"group": group, "config": label, "seed": seed, "cost": c, "feasible": bool(f),
                         "assignment": " ".join(map(str, sol))})

    for (group, label), (cfg, res) in results.items():
        cfg_dump[f"{group}/{label}"] = asdict(cfg)
        add(group, label, [r.best_cost for r in res], [r.feasible for r in res],
            [r.evaluations for r in res], [r.seconds for r in res], [r.best for r in res],
            {"final_pop_feasible_share": float(np.mean([r.hist_feas_share[-1] for r in res]))})
    add("baselines", "ГА base", base_costs, [True] * len(seeds), [base.budget] * len(seeds),
        [r.seconds for r in results[("constraints", "ремонт (base)")][1]],
        [r.best for r in results[("constraints", "ремонт (base)")][1]])
    add("baselines", "жадная эвристика + ремонт", [inst.objective(g)[0] for g in greedy_sol],
        [inst.total_violation(g)[0] == 0 for g in greedy_sol], [1] * len(seeds), greedy_t,
        greedy_sol)
    add("baselines", "случайный поиск + ремонт", [r[2] for r in rs], [r[3] == 0 for r in rs],
        [base.budget] * len(seeds), [r[5] for r in rs], [r[1] for r in rs])
    add("baselines", "MILP (HiGHS), точный оптимум", [opt_cost] * len(seeds), [True] * len(seeds),
        [0] * len(seeds), [opt_t] * len(seeds), [opt_a] * len(seeds))

    # ------------------------------------------------ таблицы
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = list(dict.fromkeys(k for s in summary for k in s))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for s in summary:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in s.items()})
    (out / "experiments_config.json").write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    f2 = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"  # noqa: E731
    md = [f"Точный оптимум (MILP): **{opt_cost:g}**, найден за {opt_t:.2f} с.\n",
          "| Группа | Конфигурация | Допустимо | Лучшее | Среднее | Медиана | Ст. откл. | Худшее | "
          "Попаданий в оптимум | Разрыв медианы, % | Доля доп. в финальной популяции | Время, с | p vs base |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        p = "—" if np.isnan(s["p_vs_base"]) else f"{s['p_vs_base']:.2g}"
        fp = s.get("final_pop_feasible_share")
        md.append(f"| {s['group']} | {s['config']} | {s['feasible_rate']:.0%} | {f2(s['best'])} | "
                  f"{f2(s['mean'])} | {f2(s['median'])} | {f2(s['std'])} | {f2(s['worst'])} | "
                  f"{s['optimum_hit_rate']:.0%} | {f2(s['gap_median_%'])} | "
                  f"{'—' if fp is None else f'{fp:.0%}'} | {s['mean_time_s']:.2f} | {p} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ------------------------------------------------ лучшее решение ГА
    base_res = results[("constraints", "ремонт (base)")][1]
    best_run = min(base_res, key=lambda r: (r.best_cost, r.seed))
    g_best = min(greedy_sol, key=lambda g: inst.objective(g)[0])
    txt = [f"# Лучшее решение ГА (seed {best_run.seed}), стоимость {best_run.best_cost:g}\n",
           f"Оптимум MILP: {opt_cost:g}. Нарушения: {viol_str(inst, best_run.best)}\n",
           "Распределение мест проекта в списках студентов: "
           + ", ".join(f"{k}: {v}" for k, v in inst.rank_histogram(best_run.best).items()) + "\n",
           inst.describe(best_run.best)]
    (out / "best_solution.md").write_text("\n".join(txt), encoding="utf-8")

    # ------------------------------------------------ примеры допустимых / недопустимых решений
    rng = np.random.default_rng(0)
    first_choice = np.array([s["prefs"][0] for s in inst.raw["students"]])
    child, _ = crossover(best_run.best, g_best, inst, "uniform", rng)
    rand = np.random.default_rng(1).integers(0, inst.P, inst.n)
    ex = ["# Примеры допустимых и недопустимых решений\n",
          "Вектор назначений: i-я позиция — номер проекта студента i (0…9).\n"]
    for title, a in [("Допустимое 1: лучшее решение ГА", best_run.best),
                     ("Допустимое 2: жадная эвристика + ремонт (лучший из 30 порядков)", g_best),
                     ("Недопустимое 1: каждому — его первый выбор", first_choice),
                     ("Недопустимое 2: потомок равномерного кроссовера двух допустимых решений", child),
                     ("Недопустимое 3: случайный вектор", rand),
                     ("То же после процедуры ремонта", inst.repair(rand))]:
        ex += [f"## {title}\n", f"`{' '.join(map(str, a))}`\n",
               f"Стоимость {inst.objective(a)[0]:g}; нарушения: {viol_str(inst, a)}\n"]
    (out / "examples.md").write_text("\n".join(ex), encoding="utf-8")

    # ------------------------------------------------ графики
    def med_curve(res):
        L = min(len(r.hist_best) for r in res)
        H = np.array([r.hist_best[:L] for r in res], dtype=float)
        return res[0].hist_fe[:L], H

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for (label, _), c in zip(configs["constraints"], colors):
        fe, H = med_curve(results[("constraints", label)][1])
        Hm = np.where(np.isinf(H), np.nan, H)
        axes[0].plot(fe, np.nanmedian(Hm, axis=0), color=c, label=label)
        axes[0].fill_between(fe, np.nanmin(Hm, axis=0), np.nanmax(Hm, axis=0), color=c, alpha=0.12)
        res = results[("constraints", label)][1]
        L = min(len(r.hist_feas_share) for r in res)
        axes[1].plot(fe[:L], np.mean([r.hist_feas_share[:L] for r in res], axis=0), color=c, label=label)
    axes[0].axhline(opt_cost, color="k", ls=":", label=f"оптимум MILP = {opt_cost:g}")
    axes[0].set(ylim=(opt_cost - 5, 150), xlabel="Вычисления фитнес-функции",
                ylabel="Лучшая допустимая стоимость", title="Сходимость: медиана и min–max (30 запусков)")
    axes[1].set(xlabel="Вычисления фитнес-функции", ylabel="Доля допустимых особей",
                title="Допустимость популяции", ylim=(-0.02, 1.02))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "constraints_convergence.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), gridspec_kw={"width_ratios": [5, 4, 2, 3]})
    for ax, group in zip(axes, ["constraints", "mutation", "crossover"]):
        labs = [l for l, _ in configs[group]]
        data = [[r.best_cost for r in results[(group, l)][1] if r.feasible] for l in labs]
        ax.boxplot(data, tick_labels=[l.replace(" (base)", "\n(base)").replace(", ", "\n") for l in labs])
        ax.axhline(opt_cost, color="k", ls=":", lw=1)
        ax.set_title(group)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3)
    data = [base_costs, [inst.objective(g)[0] for g in greedy_sol], [r[2] for r in rs]]
    axes[3].boxplot(data, tick_labels=["ГА", "жадная", "случайный\nпоиск"])
    axes[3].axhline(opt_cost, color="k", ls=":", lw=1, label="оптимум")
    axes[3].set_title("сравнение с базовыми")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)
    axes[0].set_ylabel("Итоговая стоимость (меньше — лучше)")
    fig.tight_layout()
    fig.savefig(out / "comparison_boxplots.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    cats = list(inst.rank_histogram(opt_a).keys())
    xs = np.arange(len(cats))
    for k, (name, a) in enumerate([("ГА (лучший запуск)", best_run.best),
                                   ("жадная эвристика (лучшая)", g_best), ("MILP оптимум", opt_a)]):
        ax.bar(xs + (k - 1) * 0.27, list(inst.rank_histogram(a).values()), width=0.27, label=name)
    ax.set_xticks(xs, cats)
    ax.set(ylabel="Число студентов", title="Какое по счёту желание студента исполнено")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "satisfaction.png", dpi=130)
    plt.close(fig)

    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
