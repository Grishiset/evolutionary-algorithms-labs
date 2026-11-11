"""Все эксперименты лабораторной работы № 1 (вариант 8, функция Захарова, d = 7).

    python run_experiments.py                # 30 запусков на конфигурацию, seed = 1..30
    python run_experiments.py --runs 20 --seed0 100 --out results_alt

Результаты: results/runs.csv, results/summary.csv, results/summary.md, графики *.png.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from baselines import fixed_step_local_search, random_search
from ga import GAConfig, run_ga

SUCCESS_EPS = 1e-4   # порог «успешного» запуска: f(x) < 1e-4


def stats(vals: np.ndarray) -> dict:
    return {"best": vals.min(), "mean": vals.mean(), "median": np.median(vals),
            "std": vals.std(ddof=1), "worst": vals.max(),
            "success_rate": float(np.mean(vals < SUCCESS_EPS))}


def band_plot(ax, results, label, color):
    """Медиана и диапазон min–max лучшего значения по серии запусков."""
    L = min(len(r.hist_best) for r in results)
    fe = results[0].hist_fe[:L]
    H = np.array([r.hist_best[:L] for r in results])
    ax.plot(fe, np.median(H, axis=0), color=color, label=f"{label}: медиана")
    ax.plot(fe, H.mean(axis=0), color=color, ls="--", lw=0.9, label=f"{label}: среднее")
    ax.fill_between(fe, H.min(axis=0), H.max(axis=0), color=color, alpha=0.18,
                    label=f"{label}: min–max")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--out", default=str(Path(__file__).with_name("results")))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.runs))
    base = GAConfig.from_json(args.config)

    # однофакторные сравнения: меняется ровно один параметр относительно base
    factors = {
        "selection": [("tournament", {}), ("rank", {"selection": "rank"}),
                      ("roulette", {"selection": "roulette"})],
        "mutation_scale": [("adaptive σ (1/5)", {}),
                           ("fixed σ=0.1", {"adaptive_sigma": False}),
                           ("fixed σ=0.01", {"adaptive_sigma": False, "sigma0": 0.01})],
        "pop_size": [("N=30", {"pop_size": 30}), ("N=50", {}), ("N=100", {"pop_size": 100})],
        "crossover": [("SBX", {}), ("BLX-0.5", {"crossover": "blx"}),
                      ("arithmetic", {"crossover": "arith"})],
    }

    rows, summary, cfg_dump = [], [], {"base": asdict(base), "seeds": seeds}
    cache: dict[str, list] = {}

    def run_series(name: str, cfg: GAConfig):
        key = json.dumps(asdict(cfg), sort_keys=True)
        if key not in cache:
            cache[key] = [run_ga(cfg, s) for s in seeds]
        return cache[key]

    def record(experiment, name, results, base_vals=None, cfg=None):
        vals = np.array([r.best_f for r in results])
        for r in results:
            rows.append({"experiment": experiment, "config": name, "seed": r.seed,
                         "best_f": f"{r.best_f:.6e}", "evaluations": r.evaluations,
                         "generations": r.generations, "seconds": f"{r.seconds:.4f}",
                         "best_x": " ".join(f"{v:.6g}" for v in r.best_x)})
        s = stats(vals)
        p = np.nan
        if base_vals is not None and not np.array_equal(vals, base_vals):
            p = mannwhitneyu(vals, base_vals, alternative="two-sided").pvalue
        summary.append({"experiment": experiment, "config": name, "runs": len(vals), **s,
                        "evals": results[0].evaluations,
                        "mean_time_s": np.mean([r.seconds for r in results]),
                        "p_vs_base": p})
        if cfg is not None:
            cfg_dump[f"{experiment}/{name}"] = asdict(cfg)
        return vals

    # 1. базовая серия
    base_res = run_series("base", base)
    base_vals = record("base", "GA base", base_res, cfg=base)

    # 2. сравнение с базовыми методами при том же бюджете
    rs = [random_search(s, base.budget) for s in seeds]
    ls = [fixed_step_local_search(s, base.budget) for s in seeds]
    record("baselines", "GA base", base_res)
    record("baselines", "random search", rs, base_vals)
    record("baselines", "local search, fixed step 0.05", ls, base_vals)

    # 3. однофакторные сравнения
    factor_results = {}
    for fname, variants in factors.items():
        factor_results[fname] = []
        for label, over in variants:
            cfg = replace(base, **over)
            res = run_series(label, cfg)
            vals = record(f"factor:{fname}", label, res, base_vals, cfg)
            factor_results[fname].append((label, vals, res))

    # 4. комбинация лучших уровней факторов (вывод из однофакторных сравнений)
    tuned = replace(base, pop_size=30, crossover="blx")
    record("tuned", "N=30 + BLX-0.5", run_series("tuned", tuned), base_vals, tuned)

    # ---------- таблицы
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        for s in summary:
            w.writerow({k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in s.items()})
    (out / "experiments_config.json").write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    md = ["| Эксперимент | Конфигурация | Лучшее | Среднее | Медиана | Ст. откл. | Худшее | "
          f"Доля f<{SUCCESS_EPS:g} | p (Манн–Уитни vs base) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        p = "—" if np.isnan(s["p_vs_base"]) else f"{s['p_vs_base']:.2g}"
        md.append(f"| {s['experiment']} | {s['config']} | {s['best']:.2e} | {s['mean']:.2e} | "
                  f"{s['median']:.2e} | {s['std']:.2e} | {s['worst']:.2e} | "
                  f"{s['success_rate']:.0%} | {p} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------- графики
    fig, ax = plt.subplots(figsize=(8, 5))
    band_plot(ax, base_res, "ГА", "tab:blue")
    ax.set(yscale="log", xlabel="Число вычислений f", ylabel="Лучшее f(x) к моменту",
           title=f"Сходимость ГА, функция Захарова d=7, {len(seeds)} запусков")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "convergence_base.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    band_plot(ax, base_res, "ГА", "tab:blue")
    band_plot(ax, rs, "случайный поиск", "tab:red")
    band_plot(ax, ls, "локальный поиск (шаг 0.05)", "tab:green")
    ax.set(yscale="log", xlabel="Число вычислений f", ylabel="Лучшее f(x)",
           title="ГА против базовых методов при равном бюджете")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ga_vs_baselines.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(factor_results), figsize=(4.2 * len(factor_results), 4.2))
    for ax, (fname, items) in zip(axes, factor_results.items()):
        ax.boxplot([v for _, v, _ in items], tick_labels=[l for l, _, _ in items])
        ax.set(yscale="log", title=fname)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Итоговое лучшее f(x)")
    fig.suptitle(f"Однофакторные сравнения ({len(seeds)} запусков на конфигурацию)")
    fig.tight_layout()
    fig.savefig(out / "factors_boxplot.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(factor_results), figsize=(4.2 * len(factor_results), 4.0))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, (fname, items) in zip(axes, factor_results.items()):
        for (label, _, res), c in zip(items, colors):
            L = min(len(r.hist_best) for r in res)
            H = np.array([r.hist_best[:L] for r in res])
            ax.plot(res[0].hist_fe[:L], np.median(H, axis=0), color=c, label=label)
        ax.set(yscale="log", title=fname, xlabel="вычисления f")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Медиана лучшего f(x)")
    fig.tight_layout()
    fig.savefig(out / "factors_convergence.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in base_res[:5]:
        ax.plot(r.hist_fe, r.hist_sigma, lw=1, label=f"seed {r.seed}")
    ax.set(yscale="log", xlabel="Число вычислений f", ylabel="σ (доля ширины области)",
           title="Адаптация масштаба мутации по правилу 1/5")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sigma_adaptation.png", dpi=130)
    plt.close(fig)

    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
