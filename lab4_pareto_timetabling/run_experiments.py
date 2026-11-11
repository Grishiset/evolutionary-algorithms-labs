"""ЛР № 4 (вариант 8): сравнение NSGA-II, меметического NSGA-II и взвешенной свёртки
для многокритериального расписания; экспорт фронта для интерактивного выбора.

    python run_experiments.py                     # 30 запусков × 3 алгоритма, seed 1..30, ~5 мин
    python run_experiments.py --runs 10 --out results_quick
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from moea import Config, hypervolume, igd, nondominated, nsga2, weighted_sum
from timetable import OBJ_NAMES, OBJ_SHORT, S, Instance

HERE = Path(__file__).parent
WS_SCALE = np.array([10.0, 2.0, 40.0])        # характерные масштабы критериев для свёртки
ALGOS = ["NSGA-II", "Меметический NSGA-II", "Взвешенная свёртка (10 весов)"]


def _run(args):
    algo, seed, cfg_d = args
    inst = Instance()
    cfg = Config(**cfg_d)
    ev = lambda X: inst.evaluate(X)[:2]         # noqa: E731
    if algo == ALGOS[0]:
        return algo, nsga2(ev, inst.E, S, cfg, seed)
    if algo == ALGOS[1]:
        return algo, nsga2(ev, inst.E, S, cfg, seed, memetic=True, conflicts_of=inst.conflicting_events,
                           scale=WS_SCALE)
    return algo, weighted_sum(ev, inst.E, S, cfg, seed, WS_SCALE)


def final_front(r):
    ok = r.CV == 0
    F, X = r.F[ok], r.X[ok]
    nd = nondominated(F)
    F, X = F[nd], X[nd]
    _, u = np.unique(F, axis=0, return_index=True)
    return F[u], X[u]


def a12(x, y):
    """Мера эффекта Варга–Делани: P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None]).sum() + 0.5 * (x[:, None] == y[None]).sum()) / (len(x) * len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.runs))
    cfg = Config(**{k: v for k, v in json.loads((HERE / "config.json").read_text(encoding="utf-8")).items()
                    if k in Config.__dataclass_fields__})
    inst = Instance()

    jobs = [(al, s, asdict(cfg)) for al in ALGOS for s in seeds]
    with Pool(a.workers) as pool:
        done = pool.map(_run, jobs)
    res = {al: [r for name, r in done if name == al] for al in ALGOS}

    # общая нормировка и эталонный фронт — по объединению всех найденных допустимых решений
    fronts = {al: [final_front(r) for r in rs] for al, rs in res.items()}
    allF = np.vstack([F for al in ALGOS for F, _ in fronts[al] if len(F)])
    ref_front = np.unique(allF[nondominated(allF)], axis=0)
    ideal, nadir = ref_front.min(0), ref_front.max(0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    norm = lambda F: (F - ideal) / span          # noqa: E731
    REF = np.full(3, 1.1)
    hv = lambda F: hypervolume(norm(F), REF) if len(F) else 0.0   # noqa: E731

    rows, metrics = [], {}
    for al in ALGOS:
        m = {"hv": [], "igd": [], "n": [], "feasible": [], "time": [], "evals": []}
        for r, (F, X) in zip(res[al], fronts[al]):
            m["hv"].append(hv(F))
            m["igd"].append(igd(norm(F), norm(ref_front)))
            m["n"].append(len(F))
            m["feasible"].append(len(F) > 0)
            m["time"].append(r.seconds)
            m["evals"].append(r.evaluations)
            rows.append({"algorithm": al, "seed": r.seed, "hypervolume": f"{m['hv'][-1]:.5f}",
                         "igd": f"{m['igd'][-1]:.5f}", "front_size": len(F), "evaluations": r.evaluations,
                         "seconds": f"{r.seconds:.2f}",
                         "front": json.dumps(F.tolist())})
        metrics[al] = {k: np.array(v, dtype=float) for k, v in m.items()}

    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    base = metrics[ALGOS[0]]
    md = [f"Бюджет: {cfg.budget} вычислений на запуск, {len(seeds)} запусков (seed {seeds[0]}…{seeds[-1]}). "
          f"Нормировка по эталонному фронту из {len(ref_front)} точек (объединение всех запусков всех алгоритмов): "
          f"идеал {ideal.tolist()}, надир {nadir.tolist()}, точка отсчёта HV = (1.1, 1.1, 1.1).\n",
          "| Алгоритм | HV лучший | HV средний | HV медиана | HV ст. откл. | HV худший | IGD медиана | "
          "Точек во фронте | Допустимых запусков | Время, с | p (HV) vs NSGA-II | A12 (HV) vs NSGA-II |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for al in ALGOS:
        m = metrics[al]
        p = np.nan if al == ALGOS[0] else mannwhitneyu(m["hv"], base["hv"]).pvalue
        A = np.nan if al == ALGOS[0] else a12(m["hv"], base["hv"])
        summary.append({"algorithm": al, "hv_best": m["hv"].max(), "hv_mean": m["hv"].mean(),
                        "hv_median": np.median(m["hv"]), "hv_std": m["hv"].std(ddof=1), "hv_worst": m["hv"].min(),
                        "igd_median": np.median(m["igd"]), "front_size_mean": m["n"].mean(),
                        "feasible_runs": m["feasible"].mean(), "time_s": m["time"].mean(),
                        "evaluations_mean": m["evals"].mean(), "p_hv": p, "a12_hv": A})
        md.append(f"| {al} | {m['hv'].max():.4f} | {m['hv'].mean():.4f} | {np.median(m['hv']):.4f} | "
                  f"{m['hv'].std(ddof=1):.4f} | {m['hv'].min():.4f} | {np.median(m['igd']):.4f} | "
                  f"{m['n'].mean():.1f} | {m['feasible'].mean():.0%} | {m['time'].mean():.1f} | "
                  f"{'—' if np.isnan(p) else f'{p:.2g}'} | {'—' if np.isnan(A) else f'{A:.2f}'} |")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        for s in summary:
            w.writerow({k: f"{v:.5g}" if isinstance(v, float) else v for k, v in s.items()})

    # ------------------------------------------ фронт для интерактивного выбора
    # объединяем фронты всех запусков лучшего по медиане HV алгоритма: пользователю — максимум вариантов
    best_algo = max(ALGOS, key=lambda al: np.median(metrics[al]["hv"]))
    FX = [(F, X) for F, X in fronts[best_algo] if len(F)]
    UF, UX = np.vstack([f for f, _ in FX]), np.vstack([x for _, x in FX])
    nd = nondominated(UF)
    UF, UX = UF[nd], UX[nd]
    _, u = np.unique(UF, axis=0, return_index=True)
    UF, UX = UF[u], UX[u]
    order = np.lexsort((UF[:, 2], UF[:, 1], UF[:, 0]))
    UF, UX = UF[order], UX[order]
    sols = []
    for k, (f, x) in enumerate(zip(UF, UX)):
        _, cv, hb = inst.evaluate(x[None])
        assert cv[0] == 0
        sols.append({"id": k, "objectives": f.tolist(), "slots": x.tolist(), "timetable": inst.timetable(x)})
    meta = {k: inst.raw[k] for k in ("days", "periods", "groups", "teachers", "rooms", "events")}
    (out / "front.json").write_text(json.dumps({"algorithm": best_algo, "objective_names": OBJ_NAMES,
                                                "objective_keys": OBJ_SHORT, "instance": meta,
                                                "solutions": sols}, ensure_ascii=False), encoding="utf-8")

    Fn = norm(UF)
    picks = {"Без окон для студентов": int(np.lexsort((Fn.sum(1), UF[:, 0]))[0]),
             "Удобное время (меньше 1-х и 6-х пар)": int(np.lexsort((Fn.sum(1), UF[:, 1]))[0]),
             "Удобное преподавателям": int(np.lexsort((Fn.sum(1), UF[:, 2]))[0]),
             "Сбалансированное (ближайшее к идеалу)": int(np.argmin(np.linalg.norm(Fn, axis=1)))}
    md += ["", f"### Характерные решения итогового фронта ({best_algo}, объединение {len(FX)} запусков: "
               f"{len(UF)} недоминируемых расписаний)\n",
           "| Решение | id | " + " | ".join(OBJ_NAMES) + " |", "|---|---|---|---|---|"]
    rep = ["# Характерные расписания\n"]
    for label, i in picks.items():
        md.append(f"| {label} | {i} | {UF[i, 0]:g} | {UF[i, 1]:.2f} | {UF[i, 2]:g} |")
        rep += [f"## {label} (id {i}): окна {UF[i, 0]:g}, неудобное время {UF[i, 1]:.2f}, "
                f"преподаватели {UF[i, 2]:g}\n", inst.group_grid(UX[i], 0), ""]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "representative.md").write_text("\n".join(rep), encoding="utf-8")

    # ------------------------------------------ графики
    colors = dict(zip(ALGOS, ["tab:blue", "tab:green", "tab:red"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for al in ALGOS[:2]:
        snaps = [r.hist for r in res[al]]
        L = min(len(h) for h in snaps)
        fe = [h[i][0] for h, i in zip([snaps[0]] * L, range(L))]
        H = np.array([[hv(h[i][2]) for i in range(L)] for h in snaps])
        Q = np.array([[h[i][1] for i in range(L)] for h in snaps])
        axes[0].plot(fe, np.median(H, axis=0), color=colors[al], label=al)
        axes[0].fill_between(fe, np.percentile(H, 10, axis=0), np.percentile(H, 90, axis=0),
                             color=colors[al], alpha=0.15)
        axes[1].plot(fe, Q.mean(axis=0), color=colors[al], label=al)
    axes[0].axhline(np.median(metrics[ALGOS[2]]["hv"]), color=colors[ALGOS[2]], ls="--",
                    label=f"{ALGOS[2]}: итог, медиана")
    axes[0].set(xlabel="Вычисления", ylabel="HV (норм.)", title="Динамика гиперобъёма: медиана и 10–90 %")
    axes[1].set(xlabel="Вычисления", ylabel="Доля допустимых", title="Доля допустимых расписаний в популяции")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes, ["hv", "igd"], ["Гиперобъём (больше — лучше)", "IGD (меньше — лучше)"]):
        ax.boxplot([metrics[al][key] for al in ALGOS], tick_labels=[al.replace(" (", "\n(") for al in ALGOS])
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "metrics_boxplot.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        for al in ALGOS:
            F = np.vstack([F for F, _ in fronts[al] if len(F)])
            F = F[nondominated(F)]
            ax.scatter(F[:, i], F[:, j], s=14 if al != ALGOS[2] else 30, alpha=0.6, color=colors[al],
                       marker="o" if al != ALGOS[2] else "x", label=f"{al}: объединение 30 запусков")
        ax.set(xlabel=OBJ_NAMES[i], ylabel=OBJ_NAMES[j])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Проекции недоминируемых множеств")
    fig.tight_layout()
    fig.savefig(out / "pareto_projections.png", dpi=130)
    plt.close(fig)
    (out / "experiment_config.json").write_text(json.dumps({"config": asdict(cfg), "seeds": seeds,
                                                            "ws_scale": WS_SCALE.tolist()}, ensure_ascii=False,
                                                           indent=2), encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
