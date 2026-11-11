"""ЛР № 3, трек A (вариант 8): NSGA-II для планирования зарядки электромобиля.

    python run_experiments.py               # 20 запусков на алгоритм, seed 1..20
    python run_experiments.py --runs 30 --seed0 100
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

import model
from nsga2 import MOConfig, hypervolume, nondominated, nsga2, weighted_sum_ga

HERE = Path(__file__).parent
LO, HI = np.zeros(model.T), np.full(model.T, model.P_MAX)
# нормировка критериев для гиперобъёма и свёртки: [идеал, надир] по физическим границам
F_LO = np.array([0.0, model.B.max(), -100.0])                  # пик не может быть ниже пика базовой нагрузки
# стоимость: полная зарядка 20→100 % целиком по пиковому тарифу — верхняя оценка разумных решений
E_FULL = (100 - model.SOC0) / 100 * model.CAPACITY / model.EFF
F_HI = np.array([E_FULL * model.C.max(), model.G_MAX, -model.SOC_MIN])
REF = np.full(3, 1.1)
NAMES = ["Стоимость, руб.", "Пик нагрузки, кВт", "Итоговый заряд, %"]


def ev(X):
    F, cv, _, _ = model.evaluate(X)
    return F, cv


def norm(F):
    return (F - F_LO) / (F_HI - F_LO)


def hv(F):
    return hypervolume(norm(F), REF) if len(F) else 0.0


def front_of(res):
    ok = res.CV == 0
    F, X = res.F[ok], res.X[ok]
    nd = nondominated(F)
    F, X = F[nd], X[nd]
    _, u = np.unique(np.round(F, 6), axis=0, return_index=True)
    return F[u], X[u]


def simplex_weights(m, H):
    return np.array([np.array(c) / H for c in itertools.product(range(H + 1), repeat=m) if sum(c) == H])


def show(F):                       # в таблицах — заряд положительным числом
    return np.column_stack([F[:, 0], F[:, 1], -F[:, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.runs))
    cfg = MOConfig(**json.loads((HERE / "config.json").read_text(encoding="utf-8"))["nsga2"])
    ws_cfg = MOConfig(**json.loads((HERE / "config.json").read_text(encoding="utf-8"))["weighted_sum"])
    W = simplex_weights(3, 4)                                   # 15 векторов весов
    budget = cfg.pop_size * (cfg.generations + 1)

    shift_p = json.loads((HERE / "config.json").read_text(encoding="utf-8"))["shift_mutation_prob"]
    shift = lambda Q, rng: model.shift_mutation(Q, rng, shift_p)          # noqa: E731
    MAIN = "NSGA-II + перенос энергии"
    algos = {
        MAIN: lambda s: nsga2(ev, LO, HI, cfg, s, hv_fn=hv, extra_mutation=shift),
        "NSGA-II без переноса энергии": lambda s: nsga2(ev, LO, HI, cfg, s, hv_fn=hv),
        "NSGA-II без crowding (абляция)": lambda s: nsga2(ev, LO, HI, replace(cfg, use_crowding=False), s,
                                                          hv_fn=hv, extra_mutation=shift),
        "NSGA-II, разреженная инициализация": lambda s: nsga2(ev, LO, HI, replace(cfg, init="mixed"), s,
                                                              hv_fn=hv, extra_mutation=shift),
        "Взвешенная сумма (15 весов)": lambda s: weighted_sum_ga(ev, LO, HI, W, (F_LO, F_HI - F_LO),
                                                                  replace(ws_cfg, generations=cfg.generations), s,
                                                                  extra_mutation=shift),
    }
    res = {name: [f(s) for s in seeds] for name, f in algos.items()}

    rows, summary = [], []
    hv_base = np.array([hv(front_of(r)[0]) for r in res[MAIN]])
    for name, rs in res.items():
        hvs, sizes = [], []
        for r in rs:
            F, _ = front_of(r)
            hvs.append(hv(F))
            sizes.append(len(F))
            rows.append({"algorithm": name, "seed": r.seed, "hypervolume": f"{hvs[-1]:.5f}",
                         "front_size": len(F), "evaluations": r.evaluations, "seconds": f"{r.seconds:.2f}"})
        hvs = np.array(hvs)
        p = np.nan if name == MAIN else mannwhitneyu(hvs, hv_base).pvalue
        summary.append({"algorithm": name, "runs": len(hvs), "hv_best": hvs.max(), "hv_mean": hvs.mean(),
                        "hv_median": np.median(hvs), "hv_std": hvs.std(ddof=1), "hv_worst": hvs.min(),
                        "front_size_mean": np.mean(sizes), "evaluations": rs[0].evaluations,
                        "time_s": np.mean([r.seconds for r in rs]), "p_vs_nsga2": p})

    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        for s in summary:
            w.writerow({k: f"{v:.5g}" if isinstance(v, float) else v for k, v in s.items()})
    md = ["Гиперобъём допустимого недоминируемого множества в нормированном пространстве "
          f"(точка отсчёта {REF.tolist()}; больше — лучше). Бюджет {budget} вычислений на запуск.\n",
          "| Алгоритм | Лучший HV | Средний | Медиана | Ст. откл. | Худший | Размер фронта | Время, с | p vs основной |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        p = "—" if np.isnan(s["p_vs_nsga2"]) else f"{s['p_vs_nsga2']:.2g}"
        md.append(f"| {s['algorithm']} | {s['hv_best']:.4f} | {s['hv_mean']:.4f} | {s['hv_median']:.4f} | "
                  f"{s['hv_std']:.4f} | {s['hv_worst']:.4f} | {s['front_size_mean']:.1f} | {s['time_s']:.1f} | {p} |")

    # ------------------------------------------- точность крайних точек относительно LP-ориентиров
    lp = model.lp_reference()
    md += ["", "### Крайние точки фронта против точных LP-ориентиров (медиана по запускам)\n",
           "Для уровня заряда L берутся точки фронта с зарядом ≥ L − 0.5 %.\n",
           "| Алгоритм | " + " | ".join(f"мин. стоимость @{L}% (LP {lp[f'min_cost@{L}%']:.1f})" for L in (50, 80, 100))
           + " | " + " | ".join(f"мин. пик @{L}% (LP {lp[f'min_peak@{L}%']:.2f})" for L in (50, 80, 100)) + " |",
           "|---|" + "---|" * 6]
    for name, rs in res.items():
        cells = []
        for k in (0, 1):
            for L in (50, 80, 100):
                v = []
                for r in rs:
                    Fr, _ = front_of(r)
                    sel = Fr[-Fr[:, 2] >= L - 0.5]
                    v.append(sel[:, k].min() if len(sel) else np.nan)
                v = np.array(v)
                miss = int(np.isnan(v).sum())
                cells.append(f"{np.nanmedian(v):.2f}" + (f" ({miss} без точки)" if miss else ""))
        md.append(f"| {name} | " + " | ".join(cells) + " |")

    # ------------------------------------------- характерные решения из фронта лучшего запуска NSGA-II
    best_run = max(res[MAIN], key=lambda r: hv(front_of(r)[0]))
    F, X = front_of(best_run)
    Fn = norm(F)
    picks = {
        "Дешёвое (мин. стоимость при заряде ≥ 80 %)": int(np.argmin(np.where(-F[:, 2] >= 80, F[:, 0], np.inf))),
        "Полный заряд (макс. заряд, затем мин. стоимость)": int(np.lexsort((F[:, 0], np.round(F[:, 2], 1)))[0]),
        "Щадящее для сети (мин. пик при заряде ≥ 80 %)": int(np.argmin(np.where(-F[:, 2] >= 80, F[:, 1], np.inf))),
        "Сбалансированное (ближайшее к идеалу в норм. пространстве)": int(np.argmin(np.linalg.norm(Fn, axis=1))),
        "Минимально допустимое (самое дешёвое вообще)": int(np.argmin(F[:, 0])),
    }
    md += ["", f"### Характерные решения (основной алгоритм, запуск seed {best_run.seed}, фронт из {len(F)} точек)\n",
           "| Решение | Стоимость, руб. | Пик, кВт | Заряд, % | Энергия, кВт·ч | Доля энергии ночью |",
           "|---|---|---|---|---|---|"]
    rep = {}
    for label, i in picks.items():
        _, _, actual, traj = model.evaluate(X[i][None])
        e = actual[0] * model.DT
        night = e[model.C == model.C.min()].sum() / max(e.sum(), 1e-9)
        md.append(f"| {label} | {F[i, 0]:.1f} | {F[i, 1]:.2f} | {-F[i, 2]:.1f} | {e.sum():.1f} | {night:.0%} |")
        rep[label] = (actual[0], traj[0])
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    np.savetxt(out / "best_front.csv", np.column_stack([show(F), X]), delimiter=",", fmt="%.4f",
               header="cost_rub,peak_kw,soc_final_pct," + ",".join(f"p{t}" for t in range(model.T)), comments="")
    (out / "config_used.json").write_text(json.dumps({"nsga2": asdict(cfg), "weighted_sum": asdict(ws_cfg), "shift_mutation_prob": shift_p,
                                                      "weights": W.tolist(), "seeds": seeds,
                                                      "F_LO": F_LO.tolist(), "F_HI": F_HI.tolist()},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------- графики
    union = {}
    for name in [MAIN, "Взвешенная сумма (15 весов)"]:
        allF = np.vstack([front_of(r)[0] for r in res[name]])
        union[name] = allF[nondominated(allF)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (i, j) in zip(axes, pairs):
        S = show(F)
        sc = ax.scatter(S[:, i], S[:, j], c=S[:, 3 - i - j], cmap="viridis", s=18, label=f"{MAIN} (лучший запуск)")
        Sw = show(front_of(min(res["Взвешенная сумма (15 весов)"], key=lambda r: -hv(front_of(r)[0])))[0])
        ax.scatter(Sw[:, i], Sw[:, j], marker="x", c="red", s=50, label="взвешенная сумма (лучший запуск)")
        ax.set(xlabel=NAMES[i], ylabel=NAMES[j])
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label=NAMES[3 - i - j])
    axes[0].legend(fontsize=8)
    fig.suptitle("Проекции фронта Парето (цвет — третий критерий)")
    fig.tight_layout()
    fig.savefig(out / "pareto_projections.png", dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    S = show(F)
    ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=S[:, 2], cmap="viridis", s=14)
    ax.set(xlabel=NAMES[0], ylabel=NAMES[1], zlabel=NAMES[2], title=f"Фронт Парето: {MAIN} (3D)")
    fig.tight_layout()
    fig.savefig(out / "pareto_3d.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, c in zip(res, ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
        if not res[name][0].hv_hist:
            continue
        H = np.array([[h for _, h in r.hv_hist] for r in res[name]])
        fe = [e for e, _ in res[name][0].hv_hist]
        ax.plot(fe, np.median(H, axis=0), color=c, label=f"{name}: медиана")
        ax.fill_between(fe, H.min(axis=0), H.max(axis=0), color=c, alpha=0.15)
    ws = [hv(front_of(r)[0]) for r in res["Взвешенная сумма (15 весов)"]]
    ax.axhline(np.median(ws), color="red", ls="--", label="взвешенная сумма: итог, медиана")
    ax.set(xlabel="Вычисления критериев", ylabel="Гиперобъём (норм.)", title=f"Динамика гиперобъёма, {len(seeds)} запусков")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hv_convergence.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    data = [[hv(front_of(r)[0]) for r in rs] for rs in res.values()]
    ax.boxplot(data, tick_labels=[n.replace(" (", "\n(").replace(", ", ",\n") for n in res])
    ax.set(ylabel="Гиперобъём", title="Итоговый гиперобъём по сериям запусков")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hv_boxplot.png", dpi=130)
    plt.close(fig)

    labels = [f"{int(h):02d}:{int(round((h % 1) * 60)):02d}" for h in model.HOURS]
    fig, axes = plt.subplots(len(rep), 1, figsize=(11, 2.3 * len(rep)), sharex=True)
    for ax, (label, (p, soc)) in zip(axes, rep.items()):
        x = np.arange(model.T)
        ax.bar(x, model.B, color="lightgray", label="базовая нагрузка")
        ax.bar(x, p, bottom=model.B, color="tab:blue", label="зарядка ЭМ")
        ax.axhline(model.G_MAX, color="k", lw=0.8, ls=":")
        ax2 = ax.twinx()
        ax2.plot(x, soc, color="tab:green", label="SoC, %")
        ax2.step(x, model.C * 10, where="mid", color="tab:red", alpha=0.5, label="тариф ×10")
        ax2.set_ylim(0, 105)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("кВт")
    axes[0].legend(loc="upper right", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)
    axes[-1].set_xticks(np.arange(0, model.T, 2), labels[::2], rotation=45)
    fig.tight_layout()
    fig.savefig(out / "representative_profiles.png", dpi=120)
    plt.close(fig)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
