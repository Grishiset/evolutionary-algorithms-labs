"""Сборка автономной HTML-страницы для интерактивного выбора расписания из фронта Парето.

    python build_viewer.py                         # results/front.json -> results/pareto_viewer.html

Страница работает без сети: данные встроены, графика — собственный SVG, без библиотек.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Выбор расписания из фронта Парето — ЛР4, вариант 8</title>
<style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--line:#dcdcd5;--acc:#1f5fbf;--warn:#b5462b;--ok:#2f7d4f}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:18px 24px 6px}h1{font-size:20px;margin:0 0 4px}header p{margin:2px 0;color:var(--muted);max-width:1100px}
main{display:grid;grid-template-columns:minmax(420px,560px) 1fr;gap:18px;padding:12px 24px 24px}
@media(max-width:1000px){main{grid-template-columns:1fr}}
section{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px}
h2{font-size:15px;margin:0 0 10px}label{font-size:12px;color:var(--muted)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:6px 0}
select,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:#fff}
button{cursor:pointer}button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
.slider{display:grid;grid-template-columns:190px 1fr 64px;gap:8px;align-items:center;margin:4px 0}
.slider output{font-variant-numeric:tabular-nums;text-align:right}
svg{width:100%;height:auto;display:block}.pt{cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 12px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:8px}.kpi b{font-size:20px;display:block}
.kpi small{color:var(--muted)}
table.tt{border-collapse:collapse;width:100%;table-layout:fixed}
table.tt th,table.tt td{border:1px solid var(--line);padding:4px;vertical-align:top;font-size:12px}
table.tt th{background:#f2f2ec;font-weight:600}td.bad{background:#fbefe9}td.gap{background:repeating-linear-gradient(45deg,#fff,#fff 4px,#f3f3ee 4px,#f3f3ee 8px)}
.ev{border-left:3px solid var(--acc);padding-left:4px}.ev.lab{border-color:var(--ok)}.ev.lec{border-color:#8a5cc2}
.muted{color:var(--muted)}.note{font-size:12px;color:var(--muted);margin-top:8px}
details{margin-top:10px}summary{cursor:pointer;color:var(--acc)}
</style></head><body>
<header>
<h1>Выбор расписания из фронта Парето</h1>
<p>Каждая точка — допустимое недоминируемое расписание: никакое другое найденное расписание не лучше его сразу по всем трём критериям.
Все три критерия минимизируются. Отфильтруйте неприемлемые варианты, задайте важность критериев и выберите расписание кликом.</p>
<p class="muted" id="src"></p>
</header>
<main>
<section>
<h2>Фронт Парето</h2>
<div class="row"><label>Ось X</label><select id="ax"></select><label>Ось Y</label><select id="ay"></select>
<span class="muted">цвет — третий критерий</span></div>
<svg id="plot" viewBox="0 0 540 400" role="img" aria-label="Диаграмма фронта Парето"></svg>
<h2 style="margin-top:12px">Ограничения «не хуже чем»</h2><div id="filters"></div>
<h2 style="margin-top:12px">Важность критериев</h2><div id="weights"></div>
<div class="row"><button class="primary" id="rec">Выбрать рекомендованное</button>
<span class="muted" id="count"></span></div>
<p class="note">Рекомендация — минимум взвешенной суммы нормированных критериев среди решений, прошедших фильтры.
Взвешивание здесь применяется только к уже найденному фронту: это способ выбора, а не способ поиска.</p>
</section>
<section>
<h2 id="selTitle">Выбранное расписание</h2>
<div class="kpis" id="kpis"></div>
<div class="row"><label>Показать</label><select id="viewKind"><option value="group">группу</option>
<option value="teacher">преподавателя</option><option value="room">аудиторию</option></select>
<select id="viewWho"></select><button id="dl">Скачать выбранное расписание (JSON)</button></div>
<div id="grid"></div>
<p class="note">Заштрихованные клетки — «окна» (пустые пары между занятиями в один день), розовые — первая и последняя пары.</p>
<details><summary>Как читать критерии</summary>
<ul>
<li><b>Окна студентов</b> — сумма по всем группам и дням числа пустых пар между первым и последним занятием.</li>
<li><b>Неудобное время</b> — сколько студенто-занятий (в сотнях) приходится на 09:00 и 17:20.</li>
<li><b>Неудобство преподавателей</b> — занятия в непредпочитаемые дни плюс окна преподавателей.</li>
</ul>
<p>Жёсткие ограничения выполнены во всех точках: нет пересечений у групп и преподавателей, преподаватели не стоят в недоступные слоты,
каждому занятию нашлась аудитория нужного типа и вместимости.</p></details>
</section>
</main>
<script>
const DATA = __DATA__;
const I = DATA.instance, SOL = DATA.solutions, NAMES = DATA.objective_names, M = 3;
document.getElementById('src').textContent =
  `Алгоритм: ${DATA.algorithm}. Решений во фронте: ${SOL.length}. Данные встроены в страницу, сеть не нужна.`;
const F = SOL.map(s => s.objectives);
const lo = [0,1,2].map(k => Math.min(...F.map(f => f[k]))), hi = [0,1,2].map(k => Math.max(...F.map(f => f[k])));
const fmt = (k, v) => k === 1 ? v.toFixed(2) : String(Math.round(v * 100) / 100);
let maxv = hi.slice(), w = [1, 1, 1], sel = 0;

function el(tag, attrs = {}, text) { const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; return e; }
function norm(f) { return f.map((v, k) => hi[k] > lo[k] ? (v - lo[k]) / (hi[k] - lo[k]) : 0); }
function passes(f) { return f.every((v, k) => v <= maxv[k] + 1e-9); }
function color(t) { const a = [68, 1, 84], b = [33, 145, 140], c = [253, 231, 37];
  const m = (p, q, u) => p.map((x, i) => Math.round(x + (q[i] - x) * u));
  const r = t < .5 ? m(a, b, t * 2) : m(b, c, (t - .5) * 2); return `rgb(${r})`; }

const ax = document.getElementById('ax'), ay = document.getElementById('ay');
NAMES.forEach((n, k) => { ax.add(new Option(n, k)); ay.add(new Option(n, k)); });
ax.value = 0; ay.value = 1;
ax.onchange = ay.onchange = draw;

function draw() {
  const svg = document.getElementById('plot'); svg.innerHTML = '';
  const X = +ax.value, Y = +ay.value, C = [0,1,2].find(k => k !== X && k !== Y) ?? 2;
  const L = 60, R = 520, T = 16, B = 350;
  const sx = v => L + (hi[X] > lo[X] ? (v - lo[X]) / (hi[X] - lo[X]) : .5) * (R - L);
  const sy = v => B - (hi[Y] > lo[Y] ? (v - lo[Y]) / (hi[Y] - lo[Y]) : .5) * (B - T);
  for (let i = 0; i <= 4; i++) {
    const vx = lo[X] + (hi[X] - lo[X]) * i / 4, vy = lo[Y] + (hi[Y] - lo[Y]) * i / 4;
    svg.append(el('line', {x1: sx(vx), x2: sx(vx), y1: T, y2: B, stroke: '#eee'}));
    svg.append(el('line', {x1: L, x2: R, y1: sy(vy), y2: sy(vy), stroke: '#eee'}));
    svg.append(el('text', {x: sx(vx), y: B + 16, 'text-anchor': 'middle', 'font-size': 11, fill: '#666'}, fmt(X, vx)));
    svg.append(el('text', {x: L - 6, y: sy(vy) + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#666'}, fmt(Y, vy)));
  }
  svg.append(el('text', {x: (L + R) / 2, y: 392, 'text-anchor': 'middle', 'font-size': 12}, NAMES[X] + ' →'));
  const ty = el('text', {x: 14, y: (T + B) / 2, 'text-anchor': 'middle', 'font-size': 12,
    transform: `rotate(-90 14 ${(T + B) / 2})`}, NAMES[Y] + ' →'); svg.append(ty);
  const order = SOL.map((s, i) => i).sort((a, b) => passes(F[a]) - passes(F[b]));
  for (const i of order) {
    const f = F[i], ok = passes(f), t = hi[C] > lo[C] ? (f[C] - lo[C]) / (hi[C] - lo[C]) : .5;
    const c = el('circle', {cx: sx(f[X]), cy: sy(f[Y]), r: i === sel ? 8 : 5, class: 'pt',
      fill: ok ? color(t) : '#ddd', stroke: i === sel ? '#d62728' : (ok ? '#fff' : '#ccc'), 'stroke-width': i === sel ? 3 : 1});
    c.append(el('title', {}, `#${i}: ` + f.map((v, k) => `${NAMES[k]} = ${fmt(k, v)}`).join('; ')));
    c.onclick = () => { sel = i; render(); }; svg.append(c);
  }
  document.getElementById('count').textContent = `Прошли фильтры: ${F.filter(passes).length} из ${SOL.length}`;
}

function sliders() {
  const fd = document.getElementById('filters'), wd = document.getElementById('weights');
  NAMES.forEach((n, k) => {
    const d = document.createElement('div'); d.className = 'slider';
    const step = k === 1 ? 0.01 : 1;
    d.innerHTML = `<label>${n}, не больше</label><input type="range" min="${lo[k]}" max="${hi[k]}" step="${step}" value="${hi[k]}"><output>${fmt(k, hi[k])}</output>`;
    const inp = d.querySelector('input'), o = d.querySelector('output');
    inp.oninput = () => { maxv[k] = +inp.value; o.textContent = fmt(k, maxv[k]); draw(); };
    fd.append(d);
    const e = document.createElement('div'); e.className = 'slider';
    e.innerHTML = `<label>${n}</label><input type="range" min="0" max="5" step="0.5" value="1"><output>1</output>`;
    const wi = e.querySelector('input'), wo = e.querySelector('output');
    wi.oninput = () => { w[k] = +wi.value; wo.textContent = w[k]; };
    wd.append(e);
  });
}
document.getElementById('rec').onclick = () => {
  let best = -1, bv = Infinity;
  F.forEach((f, i) => { if (!passes(f)) return; const v = norm(f).reduce((s, x, k) => s + w[k] * x, 0);
    if (v < bv) { bv = v; best = i; } });
  if (best < 0) { alert('Ни одно решение не проходит фильтры — ослабьте ограничения.'); return; }
  sel = best; render();
};

const kindSel = document.getElementById('viewKind'), whoSel = document.getElementById('viewWho');
function fillWho() {
  whoSel.innerHTML = ''; const k = kindSel.value;
  const list = k === 'group' ? I.groups : k === 'teacher' ? I.teachers : I.rooms;
  list.forEach(x => whoSel.add(new Option(x.name, x.id)));
}
kindSel.onchange = () => { fillWho(); grid(); }; whoSel.onchange = grid;

function grid() {
  const s = SOL[sel], k = kindSel.value, id = +whoSel.value;
  const items = s.timetable.filter(r => k === 'group' ? r.groups.includes(id) : k === 'teacher' ? r.teacher === id : r.room === id);
  const cell = {}; items.forEach(r => (cell[r.day + '_' + r.period] ??= []).push(r));
  let h = '<table class="tt"><tr><th style="width:60px">Пара</th>' + I.days.map(d => `<th>${d}</th>`).join('') + '</tr>';
  const busy = d => I.periods.map((_, p) => !!cell[d + '_' + p]);
  for (let p = 0; p < I.periods.length; p++) {
    h += `<tr><th>${I.periods[p]}</th>`;
    for (let d = 0; d < I.days.length; d++) {
      const b = busy(d), first = b.indexOf(true), last = b.lastIndexOf(true);
      const rs = cell[d + '_' + p] || [];
      const isGap = !rs.length && first >= 0 && p > first && p < last;
      const edge = p === 0 || p === I.periods.length - 1;
      h += `<td class="${isGap ? 'gap' : (edge && rs.length ? 'bad' : '')}">` + rs.map(r => {
        const cls = r.kind === 'лабораторная' ? 'lab' : r.kind === 'лекция' ? 'lec' : '';
        const who = k === 'teacher' ? r.groups.map(g => I.groups[g].name).join(', ') : I.teachers[r.teacher].name;
        const room = r.room >= 0 ? I.rooms[r.room].name : '—';
        return `<div class="ev ${cls}"><b>${r.title}</b><br><span class="muted">${r.kind}; ${k === 'room' ? r.groups.map(g => I.groups[g].name).join(', ') : room}<br>${who}</span></div>`;
      }).join('') + '</td>';
    }
    h += '</tr>';
  }
  document.getElementById('grid').innerHTML = h + '</table>';
}
function render() {
  const f = F[sel];
  document.getElementById('selTitle').textContent = `Выбранное расписание #${sel}`;
  document.getElementById('kpis').innerHTML = f.map((v, k) => {
    const rel = hi[k] > lo[k] ? Math.round(100 * (v - lo[k]) / (hi[k] - lo[k])) : 0;
    return `<div class="kpi"><small>${NAMES[k]}</small><b>${fmt(k, v)}</b><small>диапазон фронта ${fmt(k, lo[k])}–${fmt(k, hi[k])}; позиция ${rel}%</small></div>`;
  }).join('');
  draw(); grid();
}
document.getElementById('dl').onclick = () => {
  const blob = new Blob([JSON.stringify(SOL[sel], null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `timetable_${sel}.json`; a.click();
};
sliders(); fillWho(); document.getElementById('rec').click();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", default=str(HERE / "results" / "front.json"))
    ap.add_argument("--out", default=str(HERE / "results" / "pareto_viewer.html"))
    a = ap.parse_args()
    data = json.loads(Path(a.front).read_text(encoding="utf-8"))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"{a.out}: {len(data['solutions'])} решений")


if __name__ == "__main__":
    main()
