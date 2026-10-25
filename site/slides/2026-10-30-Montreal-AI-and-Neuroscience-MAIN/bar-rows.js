// A score per model, one row each, with its name and a note on the left: a bar from 0 (mark "bar"), or a dot on an axis
// from min to max (mark "dot": a dot encodes position, not length, so its axis need not start at 0, and close scores
// stay apart). Rows are grouped by kind in the order given, each group sorted by value; kinds "active", "passive" and
// "canvit" take their colors from talk.css (.bar-chart). With zoomMax (bars only), the rows marked core are first drawn
// alone on an axis zoomed to zoomMax; the slide's CSS then brings in the others and rescales to max (#gap).

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 1000, ROW = 46, LABEL_W = 330, BAR_W = 560, TOP = 10, GROUP_GAP = 24;

function el(name, attributes, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attributes)) node.setAttribute(k, v);
  parent?.append(node);
  return node;
}

function text(content, attributes, parent) {
  const node = el("text", attributes, parent);
  node.textContent = content;
  return node;
}

/** rows: [{name, note, value, kind, core}]; groups: the kinds, in order; ticks: the axis values, from min to max. */
export function drawBarRows(container, rows, { groups, min = 0, max, ticks, zoomMax = max, label, mark = "bar" }) {
  if (mark === "bar" && min !== 0) throw new Error("bar-rows: bars start at 0; use mark \"dot\" for an axis from another value");
  if (mark === "dot" && zoomMax !== max) throw new Error("bar-rows: zoomMax is for bars");
  if (rows.some((r) => r.value < min || r.value > max)) throw new Error(`bar-rows: a value lies outside the axis ${min}-${max}`);
  if (rows.some((r) => !groups.includes(r.kind))) throw new Error(`bar-rows: a row's kind is not one of ${groups.join(", ")}`);
  if (rows.some((r) => r.core && r.value > zoomMax)) throw new Error(`bar-rows: a core row exceeds the zoomed axis (${zoomMax})`);
  const order = (subset) => {
    const placed = new Map();
    let y = TOP;
    groups.forEach((kind, g) => {
      const members = subset.filter((r) => r.kind === kind).sort((a, b) => a.value - b.value);
      if (g && members.length && placed.size) y += GROUP_GAP;
      for (const r of members) { placed.set(r, y); y += ROW; }
    });
    return { placed, height: y };
  };
  const all = order(rows), core = order(rows.filter((r) => r.core));
  const x = (v) => LABEL_W + ((v - min) / (max - min)) * BAR_W;
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${all.height + 40}`, role: "img", "aria-label": label });
  // Zoomed, every horizontal position from the axis origin is stretched by max / zoomMax.
  const stretch = max / zoomMax;
  const shift = (px) => `translateX(${(stretch - 1) * px}px)`;
  for (const v of ticks) {
    const tick = el("g", { class: v > zoomMax ? "tick-group beyond" : "tick-group" }, svg);
    tick.style.setProperty("--zoom", shift(x(v) - x(0)));
    el("line", { x1: x(v), x2: x(v), y1: TOP - 4, y2: all.height, class: "grid" }, tick);
    text(v, { x: x(v), y: all.height + 26, "text-anchor": "middle", class: "tick" }, tick);
  }
  for (const row of rows) {
    const group = el("g", { class: `row ${row.kind}${row.core ? " core" : ""}` }, svg);
    group.style.setProperty("--y-all", `${all.placed.get(row)}px`);
    group.style.setProperty("--y-core", `${row.core ? core.placed.get(row) : all.placed.get(row)}px`);
    text(row.name, { x: LABEL_W - 14, y: 20, "text-anchor": "end", class: "name" }, group);
    text(row.note, { x: LABEL_W - 14, y: 38, "text-anchor": "end", class: "note" }, group);
    if (mark === "dot") {
      el("line", { x1: x(min), x2: x(max), y1: ROW / 2 - 2, y2: ROW / 2 - 2, class: "guide" }, group);
      el("circle", { cx: x(row.value), cy: ROW / 2 - 2, r: 11, class: "dot" }, group);
      text(row.value.toFixed(1), { x: x(row.value) + 20, y: 28, class: "value" }, group);
      continue;
    }
    el("rect", { x: x(0), y: 4, width: x(row.value) - x(0), height: ROW - 12, rx: 5, class: "bar" }, group)
      .style.setProperty("--zoom", `scaleX(${stretch})`);
    text(row.value.toFixed(1), { x: x(row.value) + 10, y: 28, class: "value" }, group)
      .style.setProperty("--zoom", shift(x(row.value) - x(0)));
  }
  container.replaceChildren(svg);
}

export async function fetchJson(src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`bar-rows: ${src} answered ${response.status}`);
  return response.json();
}

/**
 * A dot plot: one row per model, a dot at its score on an axis from min to max (a dot encodes position, so the axis
 * need not start at 0 and close scores stay apart), the name on the left and the value beside the dot; rows sorted by
 * value, emphasized rows (emphasis: true) drawn in color and bold, the others gray; a reference ({name, value}) is a
 * vertical line across the rows, named at its top. Colors from talk.css (.dot-chart).
 */
export function drawDotRows(container, rows, { min, max, ticks, axisTitle, reference, label }) {
  const W = 1000, NAME_W = 300, PLOT_W = 560, ROW_H = 56, TOP = reference ? 44 : 10;
  if ([...rows, ...(reference ? [reference] : [])].some((r) => r.value < min || r.value > max)) {
    throw new Error(`dot-rows: a value lies outside the axis ${min}-${max}`);
  }
  const sorted = [...rows].sort((a, b) => a.value - b.value);
  const x = (v) => NAME_W + ((v - min) / (max - min)) * PLOT_W;
  const bottom = TOP + sorted.length * ROW_H;
  const svg = el("svg", { viewBox: `0 0 ${W} ${bottom + 70}`, role: "img", "aria-label": label });
  for (const v of ticks) {
    el("line", { x1: x(v), x2: x(v), y1: TOP - 6, y2: bottom, class: "grid" }, svg);
    text(v, { x: x(v), y: bottom + 26, "text-anchor": "middle", class: "tick" }, svg);
  }
  el("line", { x1: x(min), x2: x(max), y1: bottom, y2: bottom, class: "axis-line" }, svg);
  text(axisTitle, { x: x((min + max) / 2), y: bottom + 58, "text-anchor": "middle", class: "axis-title" }, svg);
  if (reference) {
    el("line", { x1: x(reference.value), x2: x(reference.value), y1: TOP - 10, y2: bottom, class: "reference" }, svg);
    text(`${reference.name} ${reference.value.toFixed(1)}`, { x: x(reference.value), y: TOP - 18, "text-anchor": "middle", class: "reference-label" }, svg);
  }
  sorted.forEach((row, i) => {
    const y = TOP + i * ROW_H + ROW_H / 2;
    const group = el("g", { class: row.emphasis ? "row emphasis" : "row" }, svg);
    text(row.name, { x: NAME_W - 24, y: y + 7, "text-anchor": "end", class: "name" }, group);
    el("circle", { cx: x(row.value), cy: y, r: row.emphasis ? 11 : 8, class: "dot" }, group);
    text(row.value.toFixed(1), { x: x(row.value) + (row.emphasis ? 20 : 16), y: y + 7, class: "value" }, group);
  });
  container.replaceChildren(svg);
}
