// A score per model as a bar from 0, one row each, the model's name on the left and its score at the bar's end. Rows are
// grouped by kind in the order given, each group sorted by value; kinds "active", "passive" and "canvit" take their
// colors from talk.css (.bar-chart). With zoomMax, the rows marked core are first drawn alone on an axis zoomed to
// zoomMax; the slide's CSS then brings in the others and rescales to max (#gap).

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

/** rows: [{name, value, kind, core}]; groups: the kinds, in order; ticks: the axis values, from 0 to max. */
export function drawBarRows(container, rows, { groups, max, ticks, zoomMax = max, label }) {
  if (rows.some((r) => r.value < 0 || r.value > max)) throw new Error(`bar-rows: a value lies outside the axis 0-${max}`);
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
  const x = (v) => LABEL_W + (v / max) * BAR_W;
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
    text(row.name, { x: LABEL_W - 14, y: 29, "text-anchor": "end", class: "name" }, group);
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

