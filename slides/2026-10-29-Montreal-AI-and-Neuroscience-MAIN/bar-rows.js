// A score per model as a bar from the axis origin (min), a row each: the name on the left, the score at the bar's end.
// Rows are grouped by kind in the order given, each group sorted by value; kinds take their colors from talk.css
// (.bar-chart), CanViT's the brand's gradient. Frozen rows (a frozen backbone) are hatched inside an outline. With
// sections, consecutive groups are bracketed under a label in a left gutter. With zoomMax, the core rows are first
// drawn alone on an axis zoomed to zoomMax; the slide's CSS then brings in the others and rescales to max (#gap).

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 1000, ROW = 46, LABEL_W = 330, BAR_W = 560, TOP = 10, GROUP_GAP = 24;
const SECTION_W = 140, SECTION_GAP = 40, GROUP_GAP_IN_SECTION = 12;
let charts = 0; // ids in <defs> stay unique when several charts share a page

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

/**
 * rows: [{name, value, kind, core, frozen?}]; groups: the kinds, in order; sections: [{label, kinds}], each a run of
 * consecutive groups; ticks: the axis values, from min to max; title: the axis's title; label: the chart's accessible name.
 */
export function drawBarRows(container, rows, { groups, sections = [], min = 0, max, ticks, zoomMax = max, title, label }) {
  if (rows.some((r) => r.value < min || r.value > max)) throw new Error(`bar-rows: a value lies outside the axis ${min}-${max}`);
  if (rows.some((r) => !groups.includes(r.kind))) throw new Error(`bar-rows: a row's kind is not one of ${groups.join(", ")}`);
  if (rows.some((r) => r.core && r.value > zoomMax)) throw new Error(`bar-rows: a core row exceeds the zoomed axis (${zoomMax})`);
  const sectionOf = (kind) => sections.findIndex((s) => s.kinds.includes(kind));
  if (sections.length && groups.some((kind) => sectionOf(kind) < 0)) throw new Error("bar-rows: a group is in no section");
  // Rows placed from the top, and each section's span from its first row's top to its last row's bottom.
  const order = (subset) => {
    const placed = new Map(), spans = new Map();
    let y = TOP, previous = null;
    for (const kind of groups) {
      const members = subset.filter((r) => r.kind === kind).sort((a, b) => a.value - b.value);
      if (!members.length) continue;
      const section = sectionOf(kind);
      if (previous !== null) y += !sections.length ? GROUP_GAP : section === previous ? GROUP_GAP_IN_SECTION : SECTION_GAP;
      if (!spans.has(section)) spans.set(section, { top: y });
      for (const r of members) { placed.set(r, y); y += ROW; }
      spans.get(section).bottom = y;
      previous = section;
    }
    return { placed, spans, height: y };
  };
  const all = order(rows), core = order(rows.filter((r) => r.core));
  const left = sections.length ? SECTION_W : 0;
  const x = (v) => left + LABEL_W + ((v - min) / (max - min)) * BAR_W;
  const bottom = all.height;
  const svg = el("svg", { viewBox: `0 0 ${left + WIDTH} ${bottom + 84}`, role: "img", "aria-label": label });
  // For a slide that closes the plot around the core rows before the others arrive (#gap): how far the core rows drop
  // and the bottom axis rises to center them, and how much the vertical lines shrink about their center.
  const coreDrop = (all.height - core.height) / 2, plotHeight = bottom - (TOP - 4);
  svg.style.setProperty("--core-drop", `${coreDrop}px`);
  svg.style.setProperty("--core-squeeze", (plotHeight - 2 * coreDrop) / plotHeight);
  // The hatch, a mask of slanted stripes, so that a gradient shows through it as a flat color does; and the brand's
  // gradient, its stops colored by talk.css (.brand).
  const chart = ++charts, defs = el("defs", {}, svg);
  const id = (name) => `bar-rows-${chart}-${name}`;
  const stripes = el("pattern", { id: id("stripes"), patternUnits: "userSpaceOnUse", width: 12, height: 12,
                                  patternTransform: "rotate(45)" }, defs);
  el("rect", { width: 5, height: 12, fill: "#fff" }, stripes);
  el("rect", { width: "100%", height: "100%", fill: `url(#${id("stripes")})` }, el("mask", { id: id("hatch") }, defs));
  const brand = el("linearGradient", { id: id("brand"), class: "brand" }, defs);
  for (const offset of [0, 1]) el("stop", { offset }, brand);
  // Zoomed, every horizontal position from the axis origin is stretched by (max - min) / (zoomMax - min).
  const stretch = (max - min) / (zoomMax - min);
  const shift = (px) => `translateX(${(stretch - 1) * px}px)`;
  for (const v of ticks) {
    const tick = el("g", { class: v > zoomMax ? "tick-group beyond" : "tick-group" }, svg);
    tick.style.setProperty("--zoom", shift(x(v) - x(min)));
    el("line", { x1: x(v), x2: x(v), y1: TOP - 4, y2: bottom, class: "grid" }, tick);
    el("line", { x1: x(v), x2: x(v), y1: bottom, y2: bottom + 8, class: "tick-mark" }, tick);
    text(v, { x: x(v), y: bottom + 32, "text-anchor": "middle", class: "tick" }, tick);
  }
  // A section's label sits on its first row; its bracket runs from the first bar's top to the last bar's bottom, and
  // while only the core rows show, it is scaled to their span (--bracket-core, for the slide's CSS).
  sections.forEach((section, i) => {
    const span = all.spans.get(i);
    if (!span) return;
    const coreSpan = core.spans.get(i);
    const group = el("g", { class: `row section${coreSpan ? " core" : ""}` }, svg);
    group.style.setProperty("--y-all", `${span.top}px`);
    group.style.setProperty("--y-core", `${(coreSpan ?? span).top}px`);
    text(section.label, { x: SECTION_W - 34, y: 29, "text-anchor": "end", class: "section-label" }, group);
    const length = (s) => s.bottom - s.top - 12;
    el("line", { x1: SECTION_W - 18, x2: SECTION_W - 18, y1: 4, y2: 4 + length(span), class: "bracket" }, group)
      .style.setProperty("--bracket-core", coreSpan ? length(coreSpan) / length(span) : 1);
  });
  for (const row of rows) {
    const group = el("g", { class: `row ${row.kind}${row.core ? " core" : ""}${row.frozen ? " frozen" : ""}` }, svg);
    group.style.setProperty("--y-all", `${all.placed.get(row)}px`);
    group.style.setProperty("--y-core", `${row.core ? core.placed.get(row) : all.placed.get(row)}px`);
    const name = text(row.name, { x: left + LABEL_W - 14, y: 29, "text-anchor": "end", class: "name" }, group);
    // A frozen row's outline is a second, unmasked bar, which scales and grows with the first (talk.css, .bar).
    const shape = { x: x(min), y: 4, width: x(row.value) - x(min), height: ROW - 12, rx: 5 };
    const bars = [el("rect", { ...shape, class: "bar" }, group)];
    if (row.frozen) {
      bars[0].setAttribute("mask", `url(#${id("hatch")})`);
      bars.push(el("rect", { ...shape, class: "bar outline" }, group));
    }
    for (const bar of bars) {
      bar.style.setProperty("--zoom", `scaleX(${stretch})`);
      if (row.kind === "canvit") bar.style.setProperty(bar === bars[0] ? "fill" : "stroke", `url(#${id("brand")})`);
    }
    const value = text(row.value.toFixed(1), { x: x(row.value) + 10, y: 28, class: "value" }, group);
    value.style.setProperty("--zoom", shift(x(row.value) - x(min)));
    // CanViT's name and score in the brand's gradient, as its bar: red alone would mean the canvas.
    if (row.kind === "canvit") for (const label of [name, value]) label.style.fill = `url(#${id("brand")})`;
  }
  // Zoomed, the axis keeps its length: zoomMax then sits where max does.
  el("line", { x1: x(min), x2: x(max), y1: bottom, y2: bottom, class: "spine x" }, svg);
  el("line", { x1: x(min), x2: x(min), y1: TOP - 4, y2: bottom, class: "spine y" }, svg);
  text(title, { x: (x(min) + x(max)) / 2, y: bottom + 72, "text-anchor": "middle", class: "axis-title" }, svg);
  container.replaceChildren(svg);
}

export async function fetchJson(src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`bar-rows: ${src} answered ${response.status}`);
  return response.json();
}
