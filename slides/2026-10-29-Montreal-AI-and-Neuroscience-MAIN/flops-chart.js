// Panels B and C of the paper's fig:canvas-attention as slide charts, from experiments/canvas_attention's export of
// the paper's computation. Each is drawn at the size it is shown, so its text keeps the deck's sizes. totalFlops (B):
// total inference FLOPs against the output patch grid, log-log; DINOv3 dashed in ink, CanViT-B solid in the brand's
// gradient, the thicker line the larger model or the longer rollout, as in the paper. projectionRatio (C): the cost
// factor of canvas-side projections for a Read/Write pair, by canvas patch grid, a line per glimpse patch grid.

const SVG = "http://www.w3.org/2000/svg";
const MARGIN = { top: 16, bottom: 104, left: 120 };
const LABEL_GAP = 30;
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

/** Label heights for targets given top to bottom, at least gap apart, a run of close ones centered on its targets. */
function spread(targets, gap) {
  const runs = [];
  for (const target of targets) {
    let run = { targets: [target], top: target };
    while (runs.length && runs.at(-1).top + gap * runs.at(-1).targets.length > run.top) {
      const merged = [...runs.pop().targets, ...run.targets];
      run = { targets: merged, top: merged.reduce((a, b) => a + b) / merged.length - (gap * (merged.length - 1)) / 2 };
    }
    runs.push(run);
  }
  return runs.flatMap((run) => run.targets.map((_, i) => run.top + gap * i));
}

const SI = [[1e12, "T"], [1e9, "G"], [1e6, "M"]];
const si = (v) => { const [scale, unit] = SI.find(([s]) => v >= s); return `${+(v / scale).toPrecision(3)}${unit}`; };
const squared = (g) => `${g}²`;

/** The plot's box in a chart of this size, room for the lines' names (labels, px) on its right. */
const plotBox = ({ width, height, labels }) =>
  ({ left: MARGIN.left, right: width - labels, top: MARGIN.top, bottom: height - MARGIN.bottom });

/** A chart frame: ticks at their plot positions, spines and titles; returns the svg and a layer for the lines. */
function frame(container, plot, { width, height, xTicks, xTick, xTitle, yTicks, yTick, yTitle, label }) {
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label });
  const grid = el("g", { class: "grid" }, svg);
  for (const [v, px] of yTicks) {
    el("line", { x1: plot.left, x2: plot.right, y1: px, y2: px }, grid);
    text(yTick(v), { x: plot.left - 14, y: px + 8, "text-anchor": "end", class: "tick" }, svg);
  }
  for (const [v, px] of xTicks) {
    el("line", { x1: px, x2: px, y1: plot.bottom, y2: plot.bottom + 8, class: "tick-mark" }, svg);
    text(xTick(v), { x: px, y: plot.bottom + 38, "text-anchor": "middle", class: "tick" }, svg);
  }
  el("line", { x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom, class: "spine" }, svg);
  el("line", { x1: plot.left, x2: plot.left, y1: plot.top, y2: plot.bottom, class: "spine" }, svg);
  text(xTitle, { x: (plot.left + plot.right) / 2, y: height - 14, "text-anchor": "middle", class: "axis-title" }, svg);
  text(yTitle, { x: -(plot.top + plot.bottom) / 2, y: 30, transform: "rotate(-90)", "text-anchor": "middle", class: "axis-title" }, svg);
  container.replaceChildren(svg);
  return { svg, plot, lines: el("g", {}, svg) };
}

/** Each line's name at its right end, spread so that none overlap. */
function labelEnds(svg, plot, ends) {
  const order = [...ends].sort((a, b) => a.y - b.y);
  const heights = spread(order.map((e) => e.y + 8), LABEL_GAP);
  order.forEach((end, i) => {
    const node = text(end.label, { x: plot.right + 14, y: heights[i], class: `end-label ${end.class}` }, svg);
    if (end.fill) node.style.fill = end.fill;
  });
}

export function totalFlops(container, data, size) {
  const { grids, ticks, series } = data.total_flops;
  const values = series.flatMap((s) => s.flops);
  const [lo, hi] = [Math.min(...values), Math.max(...values)];
  const decades = [];
  for (let e = Math.ceil(Math.log10(lo)); 10 ** e <= hi; e++) decades.push(10 ** e);
  const [yLo, yHi] = [Math.log10(lo) - 0.1, Math.log10(hi) + 0.1];
  const plot = plotBox(size);
  const x = (g) => plot.left + (Math.log2(g / grids[0]) / Math.log2(grids.at(-1) / grids[0])) * (plot.right - plot.left);
  const y = (v) => plot.bottom - ((Math.log10(v) - yLo) / (yHi - yLo)) * (plot.bottom - plot.top);
  const { svg, lines } = frame(container, plot, {
    ...size, xTicks: ticks.map((g) => [g, x(g)]), xTick: squared, xTitle: "Output patch grid",
    yTicks: decades.map((v) => [v, y(v)]), yTick: si, yTitle: "Total inference FLOPs",
    label: "Total inference FLOPs against the output patch grid: DINOv3 on the whole scene and CanViT-B after 1 and 21 glimpses",
  });
  // The brand's gradient, canvas red to glimpse blue: across the plot for CanViT's lines, across each of its names.
  const chart = ++charts, defs = el("defs", {}, svg);
  const gradient = (id, attributes) => {
    const node = el("linearGradient", { id: `flops-${chart}-${id}`, class: "brand", ...attributes }, defs);
    for (const offset of [0, 1]) el("stop", { offset }, node);
    return `url(#${node.id})`;
  };
  const across = gradient("across", { gradientUnits: "userSpaceOnUse", x1: plot.left, x2: plot.right, y1: 0, y2: 0 });
  const name = gradient("name", {});
  for (const s of series) {
    const line = el("polyline", { class: `curve ${s.family} ${s.weight}`,
                                  points: grids.map((g, i) => `${x(g)},${y(s.flops[i])}`).join(" ") }, lines);
    if (s.family === "canvit") line.style.stroke = across;
  }
  labelEnds(svg, plot, series.map((s) => ({ label: s.label, y: y(s.flops.at(-1)), class: s.family,
                                            fill: s.family === "canvit" ? name : null })));
}

export function projectionRatio(container, data, size) {
  const { canvas_grids: grids, series } = data.projection_ratio;
  const hi = Math.ceil(Math.max(...series.flatMap((s) => s.ratio)) / 5) * 5;
  const plot = plotBox(size);
  const x = (g) => plot.left + (Math.log2(g / grids[0]) / Math.log2(grids.at(-1) / grids[0])) * (plot.right - plot.left);
  const y = (r) => plot.bottom - ((r - 1) / (hi - 1)) * (plot.bottom - plot.top);
  const { svg, lines } = frame(container, plot, {
    ...size, xTicks: grids.map((g) => [g, x(g)]), xTick: squared, xTitle: "Canvas patch grid",
    yTicks: [1, ...Array.from({ length: hi / 5 }, (_, i) => 5 * (i + 1))].map((v) => [v, y(v)]), yTick: (v) => `${v}×`,
    yTitle: "Cost with canvas-side projections",
    label: "How many times more a Canvas Attention Read/Write pair would cost with canvas-side projections, by canvas and glimpse patch grid",
  });
  series.forEach((s, i) => {
    const group = el("g", { class: `ratio shade-${i}` }, lines);
    el("polyline", { class: "curve", points: grids.map((g, k) => `${x(g)},${y(s.ratio[k])}`).join(" ") }, group);
    grids.forEach((g, k) => el("circle", { cx: x(g), cy: y(s.ratio[k]), r: 6 }, group));
  });
  text("Glimpse patches", { x: plot.left + 20, y: plot.top + 30, class: "key-title" }, svg);
  labelEnds(svg, plot, series.map((s, i) => ({ label: squared(s.glimpse_grid), y: y(s.ratio.at(-1)), class: `ratio shade-${i}` })));
}
