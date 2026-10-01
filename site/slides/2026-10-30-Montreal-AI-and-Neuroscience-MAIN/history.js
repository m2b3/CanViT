// Year against accuracy on one benchmark, passive against active computer vision. Color is the family (passive ink,
// active amber, CanViT canvas red); solid lines and filled points are trained end to end, dashed lines and hollow points
// are frozen features with linear decoding. Everything is labeled where it is drawn, with no key: the two passive lines
// at their ends under one "Passive" heading, each active model at its point with "Active models" under them, CanViT-B's
// results in the margin under a "CanViT-B" heading, each joined to its point. Points come from sources/sota-history.json (each read in its paper; its _about defines the series),
// CanViT-B's from the paper's macros. The passive lines are the best base-size model trained end to end (a step line)
// and the best base-size frozen features (dashed); the active models are those of the timeline slide, each at its
// paper's best number. A bracket marks the gap between the best active model and the frozen line. With data-canvit,
// CanViT-B's points are added.
// Drawn into the light DOM so that a slide's CSS can build it: its groups carry the classes passive, frozen, active,
// gap and canvit.

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 1120, HEIGHT = 540, M = { left: 56, right: 250, top: 40, bottom: 40 };
const END = 2026.6;
const CANVIT_YEAR = 2026.2;

const BENCHMARKS = {
  imagenet: { key: "imagenet_top1", unit: "ImageNet-1k top-1 accuracy (%)", years: [2012, END], range: [60, 95], step: 10,
              canvit: [{ macro: "inkFinetunedBest", label: "fine-tuned", filled: true },
                       { macro: "inkFrozenBest", label: "frozen", filled: false }] },
  ade20k: { key: "ade20k_miou", unit: "ADE20K mIoU (%)", years: [2016, END], range: [10, 70], step: 10,
            canvit: [{ macro: "adeBestMiou", label: "frozen", filled: false }] },
};

// Entries sota-history.md flags as unreproduced or as two-model systems; the frontier is drawn without them.
const FLAGGED = new Set(["OmniVec (fine-tuned)", "OmniVec2 (fine-tuned)", "ViT-P on InternImage-H + Mask2Former"]);
// The active models of the timeline slide: the entries of each paper, its label's offset from the point (px), anchor.
const ACTIVE = [
  { model: /^Saccader/, name: "Saccader", dx: -16, dy: 8, anchor: "end" },
  { model: /^GFNet/, name: "GFNet", dx: -16, dy: 8, anchor: "end" },
  { model: /^AdaGlimpse/, name: "AdaGlimpse", dx: 0, dy: 34, anchor: "middle" },
  { model: /^AdaptiveNN/, name: "AdaptiveNN", dx: -16, dy: 8, anchor: "end" },
  { model: /^AME/, name: "AME", dx: 0, dy: 34, anchor: "middle" },
];

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

const yearOf = (point) => {
  const date = String(point.date);
  return Number(date.slice(0, 4)) + (date.length >= 7 ? (Number(date.slice(5, 7)) - 0.5) / 12 : 0.5);
};

/** The best value published by each date: the points that set a new record, in time order. */
function frontier(points) {
  const records = [];
  for (const p of [...points].sort((a, b) => yearOf(a) - yearOf(b))) if (!records.length || p.value > records.at(-1).value) records.push(p);
  if (!records.length) throw new Error("history chart: an empty series");
  return records;
}

function draw(container, data, macros) {
  const spec = BENCHMARKS[container.dataset.benchmark];
  if (!spec) throw new Error(`history chart: unknown data-benchmark "${container.dataset.benchmark}"`);
  const points = data[spec.key];
  const x = (year) => M.left + (year - spec.years[0]) / (spec.years[1] - spec.years[0]) * (WIDTH - M.left - M.right);
  const y = (v) => HEIGHT - M.bottom - (v - spec.range[0]) / (spec.range[1] - spec.range[0]) * (HEIGHT - M.top - M.bottom);
  const step = (records) => records.map((p, i) => (i ? `H${x(yearOf(p))}V${y(p.value)}` : `M${x(yearOf(p))},${y(p.value)}`)).join("") + `H${x(END)}`;
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img", "aria-label": spec.unit });
  const labelX = x(END) + 18;

  const axes = el("g", { class: "axes" }, svg);
  for (let v = spec.range[0]; v <= spec.range[1]; v += spec.step) {
    el("line", { x1: M.left, x2: x(END), y1: y(v), y2: y(v), class: "grid" }, axes);
    text(v, { x: M.left - 12, y: y(v) + 7, "text-anchor": "end" }, axes);
  }
  for (let year = Math.ceil(spec.years[0] / 2) * 2; year <= 2026; year += 2) text(year, { x: x(year + 0.5), y: HEIGHT - 8, "text-anchor": "middle" }, axes);
  text(spec.unit, { x: M.left, y: M.top - 14, class: "unit" }, axes);

  // Passive lines at the active models' size: base-size backbones only (size_class, sota-history.json) [Yohaï,
  // 2026-10-01: "comparing to nonsensically large models is indeed stupid and counterproductive"].
  const base = (p) => p.kind === "passive" && p.size_class === "base";
  const passive = frontier(points.filter((p) => base(p) && p.series !== "frozen_ssl" && !FLAGGED.has(p.model)));
  const frozen = frontier(points.filter((p) => base(p) && p.series === "frozen_ssl" && p.protocol !== "linear probe + ms"));
  const passiveTop = passive.at(-1).value, frozenTop = frozen.at(-1).value;
  const passiveGroup = el("g", { class: "passive" }, svg);
  el("path", { d: step(passive), class: "line" }, passiveGroup);
  // Each line named at its end; "Passive" heads the two, above the upper one.
  const passiveY = y(passiveTop) + 7;
  text("Passive", { x: labelX, y: passiveY - 30, class: "label head" }, passiveGroup);
  text("trained end to end", { x: labelX, y: passiveY, class: "label" }, passiveGroup);
  const frozenGroup = el("g", { class: "frozen" }, svg);
  el("path", { d: step(frozen), class: "line" }, frozenGroup);
  const frozenY = Math.max(y(frozenTop) + 7, passiveY + 28);
  text("frozen + linear decoding", { x: labelX, y: frozenY, class: "label" }, frozenGroup);

  const activeGroup = el("g", { class: "active" }, svg);
  const shown = ACTIVE.flatMap((a) => {
    const entries = points.filter((p) => p.series === "active" && a.model.test(p.model));
    return entries.length ? [{ ...a, point: entries.reduce((best, p) => (p.value > best.value ? p : best)) }] : [];
  });
  if (!shown.length) throw new Error(`history chart: no active model of the timeline in ${spec.key}`);
  for (const { point, name, dx, dy, anchor } of shown) {
    el("circle", { cx: x(yearOf(point)), cy: y(point.value), r: 9 }, activeGroup);
    text(name, { x: x(yearOf(point)) + dx, y: y(point.value) + dy, "text-anchor": anchor, class: "label" }, activeGroup);
  }
  const bestActive = Math.max(...shown.map((a) => a.point.value));
  // "Active models" under the lowest name, centered under the points.
  const xs = shown.map((a) => x(yearOf(a.point))), lowest = Math.max(...shown.map((a) => y(a.point.value) + Math.max(a.dy, 8)));
  text("Active models", { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: lowest + 40, "text-anchor": "middle", class: "label head" }, activeGroup);

  // The gap, from the best active model up to the frozen line, at the plot's right edge.
  const gap = el("g", { class: "gap" }, svg);
  const gx = x(END) - 8;
  el("path", { d: `M${gx - 12},${y(frozenTop)}H${gx}V${y(bestActive)}H${gx - 12}`, class: "bracket" }, gap);
  text(`${(frozenTop - bestActive).toFixed(1)} points`, { x: gx - 20, y: (y(frozenTop) + y(bestActive)) / 2 + 10,
                                                         "text-anchor": "end", class: "gap-label" }, gap);

  if (container.hasAttribute("data-canvit")) {
    const canvit = el("g", { class: "canvit" }, svg);
    // Named in the margin like the passive lines, under a "CanViT-B" heading below them, each result joined to its
    // point by a leader.
    const values = spec.canvit.map(({ macro, ...rest }) => {
      const value = Number(macros[macro]);
      if (!Number.isFinite(value)) throw new Error(`history chart: macro ${macro} is missing`);
      return { value, ...rest };
    });
    let floor = Math.max(frozenY + 44, y(values[0].value) - 20);
    text("CanViT-B", { x: labelX, y: floor, class: "label head" }, canvit);
    for (const { value, label, filled } of values) {
      const [px, py] = [x(CANVIT_YEAR), y(value)];
      const ly = Math.max(py + 7, floor + 28);
      floor = ly;
      el("path", { d: `M${px + 14},${py}L${labelX - 8},${ly - 7}`, class: "leader" }, canvit);
      el("circle", { cx: px, cy: py, r: 12, class: filled ? "filled" : "hollow" }, canvit);
      text(`${label} ${value}`, { x: labelX, y: ly, class: "label" }, canvit);
    }
  }
  container.replaceChildren(svg);
}

/** Draw every .history-chart of the page: data-benchmark="imagenet" or "ade20k"; data-canvit adds CanViT-B. */
export async function drawHistoryCharts({ data = "sources/sota-history.json", macros = "../../assets/paper/data_macros.json" } = {}) {
  const [history, values] = await Promise.all([data, macros].map(async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`history chart: ${url} answered ${response.status}`);
    return response.json();
  }));
  for (const container of document.querySelectorAll(".history-chart")) draw(container, history, values);
}
