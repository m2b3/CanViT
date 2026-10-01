// Year against accuracy, passive and active computer vision (slides "Active computer vision was not nearly as smart
// as passive computer vision" and "Results on ADE20K and ImageNet-1k"). Points come from sources/sota-history.json
// (each read in its paper; its _about defines the series), CanViT-B's from the paper's macros. The chart is drawn
// into the light DOM, so a slide's CSS builds it click by click: its groups carry the classes passive, frozen,
// active and canvit.

const SVG = "http://www.w3.org/2000/svg";

const BENCHMARKS = {
  imagenet: { key: "imagenet_top1", title: "ImageNet-1k classification", unit: "top-1 accuracy (%)",
              years: [2012, 2026.9], range: [55, 95], ticks: [60, 70, 80, 90],
              canvit: { macro: "inkFinetunedBest" } },
  ade20k: { key: "ade20k_miou", title: "ADE20K semantic segmentation", unit: "mIoU (%)",
            years: [2016, 2026.9], range: [0, 70], ticks: [0, 20, 40, 60],
            canvit: { macro: "adeBestMiou" } },
};

// Entries sota-history.md flags as unreproduced or as two-model systems; the frontier is drawn without them.
const FLAGGED = new Set(["OmniVec (fine-tuned)", "OmniVec2 (fine-tuned)", "ViT-P on InternImage-H + Mask2Former"]);
// The sequential active models named on the chart, at their best point: label offset (px) and anchor.
const LABELED = { "Saccader": [-10, 22, "end"], "STAM": [0, -14, "middle"], "AdaGlimpse": [10, 5, "start"],
                  "AdaptiveNN": [0, -14, "middle"], "AME": [-10, 5, "end"] };

const WIDTH = 540, HEIGHT = 400, M = { left: 52, right: 18, top: 16, bottom: 40 };

function el(name, attributes, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attributes)) node.setAttribute(k, v);
  parent?.append(node);
  return node;
}

const yearOf = (point) => Number(String(point.date).slice(0, 4)) + (Number(String(point.date).slice(5, 7) || 6) - 0.5) / 12;

/** The best value published by each date: a step line through the points that set a new record. */
function frontier(points) {
  const sorted = [...points].sort((a, b) => yearOf(a) - yearOf(b));
  const records = [];
  for (const p of sorted) if (!records.length || p.value > records.at(-1).value) records.push(p);
  return records;
}

function stepPath(records, x, y, endYear) {
  let d = "";
  records.forEach((p, i) => {
    d += i === 0 ? `M${x(yearOf(p))},${y(p.value)}` : `H${x(yearOf(p))}V${y(p.value)}`;
  });
  return `${d}H${x(endYear)}`;
}

function draw(container, data, macros, benchmark) {
  const spec = BENCHMARKS[benchmark];
  if (!spec) throw new Error(`history chart: unknown benchmark "${benchmark}"`);
  const points = data[spec.key];
  if (!Array.isArray(points)) throw new Error(`history chart: sota-history.json has no "${spec.key}"`);
  const x = (year) => M.left + (year - spec.years[0]) / (spec.years[1] - spec.years[0]) * (WIDTH - M.left - M.right);
  const y = (v) => HEIGHT - M.bottom - (v - spec.range[0]) / (spec.range[1] - spec.range[0]) * (HEIGHT - M.top - M.bottom);
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img", "aria-label": spec.title });

  const axes = el("g", { class: "axes" }, svg);
  for (const tick of spec.ticks) {
    el("line", { x1: M.left, x2: WIDTH - M.right, y1: y(tick), y2: y(tick), class: "grid" }, axes);
    el("text", { x: M.left - 8, y: y(tick) + 5, "text-anchor": "end" }, axes).textContent = tick;
  }
  for (let year = Math.ceil(spec.years[0] / 2) * 2; year <= spec.years[1]; year += 2)
    el("text", { x: x(year + 0.5), y: HEIGHT - M.bottom + 24, "text-anchor": "middle" }, axes).textContent = year;
  el("text", { x: M.left, y: M.top - 2, class: "unit" }, axes).textContent = spec.unit;

  const passive = points.filter((p) => p.kind === "passive" && p.series !== "frozen_ssl" && !FLAGGED.has(p.model));
  const best = frontier(passive);
  const group = el("g", { class: "passive" }, svg);
  el("path", { d: stepPath(best, x, y, spec.years[1]), class: "line" }, group);

  const frozen = frontier(points.filter((p) => p.series === "frozen_ssl" && p.protocol !== "linear probe + ms"));
  const frozenGroup = el("g", { class: "frozen" }, svg);
  el("path", { d: stepPath(frozen, x, y, spec.years[1]), class: "line" }, frozenGroup);

  const active = el("g", { class: "active" }, svg);
  for (const p of points.filter((p) => p.series === "active")) {
    el("circle", { cx: x(yearOf(p)), cy: y(p.value), r: 6 }, active);
    const name = p.model.split(/[ (-]/)[0];
    const top = Math.max(...points.filter((q) => q.series === "active" && q.model.split(/[ (-]/)[0] === name).map((q) => q.value));
    if (name in LABELED && p.value === top) {
      const [dx, dy, anchor] = LABELED[name];
      el("text", { x: x(yearOf(p)) + dx, y: y(p.value) + dy, "text-anchor": anchor, class: "label" }, active).textContent = name;
    }
  }

  const value = Number(macros[spec.canvit.macro]);
  if (!Number.isFinite(value)) throw new Error(`history chart: macro ${spec.canvit.macro} is missing`);
  const canvit = el("g", { class: "canvit" }, svg);
  el("circle", { cx: x(2026.25), cy: y(value), r: 9 }, canvit);
  el("text", { x: x(2026.25), y: y(value) - 16, "text-anchor": "end", class: "label" }, canvit).textContent = "CanViT-B";

  container.replaceChildren(svg);
}

/** Draw every .history-chart of the page: data-benchmark="imagenet" or "ade20k". */
export async function drawHistoryCharts({ data = "sources/sota-history.json", macros = "../../assets/paper/data_macros.json" } = {}) {
  const [history, values] = await Promise.all([data, macros].map(async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`history chart: ${url} answered ${response.status}`);
    return response.json();
  }));
  for (const container of document.querySelectorAll(".history-chart")) draw(container, history, values, container.dataset.benchmark);
}
