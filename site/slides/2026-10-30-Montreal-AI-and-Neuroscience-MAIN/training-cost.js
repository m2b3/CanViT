// The training-cost slide: each active model's best accuracy against its own training compute, one chart per
// benchmark, from throwaway/training_cost/export.py's JSON (the rebuttal's FLOP accounting: teachers and pretrained
// weights counted on no side). A model is a point at its compute, a bar across a range of estimates, a bar with an arrow
// when only a lower bound is known, or a dashed line at its accuracy when its compute is not disclosed. CanViT-B is in
// canvas red, filled when fine-tuned and hollow when frozen with a linear probe, as on the history charts.

const NS = "http://www.w3.org/2000/svg";
const WIDTH = 540, HEIGHT = 420;
const PLOT = { left: 64, right: 520, top: 18, bottom: 360 };
const X_RANGE = [1, 300]; // EFLOPs, log scale
const X_TICKS = [1, 3, 10, 30, 100, 300];
const BENCHMARKS = {
  imagenet: { title: "ImageNet-1k top-1 (%)", range: [74, 86], step: 2 },
  ade20k: { title: "ADE20K mIoU (%)", range: [20, 50], step: 5 },
};

function el(name, attributes, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent.append(node);
  return node;
}

function text(content, attributes, parent) {
  const node = el("text", attributes, parent);
  node.textContent = content;
  return node;
}

function chart(container, benchmark, entries) {
  const spec = BENCHMARKS[benchmark];
  if (!spec) throw new Error(`training cost: unknown benchmark ${benchmark}`);
  const x = (eflops) => PLOT.left + (Math.log(eflops / X_RANGE[0]) / Math.log(X_RANGE[1] / X_RANGE[0])) * (PLOT.right - PLOT.left);
  const y = (value) => PLOT.bottom - ((value - spec.range[0]) / (spec.range[1] - spec.range[0])) * (PLOT.bottom - PLOT.top);
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img", "aria-label": `${spec.title} against training compute` }, container);
  const defs = el("defs", {}, svg);
  const head = el("marker", { id: `cost-arrow-${benchmark}`, viewBox: "0 0 10 10", refX: 6, refY: 5, markerWidth: 18,
                              markerHeight: 18, markerUnits: "userSpaceOnUse", orient: "auto" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 Z", class: "arrowhead" }, head);

  for (let value = spec.range[0]; value <= spec.range[1]; value += spec.step) {
    el("line", { x1: PLOT.left, x2: PLOT.right, y1: y(value), y2: y(value), class: "grid" }, svg);
    text(value, { x: PLOT.left - 10, y: y(value) + 6, "text-anchor": "end", class: "tick" }, svg);
  }
  for (const tick of X_TICKS) {
    el("line", { x1: x(tick), x2: x(tick), y1: PLOT.bottom, y2: PLOT.bottom + 6, class: "axis" }, svg);
    text(tick, { x: x(tick), y: PLOT.bottom + 26, "text-anchor": "middle", class: "tick" }, svg);
  }
  el("line", { x1: PLOT.left, x2: PLOT.right, y1: PLOT.bottom, y2: PLOT.bottom, class: "axis" }, svg);
  text("Own training compute (EFLOPs)", { x: (PLOT.left + PLOT.right) / 2, y: PLOT.bottom + 56, "text-anchor": "middle", class: "title" }, svg);
  text(spec.title, { x: -(PLOT.top + PLOT.bottom) / 2, y: 16, transform: "rotate(-90)", "text-anchor": "middle", class: "title" }, svg);

  for (const entry of entries) {
    const ours = entry.model.startsWith("CanViT");
    const group = el("g", { class: ours ? "entry canvit" : "entry prior" }, svg);
    const vy = y(entry.accuracy);
    const value = entry.accuracy.toFixed(1);
    if (entry.eflops === null) {  // compute not disclosed: a dashed line at its accuracy
      el("line", { x1: PLOT.left, x2: PLOT.right, y1: vy, y2: vy, class: "undisclosed" }, group);
      text(`${entry.model} ${value}, training compute not disclosed`, { x: PLOT.left + 8, y: vy - 9, class: "label" }, group);
      continue;
    }
    const [low, high] = entry.eflops;
    const lowerBound = high === null, range = !lowerBound && high > low;
    if (lowerBound) {  // only a lower bound: from it to the edge, with an arrow
      el("line", { x1: x(low), x2: PLOT.right - 4, y1: vy, y2: vy, class: "range", "marker-end": `url(#cost-arrow-${benchmark})` }, group);
    } else if (range) {
      el("line", { x1: x(low), x2: x(high), y1: vy, y2: vy, class: "range" }, group);
    }
    const cx = range ? (x(low) + x(high)) / 2 : x(low); // a range's point at its geometric mean
    el("circle", { cx, cy: vy, r: 8, class: ours && entry.readout.startsWith("frozen") ? "hollow" : "filled" }, group);
    const name = ours ? `${entry.model}, ${entry.readout.split(",")[0]}` : entry.model;
    const note = entry.model === "AME" ? " + pretrained ViT-L" : "";
    // Labels sit on the side of the plot with more room: left of the mark in the right half, right of it otherwise.
    const right = cx > (PLOT.left + PLOT.right) / 2;
    text(`${name} ${value}${note}`, { x: right ? x(low) - 14 : (range ? x(high) : cx) + 14, y: vy + 6,
                                      "text-anchor": right ? "end" : "start", class: "label" }, group);
  }
}

export async function drawTrainingCost(section, src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`training cost: ${src} answered ${response.status}`);
  const data = await response.json();
  for (const container of section.querySelectorAll("[data-benchmark]")) {
    const benchmark = container.dataset.benchmark;
    if (!(benchmark in data)) throw new Error(`training cost: no ${benchmark} in ${src}`);
    container.replaceChildren();
    chart(container, benchmark, data[benchmark]);
  }
}
