// <canvit-policy-curves src task="ade20k|in1k" canvas-grid y-range="LO HI" [dashed-src] [dashed-canvas-grid] [legend]
// [x-scale="linear|log"] [policies] [baselines] [t]>:
// CanViT-B's accuracy under the paper's viewing policies (all those the task evaluates, or the ids listed in
// policies), glimpse by glimpse, over a whole validation set (mean over runs), from the paper's exports: ADE20K mIoU
// (ade20k_seg.json, policy_curves) or ImageNet-1k top-1 (in1k_clf_frozen.json or in1k_clf_finetuned.json, configs).
// src at canvas-grid gives the solid curves, named at their heads. dashed-src or dashed-canvas-grid (each defaulting to
// its solid counterpart) adds the same policies dashed beside them, as the paper's Figure 3 does for a 32² canvas
// beside a 64² one and for frozen weights beside fine-tuned ones; legend names the two, "solid|dashed". y-range fixes the
// vertical axis in percent, so charts of the two tasks can share a span of points; lines are clipped to it, so a narrow
// range shows the end of the rollouts with the low starts entering from below. Lines are drawn up to glimpse t (all
// without t), each in its policy's color from the paper's figures; a slide drives t to grow them in step with recorded
// rollouts. baselines lists prior models of src's baselines, comma-separated by name ("AME (SETR), AdaGlimpse"); each
// one's best result is a dashed line across the plot, named at its right end (where rising curves have passed it),
// drawn when inside y-range.

import { POLICIES } from "./policies.js";

const SCENE_SIZE = 512;
const LABEL_GAP = 17;
// The exports by task: where their curves are, the axis title, and the policies each evaluates (EG-C2F needs the
// segmentation's per-position entropy, so ImageNet-1k has none).
const TASKS = {
  ade20k: {
    curves: (data) => data.policy_curves, baselines: (data) => data.baselines.map((b) => ({ name: b.name, value: b.miou_pct })),
    title: "ADE20K mIoU (%)", policies: Object.keys(POLICIES),
  },
  in1k: {
    curves: (data) => data.configs, baselines: () => [], title: "ImageNet-1k top-1 (%)",
    policies: Object.keys(POLICIES).filter((policy) => policy !== "entropy_coarse_to_fine"),
  },
};
const X_SCALES = ["linear", "log"];

const template = document.createElement("template");
template.innerHTML = `
<style>
  :host { display: block; color: var(--canvit-ink, #0f172a); font: 14px/1.4 var(--canvit-sans, system-ui, sans-serif); }
  svg { display: block; width: 100%; overflow: visible; }
  .grid line { stroke: #e2e8f0; }
  .axis text { fill: var(--canvit-muted, #475569); font-size: 14px; font-variant-numeric: tabular-nums; }
  .axis .title { fill: var(--canvit-ink, #0f172a); font-size: 15px; font-weight: 650; }
  .curve { fill: none; stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }
  .curve.dashed { stroke-width: 2.6; stroke-dasharray: 7 5; }
  .legend text { fill: var(--canvit-muted, #475569); font-size: 14px; font-weight: 600; }
  .legend line { stroke: var(--canvit-muted, #475569); stroke-width: 2.6; }
  .legend .dashed { stroke-dasharray: 7 5; }
  .head { stroke: var(--canvit-surface, #fff); stroke-width: 2; }
  .baseline { stroke: var(--canvit-ink, #0f172a); stroke-width: 2; stroke-dasharray: 7 5; }
  .baseline-label { fill: var(--canvit-ink, #0f172a); font-size: 15px; font-weight: 700; paint-order: stroke; stroke: var(--canvit-surface, #fff); stroke-width: 4px; }
  .label { font-size: 16px; font-weight: 800; paint-order: stroke; stroke: var(--canvit-surface, #fff); stroke-width: 5px; }
</style>
<svg role="img"></svg>`;

const NS = "http://www.w3.org/2000/svg";
function el(name, attributes, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent.append(node);
  return node;
}

function required(element, name) {
  const value = element.getAttribute(name);
  if (value === null) throw new Error(`<canvit-policy-curves>: missing ${name}`);
  return value;
}

class CanvitPolicyCurves extends HTMLElement {
  static observedAttributes = ["src", "dashed-src", "task", "canvas-grid", "dashed-canvas-grid", "legend", "policies", "baselines", "y-range", "x-scale", "t"];
  #curves = null;
  #baselines = [];
  #legend = null;
  #loadVersion = 0;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(template.content.cloneNode(true));
    this.svg = this.shadowRoot.querySelector("svg");
    new ResizeObserver(() => this.#render()).observe(this.svg);
  }

  get #task() {
    const task = required(this, "task");
    if (!(task in TASKS)) throw new Error(`<canvit-policy-curves>: task="${task}", expected ${Object.keys(TASKS).join(" or ")}`);
    return TASKS[task];
  }

  get #yRange() {
    const [low, high] = required(this, "y-range").trim().split(/\s+/).map(Number);
    if (!(low < high)) throw new Error(`<canvit-policy-curves>: y-range="${this.getAttribute("y-range")}" is not "LO HI"`);
    return [low, high];
  }

  get #xScale() {
    const scale = this.getAttribute("x-scale") ?? "linear";
    if (!X_SCALES.includes(scale)) throw new Error(`<canvit-policy-curves>: x-scale="${scale}", expected ${X_SCALES.join(" or ")}`);
    return scale;
  }

  async attributeChangedCallback(name) {
    if (name === "t" || name === "y-range" || name === "x-scale") return this.#render();
    const version = ++this.#loadVersion;
    const src = this.getAttribute("src");
    if (!src || !this.hasAttribute("task") || !this.hasAttribute("canvas-grid")) return;
    const task = this.#task;
    const grid = Number(this.getAttribute("canvas-grid"));
    const withDashed = this.hasAttribute("dashed-src") || this.hasAttribute("dashed-canvas-grid");
    const dashedSrc = this.getAttribute("dashed-src") ?? src;
    const dashedGrid = Number(this.getAttribute("dashed-canvas-grid") ?? grid);
    const legend = withDashed ? this.getAttribute("legend")?.split("|").map((label) => label.trim()) : null;
    if (withDashed && legend?.length !== 2) throw new Error(`<canvit-policy-curves>: dashed curves need legend="solid|dashed"`);
    const [data, dashedData] = await Promise.all([src, withDashed ? dashedSrc : null].map(async (url) => {
      if (url === null) return null;
      const response = await fetch(new URL(url, document.baseURI));
      if (!response.ok) throw new Error(`<canvit-policy-curves>: could not fetch ${url}: HTTP ${response.status}`);
      return response.json();
    }));
    if (version !== this.#loadVersion) return;
    const wanted = this.getAttribute("policies")?.trim().split(/\s+/) ?? task.policies;
    const unknown = wanted.filter((policy) => !task.policies.includes(policy));
    if (unknown.length) throw new Error(`<canvit-policy-curves>: policies "${unknown.join(" ")}" not evaluated on ${this.getAttribute("task")}`);
    const select = (exported, url, canvasGrid, style) => {
      const curves = task.curves(exported).filter((c) => c.scene_size === SCENE_SIZE && c.canvas_grid === canvasGrid && wanted.includes(c.policy));
      const found = curves.map((c) => c.policy).sort().join(" ");
      if (found !== [...wanted].sort().join(" ")) {
        throw new Error(`<canvit-policy-curves>: ${url} at a ${canvasGrid}² canvas has policies "${found}", expected "${wanted.join(" ")}"`);
      }
      return curves.map((curve) => ({ ...curve, style }));
    };
    const curves = [...select(data, src, grid, "solid"), ...(withDashed ? select(dashedData, dashedSrc, dashedGrid, "dashed") : [])];
    this.#legend = legend;
    const known = task.baselines(data);
    this.#baselines = (this.getAttribute("baselines")?.split(",").map((name) => name.trim()).filter(Boolean) ?? []).map((name) => {
      const runs = known.filter((b) => b.name === name);
      if (!runs.length) throw new Error(`<canvit-policy-curves>: no baseline "${name}" in ${src}`);
      return runs.reduce((a, b) => (b.value > a.value ? b : a));
    });
    this.#curves = curves;
    this.#render();
  }

  #render() {
    const width = Math.round(this.svg.clientWidth);
    if (!this.#curves || width === 0) return;
    const [low, high] = this.#yRange;
    const steps = Math.max(...this.#curves.map((c) => c.per_timestep.length));
    const shown = Math.min(steps, this.hasAttribute("t") ? Number(this.getAttribute("t")) + 1 : steps);
    const height = Math.round(width * 0.78);
    const plot = { left: 50, right: width - 78, top: 10, bottom: height - 46 };
    // Glimpse n (t = n - 1) on a linear or a log axis from glimpse 1 to the last.
    const position = this.#xScale === "log" ? (t) => Math.log(t + 1) / Math.log(steps) : (t) => t / (steps - 1);
    const x = (t) => plot.left + position(t) * (plot.right - plot.left);
    const y = (percent) => plot.bottom - ((percent - low) / (high - low)) * (plot.bottom - plot.top);
    const svg = this.svg;
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", String(height));
    svg.setAttribute("aria-label", `${this.#task.title} of CanViT-B by viewing policy, after ${shown} glimpse${shown > 1 ? "s" : ""}`);

    const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
    const tick = high - low <= 15 ? 2 : high - low <= 30 ? 5 : 10;
    for (let value = Math.ceil(low / tick) * tick; value <= high; value += tick) {
      el("line", { x1: plot.left, x2: plot.right, y1: y(value), y2: y(value) }, grid);
      el("text", { x: plot.left - 10, y: y(value) + 5, "text-anchor": "end" }, axis).textContent = String(value);
    }
    const ticks = this.#xScale === "log" ? [1, 2, 5, 10, steps] : [1, 5, 10, 15, steps];
    for (const glimpse of ticks.filter((g) => g <= steps)) {
      el("text", { x: x(glimpse - 1), y: plot.bottom + 22, "text-anchor": "middle" }, axis).textContent = String(glimpse);
    }
    el("text", { class: "title", x: (plot.left + plot.right) / 2, y: height - 6, "text-anchor": "middle" }, axis).textContent = "Glimpses";
    el("text", { class: "title", x: -(plot.top + plot.bottom) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" }, axis)
      .textContent = this.#task.title;

    const clip = el("clipPath", { id: "plot" }, el("defs", {}, svg));
    el("rect", { x: plot.left, y: plot.top - 6, width: plot.right - plot.left + 8, height: plot.bottom - plot.top + 6 }, clip);
    for (const { name, value } of this.#baselines.filter(({ value }) => value >= low && value <= high)) {
      el("line", { class: "baseline", x1: plot.left, x2: plot.right, y1: y(value), y2: y(value) }, svg);
      el("text", { class: "baseline-label", x: plot.right - 6, y: y(value) - 8, "text-anchor": "end" }, svg).textContent = `${name} ${value.toFixed(1)}`;
    }
    const lines = el("g", { "clip-path": "url(#plot)" }, svg);
    // Dashed curves are unnamed beside the solid ones; the legend says which is which.
    if (this.#legend) {
      const legend = el("g", { class: "legend" }, svg);
      [["solid", this.#legend[0]], ["dashed", this.#legend[1]]].forEach(([style, text], i) => {
        const ly = plot.top + 12 + 22 * i;
        el("line", { class: style, x1: plot.left + 14, x2: plot.left + 44, y1: ly, y2: ly }, legend);
        el("text", { x: plot.left + 52, y: ly + 5 }, legend).textContent = text;
      });
    }
    // Policies rank by their end value (to 0.1 point; a tie goes to the higher mean over the glimpses shown). The
    // higher-ranked curve and head are drawn over the others, and heads are labeled in rank order, each label at least
    // LABEL_GAP below the one above, so close lines stay legible.
    const ranked = this.#curves.map((curve) => {
      const shownPoints = curve.per_timestep.slice(0, shown);
      const means = shownPoints.map((p) => p.mean);
      return { ...POLICIES[curve.policy], style: curve.style, points: shownPoints.map((p) => [x(p.t), y(100 * p.mean)]),
               end: Math.round(1000 * means.at(-1)), mean: means.reduce((a, b) => a + b) / means.length };
    }).sort((a, b) => b.end - a.end || b.mean - a.mean);
    for (const { points, color, style } of [...ranked].reverse()) {
      el("polyline", { class: `curve ${style}`, points: points.map((p) => p.join(",")).join(" "), stroke: color }, lines);
    }
    // A head below or above the range is not drawn or named yet.
    const heads = ranked.filter(({ points, style }) => style === "solid" && points.at(-1)[1] <= plot.bottom && points.at(-1)[1] >= plot.top);
    for (const { points, color } of [...heads].reverse()) {
      const [hx, hy] = points.at(-1);
      el("circle", { class: "head", cx: hx, cy: hy, r: 5.5, fill: color }, svg);
    }
    let floor = -Infinity;
    for (const { points, label, color } of heads) {
      const [hx, hy] = points.at(-1);
      const ly = Math.max(hy + 5, floor + LABEL_GAP);
      floor = ly;
      el("text", { class: "label", x: hx + 10, y: ly, fill: color }, svg).textContent = label;
    }
  }
}

customElements.define("canvit-policy-curves", CanvitPolicyCurves);
