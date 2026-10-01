// <canvit-policy-curves src task="ade20k|in1k" canvas-grid y-range="LO HI" [x-scale="linear|log"] [t]>: CanViT-B's
// accuracy under each of the paper's viewing policies, glimpse by glimpse, over a whole validation set (mean over
// runs), from the paper's exports: ADE20K mIoU (ade20k_seg.json, policy_curves) or frozen ImageNet-1k top-1
// (in1k_clf_frozen.json, configs). y-range fixes the vertical axis in percent, so charts of the two tasks can share a
// span of points. Lines are drawn up to glimpse t (all without t), each in its policy's color from the paper's figures
// and named at its head; a slide drives t to grow them in step with recorded rollouts.

import { POLICIES } from "./policies.js";

const SCENE_SIZE = 512;
const LABEL_GAP = 17;
// The exports by task: where their curves are, the axis title, and the policies each evaluates (EG-C2F needs the
// segmentation's per-position entropy, so ImageNet-1k has none).
const TASKS = {
  ade20k: { curves: (data) => data.policy_curves, title: "ADE20K mIoU (%)", policies: Object.keys(POLICIES) },
  in1k: {
    curves: (data) => data.configs, title: "ImageNet-1k top-1 (%)",
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
  .head { stroke: var(--canvit-surface, #fff); stroke-width: 2; }
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
  static observedAttributes = ["src", "task", "canvas-grid", "y-range", "x-scale", "t"];
  #curves = null;
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
    const response = await fetch(new URL(src, document.baseURI));
    if (!response.ok) throw new Error(`<canvit-policy-curves>: could not fetch ${src}: HTTP ${response.status}`);
    const data = await response.json();
    if (version !== this.#loadVersion) return;
    const curves = task.curves(data).filter((c) => c.scene_size === SCENE_SIZE && c.canvas_grid === grid);
    const found = curves.map((c) => c.policy).sort().join(" ");
    if (found !== [...task.policies].sort().join(" ")) {
      throw new Error(`<canvit-policy-curves>: ${src} at a ${grid}² canvas has policies "${found}", expected "${task.policies.join(" ")}"`);
    }
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
    for (let value = Math.ceil(low / 10) * 10; value <= high; value += 10) {
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

    // Heads are labeled in order of height, each at least LABEL_GAP below the one above, so close lines stay legible.
    const heads = [];
    for (const curve of this.#curves) {
      const { label, color } = POLICIES[curve.policy];
      const points = curve.per_timestep.slice(0, shown).map((p) => [x(p.t), y(100 * p.mean)]);
      el("polyline", { class: "curve", points: points.map((p) => p.join(",")).join(" "), stroke: color }, svg);
      const [hx, hy] = points.at(-1);
      el("circle", { class: "head", cx: hx, cy: hy, r: 5.5, fill: color }, svg);
      heads.push({ hx, hy, label, color });
    }
    heads.sort((a, b) => a.hy - b.hy);
    let floor = -Infinity;
    for (const head of heads) {
      const ly = Math.max(head.hy + 5, floor + LABEL_GAP);
      floor = ly;
      el("text", { class: "label", x: head.hx + 10, y: ly, fill: head.color }, svg).textContent = head.label;
    }
  }
}

customElements.define("canvit-policy-curves", CanvitPolicyCurves);
