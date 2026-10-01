// <canvit-policy-race src="ade20k_seg.json" [canvas-grid="64"] [t]>: the paper's ADE20K mIoU of CanViT-B under each of
// its viewing policies, glimpse by glimpse, over the whole validation set (the export's policy_curves, mean over
// runs), drawn up to glimpse t (all 21 without t), each line in its policy's color from the paper's figures and named
// at its head. A slide drives t to grow the lines in step with recorded rollouts.

import { POLICIES } from "./policies.js";

const SCENE_SIZE = 512;
const Y_MAX = 50;
const LABEL_GAP = 17;

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

class CanvitPolicyRace extends HTMLElement {
  static observedAttributes = ["src", "canvas-grid", "t"];
  #curves = null;
  #loadVersion = 0;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(template.content.cloneNode(true));
    this.svg = this.shadowRoot.querySelector("svg");
    new ResizeObserver(() => this.#render()).observe(this.svg);
  }

  get canvasGrid() { return Number(this.getAttribute("canvas-grid") ?? 64); }

  async attributeChangedCallback(name) {
    if (name === "t") return this.#render();
    const version = ++this.#loadVersion;
    const src = this.getAttribute("src");
    if (!src) return;
    const response = await fetch(new URL(src, document.baseURI));
    if (!response.ok) throw new Error(`<canvit-policy-race>: could not fetch ${src}: HTTP ${response.status}`);
    const data = await response.json();
    if (version !== this.#loadVersion) return;
    const curves = data.policy_curves.filter((c) => c.scene_size === SCENE_SIZE && c.canvas_grid === this.canvasGrid);
    const missing = Object.keys(POLICIES).filter((policy) => !curves.some((c) => c.policy === policy));
    if (missing.length) throw new Error(`<canvit-policy-race>: ${src} has no ${this.canvasGrid}² curve for ${missing.join(", ")}`);
    this.#curves = curves;
    this.#render();
  }

  #render() {
    const width = Math.round(this.svg.clientWidth);
    if (!this.#curves || width === 0) return;
    const steps = Math.max(...this.#curves.map((c) => c.per_timestep.length));
    const shown = Math.min(steps, this.hasAttribute("t") ? Number(this.getAttribute("t")) + 1 : steps);
    const height = Math.round(width * 0.78);
    const plot = { left: 50, right: width - 78, top: 10, bottom: height - 46 };
    const x = (t) => plot.left + (t / (steps - 1)) * (plot.right - plot.left);
    const y = (miou) => plot.bottom - (miou / Y_MAX) * (plot.bottom - plot.top);
    const svg = this.svg;
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", String(height));
    svg.setAttribute("aria-label", `ADE20K mIoU of CanViT-B by viewing policy, after ${shown} glimpse${shown > 1 ? "s" : ""}`);

    const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
    for (let value = 0; value <= Y_MAX; value += 10) {
      el("line", { x1: plot.left, x2: plot.right, y1: y(value), y2: y(value) }, grid);
      el("text", { x: plot.left - 10, y: y(value) + 5, "text-anchor": "end" }, axis).textContent = String(value);
    }
    for (const t of [0, 4, 9, 14, 20].filter((t) => t < steps)) {
      el("text", { x: x(t), y: plot.bottom + 22, "text-anchor": "middle" }, axis).textContent = String(t + 1);
    }
    el("text", { class: "title", x: (plot.left + plot.right) / 2, y: height - 6, "text-anchor": "middle" }, axis).textContent = "Glimpses";
    el("text", { class: "title", x: -(plot.top + plot.bottom) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" }, axis)
      .textContent = "ADE20K mIoU (%)";

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

customElements.define("canvit-policy-race", CanvitPolicyRace);
