// <canvit-frontier src="ade20k_seg.json">: the paper's accuracy–efficiency frontier on ADE20K (Figure 3A), drawn
// from its exported data: CanViT-B's mIoU against cumulative inference FLOPs, glimpse by glimpse, for EG-C2F and
// F2C at 32² and 64² canvases, and the prior active models as points. Colors, line styles and markers follow the
// paper's figure. The chart is drawn at its displayed size, so text stays legible on narrow screens; hovering
// names the nearest point.

import { POLICIES as PAPER_POLICIES } from "./policies.js";

const POLICIES = {
  entropy_coarse_to_fine: PAPER_POLICIES.entropy_coarse_to_fine,
  fine_to_coarse: PAPER_POLICIES.fine_to_coarse,
};
const CANVASES = { 32: { label: "32² canvas", dash: "7 5" }, 64: { label: "64² canvas", dash: "" } };
const SCENE_SIZE = 512;
const BASELINES = {
  "AME (SETR)": { color: "#7f2704", marker: "triangle" },
  "AME (MAE)": { color: "#00a4b8", marker: "triangle" },
  AdaGlimpse: { color: "#6a3d9a", marker: "diamond" },
};
const Y_MAX = 50;
const NARROW_PX = 560;

const template = document.createElement("template");
template.innerHTML = `
<style>
  :host { display: block; color: var(--canvit-ink, #0f172a); font: 14px/1.4 var(--canvit-sans, system-ui, sans-serif); }
  .legend { display: grid; gap: 8px; margin-top: 14px; font-size: 14px; }
  .legend .row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 18px; }
  .legend b { min-width: 10.5em; font-weight: 700; }
  @media (max-width: 600px) { .legend b { flex-basis: 100%; } }
  .legend span { display: inline-flex; align-items: center; gap: 8px; }
  .legend svg.marker { width: 12px; height: 12px; }
  .halo { paint-order: stroke; stroke: var(--canvit-surface, #fff); stroke-width: 5px; stroke-linejoin: round; }
  .reference { stroke: #64748b; stroke-width: 1.4; stroke-dasharray: 5 5; }
  .reference-label { fill: #475569; font-size: 13.5px; font-weight: 600; }
  .end-label { font-size: 13.5px; font-weight: 700; }
  .takeaway { fill: var(--canvit-ink, #0f172a); font-size: 14px; font-weight: 650; }
  .takeaway-line { stroke: var(--canvit-ink, #0f172a); stroke-width: 1.2; }
  .legend svg { width: 24px; height: 6px; }
  .frame { position: relative; }
  .chart { display: block; width: 100%; overflow: visible; }
  .grid line { stroke: #e2e8f0; }
  .axis text { fill: var(--canvit-muted, #475569); font-size: 13.5px; font-variant-numeric: tabular-nums; }
  .axis .title { fill: var(--canvit-ink, #0f172a); font-size: 14.5px; font-weight: 650; }
  .curve { fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }
  .band { opacity: .15; }
  .baseline-label { fill: var(--canvit-ink, #0f172a); font-size: 13.5px; }
  .baseline-label tspan { fill: var(--canvit-muted, #475569); }
  .callout { fill: var(--canvit-ink, #0f172a); font-size: 15px; font-weight: 750; }
  .callout-line { stroke: var(--canvit-ink, #0f172a); stroke-width: 1.2; }
  .focus circle { fill: #fff; stroke-width: 2.5; }
  .reveal { transition: width 1.4s cubic-bezier(.3, .6, .2, 1); }
  .tooltip { position: absolute; pointer-events: none; padding: 7px 10px; border-radius: 8px; background: var(--canvit-ink, #0f172a);
             color: #fff; font-size: 13.5px; line-height: 1.45; white-space: nowrap; transform: translate(-50%, calc(-100% - 12px)); }
  .tooltip b { font-weight: 650; }
  [hidden] { display: none !important; }
  @media (prefers-reduced-motion: reduce) { .reveal { transition: none; } }
</style>
<div class="frame"><svg class="chart" role="img"></svg><div class="tooltip" hidden></div></div>
<div class="legend"></div>`;

const NS = "http://www.w3.org/2000/svg";
function el(name, attributes = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent?.append(node);
  return node;
}

function marker(shape, x, y, size, color, parent) {
  const r = size / 2;
  const points = shape === "triangle"
    ? [[x, y - r * 1.15], [x + r, y + r * 0.7], [x - r, y + r * 0.7]]
    : [[x, y - r * 1.1], [x + r * 1.1, y], [x, y + r * 1.1], [x - r * 1.1, y]];
  return el("polygon", { points: points.map((p) => p.join(",")).join(" "), fill: color }, parent);
}

/** Round tick values covering [0, max]. */
function ticks(max, target) {
  const raw = max / target;
  const step = [1, 2, 2.5, 5, 10].map((m) => m * 10 ** Math.floor(Math.log10(raw))).find((s) => s >= raw);
  return Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => i * step);
}

const lineSwatch = (color, dash) =>
  `<svg viewBox="0 0 24 6"><line x1="1" x2="23" y1="3" y2="3" stroke="${color}" stroke-width="2.6" stroke-dasharray="${dash}"/></svg>`;

class CanvitFrontier extends HTMLElement {
  static observedAttributes = ["src"];
  #data = null;
  #points = [];
  #shown = false;
  #width = 0;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(template.content.cloneNode(true));
    this.svg = this.shadowRoot.querySelector(".chart");
    this.tooltip = this.shadowRoot.querySelector(".tooltip");
    const markerSwatch = ({ color, marker: shape }) =>
      `<svg viewBox="-7 -7 14 14" class="marker">${shape === "triangle" ? `<polygon points="0,-7 6,4 -6,4" fill="${color}"/>` : `<polygon points="0,-7 7,0 0,7 -7,0" fill="${color}"/>`}</svg>`;
    this.shadowRoot.querySelector(".legend").innerHTML =
      `<div class="row"><b>CanViT-B</b>${Object.values(POLICIES).map(({ label, name, color }) => `<span>${lineSwatch(color, "")}${label} (${name})</span>`).join("")}` +
      `${Object.values(CANVASES).map(({ label, dash }) => `<span>${lineSwatch("#475569", dash)}${label}</span>`).join("")}</div>` +
      `<div class="row"><b>Prior active models</b>${Object.entries(BASELINES).map(([name, style]) => `<span>${markerSwatch(style)}${name}</span>`).join("")}</div>`;
    new ResizeObserver(() => {
      const width = Math.round(this.svg.clientWidth);
      if (this.#data && width > 0 && width !== this.#width) this.#render(width);
    }).observe(this.svg);
    new IntersectionObserver(([entry], observer) => {
      if (!entry.isIntersecting) return;
      this.#shown = true;
      this.svg.querySelector(".reveal")?.setAttribute("width", String(this.#width));
      observer.disconnect();
    }, { threshold: 0.3 }).observe(this);
  }

  async attributeChangedCallback() {
    const response = await fetch(new URL(this.getAttribute("src"), document.baseURI));
    if (!response.ok) throw new Error(`<canvit-frontier>: could not fetch ${this.getAttribute("src")}: HTTP ${response.status}`);
    this.#data = await response.json();
    const curves = this.#curves();
    if (curves.length !== Object.keys(POLICIES).length * Object.keys(CANVASES).length) {
      throw new Error(`<canvit-frontier>: expected every policy at every canvas, found ${curves.length} curves`);
    }
    this.#render(Math.round(this.svg.clientWidth));
  }

  #curves() {
    return this.#data.policy_curves.filter((c) => c.policy in POLICIES && c.scene_size === SCENE_SIZE && c.canvas_grid in CANVASES);
  }

  #render(width) {
    this.#width = width;
    const narrow = width < NARROW_PX;
    const height = Math.round(Math.min(460, Math.max(300, width * 0.58)));
    const margin = { top: 14, right: 12, bottom: 54, left: 58 };
    const svg = this.svg;
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", String(height));
    svg.setAttribute("aria-label", "ADE20K mIoU against cumulative inference FLOPs: CanViT-B's curves lie above and to the left of every prior active model.");

    const curves = this.#curves(), baselines = this.#data.baselines;
    const xMax = Math.max(...curves.flatMap((c) => c.per_timestep.map((p) => p.cum_gflops)), ...baselines.map((b) => b.gflops)) * 1.04;
    const plot = { left: margin.left, right: width - margin.right, top: margin.top, bottom: height - margin.bottom };
    const x = (gflops) => plot.left + (gflops / xMax) * (plot.right - plot.left);
    const y = (miou) => plot.bottom - (miou / Y_MAX) * (plot.bottom - plot.top);

    const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
    for (const value of ticks(Y_MAX, 5)) {
      el("line", { x1: plot.left, x2: plot.right, y1: y(value), y2: y(value) }, grid);
      el("text", { x: plot.left - 10, y: y(value) + 4, "text-anchor": "end" }, axis).textContent = String(value);
    }
    for (const value of ticks(xMax, narrow ? 3 : 5)) {
      el("text", { x: x(value), y: plot.bottom + 20, "text-anchor": "middle" }, axis).textContent = String(value);
    }
    el("line", { x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom, stroke: "#94a3b8" }, svg);
    el("text", { class: "title", x: (plot.left + plot.right) / 2, y: height - 10, "text-anchor": "middle" }, axis)
      .textContent = "Cumulative inference GFLOPs";
    el("text", { class: "title", x: -(plot.top + plot.bottom) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" }, axis)
      .textContent = "ADE20K mIoU (%)";

    // The best prior active model's mIoU, as a reference line under the data.
    const bestPrior = this.#data.best_prior;
    if (bestPrior.miou_pct !== Math.max(...baselines.map((b) => b.miou_pct))) {
      throw new Error("<canvit-frontier>: best_prior is not the highest-mIoU baseline");
    }
    el("line", { class: "reference", x1: plot.left, x2: plot.right, y1: y(bestPrior.miou_pct), y2: y(bestPrior.miou_pct) }, svg);

    // Curves are revealed left to right by a growing clip, once the chart is in view.
    const clip = el("clipPath", { id: "reveal" }, el("defs", {}, svg));
    el("rect", { class: "reveal", x: 0, y: 0, width: this.#shown ? width : 0, height }, clip);
    const drawn = el("g", { "clip-path": "url(#reveal)" }, svg);
    this.#points = [];
    for (const curve of curves) {
      const { label, color } = POLICIES[curve.policy], canvas = CANVASES[curve.canvas_grid];
      const pts = curve.per_timestep;
      if (curve.n_runs >= 2) {
        const upper = pts.map((p) => `${x(p.cum_gflops)},${y(100 * p.ci_hi)}`);
        const lower = pts.map((p) => `${x(p.cum_gflops)},${y(100 * p.ci_lo)}`).reverse();
        el("polygon", { class: "band", points: [...upper, ...lower].join(" "), fill: color }, drawn);
      }
      el("polyline", { class: "curve", points: pts.map((p) => `${x(p.cum_gflops)},${y(100 * p.mean)}`).join(" "),
                       stroke: color, "stroke-dasharray": canvas.dash }, drawn);
      for (const p of pts) {
        this.#points.push({ x: x(p.cum_gflops), y: y(100 * p.mean), color,
          html: `<b>CanViT-B</b>, ${label}, ${canvas.label}<br>${p.t + 1} glimpse${p.t ? "s" : ""}: ${(100 * p.mean).toFixed(1)}% mIoU, ${p.cum_gflops.toFixed(1)} GFLOPs` });
      }
    }

    // Baselines are labeled beside their markers on wide screens; narrow screens rely on the legend. A label goes left
    // where another baseline shares its cost or the edge is near, and above the reference line for the best prior.
    for (const b of baselines) {
      const style = BASELINES[b.name];
      if (!style) throw new Error(`<canvit-frontier>: no style for baseline "${b.name}"`);
      const [px, py] = [x(b.gflops), y(b.miou_pct)];
      marker(style.marker, px, py, 13, style.color, svg);
      this.#points.push({ x: px, y: py, color: style.color,
        html: `<b>${b.name}</b><br>${b.num_glimpses} glimpses: ${b.miou_pct.toFixed(1)}% mIoU, ${b.gflops.toFixed(1)} GFLOPs` });
      if (narrow) continue;
      const isBest = b.name === bestPrior.name && b.num_glimpses === bestPrior.num_glimpses;
      const label = el("text", { class: "baseline-label halo", x: px + 11, y: isBest ? py - 9 : py + 4 }, svg);
      label.textContent = b.name;
      el("tspan", {}, label).textContent = `, ${b.num_glimpses} glimpses`;
      const sharesCost = baselines.some((other) => other !== b && other.gflops === b.gflops);
      if (sharesCost || px + 11 + label.getComputedTextLength() > width) {
        label.setAttribute("x", String(px - 11));
        label.setAttribute("text-anchor", "end");
      }
    }
    el("text", { class: "reference-label halo", x: plot.right, y: y(bestPrior.miou_pct) - 8, "text-anchor": "end" }, svg)
      .textContent = `${narrow ? "Best prior" : "Best prior active model"}: ${bestPrior.miou_pct.toFixed(1)}%`;

    // Each 64² curve is labeled at its end as CanViT-B under its policy.
    for (const curve of curves.filter((c) => c.canvas_grid === 64)) {
      const last = curve.per_timestep.at(-1);
      const { label, color } = POLICIES[curve.policy];
      el("text", { class: "end-label halo", x: x(last.cum_gflops) + 8, y: y(100 * last.mean) + 5, fill: color }, drawn)
        .textContent = narrow ? label : `CanViT-B (${label})`;
    }

    // Even the worse-than-random fine-to-coarse order passes the best prior model, from the exported claim.
    const key = `fine_to_coarse_s${SCENE_SIZE}_c32`;
    const beat = this.#data.claims.beats_prior.find((claim) => claim.key === key);
    if (!beat) throw new Error(`<canvit-frontier>: no beats_prior claim for ${key}`);
    const [cx, cy] = [x(beat.first_beat_gflops), y(beat.first_beat_miou_pct)];
    el("circle", { cx, cy, r: 4.5, fill: "#fff", stroke: POLICIES.fine_to_coarse.color, "stroke-width": 2.2 }, drawn);
    const glimpse = `glimpse ${beat.first_beat_t + 1}`;
    const lines = narrow
      ? ["Even fine-to-coarse (F2C),", "a worse-than-random order,", "beats the best prior", `by ${glimpse}`]
      : ["Even fine-to-coarse (F2C), a worse-than-random order,", `beats the best prior model by ${glimpse}`];
    const [tx, ty] = [cx + 18, y(bestPrior.miou_pct) + 64];
    el("line", { class: "takeaway-line", x1: cx + 3, y1: cy + 6, x2: tx - 4, y2: ty - 16 }, drawn);
    const text = el("text", { class: "takeaway halo", x: tx, y: ty }, drawn);
    lines.forEach((line, i) => { el("tspan", { x: tx, dy: i ? 17 : 0 }, text).textContent = line; });

    // CanViT-B's first glimpse at the 32² canvas, the paper's single-glimpse result, labeled above the curves.
    const first = curves.find((c) => c.policy === "entropy_coarse_to_fine" && c.canvas_grid === 32).per_timestep[0];
    const [fx, fy] = [x(first.cum_gflops), y(100 * first.mean)];
    const labelY = y(Y_MAX - 1.2);
    el("line", { class: "callout-line", x1: fx, y1: fy - 7, x2: fx, y2: labelY + 5 }, svg);
    el("text", { class: "callout", x: fx + 6, y: labelY }, svg).textContent =
      `${(100 * first.mean).toFixed(1)}% from a single glimpse`;

    const focus = el("g", { class: "focus", visibility: "hidden" }, svg);
    const focusDot = el("circle", { r: 5 }, focus);
    const hit = el("rect", { x: plot.left, y: plot.top, width: plot.right - plot.left, height: plot.bottom - plot.top, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const [mx, my] = [event.clientX - box.left, event.clientY - box.top];
      const nearest = this.#points.reduce((a, b) => (Math.hypot(b.x - mx, b.y - my) < Math.hypot(a.x - mx, a.y - my) ? b : a));
      focus.setAttribute("visibility", "visible");
      focusDot.setAttribute("cx", nearest.x);
      focusDot.setAttribute("cy", nearest.y);
      focusDot.setAttribute("stroke", nearest.color);
      this.tooltip.innerHTML = nearest.html;
      this.tooltip.style.left = `${Math.min(Math.max(nearest.x, 110), width - 110)}px`;
      this.tooltip.style.top = `${nearest.y}px`;
      this.tooltip.hidden = false;
    });
    hit.addEventListener("pointerleave", () => { focus.setAttribute("visibility", "hidden"); this.tooltip.hidden = true; });
  }
}

customElements.define("canvit-frontier", CanvitFrontier);
