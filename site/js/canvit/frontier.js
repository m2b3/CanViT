// <canvit-frontier src="ade20k_seg.json">: the paper's accuracy–efficiency frontier on ADE20K (Figure 3A), drawn
// from its exported data: CanViT-B's mIoU against cumulative inference FLOPs, glimpse by glimpse, for EG-C2F and
// F2C at 32² and 64² canvases, and the prior active models as points. Colors, line styles and markers follow the
// paper's figure. The chart is drawn at its displayed size, so text stays legible on narrow screens; hovering
// names the nearest point.

const POLICIES = {
  entropy_coarse_to_fine: { label: "EG-C2F", color: "#2ca02c" },
  fine_to_coarse: { label: "F2C", color: "#d62728" },
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
  :host { display: block; color: var(--canvit-ink, #0f172a); font: 13px/1.4 var(--canvit-sans, system-ui, sans-serif); }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 20px; margin-bottom: 10px; font-size: 12.5px; }
  .legend span { display: inline-flex; align-items: center; gap: 8px; }
  .legend svg { width: 24px; height: 6px; }
  .frame { position: relative; }
  .chart { display: block; width: 100%; overflow: visible; }
  .grid line { stroke: #e2e8f0; }
  .axis text { fill: var(--canvit-muted, #64748b); font-size: 12px; font-variant-numeric: tabular-nums; }
  .axis .title { fill: var(--canvit-ink, #0f172a); font-size: 13px; font-weight: 600; }
  .curve { fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }
  .band { opacity: .15; }
  .baseline-label { fill: var(--canvit-ink, #0f172a); font-size: 12px; }
  .baseline-label tspan { fill: var(--canvit-muted, #64748b); }
  .callout { fill: var(--canvit-ink, #0f172a); font-size: 13px; font-weight: 700; }
  .callout-line { stroke: var(--canvit-ink, #0f172a); stroke-width: 1.2; }
  .focus circle { fill: #fff; stroke-width: 2.5; }
  .reveal { transition: width 1.4s cubic-bezier(.3, .6, .2, 1); }
  .tooltip { position: absolute; pointer-events: none; padding: 7px 10px; border-radius: 8px; background: var(--canvit-ink, #0f172a);
             color: #fff; font-size: 12.5px; line-height: 1.45; white-space: nowrap; transform: translate(-50%, calc(-100% - 12px)); }
  .tooltip b { font-weight: 650; }
  [hidden] { display: none !important; }
  @media (prefers-reduced-motion: reduce) { .reveal { transition: none; } }
</style>
<div class="legend"></div>
<div class="frame"><svg class="chart" role="img"></svg><div class="tooltip" hidden></div></div>`;

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
    this.shadowRoot.querySelector(".legend").innerHTML = [
      ...Object.values(POLICIES).map(({ label, color }) => `<span>${lineSwatch(color, "")}${label}</span>`),
      ...Object.values(CANVASES).map(({ label, dash }) => `<span>${lineSwatch("#64748b", dash)}${label}</span>`),
    ].join("");
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
    const margin = { top: 12, right: 12, bottom: 50, left: 54 };
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
          html: `<b>CanViT-B, ${label}</b>, ${canvas.label}<br>${p.t + 1} glimpse${p.t ? "s" : ""}: ${(100 * p.mean).toFixed(1)}% mIoU, ${p.cum_gflops.toFixed(1)} GFLOPs` });
      }
    }

    for (const b of baselines) {
      const style = BASELINES[b.name];
      if (!style) throw new Error(`<canvit-frontier>: no style for baseline "${b.name}"`);
      const [px, py] = [x(b.gflops), y(b.miou_pct)];
      marker(style.marker, px, py, 13, style.color, svg);
      // Labels go right of the marker, or left where another baseline shares its cost or the edge is near.
      const sharesCost = baselines.some((other) => other !== b && other.gflops === b.gflops);
      const label = el("text", { class: "baseline-label", x: px + 11, y: py + 4 }, svg);
      label.textContent = b.name;
      if (!narrow) el("tspan", {}, label).textContent = `, ${b.num_glimpses} glimpses`;
      if (sharesCost || px + 11 + label.getComputedTextLength() > width) {
        label.setAttribute("x", String(px - 11));
        label.setAttribute("text-anchor", "end");
      }
      this.#points.push({ x: px, y: py, color: style.color,
        html: `<b>${b.name}</b>, ${b.num_glimpses} glimpses<br>${b.miou_pct.toFixed(1)}% mIoU, ${b.gflops.toFixed(1)} GFLOPs` });
    }

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
