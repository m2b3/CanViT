// <canvit-frontier src="ade20k_seg.json" [show="canvit|prior"] [series="..."]>: the paper's accuracy–efficiency frontier on ADE20K
// (Figure 3A), drawn from its exported data: CanViT-B's mIoU against cumulative inference FLOPs, glimpse by glimpse,
// for EG-C2F and F2C at 32² and 64² canvases, and the prior active models as points. Colors, line styles and markers
// follow the paper's figure. show="prior" draws what active vision faced before CanViT instead: the prior active
// models and the passive DINOv3 ViT-B/16 teacher at input resolutions from 128 to 512 px (the export's probe_table,
// the paper's Figure 5C), on the same axes. The chart is drawn at its displayed size, so text stays legible on
// narrow screens; hovering names the nearest point. series, for a talk's step-by-step build, names what to draw:
// policy-canvas keys (f2c-32, f2c-64, egc2f-32, egc2f-64) and prior (the prior active models); each curve is then
// labeled at its end (the legend is then left out), the cost axis fits what is shown, and a change of series first
// glides the axis to its new range, then draws the new curves in. height (px) fixes the chart's height.

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
const PASSIVE = { model: "DINOv3 ViT-B/16", color: "#334155" };
const VIEWS = ["canvit", "prior"];
const SERIES = { "f2c-32": ["fine_to_coarse", 32], "f2c-64": ["fine_to_coarse", 64],
                 "egc2f-32": ["entropy_coarse_to_fine", 32], "egc2f-64": ["entropy_coarse_to_fine", 64] };
const GLIDE_MS = 900;
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
  .arrow { fill: none; stroke: var(--canvit-ink, #0f172a); stroke-width: 1.4; stroke-linecap: round; }
  .arrow-halo { fill: none; stroke: var(--canvit-surface, #fff); stroke-width: 6; stroke-linecap: round; }
  .arrowhead { fill: var(--canvit-ink, #0f172a); }
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
  .focus circle { fill: #fff; stroke-width: 2.5; }
  .reveal { transition: width 1.4s cubic-bezier(.3, .6, .2, 1); }
  .enter { stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw-in 1.2s ease forwards; }
  .enter-fade { opacity: 0; animation: fade-in .6s ease .5s forwards; }
  @keyframes draw-in { to { stroke-dashoffset: 0; } }
  @keyframes fade-in { to { opacity: 1; } }
  .tooltip { position: absolute; pointer-events: none; padding: 7px 10px; border-radius: 8px; background: var(--canvit-ink, #0f172a);
             color: #fff; font-size: 13.5px; line-height: 1.45; white-space: nowrap; transform: translate(-50%, calc(-100% - 12px)); }
  .tooltip b { font-weight: 650; }
  [hidden] { display: none !important; }
  @media (prefers-reduced-motion: reduce) { .reveal { transition: none; } .enter, .enter-fade { animation: none; stroke-dashoffset: 0; opacity: 1; } }
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

// A callout's arrow: a cubic curve from a note to the point it describes, over a halo in the surface color, so it
// reads as annotation where it crosses the data. It leaves the note along `leave` and arrives along `arrive` (unit
// vectors); its last stretch runs straight along `arrive`, which the arrowhead, oriented by the end tangent, follows.
function arrow(from, leave, to, arrive, parent) {
  const reach = 0.45 * Math.hypot(to[0] - from[0], to[1] - from[1]);
  const c1 = [from[0] + reach * leave[0], from[1] + reach * leave[1]];
  const c2 = [to[0] - reach * arrive[0], to[1] - reach * arrive[1]];
  const d = `M${from} C${c1} ${c2} ${to}`;
  el("path", { class: "arrow-halo", d }, parent);
  el("path", { class: "arrow", d, "marker-end": "url(#arrowhead)" }, parent);
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
  static observedAttributes = ["src", "show", "series", "height"];
  #data = null;
  #xMax = null; // the cost axis's current end, which a change of series glides
  #drawnSeries = new Set(); // what the last finished render showed, so a new series can be drawn in
  #glide = 0;
  #points = [];
  #shown = false;
  #width = 0;
  #loadVersion = 0;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(template.content.cloneNode(true));
    this.svg = this.shadowRoot.querySelector(".chart");
    this.tooltip = this.shadowRoot.querySelector(".tooltip");
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

  get view() {
    const view = this.getAttribute("show") ?? "canvit";
    if (!VIEWS.includes(view)) throw new Error(`<canvit-frontier>: show="${view}", expected one of ${VIEWS.join(", ")}`);
    return view;
  }

  /** The series shown: the series attribute's keys, or everything when it is absent. */
  get series() {
    const value = this.getAttribute("series");
    if (value === null) return new Set([...Object.keys(SERIES), "prior"]);
    const keys = value.split(/\s+/).filter(Boolean);
    for (const key of keys) if (!(key in SERIES) && key !== "prior") throw new Error(`<canvit-frontier>: unknown series "${key}"`);
    return new Set(keys);
  }

  get staged() { return this.hasAttribute("series"); }

  #targetXMax() {
    const shown = this.series;
    const curves = this.#curves().filter((c) => shown.has(this.#key(c)));
    const costs = [...curves.flatMap((c) => c.per_timestep.map((p) => p.cum_gflops)),
                   ...(shown.has("prior") ? this.#data.baselines.map((b) => b.gflops) : [])];
    if (!costs.length) throw new Error("<canvit-frontier>: series shows nothing");
    return Math.max(...costs) * 1.04;
  }

  #key(curve) {
    return Object.entries(SERIES).find(([, [policy, grid]]) => policy === curve.policy && grid === curve.canvas_grid)[0];
  }

  /** Glide the cost axis from its current end to the shown series', redrawing what was already shown, then draw it all. */
  #changeSeries() {
    const width = Math.round(this.svg.clientWidth);
    const from = this.#xMax ?? this.#targetXMax(), to = this.#targetXMax(), glide = ++this.#glide;
    const start = performance.now();
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const frame = (now) => {
      if (glide !== this.#glide) return;
      const t = reduced ? 1 : Math.min(1, (now - start) / GLIDE_MS);
      const eased = t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      if (t < 1) {
        this.#render(width, { xMax: from + (to - from) * eased, only: this.#drawnSeries });
        requestAnimationFrame(frame);
      } else this.#render(width, { xMax: to });
    };
    requestAnimationFrame(frame);
  }

  async attributeChangedCallback(name) {
    if (name === "series") {
      if (this.#data) this.#changeSeries();
      return;
    }
    if (name === "show" || name === "height") {
      if (this.#data) this.#render(Math.round(this.svg.clientWidth));
      return;
    }
    const version = ++this.#loadVersion;
    const src = this.getAttribute("src");
    const response = await fetch(new URL(src, document.baseURI));
    if (version !== this.#loadVersion) return;
    if (!response.ok) throw new Error(`<canvit-frontier>: could not fetch ${src}: HTTP ${response.status}`);
    const data = await response.json();
    if (version !== this.#loadVersion) return;
    this.#data = data;
    const curves = this.#curves();
    if (curves.length !== Object.keys(POLICIES).length * Object.keys(CANVASES).length) {
      throw new Error(`<canvit-frontier>: expected every policy at every canvas, found ${curves.length} curves`);
    }
    this.#render(Math.round(this.svg.clientWidth));
  }

  #curves() {
    return this.#data.policy_curves.filter((c) => c.policy in POLICIES && c.scene_size === SCENE_SIZE && c.canvas_grid in CANVASES);
  }

  /** The passive teacher's probe results, cheapest input first. */
  #passive() {
    const rows = this.#data.probe_table.filter((r) => r.model === PASSIVE.model).sort((a, b) => a.gflops - b.gflops);
    if (rows.length < 2) throw new Error(`<canvit-frontier>: no ${PASSIVE.model} rows in probe_table`);
    return rows;
  }

  #legend(view) {
    const markerSwatch = ({ color, marker: shape }) =>
      `<svg viewBox="-7 -7 14 14" class="marker">${shape === "triangle" ? `<polygon points="0,-7 6,4 -6,4" fill="${color}"/>` : `<polygon points="0,-7 7,0 0,7 -7,0" fill="${color}"/>`}</svg>`;
    const prior = `<div class="row"><b>Prior active models</b>${Object.entries(BASELINES).map(([name, style]) => `<span>${markerSwatch(style)}${name}</span>`).join("")}</div>`;
    this.shadowRoot.querySelector(".legend").innerHTML = view === "prior"
      ? `<div class="row"><b>Passive</b><span>${lineSwatch(PASSIVE.color, "")}${PASSIVE.model}, frozen, linear probe, input 128 to 512 px</span></div>${prior}`
      : `<div class="row"><b>CanViT-B</b>${Object.values(POLICIES).map(({ label, name, color }) => `<span>${lineSwatch(color, "")}${label} (${name})</span>`).join("")}` +
        `${Object.values(CANVASES).map(({ label, dash }) => `<span>${lineSwatch("#475569", dash)}${label}</span>`).join("")}</div>${prior}`;
  }

  #render(width, { xMax: fixedXMax = null, only = null } = {}) {
    // A source change while hidden must invalidate the old visible width.
    this.#width = width;
    const series = only ?? this.series, staged = this.staged, entering = only ? new Set() : new Set([...series].filter((k) => !this.#drawnSeries.has(k)));
    const margin = { top: 14, right: staged ? 96 : 12, bottom: 54, left: 58 };
    if (width <= margin.left + margin.right) return;
    const narrow = width < NARROW_PX;
    const height = this.hasAttribute("height") ? Number(this.getAttribute("height"))
      : Math.round(Math.min(460, Math.max(300, width * 0.58)));
    if (!(height > margin.top + margin.bottom)) throw new Error(`<canvit-frontier>: height="${this.getAttribute("height")}"`);
    const svg = this.svg;
    const view = this.view;
    this.#legend(view);
    this.shadowRoot.querySelector(".legend").hidden = staged;
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", String(height));
    svg.setAttribute("aria-label", view === "prior"
      ? "ADE20K mIoU against inference FLOPs: the passive DINOv3 ViT-B/16 lies above and to the left of every prior active model."
      : "ADE20K mIoU against cumulative inference FLOPs: CanViT-B's curves lie above and to the left of every prior active model.");

    const allCurves = this.#curves();
    const curves = allCurves.filter((c) => series.has(this.#key(c)));
    const baselines = series.has("prior") ? this.#data.baselines : [];
    const xMax = fixedXMax ?? (staged ? this.#targetXMax()
      : Math.max(...allCurves.flatMap((c) => c.per_timestep.map((p) => p.cum_gflops)), ...this.#data.baselines.map((b) => b.gflops)) * 1.04);
    this.#xMax = xMax;
    if (!only) this.#drawnSeries = new Set(series);
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
    if (bestPrior.miou_pct !== Math.max(...this.#data.baselines.map((b) => b.miou_pct))) {
      throw new Error("<canvit-frontier>: best_prior is not the highest-mIoU baseline");
    }
    const enterPrior = entering.has("prior") ? " enter-fade" : "";
    if (baselines.length) {
      el("line", { class: `reference${enterPrior}`, x1: plot.left, x2: plot.right, y1: y(bestPrior.miou_pct), y2: y(bestPrior.miou_pct) }, svg);
    }

    // Curves are revealed left to right by a growing clip, once the chart is in view.
    const defs = el("defs", {}, svg);
    const head = el("marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 8, markerHeight: 8,
                                markerUnits: "userSpaceOnUse", orient: "auto" }, defs);
    el("path", { class: "arrowhead", d: "M0,1 L9,5 L0,9 Z" }, head);
    const clip = el("clipPath", { id: "reveal" }, defs);
    el("rect", { class: "reveal", x: 0, y: 0, width: this.#shown || staged ? width : 0, height }, clip);
    const drawn = el("g", { "clip-path": "url(#reveal)" }, svg);
    this.#points = [];
    // Both views share the axes (curves are measured for the x range even when hidden), so they can follow each other.
    if (view === "prior") this.#drawPassive(drawn, x, y);
    else for (const curve of curves) {
      const { label, color } = POLICIES[curve.policy], canvas = CANVASES[curve.canvas_grid];
      const pts = curve.per_timestep;
      const enter = entering.has(this.#key(curve));
      if (curve.n_runs >= 2) {
        const upper = pts.map((p) => `${x(p.cum_gflops)},${y(100 * p.ci_hi)}`);
        const lower = pts.map((p) => `${x(p.cum_gflops)},${y(100 * p.ci_lo)}`).reverse();
        el("polygon", { class: `band${enter ? " enter-fade" : ""}`, points: [...upper, ...lower].join(" "), fill: color }, drawn);
      }
      // A curve drawing itself in is solid while it draws (its dash pattern would fight the draw), dashed once drawn.
      const line = el("polyline", { class: `curve${enter ? " enter" : ""}`, points: pts.map((p) => `${x(p.cum_gflops)},${y(100 * p.mean)}`).join(" "),
                                    stroke: color, "stroke-dasharray": enter ? "" : canvas.dash, pathLength: enter ? 1 : "" }, drawn);
      if (enter) line.addEventListener("animationend", () => {
        line.classList.remove("enter");
        line.removeAttribute("pathLength");
        line.setAttribute("stroke-dasharray", canvas.dash);
      });
      if (staged) {
        const last = pts.at(-1);
        el("text", { class: `end-label halo${enter ? " enter-fade" : ""}`, x: x(last.cum_gflops) + 8, y: y(100 * last.mean) + 5, fill: color }, drawn)
          .textContent = `${label}, ${curve.canvas_grid}²`;
      }
      for (const p of pts) {
        this.#points.push({ x: x(p.cum_gflops), y: y(100 * p.mean), color,
          html: `<b>CanViT-B</b>, ${label}, ${canvas.label}<br>${p.t + 1} glimpse${p.t ? "s" : ""}: ${(100 * p.mean).toFixed(1)}% mIoU, ${p.cum_gflops.toFixed(1)} GFLOPs` });
      }
    }

    // Where the F2C order, worse than random, first beats the best prior model (the exported claim). Its note sits
    // below the point and its arrow rises into it through a corridor that no baseline label may cover.
    const key = `fine_to_coarse_s${SCENE_SIZE}_c32`;
    const beat = this.#data.claims.beats_prior.find((claim) => claim.key === key);
    if (!beat) throw new Error(`<canvit-frontier>: no beats_prior claim for ${key}`);
    const [cx, cy] = [x(beat.first_beat_gflops), y(beat.first_beat_miou_pct)];
    const [tx, ty] = [cx + 30, y(bestPrior.miou_pct) + 64];
    const corridor = { left: cx - 12, right: tx + 4, top: cy + 2, bottom: ty - 12 }; // the arrow arrives from below
    const inCorridor = (box) => view === "canvit" && !staged &&
      box.x < corridor.right && box.x + box.width > corridor.left && box.y < corridor.bottom && box.y + box.height > corridor.top;

    // Baselines are labeled beside their markers on wide screens; narrow screens rely on the legend. A label goes left
    // where another baseline shares its cost or the edge is near, and above the reference line for the best prior.
    // A label across the callout's corridor puts its glimpse count on a second line, which narrows it clear.
    for (const b of baselines) {
      const style = BASELINES[b.name];
      if (!style) throw new Error(`<canvit-frontier>: no style for baseline "${b.name}"`);
      const [px, py] = [x(b.gflops), y(b.miou_pct)];
      marker(style.marker, px, py, 13, style.color, svg).setAttribute("class", enterPrior.trim());
      this.#points.push({ x: px, y: py, color: style.color,
        html: `<b>${b.name}</b><br>${b.num_glimpses} glimpses: ${b.miou_pct.toFixed(1)}% mIoU, ${b.gflops.toFixed(1)} GFLOPs` });
      if (narrow) continue;
      const isBest = b.name === bestPrior.name && b.num_glimpses === bestPrior.num_glimpses;
      const label = el("text", { class: `baseline-label halo${enterPrior}`, x: px + 11, y: isBest ? py - 9 : py + 4 }, svg);
      label.textContent = b.name;
      const count = el("tspan", {}, label);
      count.textContent = `, ${b.num_glimpses} glimpses`;
      const sharesCost = baselines.some((other) => other !== b && other.gflops === b.gflops);
      if (sharesCost || px + 11 + label.getComputedTextLength() > width) {
        label.setAttribute("x", String(px - 11));
        label.setAttribute("text-anchor", "end");
      }
      if (inCorridor(label.getBBox())) {
        count.textContent = `${b.num_glimpses} glimpses`;
        count.setAttribute("x", label.getAttribute("x"));
        count.setAttribute("dy", "1.15em");
      }
    }
    if (baselines.length) {
      el("text", { class: `reference-label halo${enterPrior}`, x: plot.right, y: y(bestPrior.miou_pct) - 8, "text-anchor": "end" }, svg)
        .textContent = `${narrow ? "Best prior" : "Best prior active model"}: ${bestPrior.miou_pct.toFixed(1)}%`;
    }
    // The project page's annotations go with its full view; a staged build says them aloud instead.
    if (view === "canvit" && !staged) this.#annotateCanvit({ svg, drawn, curves, x, y, narrow, beat: [cx, cy], note: [tx, ty] });
    this.#hover(svg, plot, width);
  }

  /** The passive teacher's probe results as a curve over input resolutions, labeled at both ends. */
  #drawPassive(parent, x, y) {
    const rows = this.#passive();
    el("polyline", { class: "curve", points: rows.map((r) => `${x(r.gflops)},${y(r.miou_pct)}`).join(" "), stroke: PASSIVE.color }, parent);
    for (const r of rows) {
      el("circle", { cx: x(r.gflops), cy: y(r.miou_pct), r: 3.5, fill: PASSIVE.color }, parent);
      this.#points.push({ x: x(r.gflops), y: y(r.miou_pct), color: PASSIVE.color,
        html: `<b>${PASSIVE.model}</b>, passive<br>${r.input_px} px input: ${r.miou_pct.toFixed(1)}% mIoU, ${r.gflops.toFixed(1)} GFLOPs` });
    }
    const [first, last] = [rows[0], rows.at(-1)];
    el("text", { class: "baseline-label halo", x: x(first.gflops) + 8, y: y(first.miou_pct) + 18 }, parent).textContent = `${first.input_px} px`;
    el("text", { class: "end-label halo", x: x(last.gflops) + 9, y: y(last.miou_pct) + 5, fill: PASSIVE.color }, parent)
      .textContent = `${PASSIVE.model}, ${last.input_px} px: ${last.miou_pct.toFixed(1)}%`;
  }

  #annotateCanvit({ svg, drawn, curves, x, y, narrow, beat: [cx, cy], note: [tx, ty] }) {
    // Each 64² curve is labeled at its end as CanViT-B under its policy.
    for (const curve of curves.filter((c) => c.canvas_grid === 64)) {
      const last = curve.per_timestep.at(-1);
      const { label, color } = POLICIES[curve.policy];
      el("text", { class: "end-label halo", x: x(last.cum_gflops) + 8, y: y(100 * last.mean) + 5, fill: color }, drawn)
        .textContent = narrow ? label : `CanViT-B (${label})`;
    }

    // The F2C callout: its point, and the paper's sentence about it.
    el("circle", { cx, cy, r: 4.5, fill: "#fff", stroke: POLICIES.fine_to_coarse.color, "stroke-width": 2.2 }, drawn);
    const lines = narrow
      ? ["CanViT's advantage holds", "even with the F2C policy's", "worse-than-random", "viewing order"]
      : ["CanViT's advantage holds even with the F2C policy's", "worse-than-random viewing order"];
    // The arrow rises from the note and reaches the point from below right.
    const diagonal = Math.SQRT1_2;
    arrow([tx - 8, ty - 5], [0, -1], [cx + 6, cy + 6], [-diagonal, -diagonal], drawn);
    const text = el("text", { class: "takeaway halo", x: tx, y: ty }, drawn);
    lines.forEach((line, i) => { el("tspan", { x: tx, dy: i ? 17 : 0 }, text).textContent = line; });

    // CanViT-B's first glimpse at the 32² canvas, the paper's single-glimpse result, labeled above the curves.
    const first = curves.find((c) => c.policy === "entropy_coarse_to_fine" && c.canvas_grid === 32).per_timestep[0];
    const [fx, fy] = [x(first.cum_gflops), y(100 * first.mean)];
    const labelY = y(Y_MAX - 1.2);
    // The arrow leaves the note leftward, then turns down into the point from above, clear of the curves.
    arrow([fx + 26, labelY - 5], [-1, 0], [fx, fy - 7], [0, 1], svg);
    el("text", { class: "callout", x: fx + 32, y: labelY }, svg).textContent =
      `${(100 * first.mean).toFixed(1)}% mIoU in a single glimpse`;
  }

  /** Hovering the plot names the nearest point. */
  #hover(svg, plot, width) {
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
