// <canvit-policy-curves src task="ade20k|in1k" canvas-grid y-range="LO HI" [dashed-src] [dashed-canvas-grid] [legend]
// [added-src] [glimpses] [x-scale="linear|log"] [policies] [baselines] [t]>:
// CanViT-B's accuracy under the paper's viewing policies (all those the task evaluates, or the ids listed in
// policies), glimpse by glimpse, over a whole validation set (mean over runs), from the paper's exports: ADE20K mIoU
// (ade20k_seg.json, policy_curves) or ImageNet-1k top-1 (in1k_clf_frozen.json or in1k_clf_finetuned.json, configs).
// src at canvas-grid gives the solid curves, named at their heads. dashed-src or dashed-canvas-grid (each defaulting to
// its solid counterpart) adds the same policies dashed beside them, as the paper's Figure 3 does for a 32² canvas
// beside a 64² one and for frozen weights beside fine-tuned ones; legend names the two, "solid|dashed". y-range fixes the
// vertical axis in percent, so charts of the two tasks can share a span of points; lines are clipped to it, so a narrow
// range shows the end of the rollouts with the low starts entering from below. Lines are drawn up to glimpse t (all
// without t), each in its policy's color from the paper's figures; a slide drives t to grow them in step with recorded
// rollouts. glimpses limits the axis to the first glimpses. added-src adds a curve that is not in the paper's exports
// read here ({label, per_timestep: [{t, mean, ci_lo, ci_hi}]}, fractions, as the exports give them), thicker, in policy
// teal, with its confidence interval as a band, in the shadow part "added" so that a slide can reveal it. baselines
// lists prior models of src's baselines, comma-separated by name ("AME (SETR), AdaGlimpse"); each one's best result is
// a marker at the glimpse count it was measured at, named beside it, drawn when inside the axes. The chart's height is
// its width times --canvit-chart-aspect (default 0.78). show="dashed-only" draws only the dashed set, still dashed, its
// policies named at its heads (a slide's step before both, so no curve changes style between steps); the dashed curves
// are the shadow part "dashed", so that a slide can set them back once both sets show;
// --canvit-chart-tick, --canvit-chart-title and --canvit-chart-label set the text sizes (14, 15 and 16 px by default).

import { POLICIES } from "./policies.js";

const SCENE_SIZE = 512;
const LABEL_GAP = 17;
// The exports by task: where their curves are, the axis title, and the policies each evaluates (EG-C2F needs the
// segmentation's per-position entropy, so ImageNet-1k has none).
const TASKS = {
  ade20k: {
    curves: (data) => data.policy_curves,
    baselines: (data) => data.baselines.map((b) => ({ name: b.name, value: b.miou_pct, glimpses: b.num_glimpses })),
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
  .axis-line { stroke: var(--canvit-ink, #0f172a); stroke-width: 1.5; }
  .axis text { fill: var(--canvit-muted, #475569); font-size: var(--canvit-chart-tick, 14px); font-variant-numeric: tabular-nums; }
  .axis .title { fill: var(--canvit-ink, #0f172a); font-size: var(--canvit-chart-title, 15px); font-weight: 650; font-variant-numeric: normal; }
  .curve { fill: none; stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }
  .curve.dashed { stroke-width: 2.6; stroke-dasharray: 7 5; }
  .legend text { fill: var(--canvit-muted, #475569); font-size: var(--canvit-chart-tick, 14px); font-weight: 600; }
  .legend line { stroke: var(--canvit-muted, #475569); stroke-width: 2.6; }
  .legend .dashed { stroke-dasharray: 7 5; }
  .head { stroke: var(--canvit-surface, #fff); stroke-width: 2; }
  .baseline { fill: var(--canvit-surface, #fff); stroke: var(--canvit-ink, #0f172a); stroke-width: 2.5; }
  .added .curve { stroke: var(--canvit-policy, #0d9488); stroke-width: 5.5; }
  .added .band { fill: var(--canvit-policy, #0d9488); opacity: .2; }
  .baseline-label { fill: var(--canvit-ink, #0f172a); font-size: 15px; font-weight: 700; paint-order: stroke; stroke: var(--canvit-surface, #fff); stroke-width: 4px; }
  .label { font-size: var(--canvit-chart-label, 16px); font-weight: 800; paint-order: stroke; stroke: var(--canvit-surface, #fff); stroke-width: 5px; }
</style>
<svg role="img"></svg>`;

const NS = "http://www.w3.org/2000/svg";
function el(name, attributes, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent.append(node);
  return node;
}

/** Label heights for targets given top to bottom: each at its target, except that labels closer than gap form a run,
 * gap apart and centered on its targets' mean, runs merging until none come closer than gap. */
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

function required(element, name) {
  const value = element.getAttribute(name);
  if (value === null) throw new Error(`<canvit-policy-curves>: missing ${name}`);
  return value;
}

class CanvitPolicyCurves extends HTMLElement {
  static observedAttributes = ["show", "src", "dashed-src", "added-src", "glimpses", "task", "canvas-grid", "dashed-canvas-grid", "legend", "policies", "baselines", "y-range", "x-scale", "t"];
  #curves = null;
  #baselines = [];
  #legend = null;
  #added = null;
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
    if (["t", "y-range", "x-scale", "glimpses", "show"].includes(name)) return this.#render();
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
    const addedSrc = this.getAttribute("added-src");
    const [data, dashedData, added] = await Promise.all([src, withDashed ? dashedSrc : null, addedSrc].map(async (url) => {
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
    if (added && !(typeof added.label === "string" && added.per_timestep?.every((p) => [p.t, p.mean, p.ci_lo, p.ci_hi].every(Number.isFinite)))) {
      throw new Error(`<canvit-policy-curves>: ${addedSrc} is not {label, per_timestep: [{t, mean, ci_lo, ci_hi}]}`);
    }
    this.#added = added;
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
    // show="dashed-only" draws the second set alone, dashed as when both are shown, as a slide's first step before both.
    const dashedOnly = this.getAttribute("show") === "dashed-only";
    if (dashedOnly && !this.#legend) throw new Error('<canvit-policy-curves>: show="dashed-only" without dashed curves');
    const curves = dashedOnly ? this.#curves.filter((c) => c.style === "dashed") : this.#curves;
    const [low, high] = this.#yRange;
    const steps = Math.min(Math.max(...curves.map((c) => c.per_timestep.length)), Number(this.getAttribute("glimpses") ?? Infinity));
    const shown = Math.min(steps, this.hasAttribute("t") ? Number(this.getAttribute("t")) + 1 : steps);
    const aspect = Number(getComputedStyle(this).getPropertyValue("--canvit-chart-aspect") || 0.78);
    const height = Math.round(width * aspect);
    const plot = { left: 72, right: width - 90, top: 10, bottom: height - 52 };
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
    const tick = high - low <= 8 ? 1 : high - low <= 15 ? 2 : high - low <= 30 ? 5 : 10;
    for (let value = Math.ceil(low / tick) * tick; value <= high; value += tick) {
      el("line", { x1: plot.left, x2: plot.right, y1: y(value), y2: y(value) }, grid);
      el("text", { x: plot.left - 10, y: y(value) + 5, "text-anchor": "end" }, axis).textContent = String(value);
    }
    const ticks = this.#xScale === "log" ? [1, 2, 5, 10, steps] : steps <= 10 ? Array.from({ length: steps }, (_, i) => i + 1) : [1, 5, 10, 15, steps];
    for (const glimpse of ticks.filter((g) => g <= steps)) {
      el("text", { x: x(glimpse - 1), y: plot.bottom + 22, "text-anchor": "middle" }, axis).textContent = String(glimpse);
    }
    el("line", { class: "axis-line", x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom }, axis);
    el("line", { class: "axis-line", x1: plot.left, x2: plot.left, y1: plot.top, y2: plot.bottom }, axis);
    el("text", { class: "title", x: (plot.left + plot.right) / 2, y: height - 6, "text-anchor": "middle" }, axis).textContent = "Glimpses";
    el("text", { class: "title", x: -(plot.top + plot.bottom) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" }, axis)
      .textContent = this.#task.title;

    const clip = el("clipPath", { id: "plot" }, el("defs", {}, svg));
    el("rect", { x: plot.left, y: plot.top - 6, width: plot.right - plot.left + 8, height: plot.bottom - plot.top + 6 }, clip);
    for (const { name, value, glimpses } of this.#baselines.filter((b) => b.value >= low && b.value <= high && b.glimpses <= steps)) {
      const [bx, by] = [x(glimpses - 1), y(value)];
      el("rect", { class: "baseline", x: bx - 6, y: by - 6, width: 12, height: 12, transform: `rotate(45 ${bx} ${by})` }, svg);
      el("text", { class: "baseline-label", x: bx + 14, y: by + 5 }, svg).textContent = `${name} ${value.toFixed(1)}`;
    }
    const lines = el("g", { "clip-path": "url(#plot)" }, svg);
    // Dashed curves are unnamed beside the solid ones; the legend says which is which. It names the sets drawn: both,
    // or the dashed set alone.
    if (this.#legend) {
      const legend = el("g", { class: "legend" }, svg);
      const entries = dashedOnly ? [["dashed", this.#legend[1]]] : [["solid", this.#legend[0]], ["dashed", this.#legend[1]]];
      entries.forEach(([style, text], i) => {
        const ly = plot.top + 12 + 22 * i;
        el("line", { class: style, x1: plot.left + 14, x2: plot.left + 44, y1: ly, y2: ly }, legend);
        el("text", { x: plot.left + 52, y: ly + 5 }, legend).textContent = text;
      });
    }
    // Policies rank by their end value (to 0.1 point; a tie goes to the higher mean over the glimpses shown). The
    // higher-ranked curve and head are drawn over the others, and heads are labeled in rank order, at least LABEL_GAP
    // apart, a group of close heads labeled around them (spread), so close lines stay legible.
    const addedGroup = this.#added && el("g", { class: "added", part: "added" }, svg);
    const rank = (curve, extra) => {
      const shownPoints = curve.per_timestep.slice(0, shown);
      const means = shownPoints.map((p) => p.mean);
      return { ...extra, points: shownPoints.map((p) => [x(p.t), y(100 * p.mean)]),
               end: Math.round(1000 * means.at(-1)), mean: means.reduce((a, b) => a + b) / means.length };
    };
    const ranked = [
      ...curves.map((curve) => rank(curve, { ...POLICIES[curve.policy], style: curve.style, group: svg })),
      ...(this.#added ? [rank(this.#added, { label: this.#added.label, color: "var(--canvit-policy, #0d9488)", style: "solid", group: addedGroup })] : []),
    ].sort((a, b) => b.end - a.end || b.mean - a.mean);
    for (const { points, color, style, group } of [...ranked].reverse()) {
      if (group !== svg) continue;
      el("polyline", { class: `curve ${style}`, points: points.map((p) => p.join(",")).join(" "), stroke: color,
                       ...(style === "dashed" ? { part: "dashed" } : {}) }, lines);
    }
    if (this.#added) {
      const band = this.#added.per_timestep.slice(0, shown);
      const edge = (bound) => band.map((p) => [x(p.t), y(100 * p[bound])]);
      el("polygon", { class: "band", points: [...edge("ci_hi"), ...edge("ci_lo").reverse()].map((p) => p.join(",")).join(" ") }, addedGroup);
      el("polyline", { class: "curve", points: band.map((p) => [x(p.t), y(100 * p.mean)].join(",")).join(" ") }, addedGroup);
    }
    // The named set's heads (the solid set's, or the dashed set's when drawn alone); a head below or above the range is
    // not drawn or named yet.
    const named = dashedOnly ? "dashed" : "solid";
    const heads = ranked.filter(({ points, style }) => style === named && points.at(-1)[1] <= plot.bottom && points.at(-1)[1] >= plot.top);
    for (const { points, color, group } of [...heads].reverse()) {
      const [hx, hy] = points.at(-1);
      el("circle", { class: "head", cx: hx, cy: hy, r: 5.5, fill: color }, group);
    }
    const heights = spread(heads.map(({ points }) => points.at(-1)[1] + 5), LABEL_GAP);
    heads.forEach(({ points, label, color, group }, i) => {
      el("text", { class: "label", x: points.at(-1)[0] + 10, y: heights[i], fill: color }, group).textContent = label;
    });
  }
}

customElements.define("canvit-policy-curves", CanvitPolicyCurves);
