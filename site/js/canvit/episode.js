// <canvit-episode src="BUNDLE [BUNDLE…]" [readout="canvas|entropy|labels|correct"] [autoplay]>: a recorded rollout
// (web bundle) played as the loop CanViT runs: the viewpoint moves on the scene, its crop becomes the glimpse,
// the glimpse goes into CanViT, which reads its canvas and writes the glimpse into it. Under the canvas, the
// pixel accuracy of the segmentation decoded from it. The first glimpses are slow and later ones faster; the
// last state holds until Replay. Several bundles in `src` become scene tabs labeled with their titles.

import { loadBundle } from "./bundle.js";
import { CORRECTNESS, colormapGradient, correctnessImage, layerImage, layerSpec } from "./layers.js";

const READOUTS = {
  canvas: { label: "Features", image: (bundle, t) => layerImage(bundle, t, "canvas") },
  entropy: { label: "Uncertainty", image: (bundle, t) => layerImage(bundle, t, "entropy") },
  labels: { label: "Segmentation", image: (bundle, t) => layerImage(bundle, t, "labels") },
  correct: { label: "Correctness", image: correctnessImage },
};
const FIRST_STEP_MS = 1600;
const STEP_RATIO = 0.74;
const SHORTEST_STEP_MS = 170;
// Within a step, as fractions of its duration: [start, end] of each movement.
const PHASES = { move: [0, 0.3], crop: [0.24, 0.42], input: [0.42, 0.6], read: [0.52, 0.7], write: [0.7, 0.9] };

const template = document.createElement("template");
template.innerHTML = `
<style>
  [hidden] { display: none !important; }
  :host { display: block; container-type: inline-size; color: var(--canvit-ink, #0f172a);
          font: 15px/1.4 var(--canvit-sans, system-ui, sans-serif);
          --glimpse: var(--canvit-glimpse, #2d6cdf); --canvas: var(--canvit-canvas, #e0483e); }
  .flow { display: grid; align-items: stretch; gap: 0 6px;
          grid-template-columns: minmax(0, 1fr) 58px minmax(0, .42fr) 58px minmax(0, .34fr) 58px minmax(0, 1fr); }
  .column { display: flex; flex-direction: column; min-width: 0; }
  .label { height: 28px; font-size: 15px; font-weight: 700; }
  .label.glimpse { color: var(--glimpse); }
  .label.canvas { color: var(--canvas); }
  .body { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; }
  canvas { display: block; width: 100%; aspect-ratio: 1; border-radius: 10px; background: var(--canvit-placeholder, #f1f5f9); }
  .glimpse canvas { box-shadow: 0 0 0 3px var(--glimpse); }
  .canvas canvas { box-shadow: 0 0 0 3px var(--canvas); image-rendering: pixelated; }
  .model { width: 100%; aspect-ratio: 1 / 1.25; border-radius: 14px; display: grid; place-content: center; gap: 4px;
           text-align: center; background: var(--canvit-ink, #0f172a); color: #fff; transition: box-shadow .12s; }
  .model b { font-size: 17px; font-weight: 750; letter-spacing: -.01em; }
  .model small { font: 13.5px ui-monospace, "JetBrains Mono", monospace; opacity: .85; }
  .model.input { box-shadow: 0 0 0 4px color-mix(in srgb, var(--glimpse) 55%, transparent); }
  .model.write { box-shadow: 0 0 0 4px color-mix(in srgb, var(--canvas) 55%, transparent); }
  svg { width: 58px; height: 22px; overflow: visible; }
  svg line { stroke-width: 2.5; stroke-linecap: round; }
  svg .head { stroke: none; }
  svg .packet { stroke: none; opacity: 0; }
  .blue line { stroke: var(--glimpse); } .blue .head, .blue .packet { fill: var(--glimpse); }
  .red line { stroke: var(--canvas); } .red .head, .red .packet { fill: var(--canvas); }
  .read { transform: scaleX(-1); }
  .tag { font-size: 13px; color: var(--canvit-muted, #475569); margin-top: -6px; }
  .bar { display: flex; align-items: center; justify-content: space-between; gap: 12px 20px; flex-wrap: wrap; margin-top: 20px; }
  .group { display: inline-flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .choice { display: inline-flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--canvit-placeholder, #f1f5f9); }
  button { font: inherit; font-size: 14.5px; font-weight: 550; border: 0; background: transparent; cursor: pointer;
           color: var(--canvit-muted, #475569); padding: 6px 12px; border-radius: 8px; }
  button:hover { color: var(--canvit-ink, #0f172a); }
  button:focus-visible { outline: 2px solid var(--glimpse); outline-offset: 2px; }
  .choice button[aria-pressed="true"] { background: #fff; color: var(--canvit-ink, #0f172a);
           box-shadow: 0 1px 2px rgb(15 23 42 / .10), 0 0 0 1px rgb(15 23 42 / .06); }
  .play { width: 34px; height: 34px; padding: 0; border-radius: 50%; display: grid; place-items: center;
          background: var(--canvit-ink, #0f172a); color: #fff; }
  .play:hover { color: #fff; opacity: .85; }
  .play svg { width: 14px; height: 14px; fill: currentColor; }
  .with-colorbar { display: flex; gap: 8px; width: 100%; }
  .with-colorbar canvas { flex: 1; min-width: 0; }
  .colorbar { display: flex; flex-direction: column; align-items: center; gap: 4px; white-space: nowrap;
              font: 12.5px/1 ui-monospace, "JetBrains Mono", monospace; color: var(--canvit-muted, #475569); }
  .colorbar-bar { flex: 1; width: 12px; border-radius: 3px; }
  /* The colorbar and the correctness legend share the slot beside the canvas, which keeps the width of the wider,
     so switching readouts moves nothing. */
  .side { display: grid; }
  .side > * { grid-area: 1 / 1; }
  .side > .off { visibility: hidden; }
  .meter { grid-column: 7; margin-top: 12px; font-size: 14px; color: var(--canvit-muted, #475569); }
  .meter-head { display: flex; align-items: baseline; gap: 8px; }
  .meter-value { color: var(--canvit-ink, #0f172a); font-size: 17px; font-weight: 750; font-variant-numeric: tabular-nums; }
  .meter-gain { color: #15803d; font-weight: 650; font-variant-numeric: tabular-nums; }
  .meter-track { position: relative; height: 8px; margin-top: 6px; border-radius: 4px; background: var(--canvit-placeholder, #f1f5f9); }
  .meter-base, .meter-gained { position: absolute; inset: 0 auto 0 0; transition: width .18s ease-out, left .18s ease-out; }
  .meter-base { background: #475569; border-radius: 4px 0 0 4px; }
  .meter-gained { background: #16a34a; }
  .meter-start { position: absolute; top: -3px; bottom: -3px; width: 2px; margin-left: -1px; background: var(--canvit-ink, #0f172a); }
  .legend { display: flex; flex-direction: column; justify-content: center; gap: 8px; font-size: 13.5px;
            color: var(--canvit-muted, #475569); white-space: nowrap; }
  .legend i { display: inline-block; width: 11px; height: 11px; margin-right: 6px; border-radius: 3px; vertical-align: -1px; }
  .error { padding: 14px; border: 1px solid #dc2626; border-radius: 10px; color: #b91c1c; font: 13px/1.5 ui-monospace, monospace; }
  @container (max-width: 760px) {
    .flow { grid-template-columns: minmax(0, 1fr); gap: 4px; justify-items: center; }
    .column { width: 100%; align-items: center; }
    .column > .body { width: 100%; }
    .column.scene, .column.canvas { max-width: 420px; }
    .column.glimpse { max-width: 200px; }
    .column.model-column { max-width: 170px; }
    .model { aspect-ratio: 2 / 1; }
    .label { text-align: center; }
    .arrow .label { display: none; }
    .arrow .body { flex-direction: row; gap: 18px; }
    .arrow svg { transform: rotate(90deg); margin: 16px 0; }
    .arrow svg.read { transform: rotate(90deg) scaleX(-1); }
    .tag { display: none; }
    .meter { grid-column: 1; width: 100%; max-width: 420px; }
  }
  @container (max-width: 480px) {
    .readouts { display: grid; grid-template-columns: 1fr 1fr; width: 100%; }
  }
</style>
<div class="flow">
  <div class="column scene"><div class="label">Scene</div><div class="body"><canvas class="scene-view"></canvas></div></div>
  <div class="column arrow"><div class="label"></div><div class="body">${arrow("crop", "blue")}<span class="tag">crop</span></div></div>
  <div class="column glimpse"><div class="label glimpse">Glimpse</div><div class="body"><canvas class="glimpse-view"></canvas></div></div>
  <div class="column arrow"><div class="label"></div><div class="body">${arrow("input", "blue")}</div></div>
  <div class="column model-column"><div class="label"></div><div class="body"><div class="model"><b>CanViT</b><small class="step"></small></div></div></div>
  <div class="column arrow"><div class="label"></div><div class="body">
    ${arrow("write", "red")}<span class="tag">write</span>${arrow("read", "red read")}<span class="tag">read</span></div></div>
  <div class="column canvas"><div class="label canvas">Canvas</div><div class="body"><div class="with-colorbar"><canvas class="canvas-view"></canvas><div class="side"><div class="colorbar" aria-hidden="true"><span class="colorbar-max"></span><div class="colorbar-bar"></div><span>0</span></div><div class="legend"></div></div></div></div></div>
  <div class="meter" hidden>
    <div class="meter-head"><span title="Share of annotated pixels whose class, decoded from the canvas, is right">Pixel accuracy</span><b class="meter-value"></b><span class="meter-gain" title="Change since the first glimpse"></span></div>
    <div class="meter-track"><span class="meter-base"></span><span class="meter-gained"></span><span class="meter-start"></span></div>
  </div>
</div>
<div class="bar">
  <div class="group"><button class="play" aria-label="Play"></button><div class="choice scenes" role="group" aria-label="Scene"></div></div>
  <div class="choice readouts" role="group" aria-label="Canvas readout"></div>
</div>`;

function arrow(name, classes) {
  return `<svg class="${classes}" data-arrow="${name}" viewBox="0 0 58 22" aria-hidden="true">
    <line x1="4" y1="11" x2="46" y2="11"/><path class="head" d="M44 5 L55 11 L44 17 Z"/>
    <circle class="packet" cx="4" cy="11" r="4.5"/></svg>`;
}

const ICONS = {
  play: `<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>`,
  pause: `<svg viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>`,
  replay: `<svg viewBox="0 0 16 16"><path d="M8 2.5a5.5 5.5 0 1 1-5.2 3.7l1.4.5A4 4 0 1 0 8 4v2.2L4.6 3.2 8 .3z"/></svg>`,
};

const easeInOut = (x) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
const progress = (u, [start, end]) => Math.min(1, Math.max(0, (u - start) / (end - start)));

/** Step durations: slow at first, each shorter than the last by STEP_RATIO, down to SHORTEST_STEP_MS. */
function schedule(count) {
  const durations = Array.from({ length: count }, (_, t) => Math.max(SHORTEST_STEP_MS, FIRST_STEP_MS * STEP_RATIO ** t));
  const starts = durations.map((_, t) => durations.slice(0, t).reduce((a, b) => a + b, 0));
  return { durations, starts, total: starts[count - 1] + durations[count - 1] };
}

function sizeToDisplay(canvas) {
  const width = Math.round(canvas.clientWidth * (window.devicePixelRatio || 1));
  if (width > 0 && canvas.width !== width) canvas.width = canvas.height = width;
  return canvas.width;
}

class CanvitEpisode extends HTMLElement {
  static observedAttributes = ["src", "readout"];

  #bundle = null;
  #scene = null;
  #schedule = null;
  #source = null;
  #time = 0;
  #playing = false;
  #pausedByUser = false;
  #frame = 0;
  #lastTime = null;
  #colors = null;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(template.content.cloneNode(true));
    const $ = (selector) => this.shadowRoot.querySelector(selector);
    this.$ = { scene: $(".scene-view"), glimpse: $(".glimpse-view"), canvas: $(".canvas-view"), model: $(".model"),
               step: $(".step"), play: $(".play"), scenes: $(".scenes"), readouts: $(".readouts"),
               meter: $(".meter"), value: $(".meter-value"), gain: $(".meter-gain"), base: $(".meter-base"),
               gained: $(".meter-gained"), start: $(".meter-start"), legend: $(".legend"),
               colorbar: $(".colorbar"), colorbarMax: $(".colorbar-max"), colorbarBar: $(".colorbar-bar"),
               arrows: Object.fromEntries([...this.shadowRoot.querySelectorAll("[data-arrow]")].map((svg) =>
                 [svg.dataset.arrow, svg.querySelector(".packet")])) };
    this.$.play.addEventListener("click", () => {
      if (this.#ended()) this.#restart();
      else {
        this.#pausedByUser = this.#playing;
        this.#setPlaying(!this.#playing);
      }
    });
    for (const [readout, { label }] of Object.entries(READOUTS)) {
      const button = Object.assign(document.createElement("button"), { textContent: label });
      button.dataset.readout = readout;
      button.addEventListener("click", () => this.setAttribute("readout", readout));
      this.$.readouts.append(button);
    }
    new ResizeObserver(() => this.#draw()).observe(this.$.scene);
    new IntersectionObserver(([entry]) => this.#setPlaying(entry.isIntersecting && this.#wantsAutoplay()),
                             { threshold: 0.25 }).observe(this);
  }

  get readout() {
    const readout = this.getAttribute("readout") ?? "canvas";
    if (!(readout in READOUTS)) throw new Error(`<canvit-episode>: unknown readout "${readout}"`);
    return readout;
  }

  attributeChangedCallback(name) {
    if (name === "src") this.#showSources();
    else this.#showReadout();
  }

  connectedCallback() {
    this.$.colorbarBar.style.background = colormapGradient(layerSpec("entropy").colormap, { angle: "0deg" });
    this.$.legend.innerHTML = Object.values(CORRECTNESS)
      .map(({ label, rgb }) => `<span><i style="background: rgb(${rgb.join(" ")})"></i>${label}</span>`).join("");
    this.#showReadout();
    this.#renderPlayButton();
  }

  #wantsAutoplay() {
    return this.hasAttribute("autoplay") && !this.#pausedByUser && !matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  #showSources() {
    const sources = (this.getAttribute("src") ?? "").split(/\s+/).filter(Boolean);
    this.$.scenes.hidden = sources.length < 2;
    this.$.scenes.replaceChildren(...sources.map((source) => {
      const button = Object.assign(document.createElement("button"), { textContent: "…" });
      button.dataset.source = source;
      button.addEventListener("click", async () => {
        await this.#select(source);
        this.#restart();
      });
      loadBundle(source).then((bundle) => { button.textContent = bundle.manifest.title; }, () => { button.textContent = source; });
      return button;
    }));
    if (sources.length > 0) this.#select(sources[0]);
  }

  async #select(source) {
    this.#source = source;
    for (const button of this.$.scenes.children) button.setAttribute("aria-pressed", String(button.dataset.source === source));
    try {
      const bundle = await loadBundle(source);
      if (this.#source !== source) return;
      const scene = new Image();
      scene.src = bundle.sceneUrl;
      await scene.decode();
      this.#bundle = bundle;
      this.#scene = scene;
      this.#schedule = schedule(bundle.glimpses.length);
      const annotated = bundle.truth !== null && bundle.glimpses.every((g) => g.pixelAccuracy !== null);
      this.$.meter.hidden = !annotated;
      this.$.colorbarMax.textContent = `log ${bundle.manifest.readout.num_classes}`;
      this.$.colorbar.title = `Entropy of the decoded class distribution, from 0 to its maximum, log ${bundle.manifest.readout.num_classes}`;
      this.$.readouts.querySelector('[data-readout="correct"]').hidden = !annotated;
      this.#time = this.#wantsAutoplay() ? 0 : this.#schedule.total;
      this.#renderPlayButton();
      this.#draw();
    } catch (error) {
      this.shadowRoot.querySelector(".flow").outerHTML = `<div class="error">${error.message}</div>`;
      throw error;
    }
  }

  #showReadout() {
    for (const button of this.$.readouts.children) button.setAttribute("aria-pressed", String(button.dataset.readout === this.readout));
    this.$.legend.classList.toggle("off", this.readout !== "correct");
    this.$.colorbar.classList.toggle("off", this.readout !== "entropy");
    this.#draw();
  }

  #ended() {
    return this.#schedule !== null && this.#time >= this.#schedule.total;
  }

  #restart() {
    this.#time = 0;
    this.#pausedByUser = false;
    this.#draw();
    this.#setPlaying(true);
  }

  #setPlaying(playing) {
    if (playing && this.#ended()) playing = false;
    if (playing === this.#playing) return this.#renderPlayButton();
    this.#playing = playing;
    this.#lastTime = null;
    this.#renderPlayButton();
    if (playing) this.#frame = requestAnimationFrame((time) => this.#tick(time));
    else cancelAnimationFrame(this.#frame);
  }

  #renderPlayButton() {
    const state = this.#playing ? "pause" : this.#ended() ? "replay" : "play";
    this.$.play.innerHTML = ICONS[state];
    this.$.play.setAttribute("aria-label", { pause: "Pause", replay: "Replay", play: "Play" }[state]);
  }

  #tick(time) {
    if (!this.#playing) return;
    if (this.#schedule && this.#lastTime !== null) {
      this.#time = Math.min(this.#schedule.total, this.#time + time - this.#lastTime);
      this.#draw();
      if (this.#ended()) return this.#setPlaying(false);
    }
    this.#lastTime = time;
    this.#frame = requestAnimationFrame((next) => this.#tick(next));
  }

  /** The glimpse under way at the current time and how far along it is, in [0, 1]. */
  #position() {
    const { durations, starts } = this.#schedule;
    let t = starts.findLastIndex((start) => start <= this.#time);
    const u = Math.min(1, (this.#time - starts[t]) / durations[t]);
    return { t, u };
  }

  #draw() {
    if (!this.#bundle) return;
    this.#colors ??= { glimpse: getComputedStyle(this).getPropertyValue("--glimpse").trim(),
                       canvas: getComputedStyle(this).getPropertyValue("--canvas").trim() };
    const { t, u } = this.#position();
    this.#drawScene(t, u);
    const glimpseShown = u >= PHASES.crop[1] ? t : t - 1, canvasShown = u >= PHASES.write[1] ? t : t - 1;
    this.#drawImage(this.$.glimpse, glimpseShown >= 0 ? layerImage(this.#bundle, glimpseShown, "crop") : null, true);
    this.#drawImage(this.$.canvas, canvasShown >= 0 ? READOUTS[this.readout].image(this.#bundle, canvasShown) : null, false);
    if (!this.$.meter.hidden) this.#drawMeter(canvasShown);
    for (const [name, packet] of Object.entries(this.$.arrows)) {
      const p = name === "read" && t === 0 ? 0 : progress(u, PHASES[name]);
      packet.setAttribute("cx", String(4 + 46 * p));
      packet.style.opacity = p > 0 && p < 1 ? "1" : "0";
    }
    const active = (phase) => u >= PHASES[phase][0] && u < PHASES[phase][1];
    this.$.model.classList.toggle("input", active("input"));
    this.$.model.classList.toggle("write", active("write"));
    this.$.step.textContent = `t = ${t}`;
  }

  #drawScene(t, u) {
    const canvas = this.$.scene;
    const size = sizeToDisplay(canvas);
    const context = canvas.getContext("2d");
    context.drawImage(this.#scene, 0, 0, size, size);
    const boxes = this.#bundle.glimpses.map((g) => g.box);
    const from = boxes[Math.max(0, t - 1)], to = boxes[t];
    const k = t === 0 ? 1 : easeInOut(progress(u, PHASES.move));
    const box = Object.fromEntries(["top", "left", "size"].map((key) => [key, from[key] + k * (to[key] - from[key])]));
    const [x, y, side] = [box.left * size, box.top * size, box.size * size];
    context.fillStyle = "rgb(15 23 42 / .4)";
    context.beginPath();
    context.rect(0, 0, size, size);
    context.rect(x, y, side, side);
    context.fill("evenodd");
    const line = Math.max(2.5, size / 150);
    context.globalAlpha = 0.28;
    context.lineWidth = line / 2.5;
    context.strokeStyle = "#fff";
    for (const past of boxes.slice(0, t)) context.strokeRect(past.left * size, past.top * size, past.size * size, past.size * size);
    context.globalAlpha = 1;
    context.lineWidth = line;
    context.strokeStyle = this.#colors.glimpse;
    context.strokeRect(x + line / 2, y + line / 2, side - line, side - line);
  }

  #drawImage(canvas, image, smooth) {
    const size = sizeToDisplay(canvas);
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, size, size);
    if (image === null) return;
    context.imageSmoothingEnabled = smooth;
    context.imageSmoothingQuality = "high";
    context.drawImage(image, 0, 0, size, size);
  }

  /** Pixel accuracy after glimpse t, on a fixed 0–100% bar: the part gained since the first glimpse in green. */
  #drawMeter(t) {
    const percent = (fraction) => `${(100 * fraction).toFixed(1)}%`;
    const first = this.#bundle.glimpses[0].pixelAccuracy;
    const now = t >= 0 ? this.#bundle.glimpses[t].pixelAccuracy : 0;
    this.$.value.textContent = t >= 0 ? percent(now) : "–";
    const gain = 100 * (now - first);
    this.$.gain.textContent = t >= 1 ? `${gain < 0 ? "−" : "+"}${Math.abs(gain).toFixed(1)}` : "";
    this.$.base.style.width = percent(Math.min(now, first));
    this.$.gained.style.left = percent(first);
    this.$.gained.style.width = percent(Math.max(0, now - first));
    this.$.start.style.left = percent(first);
    this.$.start.hidden = t < 0;
  }
}

customElements.define("canvit-episode", CanvitEpisode);
