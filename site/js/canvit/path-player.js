// <canvit-path-player src="BUNDLE [BUNDLE…]" [readout="canvas|uncertainty|segmentation"] [autoplay]>:
// CanViT looking along a recorded smooth path of viewpoints (path bundles, see path-bundle.js). Three panels:
// the scene with the viewpoint, the glimpse the model receives, and a readout of the canvas carried across
// every glimpse so far. The viewpoint moves at display rate; glimpse and readout change at each recorded
// sample. Several bundles in `src` become scene tabs labeled with their titles.

import { loadPathBundle, pointOn, tileOrigin } from "./path-bundle.js";
import { ADE20K_PALETTE } from "./ade20k.js";
import { COLORMAPS } from "./colormaps.js";

const READOUTS = {
  canvas: {
    label: "Canvas", layer: "canvas", caption: "<b class=\"canvas\">Canvas</b>, carried across every glimpse",
    note: "The canvas in PCA colors: its first three principal components as red, green and blue, with color limits fixed for the whole path.",
  },
  uncertainty: {
    label: "Uncertainty", layer: "entropy", caption: "<b>Uncertainty</b>, decoded from the <span class=\"canvas\">canvas</span>",
    note: "Entropy of the ADE20K class distribution decoded from the canvas at each position, from 0 (dark) to its maximum (bright).",
  },
  segmentation: {
    label: "Segmentation", layer: "labels", caption: "<b>Segmentation</b>, decoded from the <span class=\"canvas\">canvas</span>",
    note: "ADE20K classes decoded from the canvas by a linear probe. Point at the map to name a class.",
  },
};
const CONDITION = "carried";
const UNCERTAINTY_COLORMAP = COLORMAPS.viridis;

const template = document.createElement("template");
template.innerHTML = `
<style>
  :host { display: block; container-type: inline-size; color: var(--canvit-ink, #0f172a);
          font: 14px/1.5 var(--canvit-sans, system-ui, sans-serif); }
  .panels { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, .5fr) minmax(0, 1fr);
            gap: clamp(12px, 2.6cqi, 32px); align-items: start; }
  figure { margin: 0; }
  canvas { display: block; width: 100%; aspect-ratio: 1; border-radius: 12px; background: var(--canvit-placeholder, #f1f5f9); }
  .glimpse canvas { box-shadow: 0 0 0 2px var(--canvit-glimpse, #2d6cdf); }
  .readout canvas { image-rendering: pixelated; }
  figcaption { margin-top: 10px; font-size: 13.5px; color: var(--canvit-muted, #64748b); }
  figcaption b { color: var(--canvit-ink, #0f172a); font-weight: 600; }
  figcaption .glimpse { color: var(--canvit-glimpse, #2d6cdf); }
  figcaption .canvas { color: var(--canvit-canvas, #e0483e); font-weight: 600; }
  .bar { display: flex; align-items: center; justify-content: space-between; gap: 12px 20px; flex-wrap: wrap;
         margin-top: 18px; }
  .group { display: inline-flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .choice { display: inline-flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--canvit-placeholder, #f1f5f9); }
  button { font: inherit; font-size: 13.5px; font-weight: 500; border: 0; background: transparent; cursor: pointer;
           color: var(--canvit-muted, #64748b); padding: 6px 12px; border-radius: 8px; }
  button:hover { color: var(--canvit-ink, #0f172a); }
  button:focus-visible { outline: 2px solid var(--canvit-glimpse, #2d6cdf); outline-offset: 2px; }
  .choice button[aria-pressed="true"] { background: #fff; color: var(--canvit-ink, #0f172a);
           box-shadow: 0 1px 2px rgb(15 23 42 / .10), 0 0 0 1px rgb(15 23 42 / .06); }
  .play { width: 34px; height: 34px; padding: 0; border-radius: 50%; display: grid; place-items: center;
          background: var(--canvit-ink, #0f172a); color: #fff; }
  .play:hover { color: #fff; opacity: .85; }
  .play svg { width: 14px; height: 14px; fill: currentColor; }
  .note { margin: 12px 0 0; font-size: 13px; color: var(--canvit-muted, #64748b); min-height: 1.5em; }
  .note b { color: var(--canvit-ink, #0f172a); font-weight: 600; }
  .error { padding: 14px; border: 1px solid #dc2626; border-radius: 10px; color: #b91c1c; font: 13px/1.5 ui-monospace, monospace; }
  @container (max-width: 620px) {
    .panels { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    .scene { grid-column: 1 / -1; }
    .note { min-height: 4.5em; }
  }
</style>
<div class="panels">
  <figure class="scene"><canvas role="img" aria-label="The scene with CanViT's current viewpoint"></canvas>
    <figcaption><b>Scene</b> and the current viewpoint</figcaption></figure>
  <figure class="glimpse"><canvas role="img" aria-label="The glimpse CanViT receives"></canvas>
    <figcaption><b class="glimpse">Glimpse</b>, 128 × 128 px: all CanViT sees at this step</figcaption></figure>
  <figure class="readout"><canvas role="img" aria-label="A readout of CanViT's canvas"></canvas>
    <figcaption></figcaption></figure>
</div>
<div class="bar">
  <div class="group">
    <button class="play" aria-label="Play"></button>
    <div class="choice scenes" role="group" aria-label="Scene"></div>
  </div>
  <div class="choice readouts" role="group" aria-label="Canvas readout"></div>
</div>
<p class="note"></p>`;

const ICONS = {
  play: `<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>`,
  pause: `<svg viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>`,
};

function sizeToDisplay(canvas) {
  const scale = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * scale);
  if (width > 0 && canvas.width !== width) canvas.width = canvas.height = width;
  return canvas.width;
}

/** The readout of one sample as an ImageData of the canvas grid, from the bundle's exact layer values. */
function readoutPixels(bundle, readout, sample) {
  const { manifest } = bundle;
  const grid = manifest.canvas_grid;
  const raster = bundle.conditions[CONDITION].layers[READOUTS[readout].layer];
  const [y0, x0] = tileOrigin(manifest, sample, grid);
  const out = new ImageData(grid, grid);
  for (let row = 0; row < grid; row++) {
    for (let col = 0; col < grid; col++) {
      const at = (y0 + row) * raster.width + x0 + col;
      let rgb;
      if (readout === "canvas") rgb = raster.data.subarray(at * raster.channels, at * raster.channels + 3);
      else if (readout === "uncertainty") rgb = UNCERTAINTY_COLORMAP.subarray(3 * raster.data[at], 3 * raster.data[at] + 3);
      else rgb = ADE20K_PALETTE.subarray(3 * raster.data[at], 3 * raster.data[at] + 3);
      out.data.set(rgb, 4 * (row * grid + col));
      out.data[4 * (row * grid + col) + 3] = 255;
    }
  }
  return out;
}

class CanvitPathPlayer extends HTMLElement {
  static observedAttributes = ["src", "readout"];

  #bundles = new Map();
  #bundle = null;
  #source = null;
  #phase = 0;
  #playing = false;
  #pausedByUser = false;
  #visible = false;
  #frame = 0;
  #lastTime = null;
  #lastSample = -1;
  #pixels = document.createElement("canvas");
  #glimpseColor = null;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(template.content.cloneNode(true));
    const $ = (selector) => this.shadowRoot.querySelector(selector);
    this.$ = { scene: $(".scene canvas"), glimpse: $(".glimpse canvas"), readout: $(".readout canvas"),
               readoutCaption: $(".readout figcaption"), play: $(".play"), scenes: $(".scenes"),
               readouts: $(".readouts"), note: $(".note") };
    this.$.play.addEventListener("click", () => {
      this.#pausedByUser = this.#playing;
      this.#setPlaying(!this.#playing);
    });
    for (const [name, { label }] of Object.entries(READOUTS)) {
      const button = Object.assign(document.createElement("button"), { textContent: label });
      button.dataset.readout = name;
      button.addEventListener("click", () => this.setAttribute("readout", name));
      this.$.readouts.append(button);
    }
    this.$.readout.addEventListener("pointermove", (event) => this.#nameClass(event));
    this.$.readout.addEventListener("pointerleave", () => this.#showNote());
    new ResizeObserver(() => this.#draw(true)).observe(this.$.scene);
    new IntersectionObserver(([entry]) => {
      this.#visible = entry.isIntersecting;
      this.#setPlaying(this.#visible && this.#wantsAutoplay());
    }, { threshold: 0.25 }).observe(this);
  }

  get readout() {
    const readout = this.getAttribute("readout") ?? "canvas";
    if (!(readout in READOUTS)) throw new Error(`<canvit-path-player>: unknown readout "${readout}"`);
    return readout;
  }

  attributeChangedCallback(name) {
    if (name === "src") this.#showSources();
    else this.#showReadout();
  }

  connectedCallback() {
    this.#showReadout();
    this.#renderPlayButton();
  }

  #wantsAutoplay() {
    return this.hasAttribute("autoplay") && !this.#pausedByUser && !matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  #sources() {
    return (this.getAttribute("src") ?? "").split(/\s+/).filter(Boolean);
  }

  #showSources() {
    const sources = this.#sources();
    this.$.scenes.hidden = sources.length < 2;
    this.$.scenes.replaceChildren(...sources.map((source) => {
      const button = Object.assign(document.createElement("button"), { textContent: "…" });
      button.dataset.source = source;
      button.addEventListener("click", () => this.#select(source));
      this.#load(source).then((bundle) => { button.textContent = bundle.manifest.title; }, () => { button.textContent = source; });
      return button;
    }));
    if (sources.length > 0) this.#select(sources[0]);
  }

  #load(source) {
    if (!this.#bundles.has(source)) this.#bundles.set(source, loadPathBundle(source));
    return this.#bundles.get(source);
  }

  async #select(source) {
    this.#source = source;
    for (const button of this.$.scenes.children) button.setAttribute("aria-pressed", String(button.dataset.source === source));
    try {
      const bundle = await this.#load(source);
      if (this.#source !== source) return;
      this.#bundle = bundle;
      this.#phase = 0;
      this.#lastSample = -1;
      this.#draw(true);
      this.dispatchEvent(new CustomEvent("canvit-load", { detail: { manifest: bundle.manifest }, bubbles: true, composed: true }));
    } catch (error) {
      this.shadowRoot.querySelector(".panels").outerHTML = `<div class="error">${error.message}</div>`;
      throw error;
    }
  }

  #showReadout() {
    const readout = this.readout;
    for (const button of this.$.readouts.children) button.setAttribute("aria-pressed", String(button.dataset.readout === readout));
    this.$.readoutCaption.innerHTML = READOUTS[readout].caption;
    this.#showNote();
    this.#lastSample = -1;
    this.#draw(true);
  }

  #showNote(extra = "") {
    this.$.note.innerHTML = READOUTS[this.readout].note + extra;
  }

  #nameClass(event) {
    if (this.readout !== "segmentation" || !this.#bundle) return;
    const { manifest } = this.#bundle;
    const grid = manifest.canvas_grid;
    const box = this.$.readout.getBoundingClientRect();
    const col = Math.min(grid - 1, Math.floor(((event.clientX - box.left) / box.width) * grid));
    const row = Math.min(grid - 1, Math.floor(((event.clientY - box.top) / box.height) * grid));
    const raster = this.#bundle.conditions[CONDITION].layers.labels;
    const [y0, x0] = tileOrigin(manifest, this.#sample(), grid);
    const name = manifest.readout.class_names[raster.data[(y0 + row) * raster.width + x0 + col]];
    this.#showNote(` Here: <b>${name}</b>.`);
  }

  #setPlaying(playing) {
    if (playing === this.#playing) return;
    this.#playing = playing;
    this.#lastTime = null;
    this.#renderPlayButton();
    if (playing) this.#frame = requestAnimationFrame((time) => this.#tick(time));
    else cancelAnimationFrame(this.#frame);
  }

  #renderPlayButton() {
    this.$.play.innerHTML = this.#playing ? ICONS.pause : ICONS.play;
    this.$.play.setAttribute("aria-label", this.#playing ? "Pause" : "Play");
  }

  #tick(time) {
    if (!this.#playing) return;
    if (this.#bundle && this.#lastTime !== null) {
      this.#phase = (this.#phase + (time - this.#lastTime) / this.#bundle.manifest.path.duration_ms) % 1;
      this.#draw(false);
    }
    this.#lastTime = time;
    this.#frame = requestAnimationFrame((next) => this.#tick(next));
  }

  #sample() {
    return Math.min(this.#bundle.count - 1, Math.floor(this.#phase * this.#bundle.count));
  }

  #draw(resized) {
    if (!this.#bundle) return;
    const sample = this.#sample();
    this.#drawScene();
    if (resized || sample !== this.#lastSample) {
      this.#drawGlimpse(sample);
      this.#drawReadout(sample);
      this.#lastSample = sample;
    }
  }

  #drawScene() {
    const { scene, manifest } = this.#bundle;
    const canvas = this.$.scene;
    const size = sizeToDisplay(canvas);
    const context = canvas.getContext("2d");
    context.drawImage(scene, 0, 0, size, size);
    const [row, col, scale] = pointOn(manifest.path.segments, this.#phase);
    const x = ((col - scale + 1) / 2) * size, y = ((row - scale + 1) / 2) * size, side = scale * size;
    context.fillStyle = "rgb(15 23 42 / .38)";
    context.beginPath();
    context.rect(0, 0, size, size);
    context.rect(x, y, side, side);
    context.fill("evenodd");
    const line = Math.max(2, size / 220);
    context.lineWidth = line;
    context.strokeStyle = "#fff";
    context.strokeRect(x + line, y + line, side - 2 * line, side - 2 * line);
    this.#glimpseColor ??= getComputedStyle(this).getPropertyValue("--canvit-glimpse").trim() || "#2d6cdf";
    context.strokeStyle = this.#glimpseColor;
    context.strokeRect(x, y, side, side);
  }

  #drawGlimpse(sample) {
    const { inputs, manifest } = this.#bundle;
    const canvas = this.$.glimpse;
    const size = sizeToDisplay(canvas);
    const g = manifest.glimpse_px;
    const [y0, x0] = tileOrigin(manifest, sample, g);
    const context = canvas.getContext("2d");
    context.imageSmoothingQuality = "high";
    context.drawImage(inputs, x0, y0, g, g, 0, 0, size, size);
  }

  #drawReadout(sample) {
    const grid = this.#bundle.manifest.canvas_grid;
    this.#pixels.width = this.#pixels.height = grid;
    this.#pixels.getContext("2d").putImageData(readoutPixels(this.#bundle, this.readout, sample), 0, 0);
    const canvas = this.$.readout;
    const size = sizeToDisplay(canvas);
    const context = canvas.getContext("2d");
    context.imageSmoothingEnabled = false;
    context.drawImage(this.#pixels, 0, 0, size, size);
  }
}

customElements.define("canvit-path-player", CanvitPathPlayer);
