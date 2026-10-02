// <canvit-path src [readout="labels|canvas|entropy"] [autoplay]>: CanViT looking along a smooth path.
// The viewpoint moves continuously along the recorded Bézier curve; the maps change at the recorded
// samples, one glimpse each, shown with the canvas carried from glimpse to glimpse and reset at every
// viewpoint (schema canvit-path-bundle-df96391b-61d4-49cd-a1e5-5a4af9a42e77, canvit-pytorch/docs/viz.md).
// Changing `src` loads another bundle; hovering any scene-aligned panel marks the same point in all of them
// and names the classes there.

import { loadPathBundle, pointOn } from "./path-bundle.js";
import { UNLABELED } from "./layers.js";
import { ADE20K_PALETTE } from "./ade20k.js";
import { COLORMAPS } from "./colormaps.js";
import { frameCss, sheet } from "./view.js";

const LEGEND_CLASSES = 8;
const READOUTS = {
  labels: { title: "Segmentation" },
  canvas: { title: "Canvas features" },
  entropy: { title: "Uncertainty" },
};
const CONDITIONS = {
  carried: { title: "Canvas carried", note: "each glimpse builds on all earlier ones", line: "solid" },
  reset: { title: "Canvas reset", note: "the same glimpse, from a fresh canvas", line: "dashed" },
};

const styles = sheet(`
  :host { display: block; container-type: inline-size; font: 14px/1.45 var(--canvit-sans, system-ui, sans-serif); }
  * { box-sizing: border-box; }
  ${frameCss}
  .panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: clamp(12px, 2.2cqi, 22px) clamp(10px, 2.2cqi, 22px); }
  .readouts { grid-column: 2 / span 2; min-width: 0; }
  @container (max-width: 560px) {
    .panels { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .panels > .scene, .under, .readouts { grid-column: 1 / -1; }
  }
  figure { margin: 0; min-width: 0; }
  figcaption { display: flex; align-items: center; gap: 8px; font-weight: 650; margin-bottom: 6px; font-size: clamp(12.5px, 1.9cqi, 15px); }
  figcaption small { display: block; font-weight: 400; font-size: 12.5px; color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); }
  .caption { display: block; }
  .swatch { flex: none; width: 22px; border-top: 2.5px solid var(--canvit-canvas, #e05050); }
  .swatch.dashed { border-top-style: dashed; border-color: currentColor; opacity: .7; }
  .under { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; align-content: start; }
  .under figcaption { font-weight: 500; font-size: 12px; margin-bottom: 4px; }
  .status { margin-top: 6px; min-height: 2.9em; font: 12px/1.45 var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
            color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); font-variant-numeric: tabular-nums; }
  .status b { color: var(--canvit-ink, currentColor); font-weight: 600; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  button { font: inherit; font-size: 13px; color: inherit; cursor: pointer; }
  .chips button { padding: 5px 13px; border-radius: 999px; background: transparent;
                  border: 1px solid color-mix(in srgb, currentColor 28%, transparent); }
  .chips button[aria-pressed="true"] { background: var(--canvit-ink, currentColor); color: var(--canvit-on-ink, #111);
                                        border-color: transparent; }
  button:focus-visible, input:focus-visible { outline: 2px solid var(--canvit-focus, currentColor); outline-offset: 2px; }
  .legend { font-size: 12.5px; color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); min-height: 2.9em; margin-top: 8px; }
  .legend .classes { display: flex; flex-wrap: wrap; gap: 2px 12px; }
  .legend .class { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; color: var(--canvit-ink, currentColor); }
  .legend i { width: 10px; height: 10px; border-radius: 2px; flex: none; }
  .bar { display: inline-block; width: 140px; max-width: 30cqi; height: 9px; border-radius: 2px; vertical-align: -1px; margin: 0 6px; }
  .controls { display: flex; align-items: center; gap: 12px; margin-top: 16px; }
  .controls button { padding: 5px 12px; border-radius: 8px; border: 0; min-width: 70px;
                     background: color-mix(in srgb, currentColor 10%, transparent); }
  .controls input { flex: 1; accent-color: var(--canvit-glimpse, #4080d0); }
  svg.chart { display: block; width: 100%; height: 118px; margin-top: 10px; overflow: visible; }
  svg.chart text { font: 11px var(--canvit-sans, system-ui, sans-serif); fill: currentColor; opacity: .7; }
  svg.chart .grid { stroke: currentColor; opacity: .14; }
  svg.chart .carried { stroke: var(--canvit-canvas, #e05050); fill: none; stroke-width: 2; }
  svg.chart .reset { stroke: currentColor; opacity: .6; fill: none; stroke-width: 1.6; stroke-dasharray: 5 3; }
  svg.chart .cursor { stroke: var(--canvit-glimpse, #4080d0); stroke-width: 2; }
  .error { padding: 12px; border: 1px solid #d33; border-radius: 10px; font: 13px/1.4 var(--canvit-mono, monospace); }
`);

const percent = (fraction) => `${(100 * fraction).toFixed(1)}%`;
const signed = (value) => (value < 0 ? "−" : "+") + Math.abs(value).toFixed(2);

class CanvitPath extends HTMLElement {
  static observedAttributes = ["readout", "src"];

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).adoptedStyleSheets = [styles];
    this.elapsed = 0;
    this.playing = false;
    this.shown = null;
    this.pointer = null; // scene fractions {x, y} under the pointer, or null
    this.loading = null;
  }

  get readout() {
    const value = this.getAttribute("readout") ?? "labels";
    if (!(value in READOUTS)) throw new Error(`readout="${value}" is not one of ${Object.keys(READOUTS).join(", ")}`);
    return value;
  }

  attributeChangedCallback(name, old, value) {
    if (old === value) return;
    if (name === "src") {
      if (this.isConnected) this.#load();
      return;
    }
    this.shown = null;
    this.#chips();
    this.#legend();
    this.draw();
  }

  connectedCallback() {
    if (this.loading === null) this.#load();
    if (this.observer) return;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (this.hasAttribute("autoplay") && !reduced) {
      this.observer = new IntersectionObserver(([entry]) => {
        this.visible = entry.isIntersecting;
        if (!this.visible) this.#halt();
        else if (!this.heldByViewer) this.play();
      });
      this.observer.observe(this);
    }
  }

  async #load() {
    const src = this.getAttribute("src");
    this.loading = src;
    try {
      const bundle = await loadPathBundle(src);
      if (this.loading !== src) return; // a later src superseded this one
      this.bundle = bundle;
      this.elapsed %= bundle.manifest.path.duration_ms;
      this.shown = null;
      this.#build();
      this.draw();
      if (this.visible && !this.heldByViewer) this.play();
      this.dispatchEvent(new CustomEvent("canvit-load", { detail: { manifest: bundle.manifest } }));
    } catch (error) {
      console.error(error);
      this.shadowRoot.innerHTML = `<div class="error" role="alert"></div>`;
      this.shadowRoot.querySelector(".error").textContent = `<canvit-path>: ${error.message}`;
    }
  }

  #build() {
    const { manifest, truth, conditions } = this.bundle;
    const grid = manifest.canvas_grid;
    const pointer = `<div class="pointer" hidden></div>`;
    const panel = (name) => `<figure><figcaption><span class="swatch ${CONDITIONS[name].line}"></span>
        <span class="caption">${CONDITIONS[name].title}<small>${CONDITIONS[name].note}</small></span></figcaption>
      <div class="frame" data-aligned><canvas class="pixels" data-map="${name}" width="${grid}" height="${grid}"></canvas>${pointer}</div></figure>`;
    this.shadowRoot.innerHTML = `
      <div class="panels">
        <figure class="scene"><figcaption><span class="caption">Scene<small>the viewpoint, moving</small></span></figcaption>
          <div class="frame" data-aligned><canvas data-scene width="${manifest.scene.px}" height="${manifest.scene.px}"></canvas>${pointer}</div>
        </figure>
        ${Object.keys(conditions).map(panel).join("")}
        <div class="under">
          <figure><figcaption>Glimpse: the model's input</figcaption><div class="frame"><canvas data-input width="${manifest.glimpse_px}" height="${manifest.glimpse_px}"></canvas></div></figure>
          ${truth ? `<figure><figcaption>Annotation (ADE20K)</figcaption><div class="frame" data-aligned><canvas class="pixels" data-truth width="${grid}" height="${grid}"></canvas>${pointer}</div></figure>` : ""}
        </div>
        <div class="readouts">
          <div class="chips" role="group" aria-label="What the maps show">${Object.entries(READOUTS).map(([key, r]) =>
            `<button type="button" data-readout="${key}">${r.title}</button>`).join("")}</div>
          <div class="legend"></div>
          ${conditions.carried.pixelAccuracy ? `<svg class="chart" role="img" aria-label="Fraction of annotated pixels labeled correctly along the path, canvas carried and canvas reset"></svg>` : ""}
          <div class="status" aria-live="off"></div>
        </div>
      </div>
      <div class="controls"><button type="button" data-play>${this.playing ? "Pause" : "Play"}</button>
        <input type="range" min="0" max="${manifest.path.duration_ms - 1}" value="0" aria-label="Position along the path"></div>`;
    for (const button of this.shadowRoot.querySelectorAll("[data-readout]")) {
      button.addEventListener("click", () => this.setAttribute("readout", button.dataset.readout));
    }
    this.shadowRoot.querySelector("[data-play]").addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    const slider = this.shadowRoot.querySelector("input[type=range]");
    slider.addEventListener("input", () => { this.pause(); this.elapsed = Number(slider.value); this.draw(); });
    for (const frame of this.shadowRoot.querySelectorAll("[data-aligned]")) {
      frame.addEventListener("pointermove", (event) => {
        const rect = frame.getBoundingClientRect();
        this.pointer = { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
        this.#showPointer();
      });
      frame.addEventListener("pointerleave", (event) => {
        if (event.pointerType === "touch") return;
        this.pointer = null;
        this.#showPointer();
      });
    }
    // Read once: the canvas 2D API cannot resolve CSS variables, and reading styles every frame forces a recalculation.
    this.glimpseColor = getComputedStyle(this).getPropertyValue("--canvit-glimpse").trim() || "#4080d0";
    if (truth) this.#paintLabels(this.shadowRoot.querySelector("[data-truth]"), truth.data, 0);
    const chart = this.shadowRoot.querySelector("svg.chart");
    if (chart) {
      this.resizer?.disconnect();
      this.resizer = new ResizeObserver(() => this.#chart());
      this.resizer.observe(chart);
    }
    this.#chips();
    this.#legend();
  }

  #chips() {
    if (!this.bundle) return;
    for (const button of this.shadowRoot.querySelectorAll("[data-readout]")) {
      button.setAttribute("aria-pressed", String(button.dataset.readout === this.readout));
    }
  }

  #legend() {
    const legend = this.shadowRoot.querySelector(".legend");
    if (!legend) return;
    const { manifest, truth } = this.bundle;
    const readout = this.readout;
    if (readout === "labels") {
      const names = manifest.readout.class_names;
      let classes = "";
      if (truth) {
        const counts = new Map();
        for (const c of truth.data) if (c !== UNLABELED) counts.set(c, (counts.get(c) ?? 0) + 1);
        classes = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a)).slice(0, LEGEND_CLASSES)
          .map((c) => `<span class="class"><i style="background:rgb(${ADE20K_PALETTE.slice(3 * c, 3 * c + 3).join(" ")})"></i>${names[c]}</span>`).join("");
      }
      legend.innerHTML = `ADE20K classes decoded from the canvas by a linear probe. Hover a map to name any cell.
        ${classes ? `<div class="classes">Largest in the annotation: ${classes}</div>` : ""}`;
    } else if (readout === "canvas") {
      legend.textContent = `The canvas itself: each layer-normalized canvas token projected on three principal components, shown as RGB. ` +
        `The components and the color range come from ${manifest.pca.basis} and stay fixed, so a color means the same feature direction throughout.`;
    } else {
      const lut = COLORMAPS.viridis;
      const stops = Array.from({ length: 8 }, (_, i) => { const k = 3 * Math.round((i / 7) * 255); return `rgb(${lut[k]} ${lut[k + 1]} ${lut[k + 2]})`; });
      legend.innerHTML = `Entropy of the decoded class distribution, on a fixed scale:
        0 <span class="bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span> log ${manifest.readout.num_classes}, its maximum.`;
    }
  }

  #chart() {
    const svg = this.shadowRoot.querySelector("svg.chart");
    if (!svg || !this.bundle) return;
    const { conditions, count } = this.bundle;
    const width = Math.max(200, svg.clientWidth), height = 118, left = 36, bottom = 20;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const x = (i) => left + (i / (count - 1)) * (width - left - 4);
    const y = (v) => (height - bottom) * (1 - v) + 4;
    const line = (values) => values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    svg.innerHTML = [0, 0.5, 1].map((v) => `<line class="grid" x1="${left}" x2="${width}" y1="${y(v)}" y2="${y(v)}"/><text x="0" y="${y(v) + 4}">${v * 100}%</text>`).join("")
      + `<polyline class="reset" points="${line(conditions.reset.pixelAccuracy)}"/>`
      + `<polyline class="carried" points="${line(conditions.carried.pixelAccuracy)}"/>`
      + `<line class="cursor" y1="${y(1)}" y2="${y(0)}"/>`
      + `<text x="${left}" y="${height - 3}">Annotated pixels labeled correctly, per glimpse</text>`;
    this.cursorX = x;
    this.shown = null;
    this.draw();
  }

  #paintLabels(canvas, data, offset) {
    const grid = canvas.width;
    const out = new ImageData(grid, grid);
    for (let i = 0; i < grid * grid; i++) {
      const c = data[offset + i];
      const rgb = c === UNLABELED ? [0, 0, 0] : [ADE20K_PALETTE[3 * c], ADE20K_PALETTE[3 * c + 1], ADE20K_PALETTE[3 * c + 2]];
      out.data.set([...rgb, 255], 4 * i);
    }
    canvas.getContext("2d").putImageData(out, 0, 0);
  }

  #tileValue(raster, sample, row, col) {
    const grid = this.bundle.manifest.canvas_grid, columns = this.bundle.manifest.atlas_columns;
    return raster.data[(Math.floor(sample / columns) * grid + row) * raster.width + (sample % columns) * grid + col];
  }

  #paintTile(canvas, raster, sample, readout) {
    const grid = canvas.width;
    const columns = this.bundle.manifest.atlas_columns;
    const row0 = Math.floor(sample / columns) * grid, col0 = (sample % columns) * grid;
    const out = new ImageData(grid, grid);
    const lut = COLORMAPS.viridis;
    for (let r = 0; r < grid; r++) {
      for (let c = 0; c < grid; c++) {
        const src = (row0 + r) * raster.width + col0 + c, dst = 4 * (r * grid + c);
        if (readout === "canvas") {
          out.data.set([raster.data[3 * src], raster.data[3 * src + 1], raster.data[3 * src + 2], 255], dst);
        } else if (readout === "labels") {
          const k = raster.data[src];
          out.data.set([ADE20K_PALETTE[3 * k], ADE20K_PALETTE[3 * k + 1], ADE20K_PALETTE[3 * k + 2], 255], dst);
        } else {
          const k = 3 * raster.data[src];
          out.data.set([lut[k], lut[k + 1], lut[k + 2], 255], dst);
        }
      }
    }
    canvas.getContext("2d").putImageData(out, 0, 0);
  }

  #sample() {
    const { manifest, count } = this.bundle;
    return Math.min(count - 1, Math.floor((this.elapsed / manifest.path.duration_ms) * count));
  }

  #showPointer() {
    for (const marker of this.shadowRoot.querySelectorAll(".pointer")) {
      marker.hidden = !this.pointer;
      if (this.pointer) Object.assign(marker.style, { left: `${100 * this.pointer.x}%`, top: `${100 * this.pointer.y}%` });
    }
    this.#status();
  }

  #status() {
    const status = this.shadowRoot.querySelector(".status");
    if (!status) return;
    const { manifest, conditions, truth, count } = this.bundle;
    const sample = this.#sample();
    const [row, col, scale] = manifest.path.viewpoints[sample];
    const accuracy = conditions.carried.pixelAccuracy
      ? ` · labeled correctly: carried <b>${percent(conditions.carried.pixelAccuracy[sample])}</b>, reset <b>${percent(conditions.reset.pixelAccuracy[sample])}</b>`
      : "";
    let line = `glimpse <b>${String(sample + 1).padStart(3)}</b>/${count} · row ${signed(row)} · col ${signed(col)} · scale ${scale.toFixed(2)}${accuracy}`;
    if (this.pointer) {
      const grid = manifest.canvas_grid, names = manifest.readout.class_names;
      const r = Math.min(grid - 1, Math.max(0, Math.floor(this.pointer.y * grid)));
      const c = Math.min(grid - 1, Math.max(0, Math.floor(this.pointer.x * grid)));
      const name = (k) => (k === UNLABELED ? "unlabeled" : names[k]);
      const parts = Object.keys(conditions).map((n) => `${n} <b>${name(this.#tileValue(conditions[n].layers.labels, sample, r, c))}</b>`);
      if (truth) parts.push(`annotation <b>${name(truth.data[r * grid + c])}</b>`);
      line += `<br>here: ${parts.join(" · ")}`;
    }
    status.innerHTML = line;
  }

  draw() {
    if (!this.bundle || !this.shadowRoot.querySelector("[data-scene]")) return;
    const { manifest, scene, inputs, conditions } = this.bundle;
    const phase = this.elapsed / manifest.path.duration_ms;
    const sample = this.#sample();
    const [row, col, scale] = pointOn(manifest.path.segments, phase);

    const sceneCanvas = this.shadowRoot.querySelector("[data-scene]");
    const ctx = sceneCanvas.getContext("2d");
    const px = sceneCanvas.width;
    ctx.drawImage(scene, 0, 0, px, px);
    const box = [(col - scale + 1) * px / 2, (row - scale + 1) * px / 2, scale * px, scale * px];
    ctx.fillStyle = "rgb(0 0 0 / .38)";
    ctx.beginPath(); ctx.rect(0, 0, px, px); ctx.rect(...box); ctx.fill("evenodd");
    ctx.lineWidth = Math.max(2, px / 150);
    ctx.strokeStyle = this.glimpseColor;
    ctx.strokeRect(...box);

    const key = `${sample}|${this.readout}`;
    if (key !== this.shown) {
      this.shown = key;
      const g = manifest.glimpse_px, columns = manifest.atlas_columns;
      this.shadowRoot.querySelector("[data-input]").getContext("2d")
        .drawImage(inputs, (sample % columns) * g, Math.floor(sample / columns) * g, g, g, 0, 0, g, g);
      for (const [name, condition] of Object.entries(conditions)) {
        this.#paintTile(this.shadowRoot.querySelector(`[data-map="${name}"]`), condition.layers[this.readout], sample, this.readout);
      }
      const cursor = this.shadowRoot.querySelector(".cursor");
      if (cursor && this.cursorX) { const cx = this.cursorX(sample); cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx); }
      this.#status();
    }
    this.shadowRoot.querySelector("input[type=range]").value = String(Math.floor(this.elapsed));
  }

  play() {
    this.heldByViewer = false;
    if (this.playing || !this.bundle) return;
    this.playing = true;
    this.shadowRoot.querySelector("[data-play]").textContent = "Pause";
    let previous = null;
    const tick = (now) => {
      if (!this.playing) return;
      if (previous !== null) this.elapsed = (this.elapsed + now - previous) % this.bundle.manifest.path.duration_ms;
      previous = now;
      this.draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /** Stop until play(); scrolling back into view does not resume it. */
  pause() {
    this.heldByViewer = true;
    this.#halt();
  }

  #halt() {
    this.playing = false;
    const button = this.shadowRoot.querySelector("[data-play]");
    if (button) button.textContent = "Play";
  }
}

if (!customElements.get("canvit-path")) customElements.define("canvit-path", CanvitPath);
