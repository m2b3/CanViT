// <canvit-live model probe scene [autoload]>: CanViT-B running in the browser. Click anywhere on the scene to take
// a glimpse there, at any scale: scroll over the scene, drag out a box, or use the slider. The canvas carries over
// from glimpse to glimpse until reset, and EG-C2F can choose the next glimpse instead. Shows the model's input,
// the ADE20K classes decoded from the canvas, and their entropy on a fixed scale from 0 to log(num_classes).
// `model` and `probe` are the canvit/ and probe/ directories `python -m canvit_pytorch.viz.live export` writes,
// relative to the page; without them, the exports published on the Hub (PUBLISHED_CANVIT, PUBLISHED_PROBE). `scene`
// is an image URL, relative to the page; a button replaces it with a photo from the visitor's device. Loading starts from a button that states the download size (nothing once
// this browser has cached the files, live-model.js), or as soon as `autoload` is present (also when it is added
// later, as a deck does when the slide comes near).
// Events: canvit-load {manifest, probeManifest, backend}, canvit-glimpse {t, viewpoint, chosenBy, stepMs, runMs},
// canvit-error.

import { ADE20K_PALETTE } from "./ade20k.js";
import { COLORMAPS } from "./colormaps.js";
import { EntropyGuidedC2F } from "./entropy-guided-c2f.js";
import { LiveModel, PUBLISHED_CANVIT, PUBLISHED_PROBE, downloadBytes, loadManifests, sceneFromImage } from "./live-model.js";
import { frameCss, sheet } from "./view.js";

const DEFAULT_SCALE = 0.25;
const DRAG_THRESHOLD_PX = 5;
const WHEEL_ZOOM_PER_PIXEL = 0.0015;
const KEY_MOVE = 0.05; // scene coordinates per arrow key press
const KEY_ZOOM = 1.15;
const POLICY_PAUSE_MS = 350; // between the glimpses of an EG-C2F episode, so each one can be seen
const LEGEND_CLASSES = 6;
const SLIDER_STEPS = 1000;

const styles = sheet(`
  :host { display: block; container-type: inline-size; font: 14px/1.45 var(--canvit-sans, system-ui, sans-serif); }
  :host([hidden]) { display: none; }
  * { box-sizing: border-box; }
  ${frameCss}
  .panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: clamp(12px, 2.2cqi, 22px) clamp(10px, 2.2cqi, 22px); }
  .readouts { grid-column: 2 / span 2; min-width: 0; }
  @container (max-width: 560px) {
    .panels { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .panels > .scene, .under, .readouts { grid-column: 1 / -1; }
  }
  figure { margin: 0; min-width: 0; }
  figcaption { font-weight: 650; margin-bottom: 6px; font-size: clamp(12.5px, 1.9cqi, 15px); }
  figcaption small { display: block; font-weight: 400; font-size: 12.5px; color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); }
  .scene .frame { cursor: crosshair; touch-action: none; }
  .scene .frame.waiting { cursor: progress; }
  .frame:focus-visible { outline: 2px solid var(--canvit-focus, currentColor); outline-offset: 2px; }
  .empty { position: absolute; inset: 0; display: grid; place-items: center; padding: 12px; text-align: center;
           font-size: 12.5px; color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); }
  .empty[hidden] { display: none; }
  .under { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; align-content: start; }
  .under figcaption { font-weight: 500; font-size: 12px; margin-bottom: 4px; }
  .size { display: grid; align-content: start; gap: 6px; font-size: 12px; }
  .size label { font-weight: 500; }
  .size input { width: 100%; accent-color: var(--canvit-glimpse, #4080d0); margin: 0; }
  .size output { white-space: pre-line; font: 12px/1.45 var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
                 color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); font-variant-numeric: tabular-nums; }
  .legend { font-size: 12.5px; color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); min-height: 2.9em; }
  .legend .classes { display: flex; flex-wrap: wrap; gap: 2px 12px; margin-top: 4px; }
  .legend .class { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; color: var(--canvit-ink, currentColor); }
  .legend i { width: 10px; height: 10px; border-radius: 2px; flex: none; }
  .legend p { margin: 8px 0 0; }
  .bar { display: inline-block; width: 140px; max-width: 30cqi; height: 9px; border-radius: 2px; vertical-align: -1px; margin: 0 6px; }
  .status { margin-top: 10px; min-height: 2.9em; font: 12px/1.45 var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
            color: var(--canvit-muted, color-mix(in srgb, currentColor 60%, transparent)); font-variant-numeric: tabular-nums; }
  .status b { color: var(--canvit-ink, currentColor); font-weight: 600; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin-top: 16px; }
  [data-photo] { margin-left: auto; }
  button { font: inherit; font-size: 13px; color: inherit; cursor: pointer; padding: 6px 13px; border-radius: 8px; border: 0;
           background: color-mix(in srgb, currentColor 10%, transparent); }
  button.primary { background: var(--canvit-glimpse, #4080d0); color: #fff; }
  button:disabled { opacity: .45; cursor: default; }
  button:focus-visible, input:focus-visible { outline: 2px solid var(--canvit-focus, currentColor); outline-offset: 2px; }
  progress { width: min(260px, 50cqi); height: 8px; accent-color: var(--canvit-glimpse, #4080d0); }
  .error { padding: 12px 14px; border: 1px solid #d33; border-radius: var(--canvit-radius, 10px);
           background: color-mix(in srgb, #d33 10%, transparent); overflow-wrap: anywhere;
           font: 13px/1.45 var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace); }
`);

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const signed = (value) => (value < 0 ? "−" : "+") + Math.abs(value).toFixed(2);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const percent = (fraction) => (fraction < 0.005 ? "<1%" : `${(100 * fraction).toFixed(0)}%`);

/** A viewpoint's crop as fractions of the scene side, from the top-left corner. */
const boxOf = ({ row, col, scale }) => ({ top: (row - scale + 1) / 2, left: (col - scale + 1) / 2, size: scale });

/** The viewpoint of scale `scale` centered as near (row, col) as keeps its crop inside the scene. */
function insideScene({ row, col, scale }, minScale) {
  const s = clamp(scale, minScale, 1);
  return { row: clamp(row, s - 1, 1 - s), col: clamp(col, s - 1, 1 - s), scale: s };
}

/** logits [C, N], class-major -> Uint8Array [N] of the most likely class per cell. */
function argmaxLabels(logits, numClasses) {
  const n = logits.length / numClasses;
  const labels = new Uint8Array(n);
  const best = logits.slice(0, n);
  for (let c = 1; c < numClasses; c++) {
    for (let i = 0; i < n; i++) {
      const v = logits[c * n + i];
      if (v > best[i]) { best[i] = v; labels[i] = c; }
    }
  }
  return labels;
}

const megabytes = (bytes) => (bytes / 1e6).toFixed(0);
const paletteRgb = (c) => [ADE20K_PALETTE[3 * c], ADE20K_PALETTE[3 * c + 1], ADE20K_PALETTE[3 * c + 2]];

class CanvitLive extends HTMLElement {
  static observedAttributes = ["model", "probe", "scene", "autoload"];

  #manifest = null; // CanViT's export
  #probe = null; // the probe's export
  #downloadBytes = 0; // the files this browser has not cached
  #model = null;
  #scene = null; // {picture, tensor}
  #history = []; // viewpoints since the last reset
  #shown = null; // {labels, entropy, glimpse, stepMs, runMs, chosenBy} of the last glimpse
  #policy = null;
  #scale = DEFAULT_SCALE;
  #preview = null; // the viewpoint a click would take, or null
  #pointer = null; // hovered scene position {x, y} in fractions, or null
  #press = null; // {x, y, clientX, clientY, dragging} while the pointer is down on the scene
  #queue = Promise.resolve();
  #pendingClick = null;
  #clicking = false;
  #episode = 0; // increments to stop a running EG-C2F episode
  #generation = 0; // increments when `model` changes
  #sceneRequests = 0; // increments when `scene` changes; only the latest request is shown
  #photoUrl = null; // the object URL of the visitor's photo, while it is the scene
  #ready = Promise.withResolvers();
  #loading = false;
  #downloading = false;
  #colors = null;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).adoptedStyleSheets = [styles];
    this.#ready.promise.catch(() => {}); // failures are shown in place; awaiting `ready` still rejects
  }

  get #modelUrl() { return this.getAttribute("model") ?? PUBLISHED_CANVIT; }

  get #probeUrl() { return this.getAttribute("probe") ?? PUBLISHED_PROBE; }

  /** Resolves when the model and the scene are loaded. */
  get ready() { return this.#ready.promise; }

  get backend() { return this.#model?.backendName ?? null; }

  /** Viewpoints {row, col, scale} since the last reset, in order. */
  get history() { return this.#history.map((v) => ({ ...v })); }

  connectedCallback() {
    if (!this.#manifest && !this.#loading) this.#prepare();
  }

  attributeChangedCallback(name, old, value) {
    if (old === value || !this.isConnected) return;
    if (name === "autoload") {
      if (value !== null && this.#manifest && !this.#loading) this.load();
      return;
    }
    if (name === "model" || name === "probe") this.#prepare();
    else if (this.#manifest) this.#loadScene().catch((error) => this.#fail(error));
  }

  // Manifest and scene first: they size the panels and label the download.
  async #prepare() {
    const generation = ++this.#generation;
    this.#loading = true;
    this.#model = null;
    this.#ready = Promise.withResolvers();
    this.#ready.promise.catch(() => {});
    try {
      const manifests = await loadManifests(this.#modelUrl, this.#probeUrl);
      const bytes = await downloadBytes(manifests);
      if (generation !== this.#generation) return;
      this.#manifest = manifests.canvit.manifest;
      this.#probe = manifests.probe.manifest;
      this.#downloadBytes = bytes;
      this.#build();
      await this.#loadScene();
      if (this.hasAttribute("autoload")) await this.load();
    } catch (error) {
      if (generation === this.#generation) this.#fail(error);
    } finally {
      this.#loading = false;
    }
  }

  /** Fetch and start the model (the load button calls this). */
  async load() {
    if (this.#model || this.#downloading) return;
    this.#downloading = true;
    const generation = this.#generation;
    const manifest = this.#manifest, probeManifest = this.#probe, bytes = this.#downloadBytes;
    const button = this.shadowRoot.querySelector("[data-load]");
    button.disabled = true;
    const progress = this.shadowRoot.querySelector("progress");
    progress.hidden = false;
    try {
      const model = await LiveModel.load(this.#modelUrl, this.#probeUrl, {
        onProgress: (stage, received, total) => {
          if (stage === "fetch") {
            progress.max = total;
            progress.value = received;
            const source = bytes ? "Downloading CanViT-B and its probe" : "Reading CanViT-B and its probe from this browser's cache";
            this.#setStatus(`${source}: <b>${megabytes(received)}</b> / ${megabytes(total)} MB`);
          } else {
            progress.removeAttribute("value");
            this.#setStatus("Checked the files; starting ONNX Runtime Web…");
          }
        },
      });
      if (generation !== this.#generation) return;
      this.#model = model;
      model.setScene(this.#scene.tensor);
      this.#clearGlimpses();
      progress.hidden = true;
      button.hidden = true;
      for (const control of this.shadowRoot.querySelectorAll("[data-needs-model]")) control.hidden = false;
      this.#updateControls();
      this.#setStatus();
      this.#ready.resolve();
      this.dispatchEvent(new CustomEvent("canvit-load", { detail: { manifest, probeManifest, backend: model.backendName }, bubbles: true, composed: true }));
    } catch (error) {
      this.#fail(error);
    } finally {
      this.#downloading = false;
    }
  }

  async #loadScene() {
    const request = ++this.#sceneRequests;
    const src = this.getAttribute("scene");
    if (!src) throw new Error("needs a scene attribute: an image URL");
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = new URL(src, document.baseURI).href;
    try {
      await image.decode();
    } catch {
      if (request !== this.#sceneRequests) return; // superseded, and maybe revoked: a photo replaced by another
      throw new Error(`Could not load the scene image ${image.src}`);
    }
    const { scene_px: size, normalization: { mean, std } } = this.#manifest;
    const scene = sceneFromImage(image, { size, mean, std });
    await this.#enqueue(() => {
      if (request !== this.#sceneRequests) return;
      this.#scene = scene;
      this.#model?.setScene(scene.tensor);
      this.#clearGlimpses();
    });
  }

  /** Take a glimpse at {row, col, scale}; resolves to the step's outputs once they are shown. */
  look(viewpoint, chosenBy = "script") {
    const { row, col, scale } = viewpoint;
    const minScale = this.#manifest?.min_scale ?? 0;
    if (![row, col, scale].every(Number.isFinite) || scale < minScale || scale > 1
        || Math.abs(row) + scale > 1 + 1e-6 || Math.abs(col) + scale > 1 + 1e-6) {
      return Promise.reject(new Error(`${JSON.stringify(viewpoint)} is not a viewpoint inside the scene with scale in [${minScale}, 1]`));
    }
    return this.#enqueue(() => this.#step({ row, col, scale }, chosenBy));
  }

  /** Let EG-C2F choose the next glimpse from the current canvas. */
  policyStep() {
    return this.#enqueue(() => {
      if (this.#policy.done) return null;
      return this.#step(this.#policy.next(this.#shown?.entropy ?? null), "EG-C2F");
    });
  }

  /** Back to the initial canvas; stops a running EG-C2F episode. */
  reset() {
    this.#episode++;
    return this.#enqueue(() => this.#clearGlimpses());
  }

  /** A whole EG-C2F episode from the initial canvas, one glimpse at a time. */
  async runPolicy() {
    await this.reset();
    const episode = this.#episode;
    while (episode === this.#episode && !this.#policy.done) {
      await this.policyStep();
      if (!this.#policy.done) await sleep(POLICY_PAUSE_MS);
    }
  }

  // Steps, resets and scene changes run one at a time, in call order.
  #enqueue(task) {
    const result = this.#queue.then(task);
    this.#queue = result.catch(() => {});
    return result;
  }

  async #step(viewpoint, chosenBy) {
    if (!this.#model) throw new Error("the model is not loaded yet");
    const started = performance.now();
    const out = await this.#model.step(viewpoint);
    const labels = argmaxLabels(out.logits, this.#probe.readout.num_classes);
    const stepMs = performance.now() - started;
    this.#history.push(viewpoint);
    this.#shown = { labels, entropy: out.entropy, glimpse: out.glimpse, stepMs, runMs: out.runMs, chosenBy };
    this.#drawScene();
    this.#drawMaps();
    this.#legend();
    this.#updateControls();
    this.#setStatus();
    const t = this.#history.length - 1;
    this.dispatchEvent(new CustomEvent("canvit-glimpse", {
      detail: { t, viewpoint: { ...viewpoint }, chosenBy, stepMs, runMs: out.runMs }, bubbles: true, composed: true,
    }));
    // readCanvas reads the canvas after this glimpse until the next one starts.
    return { t, viewpoint, ...out, labels, stepMs, readCanvas: () => this.#model.readCanvas() };
  }

  #clearGlimpses() {
    this.#model?.reset();
    this.#history = [];
    this.#shown = null;
    this.#policy = new EntropyGuidedC2F({ levels: this.#probe.policy.levels, grid: this.#manifest.canvas_grid });
    this.#drawScene();
    this.#drawMaps();
    this.#legend();
    this.#updateControls();
    if (this.#model) this.#setStatus();
  }

  // ---- layout ----

  #build() {
    const m = this.#manifest;
    const grid = m.canvas_grid;
    const pointer = `<div class="pointer" hidden></div>`;
    const empty = `<div class="empty" data-empty>No glimpse yet</div>`;
    const mapFigure = (key, title, note) => `<figure><figcaption>${title}<small>${note}</small></figcaption>
      <div class="frame" data-aligned><canvas class="pixels" data-${key} width="${grid}" height="${grid}"></canvas>${empty}${pointer}</div></figure>`;
    const policy = this.#probe.policy;
    this.shadowRoot.innerHTML = `
      <div class="panels">
        <figure class="scene"><figcaption>Scene<small>click to look · scroll or drag to size the glimpse</small></figcaption>
          <div class="frame waiting" data-aligned data-scene-frame tabindex="0"
               aria-label="Scene. Click to take a glimpse. Keyboard: arrow keys move the glimpse box, + and − resize it, Enter takes the glimpse.">
            <canvas data-scene width="${m.scene_px}" height="${m.scene_px}"></canvas>${pointer}</div>
        </figure>
        ${mapFigure("labels", "Segmentation", `ADE20K classes decoded from the canvas`)}
        ${mapFigure("entropy", "Uncertainty", `entropy of the decoded classes`)}
        <div class="under">
          <figure><figcaption>Glimpse: the model's input<small>${m.glimpse_px} × ${m.glimpse_px} px</small></figcaption>
            <div class="frame"><canvas data-glimpse width="${m.glimpse_px}" height="${m.glimpse_px}"></canvas>${empty}</div></figure>
          <div class="size"><label for="scale">Glimpse size</label>
            <input id="scale" type="range" min="0" max="${SLIDER_STEPS}" step="1">
            <output for="scale" aria-live="off"></output></div>
        </div>
        <div class="readouts"><div class="legend"></div><div class="status" aria-live="polite"></div></div>
      </div>
      <div class="controls">
        <button type="button" class="primary" data-load>Run CanViT-B here (${this.#downloadBytes ? `${megabytes(this.#downloadBytes)} MB download` : "cached in this browser"})</button>
        <progress hidden></progress>
        <button type="button" data-reset data-needs-model hidden>Reset canvas</button>
        <button type="button" data-policy-step data-needs-model hidden title="${policy.description}">${policy.paper_name}: next glimpse</button>
        <button type="button" data-policy-run data-needs-model hidden title="${policy.description}">${policy.paper_name}: ${policy.num_glimpses} glimpses from a fresh canvas</button>
        <button type="button" data-photo>Use your own photo</button>
        <input type="file" accept="image/*" hidden>
      </div>`;
    const root = this.shadowRoot;
    root.querySelector("[data-load]").addEventListener("click", () => this.load());
    root.querySelector("[data-reset]").addEventListener("click", () => this.reset());
    root.querySelector("[data-policy-step]").addEventListener("click", () => { this.#episode++; this.policyStep().catch((e) => this.#fail(e)); });
    root.querySelector("[data-policy-run]").addEventListener("click", () => this.runPolicy().catch((e) => this.#fail(e)));
    const photo = root.querySelector("input[type=file]");
    root.querySelector("[data-photo]").addEventListener("click", () => photo.click());
    photo.addEventListener("change", () => {
      const [file] = photo.files;
      if (!file) return;
      const previous = this.#photoUrl;
      this.#photoUrl = URL.createObjectURL(file);
      this.setAttribute("scene", this.#photoUrl);
      if (previous) URL.revokeObjectURL(previous);
    });
    const slider = root.querySelector("#scale");
    slider.addEventListener("input", () => this.#setScale(this.#sliderScale(Number(slider.value))));
    this.#bindScene(root.querySelector("[data-scene-frame]"));
    for (const frame of root.querySelectorAll("[data-aligned]:not([data-scene-frame])")) {
      frame.addEventListener("pointermove", (event) => this.#hover(this.#fractions(event, frame)));
      frame.addEventListener("pointerleave", (event) => event.pointerType !== "touch" && this.#hover(null));
    }
    // Read once: the canvas 2D API cannot resolve CSS variables.
    const style = getComputedStyle(this);
    this.#colors = {
      glimpse: style.getPropertyValue("--canvit-glimpse").trim() || "#4080d0",
      dim: Number(style.getPropertyValue("--canvit-dim").trim() || 0.38),
    };
    this.#setScale(this.#scale);
    this.#legend();
    const cost = this.#downloadBytes ? `${megabytes(this.#downloadBytes)} MB to download` : "cached in this browser";
    this.#setStatus(`CanViT-B and its ADE20K probe, fp32: ${cost}. Runs on WebGPU where the browser has it, on WebAssembly (slower) otherwise.`);
  }

  // ---- pointer, wheel and keyboard on the scene ----

  #fractions(event, frame) {
    const r = frame.getBoundingClientRect();
    return { x: clamp((event.clientX - r.left) / r.width, 0, 1), y: clamp((event.clientY - r.top) / r.height, 0, 1) };
  }

  #viewpointAt({ x, y }, scale = this.#scale) {
    return insideScene({ row: 2 * y - 1, col: 2 * x - 1, scale }, this.#manifest.min_scale);
  }

  #bindScene(frame) {
    frame.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      frame.setPointerCapture(event.pointerId);
      this.#press = { ...this.#fractions(event, frame), clientX: event.clientX, clientY: event.clientY, dragging: false };
      this.#hover(this.#fractions(event, frame));
    });
    frame.addEventListener("pointermove", (event) => {
      const point = this.#fractions(event, frame);
      const press = this.#press;
      if (press) {
        press.dragging ||= Math.hypot(event.clientX - press.clientX, event.clientY - press.clientY) > DRAG_THRESHOLD_PX;
        if (press.dragging) {
          // A box centered where the press started, reaching the pointer.
          const scale = 2 * Math.max(Math.abs(point.x - press.x), Math.abs(point.y - press.y));
          this.#preview = this.#viewpointAt(press, scale);
          this.#pointer = point;
          this.#drawScene();
          this.#showPointer();
          this.#describeScale(this.#preview.scale);
          return;
        }
      }
      this.#hover(point);
    });
    frame.addEventListener("pointerup", (event) => {
      const press = this.#press;
      this.#press = null;
      if (!press) return;
      const viewpoint = press.dragging ? this.#preview : this.#viewpointAt(this.#fractions(event, frame));
      if (press.dragging) this.#setScale(viewpoint.scale);
      this.#click(viewpoint);
    });
    frame.addEventListener("pointercancel", () => { this.#press = null; this.#hover(null); });
    frame.addEventListener("pointerleave", (event) => { if (!this.#press && event.pointerType !== "touch") this.#hover(null); });
    frame.addEventListener("wheel", (event) => {
      event.preventDefault();
      const pixels = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 * event.deltaY : event.deltaY;
      this.#setScale(this.#scale * Math.exp(pixels * WHEEL_ZOOM_PER_PIXEL));
      this.#hover(this.#fractions(event, frame));
    }, { passive: false });
    frame.addEventListener("keydown", (event) => {
      const current = this.#preview ?? this.#viewpointAt(this.#pointer ?? { x: 0.5, y: 0.5 });
      const moves = { ArrowUp: [-KEY_MOVE, 0], ArrowDown: [KEY_MOVE, 0], ArrowLeft: [0, -KEY_MOVE], ArrowRight: [0, KEY_MOVE] };
      let next = null;
      if (event.key in moves) {
        const [dr, dc] = moves[event.key];
        next = { ...current, row: current.row + dr, col: current.col + dc };
      } else if (event.key === "+" || event.key === "=" || event.key === "-" || event.key === "−") {
        this.#setScale(this.#scale * (event.key === "-" || event.key === "−" ? 1 / KEY_ZOOM : KEY_ZOOM));
        next = { ...current, scale: this.#scale };
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        this.#click(current);
        return;
      } else {
        return;
      }
      event.preventDefault();
      this.#preview = insideScene(next, this.#manifest.min_scale);
      this.#pointer = { x: (this.#preview.col + 1) / 2, y: (this.#preview.row + 1) / 2 };
      this.#drawScene();
      this.#showPointer();
    });
  }

  #hover(point) {
    this.#pointer = point;
    this.#preview = point && this.#model ? this.#viewpointAt(point) : null;
    this.#drawScene();
    this.#showPointer();
  }

  // Clicks faster than the model runs: the latest one waits, earlier waiting ones are dropped.
  async #click(viewpoint) {
    if (!this.#model) return;
    this.#episode++;
    this.#pendingClick = viewpoint;
    if (this.#clicking) return;
    this.#clicking = true;
    try {
      while (this.#pendingClick) {
        const next = this.#pendingClick;
        this.#pendingClick = null;
        await this.look(next, "pointer");
      }
    } catch (error) {
      this.#fail(error);
    } finally {
      this.#clicking = false;
    }
  }

  // ---- glimpse size ----

  #sliderScale(value) {
    const min = this.#manifest.min_scale;
    return min * (1 / min) ** (value / SLIDER_STEPS);
  }

  #setScale(scale) {
    const min = this.#manifest.min_scale;
    this.#scale = clamp(scale, min, 1);
    const slider = this.shadowRoot.querySelector("#scale");
    slider.value = String(Math.round((SLIDER_STEPS * Math.log(this.#scale / min)) / Math.log(1 / min)));
    if (this.#preview) this.#preview = insideScene({ ...this.#preview, scale: this.#scale }, min);
    this.#describeScale(this.#scale);
    this.#drawScene();
  }

  #describeScale(scale) {
    const output = this.shadowRoot.querySelector(".size output");
    const scenePx = scale * this.#manifest.scene_px;
    output.textContent = `scale ${scale.toFixed(3)} · ${(100 * scale * scale).toFixed(1)}% of the scene\n` +
      `${scenePx.toFixed(0)} scene px → ${this.#manifest.glimpse_px} px`;
  }

  // ---- drawing ----

  #drawScene() {
    const canvas = this.shadowRoot.querySelector("[data-scene]");
    if (!canvas || !this.#scene) return;
    const ctx = canvas.getContext("2d");
    const px = canvas.width;
    const rect = (v) => { const b = boxOf(v); return [b.left * px, b.top * px, b.size * px, b.size * px]; };
    ctx.drawImage(this.#scene.picture, 0, 0, px, px);
    const current = this.#history.at(-1);
    if (current) {
      ctx.fillStyle = `rgb(0 0 0 / ${this.#colors.dim})`;
      ctx.beginPath(); ctx.rect(0, 0, px, px); ctx.rect(...rect(current)); ctx.fill("evenodd");
      ctx.setLineDash([px / 100, px / 140]);
      ctx.lineWidth = Math.max(1, px / 400);
      ctx.strokeStyle = "rgb(255 255 255 / .7)";
      for (const earlier of this.#history.slice(0, -1)) ctx.strokeRect(...rect(earlier));
      ctx.setLineDash([]);
      ctx.lineWidth = Math.max(2, px / 150);
      ctx.strokeStyle = this.#colors.glimpse;
      ctx.strokeRect(...rect(current));
    }
    if (this.#preview) {
      ctx.setLineDash([px / 60, px / 90]);
      ctx.lineWidth = Math.max(2, px / 220);
      ctx.strokeStyle = "rgb(0 0 0 / .6)";
      ctx.strokeRect(...rect(this.#preview));
      ctx.lineDashOffset = px / 60;
      ctx.strokeStyle = "#fff";
      ctx.strokeRect(...rect(this.#preview));
      ctx.lineDashOffset = 0;
      ctx.setLineDash([]);
    }
    const frame = this.shadowRoot.querySelector("[data-scene-frame]");
    frame.classList.toggle("waiting", !this.#model);
  }

  #drawMaps() {
    const m = this.#manifest;
    const shown = this.#shown;
    for (const empty of this.shadowRoot.querySelectorAll("[data-empty]")) empty.hidden = Boolean(shown);
    const paint = (selector, n, color) => {
      const canvas = this.shadowRoot.querySelector(selector);
      const ctx = canvas.getContext("2d");
      if (!shown) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
      const image = new ImageData(n, n);
      for (let i = 0; i < n * n; i++) image.data.set([...color(i), 255], 4 * i);
      ctx.putImageData(image, 0, 0);
    };
    const grid = m.canvas_grid;
    paint("[data-labels]", grid, (i) => paletteRgb(shown.labels[i]));
    // One fixed scale for every glimpse: entropy over its maximum, log(num_classes).
    const lut = COLORMAPS.viridis;
    const maxEntropy = Math.log(this.#probe.readout.num_classes);
    paint("[data-entropy]", grid, (i) => {
      const k = 3 * Math.round(255 * clamp(shown.entropy[i] / maxEntropy, 0, 1));
      return [lut[k], lut[k + 1], lut[k + 2]];
    });
    const g = m.glimpse_px, n = g * g;
    const { mean, std } = m.normalization;
    paint("[data-glimpse]", g, (i) => [0, 1, 2].map((c) => Math.round(255 * clamp(shown.glimpse[c * n + i] * std[c] + mean[c], 0, 1))));
  }

  #legend() {
    const legend = this.shadowRoot.querySelector(".legend");
    if (!legend) return;
    const lut = COLORMAPS.viridis;
    const stops = Array.from({ length: 8 }, (_, i) => { const k = 3 * Math.round((i / 7) * 255); return `rgb(${lut[k]} ${lut[k + 1]} ${lut[k + 2]})`; });
    let classes = "";
    if (this.#shown) {
      const counts = new Map();
      for (const c of this.#shown.labels) counts.set(c, (counts.get(c) ?? 0) + 1);
      const total = this.#shown.labels.length;
      classes = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a)).slice(0, LEGEND_CLASSES)
        .map((c) => `<span class="class"><i style="background:rgb(${paletteRgb(c).join(" ")})"></i>${this.#probe.readout.class_names[c]} ${percent(counts.get(c) / total)}</span>`).join("");
    }
    legend.innerHTML = `Linear probe on the layer-normalized canvas, decoded after every glimpse, including regions not yet seen. Hover a map to name any cell.
      ${classes ? `<div class="classes">${classes}</div>` : ""}
      <p>Uncertainty: 0 <span class="bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span> log ${this.#probe.readout.num_classes}, its maximum; the same scale at every glimpse.</p>`;
  }

  #showPointer() {
    for (const marker of this.shadowRoot.querySelectorAll(".pointer")) {
      marker.hidden = !this.#pointer;
      if (this.#pointer) Object.assign(marker.style, { left: `${100 * this.#pointer.x}%`, top: `${100 * this.#pointer.y}%` });
    }
    this.#setStatus();
  }

  #updateControls() {
    const root = this.shadowRoot;
    if (!this.#model || !root.querySelector("[data-policy-step]")) return;
    root.querySelector("[data-policy-step]").disabled = this.#policy.done;
  }

  #setStatus(html = null) {
    const status = this.shadowRoot.querySelector(".status");
    if (!status) return;
    if (html !== null) { status.innerHTML = html; return; }
    if (!this.#model) return;
    const model = this.#model;
    const backend = `<b>${model.backendName}</b>${model.fallback ? ` (${model.fallback})` : ""}`;
    const current = this.#history.at(-1);
    let line;
    if (!current) {
      line = `${backend} · ready · click anywhere on the scene to take a glimpse`;
    } else {
      const s = this.#shown;
      const by = s.chosenBy === "EG-C2F" ? ` · chosen by EG-C2F (${this.#policy.t}/${this.#policy.length})` : "";
      line = `${backend} · glimpse <b>${this.#history.length}</b> · row ${signed(current.row)} · col ${signed(current.col)} · ` +
        `scale ${current.scale.toFixed(3)}${by} · ${s.stepMs.toFixed(0)} ms`;
    }
    if (this.#pointer && this.#shown) {
      const grid = this.#manifest.canvas_grid;
      const i = Math.min(grid - 1, Math.floor(this.#pointer.y * grid)) * grid + Math.min(grid - 1, Math.floor(this.#pointer.x * grid));
      const entropy = this.#shown.entropy[i];
      line += `<br>here: <b>${this.#probe.readout.class_names[this.#shown.labels[i]]}</b> · ` +
        `uncertainty ${(entropy / Math.log(this.#probe.readout.num_classes)).toFixed(2)} (${entropy.toFixed(2)} nats)`;
    }
    status.innerHTML = line;
  }

  #fail(error) {
    console.error(error);
    this.#ready.reject(error);
    const box = document.createElement("div");
    box.className = "error";
    box.setAttribute("role", "alert");
    box.textContent = `<canvit-live>: ${error.message}`;
    this.shadowRoot.replaceChildren(box);
    this.dispatchEvent(new CustomEvent("canvit-error", { detail: { error }, bubbles: true, composed: true }));
  }
}

if (!customElements.get("canvit-live")) customElements.define("canvit-live", CanvitLive);
