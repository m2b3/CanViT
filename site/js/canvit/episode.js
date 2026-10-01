// <canvit-episode src="BUNDLE [BUNDLE…]" [readout="canvas|entropy|labels|correct"] [stage] [autoplay]>: a recorded
// rollout (web bundle) played as the loop CanViT runs: the viewpoint moves on the scene, its crop becomes the glimpse,
// the glimpse goes into CanViT, at the center, which reads its canvas and writes the glimpse into it. Under the canvas, the
// pixel accuracy of the segmentation decoded from it; beside a segmentation, its largest classes, named. The first
// glimpses are slow and later ones faster; the
// last state holds until Replay. Several bundles in `src` become scene tabs labeled with their titles.
// stage introduces the loop part by part, each in its place: "scene" (the scene alone), "viewpoint" (with the first
// glimpse's box), "glimpse" (the crop), "model" (CanViT), "canvas" (the canvas after the first glimpse), "all" (the
// default: everything, and it plays). Before "all" it holds at the first glimpse and does not play; released to "all",
// it plays on from that glimpse, so what the stages built stays on screen.

import { ADE20K_PALETTE } from "./ade20k.js";
import { loadBundle } from "./bundle.js";
import { CORRECTNESS, colormapGradient, correctnessImage, layerImage, layerSpec } from "./layers.js";

const READOUTS = {
  canvas: { label: "Features", image: (bundle, t) => layerImage(bundle, t, "canvas") },
  entropy: { label: "Uncertainty", image: (bundle, t) => layerImage(bundle, t, "entropy") },
  labels: { label: "Segmentation", image: (bundle, t) => layerImage(bundle, t, "labels") },
  correct: { label: "Correctness", image: correctnessImage },
};
const STAGES = ["scene", "viewpoint", "glimpse", "model", "canvas", "all"];
const NAMED_CLASSES = 6; // the segmentation's largest classes, named beside it
const FIRST_STEP_MS = 1600;
const STEP_RATIO = 0.74;
const SHORTEST_STEP_MS = 170;
// CanViT's ring lights up for input and write only in steps at least this long; faster, it would flash.
const HIGHLIGHT_MIN_STEP_MS = 450;
// Within a step, as fractions of its duration: [start, end] of each movement.
const PHASES = { move: [0, 0.3], crop: [0.24, 0.42], input: [0.42, 0.6], read: [0.52, 0.7], write: [0.7, 0.9] };

const template = document.createElement("template");
template.innerHTML = `
<style>
  [hidden] { display: none !important; }
  *, *::before, *::after { box-sizing: border-box; }
  :host { display: block; container-type: inline-size; color: var(--canvit-ink, #0f172a);
          font: 15px/1.4 var(--canvit-sans, system-ui, sans-serif);
          --glimpse: var(--canvit-glimpse, #2d6cdf); --canvas: var(--canvit-canvas, #e0483e); }
  /* CanViT sits at the center, between the scene and the canvas. The glimpse sits under it: the crop arrow runs
     from the scene into the glimpse, the input arrow up from the glimpse into CanViT, and write and read between
     CanViT and the canvas. */
  /* The canvas's colorbar or legend takes a fixed slot beside it; an empty column of the same width on the scene's
     side keeps the scene and the canvas the same size and CanViT at the center. */
  .flow { --side: 84px; --side-gap: 8px; --column-gap: 6px;
          --side-column: calc(var(--side) + var(--side-gap) - var(--column-gap));
          display: grid; align-items: center; gap: 0 var(--column-gap);
          grid-template-columns: var(--side-column) minmax(0, 1fr) 64px minmax(0, .52fr) 64px minmax(0, 1fr) var(--side-column);
          grid-template-areas:
            ". scene-label .    .             .     canvas-label ."
            ". scene       .    model         write canvas       canvas"
            ". scene       .    input         .     canvas       canvas"
            ". scene       crop glimpse       .     canvas       canvas"
            ". .           .    glimpse-label .     meter        meter"; }
  .scene-label { grid-area: scene-label; } .scene { grid-area: scene; }
  .crop { grid-area: crop; } .glimpse { grid-area: glimpse; } .glimpse-label { grid-area: glimpse-label; }
  .input { grid-area: input; } .model-cell { grid-area: model; } .write { grid-area: write; }
  .canvas-label { grid-area: canvas-label; } .canvas { grid-area: canvas; } .meter { grid-area: meter; }
  .label { align-self: end; padding-bottom: 10px; font-size: 18px; font-weight: 750; letter-spacing: -.01em; text-align: center; }
  .glimpse-label { align-self: start; padding: 8px 0 0; text-align: center; color: var(--glimpse); }
  .canvas-label { color: var(--canvas); }
  .cell { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; min-width: 0; }
  /* The scene and the canvas start level with CanViT, so write and read point into the canvas. */
  .scene, .canvas { align-self: start; }
  canvas { display: block; width: 100%; aspect-ratio: 1; border-radius: 10px; background: var(--canvit-placeholder, #f1f5f9); }
  .glimpse canvas { width: min(100%, 180px); box-shadow: 0 0 0 3px var(--glimpse); }
  .canvas canvas { box-shadow: 0 0 0 3px var(--canvas); image-rendering: pixelated; }
  .input { height: 64px; }
  .input svg { transform: rotate(-90deg); }
  /* The ring keeps its width; input and write change only its color. */
  .model { display: grid; place-items: center; gap: 6px; width: 100%; padding: 22px 12px 18px; border-radius: 18px;
           background: #fff; box-shadow: 0 0 0 3px #e2e8f0, 0 8px 24px rgb(15 23 42 / .08); transition: box-shadow .15s; }
  /* The CanViT wordmark: Can(vas) in the canvas's red, ViT in the glimpse's blue. */
  .wordmark { font-size: clamp(28px, 3.4cqi, 44px); font-weight: 800; line-height: 1; letter-spacing: -.03em;
              background: linear-gradient(90deg, var(--canvas), var(--glimpse)); -webkit-background-clip: text;
              background-clip: text; color: transparent; }
  .model small { font: 14px ui-monospace, "JetBrains Mono", monospace; color: var(--canvit-muted, #475569); }
  .model[data-phase="input"] { box-shadow: 0 0 0 3px var(--glimpse), 0 8px 24px rgb(15 23 42 / .08); }
  .model[data-phase="write"] { box-shadow: 0 0 0 3px var(--canvas), 0 8px 24px rgb(15 23 42 / .08); }
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
  .with-colorbar { display: flex; gap: var(--side-gap); width: 100%; }
  .with-colorbar canvas { flex: 1; min-width: 0; }
  .colorbar { display: flex; flex-direction: column; align-items: center; gap: 4px; white-space: nowrap;
              font: 12.5px/1 ui-monospace, "JetBrains Mono", monospace; color: var(--canvit-muted, #475569); }
  .colorbar-bar { flex: 1; width: 12px; border-radius: 3px; background: var(--colorbar-vertical); }
  /* The colorbar and the correctness legend share the slot beside the canvas, which keeps the width of the wider,
     so switching readouts moves nothing. */
  .side { display: grid; flex: none; width: var(--side); }
  .side > * { grid-area: 1 / 1; }
  .side > .off { visibility: hidden; }
  .meter { align-self: start; margin-top: 12px; font-size: 14px; color: var(--canvit-muted, #475569); }
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
  .classes { display: flex; flex-direction: column; justify-content: center; gap: 6px; font-size: 13px; white-space: nowrap;
             color: var(--canvit-ink, #0f172a); }
  .classes i { display: inline-block; width: 11px; height: 11px; margin-right: 5px; border-radius: 3px; vertical-align: -1px; }
  .error { padding: 14px; border: 1px solid #dc2626; border-radius: 10px; color: #b91c1c; font: 13px/1.5 ui-monospace, monospace; }
  /* Staged: the parts not yet introduced keep their place, invisible. */
  .flow > *, .bar { transition: opacity .6s ease; }
  :host(:is([stage="scene"], [stage="viewpoint"])) :is(.crop, .glimpse, .glimpse-label),
  :host(:is([stage="scene"], [stage="viewpoint"], [stage="glimpse"])) :is(.input, .model-cell),
  :host(:is([stage="scene"], [stage="viewpoint"], [stage="glimpse"], [stage="model"])) :is(.write, .canvas-label, .canvas),
  :host(:not([stage="all"])[stage]) :is(.meter, .bar) { opacity: 0; pointer-events: none; }
  /* Narrow: the same loop from top to bottom. */
  @container (max-width: 760px) {
    .flow { grid-template-columns: minmax(0, 1fr); justify-items: center; gap: 4px;
            grid-template-areas: "scene-label" "scene" "crop" "glimpse" "glimpse-label" "input" "model" "write"
                                 "canvas-label" "canvas" "meter"; }
    .scene, .canvas, .meter { width: 100%; max-width: 420px; }
    /* The colorbar or legend moves under the canvas, in a row, so the scene and the canvas share the full width. */
    .with-colorbar { flex-direction: column; }
    .side { width: 100%; }
    .colorbar { flex-direction: row-reverse; }
    .colorbar-bar { height: 12px; width: auto; background: var(--colorbar-horizontal); }
    .legend { flex-direction: row; justify-content: center; gap: 16px; }
    .scene-label, .canvas-label { padding-bottom: 6px; }
    .model-cell { width: 100%; max-width: 240px; }
    .crop svg, .input svg { transform: rotate(90deg); margin: 16px 0; }
    .write { flex-direction: row; gap: 18px; }
    .write svg { transform: rotate(90deg); margin: 16px 0; }
    .write svg.read { transform: rotate(90deg) scaleX(-1); }
    .tag { display: none; }
  }
  @container (max-width: 480px) {
    .readouts { display: grid; grid-template-columns: 1fr 1fr; width: 100%; }
  }
</style>
<div class="flow">
  <div class="label scene-label">Scene</div>
  <div class="cell scene"><canvas class="scene-view"></canvas></div>
  <div class="cell crop">${arrow("crop", "blue")}<span class="tag">crop</span></div>
  <div class="cell glimpse"><canvas class="glimpse-view"></canvas></div>
  <div class="label glimpse-label">Glimpse</div>
  <div class="cell input">${arrow("input", "blue")}</div>
  <div class="cell model-cell"><div class="model"><span class="wordmark">CanViT</span><small class="step"></small></div></div>
  <div class="cell write">${arrow("write", "red")}<span class="tag">write</span>${arrow("read", "red read")}<span class="tag">read</span></div>
  <div class="label canvas-label">Canvas</div>
  <div class="cell canvas"><div class="with-colorbar"><canvas class="canvas-view"></canvas><div class="side"><div class="colorbar" aria-hidden="true"><span class="colorbar-max"></span><div class="colorbar-bar"></div><span>0</span></div><div class="legend"></div><div class="classes"></div></div></div></div>
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
  static observedAttributes = ["src", "readout", "stage"];

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
  #namedAt = null; // the glimpse whose classes are named beside the canvas
  #holding = false; // held at the first glimpse by a stage: the next play goes on from there

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(template.content.cloneNode(true));
    const $ = (selector) => this.shadowRoot.querySelector(selector);
    this.$ = { scene: $(".scene-view"), glimpse: $(".glimpse-view"), canvas: $(".canvas-view"), model: $(".model"),
               step: $(".step"), play: $(".play"), scenes: $(".scenes"), readouts: $(".readouts"),
               meter: $(".meter"), value: $(".meter-value"), gain: $(".meter-gain"), base: $(".meter-base"),
               gained: $(".meter-gained"), start: $(".meter-start"), legend: $(".legend"),
               colorbar: $(".colorbar"), colorbarMax: $(".colorbar-max"), classes: $(".classes"),
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

  /** Play from where it is; a slide deck calls these as its slide is shown and left. A staged episode holds. */
  play() {
    if (this.#staged()) return;
    this.#pausedByUser = false;
    this.#setPlaying(true);
  }

  pause() { this.#setPlaying(false); }

  /** Play from the first glimpse, or on from it when a stage held it there. */
  restart() {
    if (this.#staged()) return;
    if (!this.#holding) return this.#restart();
    this.#holding = false;
    this.#pausedByUser = false;
    this.#setPlaying(true);
  }

  get stage() {
    const stage = this.getAttribute("stage") ?? "all";
    if (!STAGES.includes(stage)) throw new Error(`<canvit-episode>: unknown stage "${stage}", expected one of ${STAGES.join(", ")}`);
    return stage;
  }

  #staged() { return this.stage !== "all"; }

  /** Just before the second glimpse starts: the first one done, crop and canvas shown. */
  #held() { return this.#schedule.durations[0] - 1; }

  get readout() {
    const readout = this.getAttribute("readout") ?? "canvas";
    if (!(readout in READOUTS)) throw new Error(`<canvit-episode>: unknown readout "${readout}"`);
    return readout;
  }

  attributeChangedCallback(name) {
    if (name === "src") this.#showSources();
    else if (name === "stage") this.#showStage();
    else this.#showReadout();
  }

  #showStage() {
    if (!this.#schedule) return;
    if (this.#staged()) {
      this.#setPlaying(false);
      this.#time = this.#held();
      this.#holding = true;
    }
    this.#renderPlayButton();
    this.#draw();
  }

  connectedCallback() {
    const colormap = layerSpec("entropy").colormap;
    this.$.colorbar.style.setProperty("--colorbar-vertical", colormapGradient(colormap, { angle: "0deg" }));
    this.$.colorbar.style.setProperty("--colorbar-horizontal", colormapGradient(colormap, { angle: "90deg" }));
    this.$.legend.innerHTML = Object.values(CORRECTNESS)
      .map(({ label, rgb }) => `<span><i style="background: rgb(${rgb.join(" ")})"></i>${label}</span>`).join("");
    this.#showReadout();
    this.#renderPlayButton();
  }

  #wantsAutoplay() {
    return this.hasAttribute("autoplay") && !this.#staged() && !this.#pausedByUser && !matchMedia("(prefers-reduced-motion: reduce)").matches;
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
      this.#namedAt = null;
      const annotated = bundle.truth !== null && bundle.glimpses.every((g) => g.pixelAccuracy !== null);
      this.$.meter.hidden = !annotated;
      this.$.colorbarMax.textContent = `log ${bundle.manifest.readout.num_classes}`;
      this.$.colorbar.title = `Entropy of the decoded class distribution, from 0 to its maximum, log ${bundle.manifest.readout.num_classes}`;
      this.$.readouts.querySelector('[data-readout="correct"]').hidden = !annotated;
      this.#holding = this.#staged();
      this.#time = this.#staged() ? this.#held() : this.#wantsAutoplay() ? 0 : this.#schedule.total;
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
    this.$.classes.classList.toggle("off", this.readout !== "labels");
    this.#draw();
  }

  #ended() {
    return this.#schedule !== null && this.#time >= this.#schedule.total;
  }

  #restart() {
    this.#holding = false;
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
    if (this.readout === "labels") this.#drawClasses(canvasShown);
    for (const [name, packet] of Object.entries(this.$.arrows)) {
      const p = name === "read" && t === 0 ? 0 : progress(u, PHASES[name]);
      packet.setAttribute("cx", String(4 + 46 * p));
      packet.style.opacity = p > 0 && p < 1 ? "1" : "0";
    }
    const lit = this.#schedule.durations[t] >= HIGHLIGHT_MIN_STEP_MS;
    const active = (phase) => lit && u >= PHASES[phase][0] && u < PHASES[phase][1];
    this.$.model.dataset.phase = active("input") ? "input" : active("write") ? "write" : "";
    this.$.step.textContent = `t = ${t}`;
  }

  #drawScene(t, u) {
    const canvas = this.$.scene;
    const size = sizeToDisplay(canvas);
    const context = canvas.getContext("2d");
    context.drawImage(this.#scene, 0, 0, size, size);
    if (this.stage === "scene") return;
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

  /** The segmentation's NAMED_CLASSES largest classes after glimpse t, each with its color; nothing before glimpse 0. */
  #drawClasses(t) {
    if (t === this.#namedAt) return;
    this.#namedAt = t;
    if (t < 0) return this.$.classes.replaceChildren();
    const counts = new Map();
    for (const c of this.#bundle.glimpses[t].layers.labels.data) counts.set(c, (counts.get(c) ?? 0) + 1);
    const names = this.#bundle.manifest.readout.class_names;
    this.$.classes.replaceChildren(...[...counts].sort((a, b) => b[1] - a[1]).slice(0, NAMED_CLASSES).map(([c]) => {
      const row = document.createElement("span");
      const chip = document.createElement("i");
      chip.style.background = `rgb(${ADE20K_PALETTE[3 * c]} ${ADE20K_PALETTE[3 * c + 1]} ${ADE20K_PALETTE[3 * c + 2]})`;
      row.append(chip, names[c]);
      return row;
    }));
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
