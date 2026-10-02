// <canvit-timeline>: play/pause and a filmstrip of the glimpses that works as a slider
// (drag, click, arrow keys). It lives inside a <canvit-rollout>, which owns t.

import { CanvitView, define, sheet } from "./view.js";
import { layerImage } from "./layers.js";
import { requireLayer } from "./bundle.js";

const styles = sheet(`
  /* Button, gap and two readout lines, reserved before the bundle loads. */
  :host { min-height: calc(40px + 6px + 2 * 1.4 * 11.5px); font: 12px/1.4 var(--canvit-sans, system-ui, sans-serif); }
  .bar { display: flex; align-items: center; gap: 12px; }
  button {
    flex: none; width: 40px; height: 40px; border-radius: 50%; border: 0; cursor: pointer;
    display: grid; place-items: center; padding: 0;
    background: var(--canvit-glimpse, #4080d0); color: #fff;
  }
  button svg { width: 16px; height: 16px; fill: currentColor; }
  button:focus-visible, .strip:focus-visible { outline: 2px solid var(--canvit-focus, currentColor); outline-offset: 3px; }
  .strip {
    flex: 1; min-width: 0; display: flex; gap: 2px; cursor: pointer; touch-action: none;
    border-radius: 6px; padding: 3px 0; user-select: none; -webkit-user-select: none;
  }
  .cell { flex: 1 1 0; min-width: 0; position: relative; }
  .cell.level { margin-left: 6px; }
  /* A fixed height keeps the timeline's size independent of the glimpse count. */
  .cell canvas {
    display: block; width: 100%; height: 34px; object-fit: cover; border-radius: 3px; image-rendering: pixelated;
    opacity: .38; transition: opacity 200ms, transform 200ms;
  }
  .cell.seen canvas { opacity: .95; }
  .cell.now canvas { opacity: 1; transform: translateY(-3px);
    box-shadow: 0 0 0 2px var(--canvit-glimpse, #4080d0), 0 4px 10px rgb(0 0 0 / .3); }
  .readout {
    display: grid; margin: 6px 0 0 52px; line-height: 1.4; white-space: nowrap;
    color: var(--canvit-muted, color-mix(in srgb, currentColor 62%, transparent));
    font-family: var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 11.5px;
  }
  .readout b { color: var(--canvit-ink, currentColor); font-weight: 600; }
  .readout span { overflow: hidden; text-overflow: ellipsis; }
  @media (prefers-reduced-motion: reduce) { .cell canvas { transition: none; } }
`);

const PLAY = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9.5-5.5z"/></svg>';
const PAUSE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 2.5h3.2v11H3.5zm5.8 0h3.2v11H9.3z"/></svg>';

/** The share of the scene a box covers: "all", a unit fraction such as "1/16", or a percentage. */
function coverage({ size }) {
  const inverse = 1 / (size * size);
  if (Math.abs(inverse - 1) < 1e-6) return "all";
  return Math.abs(inverse - Math.round(inverse)) < 1e-6 ? `1/${Math.round(inverse)}` : `${(100 * size * size).toFixed(1)}%`;
}

class CanvitTimeline extends CanvitView {
  static styles = [styles];

  build(bundle) {
    if (!this.rollout) throw new Error("<canvit-timeline> must be inside a <canvit-rollout>");
    requireLayer(bundle, "crop");
    const count = bundle.glimpses.length;
    this.shadowRoot.innerHTML = `
      <div class="bar">
        <button type="button"></button>
        <div class="strip" role="slider" tabindex="0" aria-label="Glimpse" aria-valuemin="0" aria-valuemax="${count - 1}"></div>
      </div>
      <div class="readout"></div>`;
    this.button = this.shadowRoot.querySelector("button");
    this.strip = this.shadowRoot.querySelector(".strip");
    this.readout = this.shadowRoot.querySelector(".readout");
    this.cells = bundle.glimpses.map((g, t) => {
      const cell = document.createElement("div");
      cell.className = "cell";
      if (t > 0 && g.box.size !== bundle.glimpses[t - 1].box.size) cell.classList.add("level");
      const thumb = Object.assign(document.createElement("canvas"), { width: bundle.glimpsePx, height: bundle.glimpsePx });
      thumb.getContext("2d").drawImage(layerImage(bundle, t, "crop"), 0, 0);
      cell.append(thumb);
      return cell;
    });
    this.strip.append(...this.cells);
    this.shown = null;

    const emit = (type, detail) => this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true, composed: true }));
    this.button.addEventListener("click", () => emit("canvit-toggle"));
    const seek = (event) => {
      // The cell whose center is nearest the pointer (cells have gaps between levels).
      const centers = this.cells.map((c) => { const r = c.getBoundingClientRect(); return r.left + r.width / 2; });
      const t = centers.reduce((best, x, i) => (Math.abs(x - event.clientX) < Math.abs(centers[best] - event.clientX) ? i : best), 0);
      emit("canvit-seek", { t });
    };
    this.strip.addEventListener("pointerdown", (event) => {
      this.strip.setPointerCapture(event.pointerId);
      seek(event);
    });
    this.strip.addEventListener("pointermove", (event) => this.strip.hasPointerCapture(event.pointerId) && seek(event));
  }

  render(t) {
    const { glimpses } = this.bundle;
    const { viewpoint, box, pixelAccuracy } = glimpses[t];
    const playing = this.rollout.hasAttribute("playing");
    if (this.shown === `${t}|${playing}`) return; // pointer moves need no redraw
    this.shown = `${t}|${playing}`;
    this.button.innerHTML = playing ? PAUSE : PLAY;
    this.button.setAttribute("aria-label", playing ? "Pause" : "Play");
    this.cells.forEach((cell, i) => {
      cell.classList.toggle("seen", i < t);
      cell.classList.toggle("now", i === t);
    });
    const text = `glimpse ${t + 1} of ${glimpses.length}, scale ${viewpoint.scale}`;
    this.strip.setAttribute("aria-valuenow", t);
    this.strip.setAttribute("aria-valuetext", text);
    this.readout.innerHTML = `<span><b>t = ${t}</b> · glimpse ${t + 1}/${glimpses.length}</span>` +
      `<span>scale s = ${viewpoint.scale} · covers ${coverage(box)} of the scene` +
      (pixelAccuracy === null ? "" : ` · annotated pixels labeled correctly: ${(100 * pixelAccuracy).toFixed(1)}%`) + `</span>`;
  }
}

define("canvit-timeline", CanvitTimeline);
