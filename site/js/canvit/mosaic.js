// <canvit-mosaic src t>: everything the model has seen up to glimpse t, each glimpse
// pasted into its box at the resolution the model saw it, the
// sharpest view of a region on top. Regions never glimpsed stay empty.

import { CanvitView, define, frameCss, place, sheet } from "./view.js";
import { layerImage } from "./layers.js";
import { requireLayer } from "./bundle.js";

const styles = sheet(`
  ${frameCss}
  :host { aspect-ratio: 1; }
  .frame {
    background: repeating-linear-gradient(45deg,
      color-mix(in srgb, currentColor 6%, transparent) 0 6px,
      color-mix(in srgb, currentColor 12%, transparent) 6px 12px);
  }
  .glimpse { border: 2px solid var(--canvit-glimpse, #4080d0); box-shadow: 0 0 0 1px rgb(0 0 0 / .55); }
`);

class CanvitMosaic extends CanvitView {
  static styles = [styles];

  build(bundle) {
    requireLayer(bundle, "crop");
    const frame = document.createElement("div");
    frame.className = "frame";
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    this.glimpse = document.createElement("div");
    this.glimpse.className = "box glimpse";
    this.marker = document.createElement("div");
    this.marker.className = "pointer";
    frame.append(this.canvas, this.glimpse, this.marker);
    this.shadowRoot.replaceChildren(frame);
    this.drawn = null;
    this.resizer?.disconnect();
    this.resizer = new ResizeObserver(() => this.update());
    this.resizer.observe(frame);
    this.trackPointer(frame);
  }

  disconnectedCallback() { this.resizer?.disconnect(); this.invalidate(); }

  render(t) {
    const { glimpses } = this.bundle;
    // Draw at device resolution so upscaled glimpse pixels stay crisp squares.
    const side = Math.round(this.canvas.clientWidth * devicePixelRatio);
    const key = `${t}|${side}`;
    if (side > 0 && key !== this.drawn) {
      this.canvas.width = this.canvas.height = side;
      const context = this.canvas.getContext("2d");
      context.imageSmoothingEnabled = false;
      const order = glimpses.slice(0, t + 1).sort((a, b) => b.box.size - a.box.size || a.t - b.t);
      for (const { t: i, box } of order) {
        const x = Math.round(box.left * side), y = Math.round(box.top * side);
        const end = (start) => Math.round((start + box.size) * side);
        context.drawImage(layerImage(this.bundle, i, "crop"), x, y, end(box.left) - x, end(box.top) - y);
      }
      this.canvas.setAttribute("aria-label", `The ${t + 1} glimpses seen up to glimpse ${t}, pasted where they were taken`);
      this.drawn = key;
    }
    place(this.glimpse, glimpses[t].box);
    const p = this.pointer;
    this.marker.hidden = !p;
    if (p) Object.assign(this.marker.style, { left: `${100 * p.x}%`, top: `${100 * p.y}%` });
  }
}

define("canvit-mosaic", CanvitMosaic);
