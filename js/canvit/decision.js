// <canvit-decision src t>: how Entropy-Guided C2F (EG-C2F) chose glimpse t + 1. The
// uncertainty map after glimpse t, split into the candidate tiles of the next glimpse's
// quadtree level; each unvisited tile shows its mean uncertainty, and the recorded next
// glimpse is outlined. The tile means are recomputed here from the 8-bit entropy layer.

import { CanvitView, define, frameCss, place, sheet } from "./view.js";
import { renderLayer } from "./layers.js";
import { requireLayer } from "./bundle.js";

const styles = sheet(`
  ${frameCss}
  :host { aspect-ratio: 1; }
  .tile { position: absolute; display: grid; place-items: center; outline: 1px solid rgb(255 255 255 / .55); outline-offset: -.5px; }
  .tile span {
    padding: 1px 5px; border-radius: 4px; background: rgb(10 8 18 / .72); color: #fff;
    font: 600 clamp(9px, 2.6cqi, 12px)/1.3 var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
  }
  .frame { container-type: inline-size; }
  .tile.visited { background: repeating-linear-gradient(135deg, rgb(10 8 18 / .62) 0 5px, rgb(10 8 18 / .38) 5px 10px); }
  .tile.visited span { background: none; color: rgb(255 255 255 / .8); font-weight: 500; }
  .tile.chosen { outline: 3px solid var(--canvit-glimpse, #4080d0); outline-offset: -1.5px; z-index: 1;
                 box-shadow: 0 0 0 1px rgb(0 0 0 / .5), 0 0 24px rgb(255 176 32 / .45); }
  .tile.chosen span { background: var(--canvit-glimpse, #4080d0); color: #fff; }
  .done {
    position: absolute; inset: auto 0 0 0; padding: 8px; text-align: center; color: #fff;
    background: rgb(10 8 18 / .72); font: 600 12px/1.3 var(--canvit-sans, system-ui, sans-serif);
  }
`);

const same = (a, b) => Math.abs(a - b) < 1e-6;

class CanvitDecision extends CanvitView {
  static styles = [styles];

  build(bundle) {
    const policy = bundle.manifest.policy?.name;
    if (policy !== "entropy_coarse_to_fine") {
      throw new Error(`shows EG-C2F decisions, but ${bundle.url} was recorded with policy "${policy}"`);
    }
    requireLayer(bundle, "entropy");
    this.frame = document.createElement("div");
    this.frame.className = "frame";
    this.canvas = Object.assign(document.createElement("canvas"), { width: bundle.grid, height: bundle.grid, className: "pixels" });
    this.canvas.setAttribute("role", "img");
    this.tiles = document.createElement("div");
    this.marker = document.createElement("div");
    this.marker.className = "pointer";
    this.frame.append(this.canvas, this.tiles, this.marker);
    this.shadowRoot.replaceChildren(this.frame);
    this.shown = null;
    this.trackPointer(this.frame);
  }

  render(t) {
    const p = this.pointer;
    this.marker.hidden = !p;
    if (p) Object.assign(this.marker.style, { left: `${100 * p.x}%`, top: `${100 * p.y}%` });
    if (t !== this.shown) this.#draw(t);
    this.shown = t;
  }

  #draw(t) {
    const { bundle } = this;
    const { grid, glimpses } = bundle;
    this.canvas.getContext("2d").putImageData(renderLayer(bundle, t, "entropy"), 0, 0);
    const next = glimpses[t + 1];
    this.tiles.replaceChildren();
    if (!next) {
      const done = document.createElement("div");
      done.className = "done";
      done.textContent = `t = ${t} was the last glimpse`;
      this.tiles.append(done);
      this.canvas.setAttribute("aria-label", `Uncertainty after the last glimpse, t = ${t}`);
      return;
    }

    const n = Math.round(1 / next.box.size);
    if (!same(n * next.box.size, 1) || grid % n !== 0) {
      throw new Error(`glimpse ${t + 1} (size ${next.box.size}) is not a tile of a ${grid}-cell quadtree level`);
    }
    const f = glimpses[t].layers.entropy.fractions;
    const cells = grid / n;
    const visited = glimpses.slice(1, t + 1).filter((g) => same(g.box.size, next.box.size));
    let best = null;
    const tiles = [];
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const box = { top: row / n, left: col / n, size: 1 / n };
        let sum = 0;
        for (let y = row * cells; y < (row + 1) * cells; y++) {
          for (let x = col * cells; x < (col + 1) * cells; x++) sum += f[y * grid + x];
        }
        const tile = { box, score: sum / (cells * cells),
                       visited: visited.some((g) => same(g.box.top, box.top) && same(g.box.left, box.left)),
                       chosen: same(next.box.top, box.top) && same(next.box.left, box.left) };
        if (!tile.visited && (!best || tile.score > best.score)) best = tile;
        tiles.push(tile);
      }
    }
    const chosen = tiles.find((tile) => tile.chosen);
    if (!chosen) throw new Error(`glimpse ${t + 1} box ${JSON.stringify(next.box)} is not on the ${n} × ${n} tile grid`);
    if (chosen !== best) {
      // A near-tie can flip when entropy is rounded to 8 bits; the recorded choice stays authoritative.
      console.warn(`EG-C2F at t=${t}: recorded tile scores ${chosen.score.toFixed(4)} on the 8-bit layer, ` +
                   `max unvisited is ${best.score.toFixed(4)}`);
    }
    for (const tile of tiles) {
      const el = document.createElement("div");
      el.className = `tile${tile.visited ? " visited" : ""}${tile.chosen ? " chosen" : ""}`;
      place(el, tile.box);
      const label = document.createElement("span");
      label.textContent = tile.visited ? "seen" : tile.score.toFixed(2);
      el.append(label);
      this.tiles.append(el);
    }
    this.canvas.setAttribute("aria-label",
      `Uncertainty after glimpse ${t}, split into ${n * n} candidate tiles; the next glimpse goes to the outlined tile`);
  }
}

define("canvit-decision", CanvitDecision);
