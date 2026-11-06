// <title-teaser scenes="BUNDLE BUNDLE…" [glimpses]>: the title slide's teaser. Each recorded rollout in turn, its
// first few glimpses (glimpses, default 5): the viewpoint glides to its next position, the glimpse is taken there (a
// shutter flash, a faint outline left behind), and the canvas below crossfades to its state after that glimpse. Then
// the next scene. play() and pause() follow the slide (data-play).

import { loadBundle } from "../../js/canvit/bundle.js";
import { layerImage } from "../../js/canvit/layers.js";

const GLIDE_MS = 1100;
const HOLD_MS = 1600;
const SCENE_HOLD_MS = 1400;
const FADE_MS = 600;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function place(element, { top, left, size }) {
  Object.assign(element.style, { top: `${100 * top}%`, left: `${100 * left}%`, width: `${100 * size}%`, height: `${100 * size}%` });
}

class TitleTeaser extends HTMLElement {
  #run = null; // the current playback's token; pause() cancels it
  #scene = 0;

  connectedCallback() {
    if (this.firstChild) return;
    this.innerHTML = `
      <figure class="scene"><img alt=""><div class="traces"></div><div class="box"><span class="flash"></span></div></figure>
      <figure class="canvas"><canvas class="back"></canvas><canvas class="front"></canvas></figure>`;
    const scenes = this.getAttribute("scenes")?.trim().split(/\s+/);
    if (!scenes?.length) throw new Error("<title-teaser>: missing scenes");
    this.scenes = scenes;
  }

  get glimpses() { return Number(this.getAttribute("glimpses") ?? 5); }

  play() {
    if (this.#run) return;
    const run = { cancelled: false };
    this.#run = run;
    this.#loop(run).catch((error) => { this.#run = null; throw error; });
  }

  pause() {
    if (this.#run) this.#run.cancelled = true;
    this.#run = null;
  }

  async #loop(run) {
    const $ = (selector) => this.querySelector(selector);
    const [img, traces, box, back, front] = [$(".scene img"), $(".traces"), $(".box"), $(".canvas .back"), $(".canvas .front")];
    const wait = async (ms) => { await sleep(ms); return run.cancelled; };
    while (!run.cancelled) {
      const bundle = await loadBundle(this.scenes[this.#scene]);
      if (run.cancelled) return;
      img.src = bundle.sceneUrl;
      img.alt = `${bundle.manifest.title}, with CanViT's glimpses`;
      traces.replaceChildren();
      box.classList.add("hidden");
      // Sizing a canvas clears it: both start blank, at the canvas grid's size.
      const { width, height } = layerImage(bundle, 0, "canvas");
      for (const canvas of [back, front]) Object.assign(canvas, { width, height });
      front.classList.remove("shown");
      this.classList.remove("faded");
      const count = Math.min(this.glimpses, bundle.glimpses.length);
      for (let t = 0; t < count; t++) {
        // Glide to the viewpoint; the first glimpse appears in place.
        box.classList.toggle("gliding", t > 0);
        place(box, bundle.glimpses[t].box);
        box.classList.remove("hidden", "taken");
        if (await wait(t > 0 ? GLIDE_MS : FADE_MS)) return;
        // Take the glimpse: the shutter flashes, an outline stays, the canvas takes in what was seen.
        void box.offsetWidth; // restart the flash animation
        box.classList.add("taken");
        const trace = document.createElement("div");
        trace.className = "trace";
        place(trace, bundle.glimpses[t].box);
        traces.append(trace);
        back.getContext("2d").drawImage(front, 0, 0);
        front.getContext("2d").drawImage(layerImage(bundle, t, "canvas"), 0, 0);
        front.classList.remove("shown");
        void front.offsetWidth;
        front.classList.add("shown");
        if (await wait(HOLD_MS)) return;
      }
      if (await wait(SCENE_HOLD_MS)) return;
      this.classList.add("faded");
      if (await wait(FADE_MS)) return;
      this.#scene = (this.#scene + 1) % this.scenes.length;
    }
  }
}

customElements.define("title-teaser", TitleTeaser);
