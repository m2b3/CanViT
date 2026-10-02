// <pretraining-rollout src="rollouts.json" scene="NAME" policy="random|full_then_random" [period] [hold]>: a scene
// and the viewpoints the pretraining sampler drew for it (experiments/pretraining/export.py), the boxes appearing one
// by one in the policy's color from the paper's figures, the earlier ones fading, then starting over. play(), pause()
// and restart() for data-play (deck.js).

import { POLICIES } from "../../js/canvit/policies.js";

const SVG = "http://www.w3.org/2000/svg";
const records = new Map();

function record(src) {
  if (!records.has(src)) {
    records.set(src, fetch(src).then((r) => {
      if (!r.ok) throw new Error(`<pretraining-rollout>: ${src} answered ${r.status}`);
      return r.json();
    }));
  }
  return records.get(src);
}

class PretrainingRollout extends HTMLElement {
  #timer = null;
  #shown = 0;
  #boxes = [];

  async connectedCallback() {
    const src = this.getAttribute("src");
    const { scenes } = await record(src);
    const scene = scenes.find((s) => s.name === this.getAttribute("scene"));
    if (!scene) throw new Error(`<pretraining-rollout>: no scene "${this.getAttribute("scene")}" in ${src}`);
    const policy = this.getAttribute("policy");
    const viewpoints = scene.rollouts[policy];
    if (!viewpoints || !POLICIES[policy]) throw new Error(`<pretraining-rollout>: no ${policy} rollout for ${scene.name}`);
    const image = document.createElement("img");
    image.src = new URL(scene.image, new URL(src, document.baseURI)).href;
    image.alt = "";
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", "0 0 2 2");
    this.style.setProperty("--policy-color", POLICIES[policy].color);
    this.#boxes = viewpoints.map(([row, col, scale]) => {
      const box = document.createElementNS(SVG, "rect");
      box.setAttribute("x", col - scale + 1);
      box.setAttribute("y", row - scale + 1);
      box.setAttribute("width", 2 * scale);
      box.setAttribute("height", 2 * scale);
      svg.append(box);
      return box;
    });
    this.replaceChildren(image, svg);
    this.#render();
  }

  get #period() { return Number(this.getAttribute("period") ?? 900); }
  get #hold() { return Number(this.getAttribute("hold") ?? 2200); }

  #render() {
    this.#boxes.forEach((box, i) => box.setAttribute("class", i < this.#shown - 1 ? "past" : i === this.#shown - 1 ? "current" : "hidden"));
  }

  #tick = () => {
    const done = this.#shown >= this.#boxes.length;
    this.#timer = setTimeout(() => {
      this.#shown = done ? 0 : this.#shown + 1;
      this.#render();
      this.#tick();
    }, done ? this.#hold : this.#period);
  };

  play() { if (this.#timer === null) this.#tick(); }
  pause() { clearTimeout(this.#timer); this.#timer = null; }
  restart() { this.pause(); this.#shown = 0; this.#render(); this.play(); }
  get playing() { return this.#timer !== null; }
}

customElements.define("pretraining-rollout", PretrainingRollout);

// Row labels name their policy as the paper does, in its color: <p data-policy-label="random">.
for (const label of document.querySelectorAll("[data-policy-label]")) {
  const policy = POLICIES[label.dataset.policyLabel];
  if (!policy) throw new Error(`data-policy-label="${label.dataset.policyLabel}" is not in js/canvit/policies.js`);
  label.textContent = policy.label;
  label.style.color = policy.color;
}
