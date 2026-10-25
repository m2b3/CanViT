// <foveated-scene src="IMAGE" fixations="x y, x y, ..." [fovea="0.06"]>: a photograph and a fixation cross. Stopped,
// the photograph is shown at full resolution with the cross at its center. Playing, it is seen around the fixation,
// sharp at the fovea and blurrier with eccentricity, the fixation jumping from point to point (a quick saccade, then a
// hold), looping. Fixations are fractions of the image's width and height; fovea is the sharp disk's radius as a
// fraction of the width. Layers of the same image, each blurrier, are stacked under masks that widen with the blur
// (talk.css), and the class "playing" turns the blur and masks on; --fx and --fy, registered so they animate, carry the
// fixation. play(), pause() and restart() for data-play (deck.js).

const HOLD_MS = 900;
// From the far periphery up to the fovea: blur as a fraction of the width, then the mask's inner and outer radii in
// fovea radii (the base layer is unmasked).
const LEVELS = [
  { blur: 0.022, inner: null, outer: null },
  { blur: 0.01, inner: 2.2, outer: 4 },
  { blur: 0.004, inner: 1.2, outer: 2.2 },
  { blur: 0, inner: 0.7, outer: 1.2 },
];

for (const name of ["--fx", "--fy"]) CSS.registerProperty({ name, syntax: "<percentage>", inherits: true, initialValue: "50%" });

function parseFixations(text) {
  const fixations = (text ?? "").split(",").map((pair) => pair.trim().split(/\s+/).map(Number));
  const valid = fixations.length > 1 && fixations.every((f) => f.length === 2 && f.every((v) => v >= 0 && v <= 1));
  if (!valid) throw new Error(`<foveated-scene>: fixations must be two or more "x y" pairs in [0, 1], got "${text}"`);
  return fixations;
}

class FoveatedScene extends HTMLElement {
  #fixations = [];
  #index = 0;
  #timer = null;

  connectedCallback() {
    this.#fixations = parseFixations(this.getAttribute("fixations"));
    this.style.setProperty("--fovea", this.getAttribute("fovea") ?? "0.06");
    const layers = LEVELS.map(({ blur, inner, outer }) => {
      const layer = Object.assign(document.createElement("img"), { src: this.getAttribute("src"), alt: "" });
      layer.style.setProperty("--blur", blur);
      if (inner !== null) {
        layer.classList.add("masked");
        layer.style.setProperty("--inner", inner);
        layer.style.setProperty("--outer", outer);
      }
      return layer;
    });
    layers[0].alt = this.getAttribute("alt") ?? "";
    this.replaceChildren(...layers, Object.assign(document.createElement("span"), { className: "fixation" }));
    this.pause();
  }

  play() {
    if (this.#timer !== null) return;
    this.classList.add("playing");
    this.#fixate(this.#index);
    this.#timer = setInterval(() => this.#fixate(this.#index + 1), HOLD_MS);
  }

  pause() {
    clearInterval(this.#timer);
    this.#timer = null;
    this.classList.remove("playing");
    this.#index = 0;
    this.#place(0.5, 0.5);
  }

  restart() {
    this.pause();
    this.play();
  }

  #fixate(index) {
    this.#index = index % this.#fixations.length;
    this.#place(...this.#fixations[this.#index]);
  }

  #place(x, y) {
    this.style.setProperty("--fx", `${100 * x}%`);
    this.style.setProperty("--fy", `${100 * y}%`);
  }
}

customElements.define("foveated-scene", FoveatedScene);
