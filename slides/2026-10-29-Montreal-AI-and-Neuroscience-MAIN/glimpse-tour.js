// <glimpse-tour src="…/"> (#next): what CanViT receives of a still scene as its glimpse moves and zooms, unseen regions
// black. Each stop (experiments/glimpse_tour/export.py, manifest.json) is the glimpse drawn in its box with its
// patches; between stops the box glides empty and the next glimpse appears on arrival. play(), pause() and restart()
// for data-play (deck.js); a call before the stops have loaded takes effect once they have.

const HOLD_MS = 1400; // a stop, its glimpse shown
const MOVE_MS = 600; // the box gliding to the next stop: talk.css's transition on #next glimpse-tour .box
const pct = (v) => `${(v * 100).toFixed(3)}%`;

class GlimpseTour extends HTMLElement {
  #stops = [];
  #box = null;
  #img = null;
  #index = 0;
  #playing = false;
  #timer = null;
  #arrival = null;

  async connectedCallback() {
    if (this.#box) return;
    const base = this.getAttribute("src");
    const response = await fetch(`${base}manifest.json`);
    if (!response.ok) throw new Error(`<glimpse-tour>: ${base}manifest.json answered ${response.status}`);
    const manifest = await response.json();
    if (!manifest.stops?.length) throw new Error(`<glimpse-tour>: ${base}manifest.json has no stops`);
    this.#stops = manifest.stops.map((stop) => ({ box: stop.box, url: base + stop.glimpse }));
    await Promise.all(this.#stops.map((stop) => Object.assign(new Image(), { src: stop.url }).decode()));
    this.#box = Object.assign(document.createElement("div"), { className: "box" });
    this.#img = Object.assign(document.createElement("img"), { alt: "" });
    this.#box.append(this.#img, Object.assign(document.createElement("span"), { className: "patches" }));
    this.replaceChildren(this.#box);
    this.#place(this.#index);
    if (this.#playing) this.#start();
  }

  /** Stop i at once: its box in place, its glimpse in it. */
  #place(i) {
    this.#index = i;
    const { box, url } = this.#stops[i];
    this.#box.classList.add("snap");
    Object.assign(this.#box.style, { top: pct(box.top), left: pct(box.left), width: pct(box.size), height: pct(box.size) });
    this.#img.src = url;
    this.#box.classList.remove("empty");
    void this.#box.offsetWidth; // the snap lands before the transition returns
    this.#box.classList.remove("snap");
  }

  /** The next stop: the box glides there empty, and its glimpse appears on arrival. */
  #advance() {
    const i = (this.#index + 1) % this.#stops.length;
    this.#index = i;
    const { box, url } = this.#stops[i];
    this.#box.classList.add("empty");
    Object.assign(this.#box.style, { top: pct(box.top), left: pct(box.left), width: pct(box.size), height: pct(box.size) });
    this.#arrival = setTimeout(() => {
      this.#img.src = url;
      this.#box.classList.remove("empty");
    }, MOVE_MS);
  }

  #start() {
    if (this.#timer === null) this.#timer = setInterval(() => this.#advance(), HOLD_MS + MOVE_MS);
  }

  #stop() {
    clearInterval(this.#timer);
    clearTimeout(this.#arrival);
    this.#timer = null;
  }

  play() {
    this.#playing = true;
    if (this.#box) this.#start();
  }

  pause() {
    this.#playing = false;
    this.#stop();
    if (this.#box) this.#place(this.#index);
  }

  restart() {
    this.#stop();
    this.#index = 0;
    if (this.#box) this.#place(0);
    this.play();
  }
}

if (!customElements.get("glimpse-tour")) customElements.define("glimpse-tour", GlimpseTour);
