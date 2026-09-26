// <canvit-rollout src [t] [autoplay] [interval] [layers]>: owns the glimpse index t and
// playback for the views inside it, and shares the hovered scene position among them.
// Empty, it shows the default layout: scene, glimpse, maps with legends, timeline.
// With children, it drives those instead (add a <canvit-timeline> for controls).
// Events: canvit-change {t} when t changes; canvit-load; canvit-error.

import { CanvitView, VIEW_TAGS, define, sheet } from "./view.js";
import { layerSpec } from "./layers.js";

const DEFAULT_LAYERS = "canvas labels entropy change";

const styles = sheet(`
  :host { display: block; container-type: inline-size; outline: none; }
  figure { margin: 0; min-width: 0; }
  figcaption { display: flex; align-items: baseline; gap: 8px; margin: 0 0 6px; font: 12px/1.3 var(--canvit-sans, system-ui, sans-serif); }
  figcaption b { font-weight: 650; letter-spacing: .01em; }
  figcaption span { color: var(--canvit-muted, color-mix(in srgb, currentColor 62%, transparent)); }
  .layout { display: grid; gap: 14px 12px; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr);
            grid-template-areas: "scene crop" "maps maps" "time time"; }
  .scene { grid-area: scene; }
  .crop { grid-area: crop; align-self: end; }
  .maps { grid-area: maps; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 12px; }
  canvit-timeline { grid-area: time; }
  canvit-legend { margin-top: 6px; }
  /* Medium: scene and glimpse side by side at equal size, the maps in one row. */
  @container (min-width: 600px) {
    .layout { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    .maps { grid-template-columns: none; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); }
  }
  /* Wide: scene, glimpse and a 2-column grid of maps. */
  @container (min-width: 960px) {
    .layout { gap: 18px 22px; grid-template-columns: minmax(0, 1.3fr) minmax(0, .52fr) minmax(0, 1.25fr);
              grid-template-areas: "scene crop maps" "time time time"; align-items: center; }
    .crop { align-self: center; }
    .maps { gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-flow: row; }
  }
`);

class CanvitRollout extends CanvitView {
  static observedAttributes = [...CanvitView.observedAttributes, "layers"];
  static styles = [styles];

  pointer = null;
  #timer = null;
  #userPaused = false;
  #visible = false;

  constructor() {
    super();
    this.addEventListener("canvit-seek", (event) => { event.stopPropagation(); this.#takeControl(); this.seek(event.detail.t); });
    this.addEventListener("canvit-toggle", (event) => {
      event.stopPropagation();
      this.#userPaused = this.hasAttribute("playing");
      if (this.#userPaused) this.pause();
      else this.play();
    });
    this.addEventListener("canvit-pointer", (event) => {
      event.stopPropagation();
      this.pointer = event.detail;
      this.update();
    });
    this.addEventListener("keydown", (event) => this.#key(event));
    this.addEventListener("canvit-load", (event) => event.composedPath()[0] === this && this.#autoplay());
  }

  /** A rollout is the root of its views. */
  get rollout() { return null; }

  connectedCallback() {
    if (!this.shadowRoot.firstChild) this.#layout();
    super.connectedCallback();
    if (this.hasAttribute("autoplay")) {
      this.observer ??= new IntersectionObserver(([entry]) => {
        this.#visible = entry.isIntersecting;
        this.#autoplay();
      }, { threshold: 0.3 });
      this.observer.observe(this);
    }
  }

  disconnectedCallback() {
    this.observer?.disconnect();
    this.pause();
  }

  attributeChangedCallback(name, old, value) {
    if (name === "layers" && old !== value && this.shadowRoot.firstChild) this.#layout();
    super.attributeChangedCallback(name, old, value);
    if (name === "src" && old !== value) for (const view of this.#views()) view.refresh();
  }

  build() {
    if (!this.shadowRoot.querySelector("slot")) this.#layout();
  }

  #views() {
    const selector = [...VIEW_TAGS].filter((tag) => tag !== this.localName).join(",");
    return [...this.querySelectorAll(selector), ...this.shadowRoot.querySelectorAll(selector)];
  }

  render() {
    for (const view of this.#views()) view.update();
    const px = this.shadowRoot.querySelector(".crop figcaption span");
    if (px) px.textContent = `the ${this.bundle.glimpsePx} px input`;
  }

  seek(t) {
    const count = this.bundle?.glimpses.length;
    if (!count) return;
    this.setAttribute("t", String(Math.min(count - 1, Math.max(0, t))));
  }

  play() {
    if (this.#timer || !this.bundle) return;
    if (!(this.interval > 0)) return this.fail(new Error(`interval="${this.getAttribute("interval")}" is not a positive number of milliseconds`));
    this.setAttribute("playing", "");
    this.#schedule();
    this.update();
  }

  pause() {
    clearTimeout(this.#timer);
    this.#timer = null;
    if (this.hasAttribute("playing")) {
      this.removeAttribute("playing");
      this.update();
    }
  }

  get interval() { return Number(this.getAttribute("interval") ?? 1200); }

  #schedule() {
    const count = this.bundle.glimpses.length;
    const last = this.t === count - 1;
    this.#timer = setTimeout(() => {
      this.seek(last ? 0 : this.t + 1);
      this.#schedule();
    }, last ? 2.5 * this.interval : this.interval);
  }

  #autoplay() {
    if (!this.hasAttribute("autoplay") || !this.bundle) return;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (this.#visible && !this.#userPaused && !reduced) this.play();
    else this.pause();
  }

  #takeControl() {
    this.#userPaused = true;
    this.pause();
  }

  #key(event) {
    if (!this.bundle || event.defaultPrevented || event.composedPath()[0].localName === "button") return;
    const t = this.t, last = this.bundle.glimpses.length - 1;
    const target = { ArrowLeft: t - 1, ArrowDown: t - 1, ArrowRight: t + 1, ArrowUp: t + 1, Home: 0, End: last }[event.key];
    if (target !== undefined) {
      this.#takeControl();
      this.seek(target);
    } else if (event.key === " ") {
      this.dispatchEvent(new CustomEvent("canvit-toggle"));
    } else return;
    event.preventDefault();
    event.stopPropagation();
  }

  /** A slot for author-composed views; the default layout while the slot is empty. */
  #layout() {
    const slot = document.createElement("slot");
    slot.addEventListener("slotchange", () => { this.#fill(slot); this.update(); });
    this.shadowRoot.replaceChildren(slot);
    try {
      this.#fill(slot);
    } catch (error) {
      this.fail(error);
    }
  }

  #fill(slot) {
    const layout = this.shadowRoot.querySelector(".layout");
    const wanted = slot.assignedElements().length === 0;
    if (wanted && !layout) this.shadowRoot.append(this.#defaultLayout());
    if (!wanted && layout) layout.remove();
  }

  #defaultLayout() {
    const figure = (className, caption, ...content) => {
      const f = document.createElement("figure");
      f.className = className;
      if (caption) {
        const c = document.createElement("figcaption");
        c.innerHTML = caption;
        f.append(c);
      }
      f.append(...content);
      return f;
    };
    const element = (tag, attributes = {}) => {
      const e = document.createElement(tag);
      for (const [k, v] of Object.entries(attributes)) e.setAttribute(k, v);
      return e;
    };
    const maps = document.createElement("div");
    maps.className = "maps";
    for (const layer of (this.getAttribute("layers") ?? DEFAULT_LAYERS).split(/\s+/).filter(Boolean)) {
      const spec = layerSpec(layer);
      maps.append(figure("map", `<b>${spec.title}</b>`, element("canvit-map", { layer }), element("canvit-legend", { layer })));
    }
    const layout = document.createElement("div");
    layout.className = "layout";
    layout.append(
      figure("scene", "<b>Scene</b><span>glimpse t outlined</span>", element("canvit-scene")),
      figure("crop", "<b>Glimpse</b><span></span>", element("canvit-map", { layer: "crop" })),
      maps,
      element("canvit-timeline"),
    );
    return layout;
  }
}

define("canvit-rollout", CanvitRollout);
