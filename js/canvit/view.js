// The base of every <canvit-*> element: a view of one bundle at one glimpse index t.
// Inside a <canvit-rollout>, a view takes `src` (unless it sets its own) and `t` from
// the rollout; standalone, from its own attributes.

import { loadBundle } from "./bundle.js";

export const VIEW_TAGS = new Set();

export function define(tag, elementClass) {
  VIEW_TAGS.add(tag);
  if (!customElements.get(tag)) customElements.define(tag, elementClass);
}

/** The nearest <canvit-rollout> above `element`, across shadow roots. */
export function rolloutOf(element) {
  for (let node = element.parentNode; node; node = node instanceof ShadowRoot ? node.host : node.parentNode) {
    if (node.localName === "canvit-rollout") return node;
  }
  return null;
}

export function sheet(css) {
  const s = new CSSStyleSheet();
  s.replaceSync(css);
  return s;
}

const base = sheet(`
  :host { display: block; position: relative; min-width: 0; }
  :host([hidden]) { display: none; }
  * { box-sizing: border-box; }
  .error {
    padding: 12px 14px; border: 1px solid #d33; border-radius: var(--canvit-radius, 10px);
    background: color-mix(in srgb, #d33 10%, transparent); color: inherit;
    font: 13px/1.45 var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    overflow-wrap: anywhere;
  }
`);

/** Shared look of a square panel: a map, the scene, the mosaic. */
export const frameCss = `
  .frame {
    position: relative; width: 100%; aspect-ratio: 1; overflow: hidden;
    border-radius: var(--canvit-radius, 10px);
    background: var(--canvit-placeholder, color-mix(in srgb, currentColor 8%, transparent));
  }
  .frame > canvas, .frame > img {
    position: absolute; inset: 0; width: 100%; height: 100%; display: block;
  }
  canvas.pixels { image-rendering: pixelated; }
  .box {
    position: absolute; pointer-events: none;
    transition-property: left, top, width, height, opacity;
    transition-duration: var(--canvit-move, 520ms);
    transition-timing-function: cubic-bezier(.65, 0, .25, 1);
  }
  .pointer {
    position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px; pointer-events: none;
    border: 2px solid #fff; border-radius: 50%;
    box-shadow: 0 0 0 1.5px rgb(0 0 0 / .65), inset 0 0 0 1.5px rgb(0 0 0 / .65);
  }
  @media (prefers-reduced-motion: reduce) { .box { transition: none; } }
`;

/** Position an absolutely placed element over a box given as scene fractions. */
export function place(element, { top, left, size }) {
  Object.assign(element.style, { top: `${100 * top}%`, left: `${100 * left}%`,
                                 width: `${100 * size}%`, height: `${100 * size}%` });
}

export class CanvitView extends HTMLElement {
  static observedAttributes = ["src", "t"];
  static styles = [];

  bundle = null;
  #loading = null;
  #ownPointer = null;
  #built = false;

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).adoptedStyleSheets = [base, ...this.constructor.styles];
  }

  connectedCallback() {
    this.refresh();
    this.update();
  }

  attributeChangedCallback(name, old, value) {
    if (old === value) return;
    if (name === "src") return this.refresh();
    this.update();
    if (name === "t" && this.#built) {
      this.dispatchEvent(new CustomEvent("canvit-change", { detail: { t: this.t }, bubbles: true, composed: true }));
    }
  }

  get rollout() { return rolloutOf(this); }

  get src() { return this.getAttribute("src") ?? this.rollout?.getAttribute("src") ?? null; }

  /** The glimpse index shown, checked against the bundle. */
  get t() {
    const owner = this.rollout ?? this;
    const raw = owner.getAttribute("t") ?? "0";
    const t = Number(raw);
    const count = this.bundle?.glimpses.length ?? Infinity;
    if (!Number.isInteger(t) || t < 0 || t >= count) {
      throw new Error(`<${owner.localName}> t="${raw}" is not a glimpse index (0 to ${count - 1})`);
    }
    return t;
  }

  /** The hovered scene position {x, y} as fractions of the scene side, or null. */
  get pointer() {
    const rollout = this.rollout;
    return rollout ? rollout.pointer ?? null : this.#ownPointer;
  }

  /** Report a hovered scene position; a rollout shares it with all its views. */
  setPointer(point) {
    this.#ownPointer = point;
    this.dispatchEvent(new CustomEvent("canvit-pointer", { detail: point, bubbles: true, composed: true }));
    if (!this.rollout) this.update();
  }

  /** (Re)load when the effective src changed. */
  refresh() {
    const src = this.src;
    if (!this.isConnected || src === this.#loading) return;
    this.#loading = src;
    this.bundle = null;
    if (!src) return this.fail(new Error(`<${this.localName}> needs a src attribute or a <canvit-rollout src> ancestor`));
    loadBundle(src).then(
      (bundle) => {
        if (this.#loading !== src) return;
        this.bundle = bundle;
        this.#built = false;
        this.update();
        this.dispatchEvent(new CustomEvent("canvit-load", { detail: { bundle }, bubbles: true, composed: true }));
      },
      (error) => this.#loading === src && this.fail(error),
    );
  }

  /** Rebuild the shadow DOM at the next update (after a structural attribute changed). */
  invalidate() { this.#built = false; }

  update() {
    if (!this.bundle || !this.isConnected) return;
    try {
      if (!this.#built) {
        this.build(this.bundle);
        this.#built = true;
      }
      this.render(this.t);
    } catch (error) {
      this.fail(error);
    }
  }

  fail(error) {
    this.#built = false;
    console.error(error);
    const box = document.createElement("div");
    box.className = "error";
    box.setAttribute("role", "alert");
    box.textContent = `<${this.localName}>: ${error.message}`;
    this.shadowRoot.replaceChildren(box);
    this.dispatchEvent(new CustomEvent("canvit-error", { detail: { error }, bubbles: true, composed: true }));
  }

  /** Scene fractions under a pointer event over `target`, whose box in the scene is `box`. */
  scenePoint(event, target, box = { top: 0, left: 0, size: 1 }) {
    const r = target.getBoundingClientRect();
    const u = (event.clientX - r.left) / r.width, v = (event.clientY - r.top) / r.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { x: box.left + u * box.size, y: box.top + v * box.size };
  }

  /** Report the scene position under the pointer while it moves over `target`. */
  trackPointer(target, boxOf = () => undefined) {
    const move = (event) => this.setPointer(this.scenePoint(event, target, boxOf()));
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerdown", move);
    // A tap leaves its readout in place; a mouse leaving clears it.
    target.addEventListener("pointerleave", (event) => event.pointerType !== "touch" && this.setPointer(null));
  }

  /** Create the shadow DOM for a loaded bundle (replacing any error message). */
  build(bundle) { throw new Error(`${this.localName} must implement build(bundle)`); }

  render(t) { throw new Error(`${this.localName} must implement render(t)`); }
}
