// <talk-slide talk="slides/TALK/" slide="ID" [step-ms] [hold-ms]>: one slide of a talk, shown on a page as the talk
// shows it. The slide's markup comes from the talk's index.html and its drawing from the talk's talk.js (drawSlide), so
// the page and the talk show the same thing from the same files. It is drawn when it nears the viewport; in view, it
// plays its clicks one by one (step-ms apart), holds the last (hold-ms), and starts over. Its controls step back and
// forward and replay; stepping by hand stops the playback until replay. The page links reveal.js's reveal.css,
// slides/deck.css and the talk's talk.css: the slide sits in .reveal > .slides as on its deck, at the deck's size
// (1280 × 720), scaled to the element's width.

import { fillCode, playsNow, syncSlide } from "../slides/slide-state.js";

const WIDTH = 1280, HEIGHT = 720;
// Attributes that hold URLs, relative in a talk's markup to the talk's directory; some hold several, space-separated.
const URL_ATTRIBUTES = ["src", "href", "data-src", "scenes", "dashed-src", "added-src"];
const ABSOLUTE = /^(?:[a-z][a-z0-9+.-]*:|#|\/)/i;

const talks = new Map(); // talk URL -> its parsed index.html

async function talkDocument(base) {
  if (!talks.has(base)) {
    talks.set(base, fetch(new URL("index.html", base)).then(async (response) => {
      if (!response.ok) throw new Error(`<talk-slide>: ${base}index.html answered ${response.status}`);
      return new DOMParser().parseFromString(await response.text(), "text/html");
    }));
  }
  return talks.get(base);
}

/** Every relative URL under root, resolved against the talk's directory. */
function rebase(root, base) {
  for (const element of [root, ...root.querySelectorAll("*")]) {
    for (const name of URL_ATTRIBUTES) {
      const value = element.getAttribute(name);
      if (!value) continue;
      element.setAttribute(name, value.split(/\s+/).filter(Boolean)
        .map((token) => (ABSOLUTE.test(token) ? token : new URL(token, base).href)).join(" "));
    }
  }
}

/** The slide's fragments as reveal.js groups them: by data-fragment-index, then the others one by one, in order. */
function fragmentGroups(slide) {
  const indexed = [], rest = [];
  for (const fragment of slide.querySelectorAll(".fragment")) {
    if (fragment.hasAttribute("data-fragment-index")) (indexed[Number(fragment.dataset.fragmentIndex)] ??= []).push(fragment);
    else rest.push([fragment]);
  }
  return [...indexed.filter(Boolean), ...rest];
}

class TalkSlide extends HTMLElement {
  #slide = null;
  #groups = [];
  #step = 0;
  #timer = null;
  #autoplay = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  #inView = false;
  #started = new Set(); // elements playing since they were shown

  connectedCallback() {
    if (this.firstChild) return;
    const frame = Object.assign(document.createElement("div"), { className: "talk-slide-frame" });
    this.append(frame);
    new ResizeObserver(() => this.#fit()).observe(this);
    this.#fit();
    // Drawn ahead of the viewport, so it is ready when it comes into view.
    new IntersectionObserver(([entry], observer) => {
      if (!entry.isIntersecting) return;
      observer.disconnect();
      this.#load().catch((error) => this.#fail(error));
    }, { rootMargin: "800px 0px" }).observe(this);
    new IntersectionObserver(([entry]) => {
      this.#inView = entry.isIntersecting;
      if (this.#inView) this.#resume();
      else this.#pause();
    }, { threshold: 0.5 }).observe(this);
  }

  get #stepMs() { return Number(this.getAttribute("step-ms") ?? 2600); }
  get #holdMs() { return Number(this.getAttribute("hold-ms") ?? 6000); }

  #fit() {
    const frame = this.querySelector(".talk-slide-frame");
    const scale = this.clientWidth / WIDTH;
    frame.style.height = `${HEIGHT * scale}px`;
    frame.style.setProperty("--scale", String(scale));
  }

  async #load() {
    const base = new URL(this.getAttribute("talk"), document.baseURI).href;
    const id = this.getAttribute("slide");
    const source = (await talkDocument(base)).getElementById(id);
    if (!source || source.tagName !== "SECTION") throw new Error(`<talk-slide>: ${base}index.html has no slide #${id}`);
    // Rebased while inert, in the parsed document: custom elements upgrade only once the slide is in this page, so none
    // ever sees the talk's relative URLs.
    const inert = source.cloneNode(true);
    inert.querySelectorAll("aside.notes").forEach((notes) => notes.remove());
    rebase(inert, base);
    const slide = document.adoptNode(inert);
    slide.classList.add("present", "entering");
    const reveal = Object.assign(document.createElement("div"), { className: "reveal" });
    const slides = Object.assign(document.createElement("div"), { className: "slides" });
    slides.append(slide);
    reveal.append(slides);
    this.querySelector(".talk-slide-frame").append(reveal);
    this.#slide = slide;
    this.#groups = fragmentGroups(slide);
    await fillCode(slide);
    for (const code of slide.querySelectorAll("pre code[class*=language-]")) {
      const { default: RevealHighlight } = await import("../slides/node_modules/reveal.js/dist/plugin/highlight.mjs");
      RevealHighlight().hljs.highlightElement(code);
    }
    const { drawSlide } = await import(new URL("talk.js", base).href);
    await drawSlide(slide);
    if (this.#groups.length) this.#controls();
    this.#show(this.#autoplay ? 0 : this.#groups.length, { jump: true });
    if (this.#inView) this.#resume();
  }

  #fail(error) {
    console.error(error);
    this.querySelector(".talk-slide-frame").replaceChildren(Object.assign(document.createElement("p"), {
      className: "talk-slide-error", textContent: String(error.message ?? error),
    }));
  }

  #controls() {
    const bar = Object.assign(document.createElement("div"), { className: "talk-slide-controls" });
    const button = (label, text, action) => {
      const b = Object.assign(document.createElement("button"), { type: "button", textContent: text });
      b.setAttribute("aria-label", label);
      b.addEventListener("click", action);
      return b;
    };
    const dots = Object.assign(document.createElement("span"), { className: "dots" });
    for (let k = 0; k <= this.#groups.length; k++) {
      const dot = button(`Step ${k + 1} of ${this.#groups.length + 1}`, "", () => this.#manual(k));
      dot.className = "dot";
      dots.append(dot);
    }
    bar.append(
      button("Previous step", "‹", () => this.#manual(this.#step - 1)),
      dots,
      button("Next step", "›", () => this.#manual(this.#step + 1)),
      button("Replay", "↻", () => { this.#autoplay = true; this.#show(0, { jump: true }); this.#resume(); }),
    );
    // The same slide in its deck, full screen.
    const open = Object.assign(document.createElement("a"), { className: "open", textContent: "⤢" });
    open.href = `${this.getAttribute("talk")}#/${this.getAttribute("slide")}`;
    open.setAttribute("aria-label", "Open this slide in the talk");
    open.title = "Open this slide in the talk";
    bar.append(open);
    this.append(bar);
  }

  #manual(step) {
    this.#autoplay = false;
    this.#pause();
    this.#show(Math.max(0, Math.min(this.#groups.length, step)));
  }

  /** Shows the first `step` fragment groups. A jump lands in its state without transitions, as a deck's entered slide. */
  #show(step, { jump = false } = {}) {
    const slide = this.#slide;
    if (!slide) return;
    this.#step = step;
    if (jump) slide.classList.add("entering");
    this.#groups.forEach((group, k) => group.forEach((fragment) => fragment.classList.toggle("visible", k < step)));
    syncSlide(slide);
    if (jump) {
      slide.getBoundingClientRect();
      this.#started.clear();
      requestAnimationFrame(() => requestAnimationFrame(() => slide.classList.remove("entering")));
    }
    this.#playback();
    this.querySelectorAll(".talk-slide-controls .dot").forEach((dot, k) => dot.classList.toggle("on", k === step));
  }

  /** Plays the animations the slide's state shows, each from its start when it appears; pauses the others. */
  async #playback() {
    const elements = [...this.#slide.querySelectorAll("[data-play]")];
    await Promise.all(elements.map((e) => customElements.whenDefined(e.localName).catch(() => {})));
    for (const element of elements) {
      const shown = this.#inView && playsNow(element, this.#slide);
      if (shown && !this.#started.has(element)) {
        this.#started.add(element);
        (element.restart ?? element.play)?.call(element);
      } else if (!shown) {
        this.#started.delete(element);
        element.pause?.();
      }
    }
  }

  #resume() {
    if (!this.#slide) return;
    this.#playback();
    if (!this.#autoplay || this.#timer || !this.#groups.length) return;
    const tick = () => {
      if (this.#step < this.#groups.length) {
        this.#show(this.#step + 1);
        this.#timer = setTimeout(tick, this.#step < this.#groups.length ? this.#stepMs : this.#holdMs);
      } else {
        this.#show(0, { jump: true });
        this.#timer = setTimeout(tick, this.#stepMs);
      }
    };
    this.#timer = setTimeout(tick, this.#step === 0 ? this.#stepMs : this.#holdMs);
  }

  #pause() {
    clearTimeout(this.#timer);
    this.#timer = null;
    if (this.#slide) this.#playback();
  }
}

customElements.define("talk-slide", TalkSlide);
