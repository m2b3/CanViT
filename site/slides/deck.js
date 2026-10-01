// The deck every talk in this directory runs on: reveal.js at 1280 × 720, with the project page's components on the
// slides. A talk's page imports this module once, after its slides are in the document:
//
//   <script type="module">import { startDeck } from "../deck.js"; await startDeck();</script>
//
// Slide markup conventions, all handled here:
//   data-play on an element      play() when its slide is shown (restart() when it has one), pause() when it is left
//   data-canvit-target="SEL"     on a fragment: while shown, set t="data-canvit-t" on the slide's elements matching SEL;
//     data-canvit-t="T"          with none shown, a target keeps the t of its markup
//   <section data-step>          gets data-step="N" and classes step-1 … step-N, N the number of fragments shown, so
//                                its CSS can build a diagram click by click
//   <section data-status="…">    how far the slide is from presentable: shown as a corner badge unless ?present
//   data-load-near               on an element that loads something heavy when given `autoload` (<canvit-live>): the
//                                attribute is added once its slide is the current one or the next, never at page load
// Elements a slide can use besides the project page's components: <deck-sequence> (sequence.js), its children shown
// one at a time, played with data-play.

import Reveal from "./node_modules/reveal.js/dist/reveal.mjs";
import RevealNotes from "./node_modules/reveal.js/dist/plugin/notes.mjs";

const STATUSES = {
  ready: "ready",
  draft: "draft: content in place, visuals or wording to finish",
  aspirational: "aspirational: shows what we want; the data or code behind it does not exist yet",
};

const params = new URLSearchParams(location.search);
const presenting = params.has("present") || params.has("receiver");
const printing = params.has("print-pdf");

// Clicks on these never advance the deck: controls, links and the live model's scene.
const INTERACTIVE = "a, button, input, select, textarea, label, video[controls], canvit-live, canvit-episode, canvit-frontier, canvit-rollout, [data-no-advance]";

/** Play the slide's animations from the start; pause every other slide's. */
async function syncPlayback(deck) {
  const current = deck.getCurrentSlide();
  for (const element of document.querySelectorAll(".reveal .slides [data-play]")) {
    await customElements.whenDefined(element.localName).catch(() => {});
    const shown = !printing && current?.contains(element);
    if (shown) (element.restart ?? element.play).call(element);
    else element.pause?.();
  }
}

const initialT = new WeakMap();

/** Heavy elements load when their slide comes near: the current slide or the next one. */
function loadNear(deck) {
  const slides = deck.getSlides();
  const index = slides.indexOf(deck.getCurrentSlide());
  for (const slide of slides.slice(index, index + 2)) {
    for (const element of slide.querySelectorAll("[data-load-near]:not([autoload])")) element.setAttribute("autoload", "");
  }
}

/** Every data-canvit-target fragment of the current slide sets t on its targets; none shown restores the markup's t. */
function syncTargets(deck) {
  const slide = deck.getCurrentSlide();
  if (!slide) return;
  const fragments = [...slide.querySelectorAll("[data-canvit-target]")]
    .sort((a, b) => Number(a.dataset.fragmentIndex ?? 0) - Number(b.dataset.fragmentIndex ?? 0));
  for (const selector of new Set(fragments.map((f) => f.dataset.canvitTarget))) {
    const targets = slide.querySelectorAll(selector);
    if (targets.length === 0) throw new Error(`data-canvit-target="${selector}" matches nothing on its slide`);
    const shown = fragments.filter((f) => f.dataset.canvitTarget === selector && f.classList.contains("visible"));
    for (const target of targets) {
      if (!initialT.has(target)) initialT.set(target, target.getAttribute("t") ?? "0");
      target.setAttribute("t", shown.length ? shown.at(-1).dataset.canvitT : initialT.get(target));
    }
  }
}

function syncSteps(deck) {
  const slide = deck.getCurrentSlide();
  if (!slide?.hasAttribute("data-step")) return;
  const fragments = slide.querySelectorAll(".fragment");
  const step = slide.querySelectorAll(".fragment.visible").length;
  slide.dataset.step = String(step);
  for (let n = 1; n <= fragments.length; n++) slide.classList.toggle(`step-${n}`, n <= step);
}

function statusBadges(slides) {
  for (const section of slides.querySelectorAll("section[data-status]")) {
    const status = section.dataset.status;
    if (!(status in STATUSES)) throw new Error(`data-status="${status}": expected one of ${Object.keys(STATUSES).join(", ")}`);
    if (presenting || printing || status === "ready") continue;
    const badge = document.createElement("div");
    badge.className = `status-badge ${status}`;
    badge.textContent = STATUSES[status];
    section.append(badge);
  }
}

/** A button that starts the current slide over: fragments hidden, animations from the beginning. */
function replayButton(deck) {
  const button = document.createElement("button");
  button.className = "replay";
  button.type = "button";
  button.setAttribute("aria-label", "Replay this slide");
  button.textContent = "↻";
  document.querySelector(".reveal").append(button);
  const sync = () => {
    const slide = deck.getCurrentSlide();
    button.hidden = printing || !slide?.querySelector("[data-play], .fragment");
  };
  deck.on("ready", sync);
  deck.on("slidechanged", sync);
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    button.blur();
    const { h, v } = deck.getIndices();
    deck.slide(h, v, -1);
    syncPlayback(deck);
  });
}

export async function startDeck(options = {}) {
  const slides = document.querySelector(".reveal .slides");
  statusBadges(slides);
  const deck = new Reveal({
    width: 1280,
    height: 720,
    margin: 0,
    // Up to 4K at scale 3; the components size their canvases from their displayed size, so they stay sharp.
    maxScale: 4,
    center: false,
    hash: true,
    controls: false,
    progress: true,
    slideNumber: "c/t",
    transition: "fade",
    transitionSpeed: "fast",
    pdfSeparateFragments: false,
    plugins: [RevealNotes],
    ...options,
  });
  const sync = () => {
    syncTargets(deck);
    syncSteps(deck);
  };
  for (const event of ["ready", "slidechanged", "fragmentshown", "fragmenthidden"]) deck.on(event, sync);
  for (const event of ["ready", "slidechanged"]) deck.on(event, () => syncPlayback(deck));
  // The speaker view's previews (?receiver) and print never load heavy elements: each preview is a whole deck.
  if (!printing && !params.has("receiver")) for (const event of ["ready", "slidechanged"]) deck.on(event, () => loadNear(deck));
  replayButton(deck);
  await deck.initialize();
  // A click on a slide advances like the space bar, except on controls and interactive figures, after a text
  // selection, with a modifier held, and in the speaker view's previews.
  if (!params.has("receiver")) {
    slides.addEventListener("click", (event) => {
      if (event.button !== 0 || event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (event.composedPath().some((node) => node instanceof Element && node.matches(INTERACTIVE))) return;
      if (window.getSelection()?.toString()) return;
      deck.next();
    });
  }
  window.deck = deck;
  return deck;
}
