// The deck every talk in this directory runs on: reveal.js at 1280 × 720, with the project page's components on the
// slides. A talk's page imports this module once, after its slides are in the document:
//
//   <script type="module">import { startDeck } from "../deck.js"; await startDeck();</script>
//
// Slide markup conventions, all handled here:
//   data-play on an element      play() when its slide is shown (restart() when it has one), pause() when it is left;
//     data-play="NAME"           only while its slide shows the named state (data-shows below)
//   data-canvit-target="SEL"     on a fragment: while it is the last shown one for SEL, set each of its data-canvit-NAME
//     data-canvit-NAME="V"       attributes as NAME="V" on the slide's elements matching SEL (t on a rollout, series on
//                                a frontier); with none shown, a target keeps the values of its markup
//   data-shows="NAME"            on a fragment: while it is shown, its slide has the class shows-NAME, so the slide's CSS
//                                builds a diagram click by click, keyed on named states
//   <section data-status="…">    how far the slide is from presentable: shown as a corner badge unless ?present
//   data-load-near               on an element that loads something heavy when given `autoload` (<canvit-live>): the
//                                attribute is added once its slide is the current one or the next, never at page load
//   <code data-src="FILE">       filled with the file's text before reveal.js highlights it (class="language-…"), so
//                                a snippet on a slide is a file that runs
//   data-lines="A-B"             on a fragment: while it is the last shown one with data-lines, a band marks lines A
//                                to B of its slide's code block
// Elements a slide can use besides the project page's components: <deck-sequence> (sequence.js), its children shown
// one at a time, played with data-play.

import Reveal from "./node_modules/reveal.js/dist/reveal.mjs";
import RevealNotes from "./node_modules/reveal.js/dist/plugin/notes.mjs";
import RevealHighlight from "./node_modules/reveal.js/dist/plugin/highlight.mjs";

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

// The animations playing since their slide or state was shown; they start over only when shown anew.
const started = new Set();

/** Play the animations the current slide shows, each from the start when it appears; pause every other one. */
async function syncPlayback(deck) {
  const current = deck.getCurrentSlide();
  for (const element of document.querySelectorAll(".reveal .slides [data-play]")) {
    await customElements.whenDefined(element.localName).catch(() => {});
    const state = element.dataset.play;
    const shown = !printing && !!current?.contains(element) && (!state || current.classList.contains(`shows-${state}`));
    if (shown && !started.has(element)) {
      started.add(element);
      (element.restart ?? element.play).call(element);
    } else if (!shown) {
      started.delete(element);
      element.pause?.();
    }
  }
}

const initialValues = new WeakMap(); // target -> {name: its markup's value}

/** Heavy elements load when their slide comes near: the current slide or the next one. */
function loadNear(deck) {
  const slides = deck.getSlides();
  const index = slides.indexOf(deck.getCurrentSlide());
  for (const slide of slides.slice(index, index + 2)) {
    for (const element of slide.querySelectorAll("[data-load-near]:not([autoload])")) element.setAttribute("autoload", "");
  }
}

/** The attributes a data-canvit-target fragment sets: its data-canvit-NAME attributes other than the target. */
const targetAttributes = (fragment) =>
  [...fragment.attributes].filter((a) => a.name.startsWith("data-canvit-") && a.name !== "data-canvit-target")
    .map((a) => [a.name.slice("data-canvit-".length), a.value]);

/** Every data-canvit-target fragment of the current slide sets its attributes on its targets; none shown restores the
 * markup's values. */
function syncTargets(deck) {
  const slide = deck.getCurrentSlide();
  if (!slide) return;
  const fragments = [...slide.querySelectorAll("[data-canvit-target]")]
    .sort((a, b) => Number(a.dataset.fragmentIndex ?? 0) - Number(b.dataset.fragmentIndex ?? 0));
  for (const selector of new Set(fragments.map((f) => f.dataset.canvitTarget))) {
    const targets = slide.querySelectorAll(selector);
    if (targets.length === 0) throw new Error(`data-canvit-target="${selector}" matches nothing on its slide`);
    const mine = fragments.filter((f) => f.dataset.canvitTarget === selector);
    const names = new Set(mine.flatMap((f) => targetAttributes(f).map(([name]) => name)));
    const last = mine.filter((f) => f.classList.contains("visible")).at(-1);
    for (const target of targets) {
      if (!initialValues.has(target)) {
        initialValues.set(target, Object.fromEntries([...names].map((name) => [name, target.getAttribute(name)])));
      }
      const values = last ? Object.fromEntries(targetAttributes(last)) : initialValues.get(target);
      for (const name of names) {
        const value = values[name] ?? initialValues.get(target)[name];
        if (value === null) target.removeAttribute(name);
        else if (target.getAttribute(name) !== value) target.setAttribute(name, value);
      }
    }
  }
}

/** A shown fragment with data-shows="NAME" gives its slide the class shows-NAME; hidden, it takes it away. */
function syncStates(deck) {
  const slide = deck.getCurrentSlide();
  if (!slide) return;
  for (const fragment of slide.querySelectorAll(".fragment[data-shows]")) {
    slide.classList.toggle(`shows-${fragment.dataset.shows}`, fragment.classList.contains("visible"));
  }
}

async function fillCode(slides) {
  for (const code of slides.querySelectorAll("code[data-src]")) {
    const response = await fetch(code.dataset.src);
    if (!response.ok) throw new Error(`<code data-src="${code.dataset.src}">: HTTP ${response.status}`);
    code.textContent = (await response.text()).trimEnd();
  }
}

/** The last shown data-lines fragment of the current slide puts a band behind those lines of the slide's <pre>. */
function syncLines(deck) {
  const slide = deck.getCurrentSlide();
  const fragments = [...(slide?.querySelectorAll(".fragment[data-lines]") ?? [])];
  if (fragments.length === 0) return;
  const pre = slide.querySelector("pre");
  if (!pre) throw new Error(`#${slide.id}: data-lines fragments, but no <pre> to mark`);
  const range = fragments.filter((f) => f.classList.contains("visible")).at(-1)?.dataset.lines;
  pre.classList.toggle("marked", Boolean(range));
  if (!range) return;
  const match = /^(\d+)-(\d+)$/.exec(range);
  if (!match || Number(match[2]) < Number(match[1])) throw new Error(`#${slide.id}: data-lines="${range}" is not "A-B" with A ≤ B`);
  pre.style.setProperty("--line-from", match[1]);
  pre.style.setProperty("--line-count", String(Number(match[2]) - Number(match[1]) + 1));
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
    for (const element of deck.getCurrentSlide().querySelectorAll("[data-play]")) started.delete(element);
    syncPlayback(deck);
  });
}

export async function startDeck(options = {}) {
  const slides = document.querySelector(".reveal .slides");
  statusBadges(slides);
  await fillCode(slides);
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
    plugins: [RevealNotes, RevealHighlight],
    ...options,
  });
  const sync = () => {
    syncTargets(deck);
    syncStates(deck);
    syncLines(deck);
  };
  for (const event of ["ready", "slidechanged", "fragmentshown", "fragmenthidden"]) deck.on(event, sync);
  for (const event of ["ready", "slidechanged", "fragmentshown", "fragmenthidden"]) deck.on(event, () => syncPlayback(deck));
  // The speaker view's previews (?receiver) and print never load heavy elements: each preview is a whole deck.
  if (!printing && !params.has("receiver")) for (const event of ["ready", "slidechanged"]) deck.on(event, () => loadNear(deck));
  replayButton(deck);
  await deck.initialize();
  // reveal.js positions every slide absolutely; a slide given another position joins the page's flow and pushes every
  // later slide below the viewport, blank.
  if (!printing) for (const section of slides.querySelectorAll(":scope > section")) {
    const { position } = getComputedStyle(section);
    if (position !== "absolute") throw new Error(`#${section.id}: position: ${position}; reveal.js needs every slide absolute`);
  }
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
