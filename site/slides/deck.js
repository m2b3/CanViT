// The deck every talk in this directory runs on: reveal.js at 1280 × 720, with the project page's components on the
// slides. A talk's page imports this module once, after its slides are in the document:
//
//   <script type="module">import { startDeck } from "../deck.js"; await startDeck();</script>
//
// Slide markup conventions, all handled here:
//   data-play on an element      play() when its slide is shown (restart() when it has one), pause() when it is left;
//     data-play="NAME"           only while its slide shows the named state (data-shows below)
//     data-play="!NAME"          only while its slide does not show the named state
//     data-stills="MS,MS,…"      ms after a restart at which a still capture (shoot.py --steps) takes a page each
//   data-canvit-target="SEL"     on a fragment: while it is the last shown one for SEL, set each of its data-canvit-NAME
//     data-canvit-NAME="V"       attributes as NAME="V" on the slide's elements matching SEL (t on a rollout, series on
//                                a frontier); with none shown, a target keeps the values of its markup
//   data-shows="NAME"            on a fragment: while it is shown, its slide has the class shows-NAME, so the slide's CSS
//                                builds a diagram click by click, keyed on named states
//   <section data-status="…">    how far the slide is from presentable: ready, draft or aspirational
//   <div class="slides" data-talk-seconds="N">
//                                the talk's time, shared among the slides before <section id="appendix"> in proportion to
//                                the spoken words of their notes (spokenWords); each slide's range of the clock heads
//                                its notes
//   data-rehearsal="FILE"        on the same element: a measured run-through (experiments/rehearsal_timing.py), each
//                                slide's range of that run's clock shown under its budget
//   <section id="appendix">      opens the slides for questions: it and every slide after it are uncounted
//   data-load-near               on an element that loads something heavy when given `autoload` (<canvit-live>): the
//                                attribute is added once its slide is the current one or the next, never at page load
//   <code data-src="FILE">       filled with the file's text before reveal.js highlights it (class="language-…")
//   data-lines="A-B"             on a fragment: while it is the last shown one with data-lines, a band marks lines A
//                                to B of its slide's code block

import Reveal from "./node_modules/reveal.js/dist/reveal.mjs";
import RevealNotes from "./node_modules/reveal.js/dist/plugin/notes.mjs";
import RevealHighlight from "./node_modules/reveal.js/dist/plugin/highlight.mjs";
import { fillCode, playsNow, syncSlide } from "./slide-state.js";

const STATUSES = ["ready", "draft", "aspirational"];

const params = new URLSearchParams(location.search);
const printing = params.has("print-pdf");
const capturing = params.has("capture"); // a still capture (shoot.py, as pdf_backup.py runs it): no on-screen controls

// Clicks on these never advance the deck: controls, links and the live model's scene.
const INTERACTIVE = "a, button, input, select, textarea, label, video[controls], canvit-live, canvit-episode, canvit-frontier, canvit-rollout, [data-no-advance]";

// The animations playing since their slide or state was shown; they start over only when shown anew.
const started = new Set();

/** Play the animations the current slide shows, each from the start when it appears; pause every other one. The
 * current slide is read after awaiting element definitions, so racing calls all apply the slide shown when they resume. */
async function syncPlayback(deck) {
  const elements = [...document.querySelectorAll(".reveal .slides [data-play]")];
  await Promise.all(elements.map((element) => customElements.whenDefined(element.localName).catch(() => {})));
  const current = deck.getCurrentSlide();
  for (const element of elements) {
    const shown = !printing && !!current?.contains(element) && playsNow(element, current);
    if (shown && !started.has(element)) {
      started.add(element);
      (element.restart ?? element.play).call(element);
    } else if (!shown) {
      started.delete(element);
      element.pause?.();
    }
  }
}

/** Heavy elements load when their slide is the current one or the next. */
function loadNear(deck) {
  const slides = deck.getHorizontalSlides(); // every slide, the appendix's too (getSlides() leaves uncounted ones out)
  const index = slides.indexOf(deck.getCurrentSlide());
  for (const slide of slides.slice(index, index + 2)) {
    for (const element of slide.querySelectorAll("[data-load-near]:not([autoload])")) element.setAttribute("autoload", "");
  }
}

function checkStatuses(slides) {
  for (const section of slides.querySelectorAll("section[data-status]")) {
    const status = section.dataset.status;
    if (!STATUSES.includes(status)) throw new Error(`#${section.id}: data-status="${status}": expected one of ${STATUSES.join(", ")}`);
  }
}

/** The spoken words of a slide's notes: bracketed markers such as [CLICK] and lines in parentheses are not said. */
function spokenWords(section) {
  let words = 0;
  for (const line of section.querySelectorAll("aside.notes li")) {
    const text = line.textContent.replace(/\[[^\]]*\]/g, " ").trim();
    if (text.startsWith("(") && text.endsWith(")")) continue;
    words += text.split(/\s+/).filter(Boolean).length;
  }
  return words;
}

/** Each slide before the appendix gets the talk's time in proportion to its notes' words, back to back from 00:00; with a
 * measured run-through, its range in that run follows. */
async function timeRanges(slides) {
  if (!slides.hasAttribute("data-talk-seconds")) return;
  const total = Number(slides.dataset.talkSeconds);
  if (!(total > 0)) throw new Error(`data-talk-seconds="${slides.dataset.talkSeconds}"`);
  const timed = [];
  for (const section of slides.querySelectorAll(":scope > section")) {
    if (section.id === "appendix") break;
    timed.push(section);
  }
  const words = timed.map(spokenWords), all = words.reduce((a, b) => a + b, 0);
  if (!all) throw new Error("data-talk-seconds: the notes of the slides before the appendix have no words");
  const clock = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  const span = (start, end) => `${clock(Math.round(start))}–${clock(Math.round(end))}`;
  let run = null;
  if (slides.dataset.rehearsal) {
    const response = await fetch(slides.dataset.rehearsal);
    if (!response.ok) throw new Error(`data-rehearsal: ${slides.dataset.rehearsal} answered ${response.status}`);
    run = await response.json();
  }
  let said = 0, ran = 0;
  timed.forEach((section, i) => {
    const start = (total * said) / all;
    said += words[i];
    let line = `⏱ ${span(start, (total * said) / all)}`;
    if (run) {
      const seconds = run.seconds[section.id];
      line += seconds === undefined ? " · not in the run" : ` · run ${span(ran, ran + seconds)}`;
      ran += seconds ?? 0;
    }
    section.querySelector("aside.notes")?.prepend(Object.assign(document.createElement("p"), { textContent: line }));
  });
}

/** The appendix and what follows it count toward neither the slide numbers nor the progress bar. reveal.js requires
 * uncounted slides to close the deck. */
function uncountAppendix(slides) {
  const appendix = slides.querySelector(":scope > section#appendix");
  for (let slide = appendix; slide; slide = slide.nextElementSibling) {
    if (slide.tagName === "SECTION") slide.dataset.visibility = "uncounted";
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
    button.hidden = printing || capturing || !slide?.querySelector("[data-play], .fragment");
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
  checkStatuses(slides);
  await timeRanges(slides);
  uncountAppendix(slides);
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
    // The talk's slides numbered n / total; the appendix's A1, A2, … after its unnumbered divider.
    slideNumber: (slide) => {
      if (slide?.dataset.visibility !== "uncounted") return [deck.getSlidePastCount(slide) + 1, "/", deck.getTotalSlides()];
      const appendix = [...slides.querySelectorAll(":scope > section[data-visibility=uncounted]")];
      const k = appendix.indexOf(slide);
      return [k > 0 ? `A${k}` : ""];
    },
    // A still capture has no transition between slides: each capture shows one slide, settled.
    transition: capturing ? "none" : "fade",
    transitionSpeed: "fast",
    pdfSeparateFragments: false,
    plugins: [RevealNotes, RevealHighlight],
    ...options,
  });
  // Past the talk's last slide, in the appendix, the progress bar stays full (deck.css, .in-appendix).
  for (const event of ["ready", "slidechanged"]) {
    deck.on(event, () => slides.closest(".reveal").classList.toggle("in-appendix", deck.getCurrentSlide()?.dataset.visibility === "uncounted"));
  }
  // A slide just entered lands in its state without transitions: syncSlide only updates the current slide, so a slide
  // left at its last state and re-entered would otherwise animate back to its start. While "entering", transitions are
  // off and components skip their glides (deck.css sets --canvit-move and --canvit-glide, which reach their shadow DOM;
  // components also check closest(".entering")). The class leaves two frames later, once the new state has rendered.
  // A still capture keeps every slide "entering", so it waits only for loading, never for animations.
  let entered = null;
  const sync = () => {
    const slide = deck.getCurrentSlide();
    const entering = slide && slide !== entered;
    if (entering) {
      entered = slide;
      slide.classList.add("entering");
    }
    if (slide) syncSlide(slide);
    if (capturing) slide.classList.add("entering");
    else if (entering) {
      slide.getBoundingClientRect();
      requestAnimationFrame(() => requestAnimationFrame(() => slide.classList.remove("entering")));
    }
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
