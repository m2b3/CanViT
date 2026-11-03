// A slide's build state from its shown fragments, shared by the deck (deck.js) and by pages that show slides outside a
// deck (../js/talk-slide.js). The conventions are deck.js's: data-shows, data-canvit-target, data-lines, data-play.

const initialValues = new WeakMap(); // target -> {name: its markup's value}

/** The attributes a data-canvit-target fragment sets: its data-canvit-NAME attributes other than the target. */
const targetAttributes = (fragment) =>
  [...fragment.attributes].filter((a) => a.name.startsWith("data-canvit-") && a.name !== "data-canvit-target")
    .map((a) => [a.name.slice("data-canvit-".length), a.value]);

/** Every data-canvit-target fragment of the slide sets its attributes on its targets; none shown restores the markup's
 * values. */
export function syncTargets(slide) {
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
export function syncStates(slide) {
  for (const fragment of slide.querySelectorAll(".fragment[data-shows]")) {
    slide.classList.toggle(`shows-${fragment.dataset.shows}`, fragment.classList.contains("visible"));
  }
}

/** The last shown data-lines fragment of the slide puts a band behind those lines of the slide's <pre>. */
export function syncLines(slide) {
  const fragments = [...slide.querySelectorAll(".fragment[data-lines]")];
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

/** The slide's state from its shown fragments: targets, named states and marked code lines. */
export function syncSlide(slide) {
  syncTargets(slide);
  syncStates(slide);
  syncLines(slide);
}

/** Whether an element with data-play plays in the slide's current state ("NAME": only while the slide shows NAME;
 * "!NAME": only while it does not). */
export function playsNow(element, slide) {
  const state = element.dataset.play;
  const shows = (name) => slide.classList.contains(`shows-${name}`);
  return !state || (state.startsWith("!") ? !shows(state.slice(1)) : shows(state));
}

/** Fills each <code data-src> under root with its file's text. */
export async function fillCode(root) {
  for (const code of root.querySelectorAll("code[data-src]")) {
    const response = await fetch(code.dataset.src);
    if (!response.ok) throw new Error(`<code data-src="${code.dataset.src}">: HTTP ${response.status}`);
    code.textContent = (await response.text()).trimEnd();
  }
}
