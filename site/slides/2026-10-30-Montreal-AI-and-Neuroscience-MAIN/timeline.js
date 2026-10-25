// The deep active vision timeline: each model of sources/active-vision-timeline.json at its release date on a year
// axis, its label (name, citation) above or below on a dashed leader, in the style of Yohaï's typst
// timeline (github.com/yberreby/typst-snippets, timeline.typ).
// The slide opens on a short axis with the pioneer alone (RAM, its paper's figures beside it); when the slide shows
// rest, the axis grows to its full length, every mark moving to its place on the full scale (class grows, by
// --grow-dx), and the later models arrive one after another in date order (class later-model, --i), without clicks.
// The file's neuro_ai models (built to compare with brain or behavioral data) are drawn the same way, in the group
// class neuro, which the slide reveals on its own step.
// The axis keeps its width: it first ends late in 2025; when the slide shows canvit, the years squeeze to make room
// for 2026, every mark slides to its place on the new scale (class slides, by --dx), and the 2026 tick (class
// after-only) and CanViT's entry (class canvit, the file's `canvit`) appear.

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 1152, HEIGHT = 556, AXIS_Y = 280, LEFT = 40, RIGHT = 1112;
const START = 2013.5;
const ENDS = { before: 2025.95, after: 2026.4 }; // the year at the axis's end, before and after CanViT arrives
const SHORT = { end: 2015.4, right: 470 }; // the short axis the slide opens on, beside the pioneer's figures
const ROW_OFFSET = { "-3": -235, "-2": -150, "-1": -60, 1: 60, 2: 150 };
const LINE = 25;

function el(name, attributes, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attributes)) node.setAttribute(k, v);
  parent?.append(node);
  return node;
}

function text(content, attributes, parent) {
  const node = el("text", attributes, parent);
  node.textContent = content;
  return node;
}

const yearOf = (date) => Number(date.slice(0, 4)) + (Number(date.slice(5, 7)) - 1 + (Number(date.slice(8, 10)) - 1) / 31) / 12;
const scale = (end) => (year) => LEFT + (year - START) / (end - START) * (RIGHT - LEFT);
const before = scale(ENDS.before), after = scale(ENDS.after);
const short = (year) => LEFT + (year - START) / (SHORT.end - START) * (SHORT.right - LEFT);

/** A group drawn at the year's place on the full axis, which sits at its place on the short axis until it grows. */
function growing(year, parent) {
  const group = el("g", { class: year <= SHORT.end ? "grows" : "grows later" }, parent);
  group.style.setProperty("--grow-dx", `${short(year) - before(year)}px`);
  return group;
}

/** A group drawn at the year's place before CanViT, which slides to its place after. */
function sliding(year, parent) {
  const group = el("g", { class: "slides" }, parent);
  group.style.setProperty("--dx", `${after(year) - before(year)}px`);
  return group;
}

/** An entry that arrives in turn once the axis has grown. */
function laterModel(parent, i) {
  const group = el("g", { class: "entry later-model" }, parent);
  group.style.setProperty("--i", i);
  return group;
}

/** Draw the timeline into container from the models of a sources/active-vision-timeline.json file. */
export async function drawTimeline(container, { src = "sources/active-vision-timeline.json" } = {}) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`timeline: ${src} answered ${response.status}`);
  const { models, neuro_ai: neuroAi, canvit } = await response.json();
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img", "aria-label": "Deep active vision models from 2014 to 2025" });

  const end = RIGHT + 18;
  const years = []; // [year, x, shown before CanViT], labeled after the entries so their leaders pass behind
  svg.style.setProperty("--axis-k", `${(SHORT.right + 18 - LEFT) / (end - LEFT)}`);
  svg.style.setProperty("--head-dx", `${SHORT.right - RIGHT}px`);
  el("path", { d: `M${LEFT},${AXIS_Y}H${end}`, class: "axis axis-line" }, svg);
  el("path", { d: `M${end - 12},${AXIS_Y - 9}L${end},${AXIS_Y}L${end - 12},${AXIS_Y + 9}`, class: "axis axis-head" }, svg);
  for (let year = Math.ceil(START); year <= Math.floor(ENDS.after); year++) {
    const shownBefore = year <= ENDS.before;
    const parent = shownBefore ? sliding(year, growing(year, svg)) : el("g", { class: "after-only" }, svg);
    const tx = shownBefore ? before(year) : after(year);
    el("line", { x1: tx, x2: tx, y1: AXIS_Y - 6, y2: AXIS_Y + 6, class: "tick" }, parent);
    if (year % 2 === 0) years.push([year, tx, shownBefore]);
  }

  const brand = el("linearGradient", { id: "timeline-brand" }, el("defs", {}, svg));
  el("stop", { offset: "0", "stop-color": "var(--canvas)" }, brand);
  el("stop", { offset: "1", "stop-color": "var(--glimpse)" }, brand);
  let later = 0; // the order in which the later models arrive
  for (const model of [...models, ...neuroAi.map((m) => ({ ...m, neuro: true })), { ...canvit, ours: true }]) {
    if (!(String(model.row) in ROW_OFFSET)) throw new Error(`timeline: ${model.name} has row ${model.row}`);
    const year = yearOf(model.date);
    const mx = model.ours ? after(year) : before(year), above = model.row < 0, ly = AXIS_Y + ROW_OFFSET[model.row];
    // Labels near an edge grow inward, so they stay on the slide.
    const anchor = mx < 200 ? "start" : mx > WIDTH - 200 ? "end" : "middle";
    const tx = { start: mx - 12, end: mx + 12, middle: mx }[anchor];
    const top = above ? ly - LINE - 12 : ly + 22; // the name's baseline
    const entry = model.ours ? el("g", { class: "entry canvit" }, svg)
      : model.neuro ? sliding(year, growing(year, el("g", { class: "entry neuro" }, svg)))
      : model.pioneer ? sliding(year, growing(year, el("g", { class: "entry pioneer" }, svg)))
      : sliding(year, growing(year, laterModel(svg, later++)));
    el("line", { x1: mx, x2: mx, y1: AXIS_Y + (above ? -10 : 10), y2: above ? ly + 4 : ly, class: "leader" }, entry);
    [[model.name, "name"], [model.cite, "cite"]]
      .forEach(([content, kind], i) => text(content, { x: tx, y: top + i * LINE, "text-anchor": anchor, class: kind }, entry));
    if (model.pioneer) el("circle", { cx: mx, cy: AXIS_Y, r: 20, class: "pioneer-ring" }, entry);
    el("circle", { cx: mx, cy: AXIS_Y, r: model.ours ? 14 : 9,
                   class: model.ours ? "marker ours" : model.neuro ? "marker neuro" : "marker" }, entry);
  }
  for (const [year, tx, shownBefore] of years) {
    const parent = shownBefore ? sliding(year, growing(year, svg)) : el("g", { class: "after-only" }, svg);
    text(year, { x: tx, y: AXIS_Y + 28, "text-anchor": "middle", class: "year" }, parent);
  }
  container.replaceChildren(svg);
}
