// The deep active vision timeline: each model of sources/active-vision-timeline.json at its release date on a year
// axis, its label above or below on a dashed leader (style of github.com/yberreby/typst-snippets, timeline.typ).
// The slide opens on a short axis with RAM alone; on "rest" the axis grows (class grows, --grow-dx) and the later
// models arrive in date order (class later-model, --i); the neuro_ai models form the group class neuro, revealed on
// its own step.

const SRC = "sources/active-vision-timeline.json";
const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 1152, HEIGHT = 556, AXIS_Y = 280, LEFT = 40, RIGHT = 1112;
const START = 2013.5;
const END = 2025.95; // the year at the axis's end
const SHORT = { end: 2015.4, right: 470 }; // the short axis the slide opens on, beside the pioneer's figures
const ROW_OFFSET = { "-3": -235, "-2": -150, "-1": -60, 1: 60, 2: 135, 3: 210 };
const LINE = 25;
const HEAD_ROOM = 34; // the axis runs past its last year, so the arrowhead clears the last marker

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
const full = (year) => LEFT + (year - START) / (END - START) * (RIGHT - LEFT);
const short = (year) => LEFT + (year - START) / (SHORT.end - START) * (SHORT.right - LEFT);

/** A group drawn at the year's place on the full axis, which sits at its place on the short axis until it grows. */
function growing(year, parent) {
  const group = el("g", { class: year <= SHORT.end ? "grows" : "grows later" }, parent);
  group.style.setProperty("--grow-dx", `${short(year) - full(year)}px`);
  return group;
}

/** An entry that arrives in turn once the axis has grown. */
function laterModel(parent, i) {
  const group = el("g", { class: "entry later-model" }, parent);
  group.style.setProperty("--i", i);
  return group;
}

/** Draw the timeline into container from the models of SRC. */
export async function drawTimeline(container) {
  const response = await fetch(SRC);
  if (!response.ok) throw new Error(`timeline: ${SRC} answered ${response.status}`);
  const { models, neuro_ai: neuroAi } = await response.json();
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img", "aria-label": "Deep active vision models from 2014 to 2025" });

  const end = RIGHT + HEAD_ROOM;
  const years = []; // [year, x], labeled after the entries so their leaders pass behind
  svg.style.setProperty("--axis-k", `${(SHORT.right + HEAD_ROOM - LEFT) / (end - LEFT)}`);
  svg.style.setProperty("--head-dx", `${SHORT.right - RIGHT}px`);
  el("path", { d: `M${LEFT},${AXIS_Y}H${end}`, class: "axis axis-line" }, svg);
  el("path", { d: `M${end - 12},${AXIS_Y - 9}L${end},${AXIS_Y}L${end - 12},${AXIS_Y + 9}`, class: "axis axis-head" }, svg);
  for (let year = Math.ceil(START); year <= END; year++) {
    const tx = full(year);
    el("line", { x1: tx, x2: tx, y1: AXIS_Y - 6, y2: AXIS_Y + 6, class: "tick" }, growing(year, svg));
    if (year % 2 === 0) years.push([year, tx]);
  }

  let later = 0; // the order in which the later models arrive
  for (const model of [...models, ...neuroAi.map((m) => ({ ...m, neuro: true }))]) {
    if (!(String(model.row) in ROW_OFFSET)) throw new Error(`timeline: ${model.name} has row ${model.row}`);
    const year = yearOf(model.date);
    const mx = full(year), above = model.row < 0, ly = AXIS_Y + ROW_OFFSET[model.row];
    // Labels near an edge grow inward, so they stay on the slide.
    const anchor = mx < 200 ? "start" : mx > WIDTH - 200 ? "end" : "middle";
    const tx = { start: mx - 12, end: mx + 12, middle: mx }[anchor];
    const top = above ? ly - LINE - 12 : ly + 22; // the name's baseline
    const entry = model.neuro ? growing(year, el("g", { class: "entry neuro" }, svg))
      : model.pioneer ? growing(year, el("g", { class: "entry pioneer" }, svg))
      : growing(year, laterModel(svg, later++));
    el("line", { x1: mx, x2: mx, y1: AXIS_Y + (above ? -10 : 10), y2: above ? ly + 4 : ly, class: "leader" }, entry);
    [[model.name, "name"], [model.cite, "cite"]]
      .forEach(([content, kind], i) => text(content, { x: tx, y: top + i * LINE, "text-anchor": anchor, class: kind }, entry));
    if (model.pioneer) el("circle", { cx: mx, cy: AXIS_Y, r: 20, class: "pioneer-ring" }, entry);
    el("circle", { cx: mx, cy: AXIS_Y, r: 9, class: model.neuro ? "marker neuro" : "marker" }, entry);
  }
  for (const [year, tx] of years) {
    text(year, { x: tx, y: AXIS_Y + 28, "text-anchor": "middle", class: "year" }, growing(year, svg));
  }
  container.replaceChildren(svg);
}
