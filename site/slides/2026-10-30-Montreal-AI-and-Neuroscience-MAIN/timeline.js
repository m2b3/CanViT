// The deep active vision timeline: each model of sources/active-vision-timeline.json at its release date on a year
// axis, its label (name, what it did, citation) above or below on a dashed leader, in the style of Yohaï's typst
// timeline (github.com/yberreby/typst-snippets, timeline.typ). Every label is a fragment, so the models arrive one per
// click in date order. Markers of models that learn where to look by reinforcement learning carry the class rl.

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 1152, HEIGHT = 556, AXIS_Y = 280, LEFT = 40, RIGHT = 1112;
const YEARS = [2013.5, 2026.4];
const ROW_OFFSET = { "-2": -150, "-1": -60, 1: 60, 2: 150 };
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
const x = (year) => LEFT + (year - YEARS[0]) / (YEARS[1] - YEARS[0]) * (RIGHT - LEFT);

/** Draw the timeline into container from the models of a sources/active-vision-timeline.json file. */
export async function drawTimeline(container, { src = "sources/active-vision-timeline.json" } = {}) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`timeline: ${src} answered ${response.status}`);
  const { models } = await response.json();
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img", "aria-label": "Deep active vision models from 2014 to 2025" });

  el("path", { d: `M${LEFT},${AXIS_Y}H${RIGHT + 18}M${RIGHT + 8},${AXIS_Y - 9}L${RIGHT + 20},${AXIS_Y}L${RIGHT + 8},${AXIS_Y + 9}`, class: "axis" }, svg);
  for (let year = Math.ceil(YEARS[0]); year <= Math.floor(YEARS[1]); year++) {
    el("line", { x1: x(year), x2: x(year), y1: AXIS_Y - 6, y2: AXIS_Y + 6, class: "tick" }, svg);
    if (year % 2 === 0) text(year, { x: x(year), y: AXIS_Y + 28, "text-anchor": "middle", class: "year" }, svg);
  }

  for (const model of models) {
    if (!(String(model.row) in ROW_OFFSET)) throw new Error(`timeline: ${model.name} has row ${model.row}`);
    const mx = x(yearOf(model.date)), above = model.row < 0, ly = AXIS_Y + ROW_OFFSET[model.row];
    // Labels near an edge grow inward, so they stay on the slide.
    const anchor = mx < 200 ? "start" : mx > WIDTH - 200 ? "end" : "middle";
    const tx = { start: mx - 12, end: mx + 12, middle: mx }[anchor];
    const top = above ? ly - 2 * LINE - 12 : ly + 22; // the name's baseline
    const entry = el("g", { class: "fragment entry" }, svg);
    el("line", { x1: mx, x2: mx, y1: AXIS_Y + (above ? -10 : 10), y2: above ? ly + 4 : ly, class: "leader" }, entry);
    [[model.name, "name"], [model.did, "did"], [model.cite, "cite"]]
      .forEach(([content, kind], i) => text(content, { x: tx, y: top + i * LINE, "text-anchor": anchor, class: kind }, entry));
    el("circle", { cx: mx, cy: AXIS_Y, r: 9, class: model.policy === "rl" ? "marker rl" : "marker" }, entry);
  }
  container.replaceChildren(svg);
}
