// What learning where to look bought, on "What's in an active-vision model?": prior active models' own ablations
// (sources/policy-ablations.json), each a row from the accuracy with random viewpoints to the accuracy with the paper's
// chosen ones. The rows come from different benchmarks, so each starts at the same x and only the gain compares, on one
// scale of PX_PER_POINT; each value under its dot, the gain in points at the right.

const NS = "http://www.w3.org/2000/svg";
const WIDTH = 1100, ROW = 76, TOP = 30, START = 560;
const PX_PER_POINT = 140; // accuracy points, the same for every row

function el(name, attributes, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent.append(node);
  return node;
}

function text(content, attributes, parent) {
  const node = el("text", attributes, parent);
  node.textContent = content;
  return node;
}

export async function drawPolicyAblations(container, src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`policy ablations: ${src} answered ${response.status}`);
  const { ablations } = await response.json();
  const height = TOP + ablations.length * ROW;
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${height}`, role: "img",
                          "aria-label": "Accuracy with random viewpoints against chosen ones, in two prior active models" }, container);
  ablations.forEach((a, i) => {
    if (!(a.chosen > a.random && a.chosen - a.random < 3)) throw new Error(`policy ablations: ${a.model}'s gain is off the scale`);
    const x = (value) => START + (value - a.random) * PX_PER_POINT;
    const y = TOP + i * ROW + ROW / 2;
    const row = el("g", { class: "row", style: `--row: ${i}` }, svg);
    text(a.model, { x: 0, y: y - 6, class: "model" }, row);
    text(`${a.task} · ${a.citation}`, { x: 0, y: y + 22, class: "where" }, row);
    el("line", { x1: x(a.random), x2: x(a.chosen), y1: y, y2: y, class: "span" }, row);
    el("circle", { cx: x(a.random), cy: y, r: 11, class: "random" }, row);
    el("circle", { cx: x(a.chosen), cy: y, r: 11, class: "chosen" }, row);
    text(`${a.random.toFixed(1)}%`, { x: x(a.random), y: y + 32, "text-anchor": "middle", class: "value" }, row);
    text(`${a.chosen.toFixed(1)}%`, { x: x(a.chosen), y: y + 32, "text-anchor": "middle", class: "value" }, row);
    text(`+${(a.chosen - a.random).toFixed(1)} points`, { x: x(a.chosen) + 24, y: y + 9, class: "delta" }, row);
    if (i === 0) {
      text("random", { x: x(a.random), y: y - 22, "text-anchor": "middle", class: "key random-key" }, row);
      text("chosen", { x: x(a.chosen), y: y - 22, "text-anchor": "middle", class: "key chosen-key" }, row);
    }
  });
}
