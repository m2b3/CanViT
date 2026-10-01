// Whether the uncertainty read out from the canvas tracks the segmentation's errors: accuracy of the decoded class in
// each tenth of canvas cells ranked by entropy, over ADE20K validation, from throwaway/metacognition/calibration.py's
// JSON, after the full-scene glimpse (1) and after C2F's first five glimpses (5).

const NS = "http://www.w3.org/2000/svg";
const WIDTH = 1100, HEIGHT = 470;
const PLOT = { left: 80, right: 1080, top: 20, bottom: 400 };
const SERIES = [{ glimpses: "1", label: "after the full-scene glimpse" }, { glimpses: "5", label: "after five C2F glimpses" }];

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

export async function drawUncertaintyCalibration(container, src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`uncertainty calibration: ${src} answered ${response.status}`);
  const { by_glimpses: byGlimpses } = await response.json();
  const series = SERIES.map((s) => {
    if (!(s.glimpses in byGlimpses)) throw new Error(`uncertainty calibration: no ${s.glimpses}-glimpse result in ${src}`);
    return { ...s, ...byGlimpses[s.glimpses] };
  });
  const deciles = series[0].accuracy_by_entropy_decile.length;
  const band = (PLOT.right - PLOT.left) / deciles, bar = band * 0.36;
  const y = (accuracy) => PLOT.bottom - accuracy * (PLOT.bottom - PLOT.top);
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img",
                          "aria-label": "Segmentation accuracy in each tenth of canvas cells, from the surest to the least sure" }, container);
  for (const value of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: PLOT.left, x2: PLOT.right, y1: y(value), y2: y(value), class: "grid" }, svg);
    text(`${100 * value}%`, { x: PLOT.left - 10, y: y(value) + 6, "text-anchor": "end", class: "tick" }, svg);
  }
  series.forEach((s, k) => {
    s.accuracy_by_entropy_decile.forEach((accuracy, d) => {
      const x = PLOT.left + d * band + band / 2 + (k - 0.5) * bar;
      el("rect", { x: x - bar / 2, y: y(accuracy), width: bar, height: PLOT.bottom - y(accuracy), class: `bar series-${k}` }, svg);
    });
  });
  text("surest tenth of cells", { x: PLOT.left + band / 2, y: PLOT.bottom + 30, "text-anchor": "middle", class: "tick" }, svg);
  text("least sure tenth", { x: PLOT.right - band / 2, y: PLOT.bottom + 30, "text-anchor": "middle", class: "tick" }, svg);
  text("canvas cells ranked by the entropy of their decoded classes →", { x: (PLOT.left + PLOT.right) / 2, y: PLOT.bottom + 62,
                                                                          "text-anchor": "middle", class: "title" }, svg);
  series.forEach((s, k) => {
    const key = el("g", { class: "key" }, svg);
    el("rect", { x: PLOT.right - 330, y: PLOT.top + 10 + k * 30, width: 18, height: 18, class: `bar series-${k}` }, key);
    text(s.label, { x: PLOT.right - 302, y: PLOT.top + 25 + k * 30, class: "tick" }, key);
  });
}
