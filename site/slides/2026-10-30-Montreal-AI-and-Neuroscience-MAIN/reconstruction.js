// The reconstruction slide's chart: cosine similarity between CanViT's guess and DINOv3's target, over all patches
// and over the patches no glimpse ever covered, against the number of glimpses on a log axis (the export's
// cosine.json, throwaway/distillation). It grows with the slide's <deck-sequence>: drawn up to the glimpse shown.
// The vertical axis is spaced by -log(1 - cosine), labeled in cosine, so gains near 1 stay visible and closer is up.

const SVG = "http://www.w3.org/2000/svg";
const W = 1000, H = 210, M = { left: 70, right: 190, top: 34, bottom: 36 };
const Y_RANGE = [0.7, 0.95];
const Y_TICKS = [0.7, 0.8, 0.9, 0.95];
const spaced = (cosine) => -Math.log(1 - cosine);

function el(name, attributes, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attributes)) node.setAttribute(k, v);
  parent?.append(node);
  return node;
}

export async function drawReconstruction(section, src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`reconstruction: ${src} answered ${response.status}`);
  const { all, never_seen: neverSeen } = await response.json();
  const count = all.length;
  const x = (glimpses) => M.left + Math.log(glimpses) / Math.log(count) * (W - M.left - M.right);
  const [low, high] = Y_RANGE.map(spaced);
  const y = (cosine) => H - M.bottom - (spaced(cosine) - low) / (high - low) * (H - M.top - M.bottom);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Cosine similarity to the teacher against glimpses" });
  for (const v of Y_TICKS) {
    el("line", { x1: M.left, x2: W - M.right, y1: y(v), y2: y(v), class: "grid" }, svg);
    el("text", { x: M.left - 10, y: y(v) + 6, "text-anchor": "end" }, svg).textContent = String(v);
  }
  el("text", { x: M.left, y: 12, "text-anchor": "start", class: "axis-title" }, svg).textContent = "similarity to the target";
  for (const g of [1, 2, 5, 10, 20].filter((g) => g <= count)) {
    el("text", { x: x(g), y: H - 10, "text-anchor": "middle" }, svg).textContent = g;
  }
  el("text", { x: (M.left + W - M.right) / 2, y: H + 12, "text-anchor": "middle", class: "axis-title" }, svg).textContent = "glimpses";
  const lines = [["all", all, "all patches"], ["never-seen", neverSeen, "never in a glimpse"]].map(([name, values, label]) => {
    const path = el("path", { class: `line ${name}` }, svg);
    const head = el("circle", { r: 6, class: `head ${name}` }, svg);
    const text = el("text", { class: `label ${name}` }, svg);
    text.textContent = label;
    return { values, path, head, text };
  });
  section.querySelector(".cosine-chart").replaceChildren(svg);

  const show = (index) => {
    for (const { values, path, head, text } of lines) {
      const points = values.slice(0, index + 1).map((v, i) => [x(i + 1), y(v)]);
      path.setAttribute("d", points.map(([px, py], i) => `${i ? "L" : "M"}${px},${py}`).join(""));
      const [hx, hy] = points.at(-1);
      head.setAttribute("cx", hx);
      head.setAttribute("cy", hy);
      text.setAttribute("x", hx + 12);
      text.setAttribute("y", hy + 6);
    }
  };
  const sequence = section.querySelector("deck-sequence");
  if (sequence.children.length !== count) throw new Error(`reconstruction: ${sequence.children.length} frames, ${count} cosines`);
  sequence.addEventListener("deck-sequence-change", (event) => show(event.detail.index));
  show(0);
}
