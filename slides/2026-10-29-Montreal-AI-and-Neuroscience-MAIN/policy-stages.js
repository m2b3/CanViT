// #c2f-and-f2c: each policy's rollout as a row of stages, a stage being a run of consecutive glimpses of the same size
// (a quadtree level): the scene with that stage's boxes, labeled with its glimpses, arrows between stages. Viewpoints
// from the recorded bundles' manifests; labels and colors from js/canvit/policies.js.

import { POLICIES } from "../../js/canvit/policies.js";

const SVG = "http://www.w3.org/2000/svg";

function el(name, attributes, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent.append(node);
  return node;
}

/** Runs of consecutive glimpses whose boxes have the same size: [{first, last, boxes}], glimpses counted from 1. */
function stages(glimpses) {
  const runs = [];
  glimpses.forEach(({ box }, i) => {
    const run = runs.at(-1);
    if (run && Math.abs(run.boxes[0].size - box.size) < 1e-6) {
      run.boxes.push(box);
      run.last = i + 1;
    } else runs.push({ first: i + 1, last: i + 1, boxes: [box] });
  });
  return runs;
}

export async function drawPolicyStages(container, bundles) {
  const rows = await Promise.all(bundles.map(async (src) => {
    const response = await fetch(`${src}/manifest.json`);
    if (!response.ok) throw new Error(`policy stages: ${src}/manifest.json answered ${response.status}`);
    const manifest = await response.json();
    const policy = POLICIES[manifest.policy.name];
    if (!policy) throw new Error(`policy stages: ${src}: policy "${manifest.policy.name}" is not in js/canvit/policies.js`);
    const row = document.createElement("div");
    row.className = "stage-row";
    row.style.setProperty("--policy-color", policy.color);
    row.append(Object.assign(document.createElement("p"), { className: "label", textContent: policy.label }));
    stages(manifest.glimpses).forEach((stage, i) => {
      if (i > 0) row.append(Object.assign(document.createElement("span"), { className: "arrow" }));
      const figure = document.createElement("figure");
      const frame = Object.assign(document.createElement("div"), { className: "frame" });
      frame.append(Object.assign(document.createElement("img"), { src: `${src}/${manifest.scene.image}`, alt: "" }));
      const svg = el("svg", { viewBox: "0 0 1 1", preserveAspectRatio: "none" }, frame);
      // Boxes lie on their bounds, so neighbors share an edge; the stroke thins with the glimpse (CSS --stroke, px).
      svg.style.setProperty("--stroke", (2 + 3 * stage.boxes[0].size).toFixed(2));
      const rect = ({ top, left, size }, className) => el("rect", { class: className, x: left, y: top, width: size, height: size }, svg);
      for (const box of stage.boxes) rect(box, "halo");
      for (const box of stage.boxes) rect(box, "box");
      const caption = stage.first === stage.last ? `Glimpse ${stage.first}` : `Glimpses ${stage.first}–${stage.last}`;
      figure.append(frame, Object.assign(document.createElement("figcaption"), { textContent: caption }));
      row.append(figure);
    });
    return row;
  }));
  container.replaceChildren(...rows);
}
