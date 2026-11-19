// The static "Pretraining policies" slide: for a scene of experiments/pretraining's rollouts.json, a row per pretraining
// policy: the scene with every glimpse's box numbered by its step, then the glimpses themselves (crops.py's images,
// as the model receives them), numbered, in the policy's color from js/canvit/policies.js.

import { POLICIES } from "../../js/canvit/policies.js";

const SVG = "http://www.w3.org/2000/svg";

function el(name, attributes, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attributes)) node.setAttribute(k, v);
  parent?.append(node);
  return node;
}

export async function drawPretrainingGrid(container, src, sceneName, policies) {
  const response = await fetch(`${src}/rollouts.json`);
  if (!response.ok) throw new Error(`pretraining grid: ${src}/rollouts.json answered ${response.status}`);
  const { scenes } = await response.json();
  const scene = scenes.find((s) => s.name === sceneName);
  if (!scene) throw new Error(`pretraining grid: no scene ${sceneName} in ${src}`);
  const rows = policies.map((policy) => {
    const viewpoints = scene.rollouts[policy];
    if (!viewpoints || !POLICIES[policy]) throw new Error(`pretraining grid: no ${policy} rollout for ${sceneName}`);
    const row = document.createElement("div");
    row.className = `grid-row ${policy}`;
    row.style.setProperty("--policy-color", POLICIES[policy].color);
    const label = Object.assign(document.createElement("p"), { className: "label" });
    // The pretraining policies differ only in their first glimpse, the word before "-then-IID": it stands out.
    const [first, rest] = POLICIES[policy].name.split(/(?=-then-)/);
    if (!rest) throw new Error(`pretraining grid: policy name "${POLICIES[policy].name}" has no "-then-"`);
    const name = document.createElement("span");
    name.append(Object.assign(document.createElement("em"), { textContent: first }), rest);
    label.append(Object.assign(document.createElement("b"), { textContent: POLICIES[policy].label }), name);
    const frame = Object.assign(document.createElement("div"), { className: "scene" });
    frame.append(Object.assign(document.createElement("img"), { src: `${src}/${scene.image}`, alt: "" }));
    const svg = el("svg", { viewBox: "-1 -1 2 2" }, frame);
    viewpoints.forEach(([r, c, s]) => {
      el("rect", { x: c - s, y: r - s, width: 2 * s, height: 2 * s, class: "halo" }, svg);
      el("rect", { x: c - s, y: r - s, width: 2 * s, height: 2 * s }, svg);
    });
    // The glimpses' numbers over every box, each in its box's corner, moved along when it would cover an earlier one.
    const placed = [];
    viewpoints.forEach(([r, c, s], i) => {
      let [x, y] = [c - s + 0.04, r - s + 0.13];
      while (placed.some(([px, py]) => Math.abs(px - x) < 0.12 && Math.abs(py - y) < 0.12)) x += 0.13;
      placed.push([x, y]);
      el("text", { x, y }, svg).textContent = i + 1;
    });
    const glimpses = Object.assign(document.createElement("div"), { className: "glimpses" });
    viewpoints.forEach((_, i) => {
      const figure = document.createElement("figure");
      figure.append(Object.assign(document.createElement("img"), { src: `${src}/crops/${sceneName}-${policy}-${i + 1}.png`, alt: `Glimpse ${i + 1}` }),
                    Object.assign(document.createElement("figcaption"), { textContent: i + 1 }));
      glimpses.append(figure);
    });
    row.append(label, frame, glimpses);
    return row;
  });
  container.replaceChildren(...rows);
}
