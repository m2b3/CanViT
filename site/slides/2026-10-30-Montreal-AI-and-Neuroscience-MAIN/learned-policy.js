// The learned-policy slide: ADE20K mIoU over the first five glimpses for the paper's policies on the 64 x 64 canvas
// (assets/paper/ade20k_seg.json) and for a viewing policy learned on the frozen CanViT-B (sources/learned-policy.json:
// the mean over 8 training seeds, the band one standard deviation). The baselines are drawn first; the learned
// policy appears when the slide shows "learned".

import { POLICIES } from "../../js/canvit/policies.js";

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 760, HEIGHT = 470, M = { left: 70, right: 150, top: 20, bottom: 56 };
const Y = [39, 45.5];
const BASELINES = ["full_then_random", "coarse_to_fine", "entropy_coarse_to_fine"];
const CANVAS_GRID = 64;

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

export async function drawLearnedPolicy(container, { paper, learned }) {
  const [curves, ours] = await Promise.all([paper, learned].map(async (src) => {
    const response = await fetch(src);
    if (!response.ok) throw new Error(`learned-policy: ${src} answered ${response.status}`);
    return response.json();
  }));
  const n = ours.t.length;
  const x = (t) => M.left + (t / (n - 1)) * (WIDTH - M.left - M.right);
  const y = (miou) => HEIGHT - M.bottom - ((miou - Y[0]) / (Y[1] - Y[0])) * (HEIGHT - M.top - M.bottom);
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img",
                          "aria-label": "ADE20K mIoU over five glimpses: a learned policy against the paper's policies" });

  for (let v = Math.ceil(Y[0]); v <= Y[1]; v++) {
    el("line", { x1: M.left, x2: WIDTH - M.right, y1: y(v), y2: y(v), class: "grid" }, svg);
    text(v, { x: M.left - 12, y: y(v) + 6, "text-anchor": "end", class: "tick" }, svg);
  }
  for (const t of ours.t) text(t + 1, { x: x(t), y: HEIGHT - M.bottom + 28, "text-anchor": "middle", class: "tick" }, svg);
  text("Glimpses", { x: (M.left + WIDTH - M.right) / 2, y: HEIGHT - 6, "text-anchor": "middle", class: "axis-title" }, svg);
  text("ADE20K mIoU (%)", { x: -(M.top + HEIGHT - M.bottom) / 2, y: 18, transform: "rotate(-90)", "text-anchor": "middle",
                            class: "axis-title" }, svg);

  const line = (points) => points.map(([t, v], i) => `${i ? "L" : "M"}${x(t)},${y(v)}`).join("");
  const labels = [];
  for (const name of BASELINES) {
    const curve = curves.policy_curves.find((c) => c.policy === name && c.canvas_grid === CANVAS_GRID);
    if (!curve) throw new Error(`learned-policy: no ${name} curve at canvas ${CANVAS_GRID} in ${paper}`);
    const points = ours.t.map((t) => [t, 100 * curve.per_timestep.find((p) => p.t === t).mean]);
    const policy = POLICIES[name];
    el("path", { d: line(points), class: "curve", stroke: policy.color }, svg);
    labels.push({ v: points.at(-1)[1], label: policy.label, color: policy.color, group: svg });
  }
  const learnedGroup = el("g", { class: "learned" }, svg);
  const hi = ours.t.map((t, i) => [t, ours.miou_mean[i] + ours.miou_sd[i]]);
  const lo = ours.t.map((t, i) => [t, ours.miou_mean[i] - ours.miou_sd[i]]).reverse();
  el("path", { d: `${line(hi)}${line(lo).replace("M", "L")}Z`, class: "band" }, learnedGroup);
  el("path", { d: line(ours.t.map((t, i) => [t, ours.miou_mean[i]])), class: "curve ours" }, learnedGroup);
  labels.push({ v: ours.miou_mean.at(-1), label: "Learned", color: "var(--policy)", group: learnedGroup, ours: true });

  // End labels, pushed apart where curves end close together.
  labels.sort((a, b) => b.v - a.v);
  let last = -Infinity;
  for (const item of labels) {
    const ly = Math.max(y(item.v) + 7, last + 26);
    last = ly;
    text(item.label, { x: WIDTH - M.right + 14, y: ly, fill: item.color, class: item.ours ? "end-label ours" : "end-label" }, item.group);
  }
  container.replaceChildren(svg);
}
