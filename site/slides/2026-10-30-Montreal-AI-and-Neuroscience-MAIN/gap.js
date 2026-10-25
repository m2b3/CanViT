// The gap on ADE20K, as bars of mIoU. The paper's export (assets/paper/ade20k_seg.json) supplies the frozen DINOv3
// ViT-B/16 read by the paper's linear probe at 512 px (probe_table), DINOv3 ViT-S/16 likewise, and the prior active
// models (baselines, each at its best glimpse count); sources/ade20k-passive-references.json supplies other passive
// results, each with its source (those marked shown). The bars state shows the active models alone, on an axis zoomed to
// ZOOM_MAX; the "more" state brings in the passive models: the axis rescales to MAX, the active bars shrink with it,
// and every row glides to its place, active models first, then passive ones, each group by value.

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 1000, ROW = 46, LABEL_W = 330, BAR_W = 560, TOP = 10, GROUP_GAP = 24;
const MAX = 70, ZOOM_MAX = 30;

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

async function json(src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`gap: ${src} answered ${response.status}`);
  return response.json();
}

function bestPrior(baselines, name) {
  const runs = baselines.filter((b) => b.name === name);
  if (!runs.length) throw new Error(`gap: no baseline ${name}`);
  return runs.reduce((a, b) => (b.miou_pct > a.miou_pct ? b : a));
}

function probe(table, model) {
  const row = table.find((e) => e.model === model && e.input_px === 512);
  if (!row) throw new Error(`gap: no ${model} at 512 px in the paper's probe_table`);
  return row.miou_pct;
}

export async function drawGap(container, { paper, references }) {
  const [data, refs] = await Promise.all([json(paper), json(references)]);
  const prior = (name, label, core) => {
    const run = bestPrior(data.baselines, name);
    return { name: label, note: `active, ${run.num_glimpses} glimpses`, value: run.miou_pct, kind: "active", core };
  };
  const rows = [
    prior("AME (MAE)", "AME (MAE ViT-L)", true),
    prior("AdaGlimpse", "AdaGlimpse", true),
    prior("AME (SETR)", "AME (SETR ViT-L)", true),
    { name: "DINOv3 ViT-S", note: "passive, frozen, linear", value: probe(data.probe_table, "DINOv3 ViT-S/16"), kind: "passive", core: false },
    { name: "DINOv3 ViT-B", note: "passive, frozen, linear", value: probe(data.probe_table, "DINOv3 ViT-B/16"), kind: "passive", core: false },
    ...refs.references.filter((r) => r.shown).map((r) => ({ name: r.name, note: r.note, value: r.value, kind: "passive", core: false })),
  ];
  if (rows.some((r) => r.core && r.value > ZOOM_MAX)) throw new Error(`gap: an active model exceeds the zoomed axis (${ZOOM_MAX})`);
  const order = (subset) => {
    const sorted = (kind) => subset.filter((r) => r.kind === kind).sort((a, b) => a.value - b.value);
    const placed = new Map();
    let y = TOP;
    for (const [g, kind] of ["active", "passive"].entries()) {
      if (g) y += GROUP_GAP;
      for (const r of sorted(kind)) { placed.set(r, y); y += ROW; }
    }
    return { placed, height: y };
  };
  const all = order(rows), core = order(rows.filter((r) => r.core));
  const x = (v) => LABEL_W + (v / MAX) * BAR_W;
  const height = all.height + 40;
  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${height}`, role: "img", "aria-label": "ADE20K mIoU of passive and active models" });
  // Zoomed, every horizontal position from the axis origin is stretched by MAX / ZOOM_MAX.
  const stretch = MAX / ZOOM_MAX;
  const shift = (px) => `translateX(${(stretch - 1) * px}px)`;
  for (const v of [0, 10, 20, 30, 40, 50, 60, 70]) {
    const tick = el("g", { class: v > ZOOM_MAX ? "tick-group beyond" : "tick-group" }, svg);
    tick.style.setProperty("--zoom", shift(x(v) - x(0)));
    el("line", { x1: x(v), x2: x(v), y1: TOP - 4, y2: all.height, class: "grid" }, tick);
    text(v, { x: x(v), y: all.height + 26, "text-anchor": "middle", class: "tick" }, tick);
  }
  for (const row of rows) {
    const group = el("g", { class: `row ${row.kind}${row.core ? " core" : ""}` }, svg);
    group.style.setProperty("--y-all", `${all.placed.get(row)}px`);
    group.style.setProperty("--y-core", `${row.core ? core.placed.get(row) : all.placed.get(row)}px`);
    text(row.name, { x: LABEL_W - 14, y: 20, "text-anchor": "end", class: "name" }, group);
    text(row.note, { x: LABEL_W - 14, y: 38, "text-anchor": "end", class: "note" }, group);
    el("rect", { x: x(0), y: 4, width: x(row.value) - x(0), height: ROW - 12, rx: 5, class: "bar" }, group)
      .style.setProperty("--zoom", `scaleX(${stretch})`);
    text(row.value.toFixed(1), { x: x(row.value) + 10, y: 28, class: "value" }, group)
      .style.setProperty("--zoom", shift(x(row.value) - x(0)));
  }
  container.replaceChildren(svg);
}
