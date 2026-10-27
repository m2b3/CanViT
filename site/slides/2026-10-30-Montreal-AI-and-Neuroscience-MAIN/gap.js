// The gap on ADE20K, as bars of mIoU (bar-rows.js). The paper's export (assets/paper/ade20k_seg.json) supplies the
// frozen DINOv3 ViT-B/16 read by the paper's linear probe at 512 px (probe_table), DINOv3 ViT-S/16 likewise, and the
// prior active models (baselines, each at its best); sources/ade20k-passive-references.json supplies other passive
// results, each with its source (those marked shown). The bars state shows the active models alone, on an axis zoomed
// to ZOOM_MAX; the "more" state brings in the passive models: the axis rescales to MAX, the active bars shrink with
// it, and every row glides to its place, active models first, then passive ones, each group by value. With canvit,
// the chart is the "more" state's from the start, and CanViT-B's best result (the export's best policy curve point)
// waits for its own row between the two groups (#state-of-the-art).

import { drawBarRows, fetchJson } from "./bar-rows.js";

const MAX = 70, ZOOM_MAX = 30;

export function bestPrior(baselines, name) {
  const runs = baselines.filter((b) => b.name === name);
  if (!runs.length) throw new Error(`gap: no baseline ${name}`);
  return runs.reduce((a, b) => (b.miou_pct > a.miou_pct ? b : a));
}

export function probe(table, model) {
  const row = table.find((e) => e.model === model && e.input_px === 512);
  if (!row) throw new Error(`gap: no ${model} at 512 px in the paper's probe_table`);
  return row.miou_pct;
}

/** CanViT-B's best mIoU over every policy, canvas and glimpse of the export, which must be the paper's claim. */
export function bestCanvit(data) {
  const points = data.policy_curves.flatMap((curve) => curve.per_timestep.map((p) => ({ ...p, curve })));
  const best = points.reduce((a, b) => (b.mean > a.mean ? b : a));
  if ((100 * best.mean).toFixed(1) !== data.claims.best_miou_pct.toFixed(1)) {
    throw new Error(`gap: the export's best point (${100 * best.mean}) is not its claim (${data.claims.best_miou_pct})`);
  }
  return best;
}

export async function drawGap(container, { paper, references, canvit = false }) {
  const [data, refs] = await Promise.all([fetchJson(paper), fetchJson(references)]);
  const prior = (name, label) => {
    const run = bestPrior(data.baselines, name);
    return { name: label, note: `${run.num_glimpses} glimpses`, value: run.miou_pct, kind: "active", core: true };
  };
  const rows = [
    prior("AME (MAE)", "AME (MAE ViT-L)"),
    prior("AdaGlimpse", "AdaGlimpse"),
    prior("AME (SETR)", "AME (SETR ViT-L)"),
    { name: "DINOv3 ViT-S", note: "frozen, linear", value: probe(data.probe_table, "DINOv3 ViT-S/16"), kind: "passive", core: canvit },
    { name: "DINOv3 ViT-B", note: "frozen, linear", value: probe(data.probe_table, "DINOv3 ViT-B/16"), kind: "passive", core: canvit },
    ...refs.references.filter((r) => r.shown).map((r) => ({ name: r.name, note: r.note, value: r.value, kind: "passive", core: canvit })),
  ];
  if (canvit) {
    const best = bestCanvit(data);
    rows.push({ name: "CanViT-B", note: `frozen, linear, ${best.t + 1} glimpses`, value: 100 * best.mean, kind: "canvit",
                kindLabel: "active", core: false });
  }
  drawBarRows(container, rows, { groups: ["active", "canvit", "passive"], max: MAX, zoomMax: canvit ? MAX : ZOOM_MAX,
                                 ticks: [0, 10, 20, 30, 40, 50, 60, 70], label: "ADE20K mIoU of passive and active models" });
}
