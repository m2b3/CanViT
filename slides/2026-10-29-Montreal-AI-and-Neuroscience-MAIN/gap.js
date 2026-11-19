// The gap on ADE20K as bars of mIoU (bar-rows.js). assets/paper/ade20k_seg.json supplies frozen DINOv3 ViT-S/16 and
// ViT-B/16 read by the paper's 512 px linear probe (probe_table) and the prior active models at their best (baselines);
// sources/ade20k-passive-references.json supplies other passive results with their sources (those marked shown). The
// bars state shows the active models alone on an axis zoomed to ZOOM_MAX; "more" rescales to MAX and brings in the
// passive models. With canvit (#state-of-the-art), the chart starts in the "more" state and CanViT-B's best result
// waits for its own row between the groups.

import { drawBarRows, fetchJson } from "./bar-rows.js";

const MAX = 70, ZOOM_MAX = 30;

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

/** CanViT-B's best mIoU over every policy, canvas and glimpse of the export, which must be the paper's claim. */
function bestCanvit(data) {
  const points = data.policy_curves.flatMap((curve) => curve.per_timestep.map((p) => ({ ...p, curve })));
  const best = points.reduce((a, b) => (b.mean > a.mean ? b : a));
  if ((100 * best.mean).toFixed(1) !== data.claims.best_miou_pct.toFixed(1)) {
    throw new Error(`gap: the export's best point (${100 * best.mean}) is not its claim (${data.claims.best_miou_pct})`);
  }
  return best;
}

// Rows are bracketed as active or passive models (SECTIONS); frozen backbones (read by linear probes, or DINOv3 7B
// under Mask2Former) are hatched, models trained end to end solid.
const SECTIONS = [{ label: "Active", kinds: ["active", "canvit"] }, { label: "Passive", kinds: ["passive"] }];

export async function drawGap(container, { paper, references, canvit = false }) {
  const [data, refs] = await Promise.all([fetchJson(paper), fetchJson(references)]);
  const prior = (name, label) => {
    const run = bestPrior(data.baselines, name);
    return { name: label, value: run.miou_pct, kind: "active", core: true };
  };
  const rows = [
    prior("AME (MAE)", "AME (MAE ViT-L)"),
    prior("AdaGlimpse", "AdaGlimpse"),
    prior("AME (SETR)", "AME (SETR ViT-L)"),
    { name: "DINOv3 ViT-S", value: probe(data.probe_table, "DINOv3 ViT-S/16"), kind: "passive", core: canvit, frozen: true },
    { name: "DINOv3 ViT-B", value: probe(data.probe_table, "DINOv3 ViT-B/16"), kind: "passive", core: canvit, frozen: true },
    ...refs.references.filter((r) => r.shown).map((r) => ({ name: r.name, value: r.value, kind: "passive", core: canvit,
                                                             frozen: r.note === "frozen backbone" })),
  ];
  if (canvit) {
    rows.push({ name: "CanViT-B", value: 100 * bestCanvit(data).mean, kind: "canvit", core: false, frozen: true });
  }
  drawBarRows(container, rows, { groups: ["active", "canvit", "passive"], sections: SECTIONS, max: MAX,
                                 zoomMax: canvit ? MAX : ZOOM_MAX, ticks: [0, 10, 20, 30, 40, 50, 60, 70],
                                 title: "ADE20K mIoU (%)",
                                 label: "ADE20K mIoU of passive and active models" });
}

// ImageNet-1k top-1 in the same place (#state-of-the-art). The accuracies crowd near the top, so the axis starts at
// IN1K_MIN. Sources: the paper's macros for CanViT-B (frozen with a linear probe, and fine-tuned), its export of the
// DINOv3 probes (dinov3_probe_data.json), and sources/in1k-active-references.json for the prior active models as the
// paper's tab:active-vision-comparison prints them (site/check_paper_numbers.py checks them against it). CanViT-B's
// rows wait for their own state.

const IN1K_MIN = 70, IN1K_MAX = 90;

function dinov3Top1(probes, model) {
  const row = probes.find((p) => p.model === model);
  if (!row) throw new Error(`gap: no DINOv3 ImageNet-1k probe for ${model}`);
  return row.in1k_top1;
}

export async function drawIn1kGap(container, { macros, probes, references }) {
  const [paper, dinov3, refs] = await Promise.all([fetchJson(macros), fetchJson(probes), fetchJson(references)]);
  const rows = [
    ...refs.references.map((r) => ({ name: r.name, value: r.top1, kind: "active", core: true })),
    { name: "CanViT-B, frozen", value: Number(paper.inkFrozenBest), kind: "canvit", core: false, frozen: true },
    { name: "CanViT-B, fine-tuned", value: Number(paper.inkFinetunedBest), kind: "canvit", core: false },
    { name: "DINOv3 ViT-S", value: dinov3Top1(dinov3.probes, "ViT-S/16"), kind: "passive", core: true, frozen: true },
    { name: "DINOv3 ViT-B", value: dinov3Top1(dinov3.probes, "ViT-B/16"), kind: "passive", core: true, frozen: true },
  ];
  drawBarRows(container, rows, { groups: ["active", "canvit", "passive"], sections: SECTIONS, min: IN1K_MIN,
                                 max: IN1K_MAX, ticks: [70, 75, 80, 85, 90], title: "ImageNet-1k top-1 accuracy (%)",
                                 label: "ImageNet-1k top-1 accuracy of passive and active models" });
}
