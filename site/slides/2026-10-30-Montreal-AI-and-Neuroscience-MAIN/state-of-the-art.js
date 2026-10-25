// "A new active-vision state of the art": the paper's Table 1 as dot plots (bar-rows.js), a panel per task shown in
// turn, each with the prior active models' best results, CanViT-B's, and the passive DINOv3 ViT-B read out linearly
// from frozen features, the gap CanViT narrows; the axes span where the scores lie. ADE20K: prior models and DINOv3 from the paper's export
// (assets/paper/ade20k_seg.json), CanViT-B from its macros (data_macros.json). ImageNet-1k: prior models from
// sources/in1k-active-references.json, CanViT-B frozen and fine-tuned from the macros, DINOv3 ViT-B from the paper's
// probe export (dinov3_probe_data.json).

import { drawBarRows, fetchJson } from "./bar-rows.js";
import { bestPrior, probe } from "./gap.js";

const macro = (macros, name) => {
  const value = Number(macros[name]);
  if (!Number.isFinite(value)) throw new Error(`state-of-the-art: no number in macro ${name}`);
  return value;
};

export async function drawStateOfTheArt(section, { paper, macros, in1kReferences, dinov3Probes }) {
  const [ade, m, in1k, probes] = await Promise.all([paper, macros, in1kReferences, dinov3Probes].map(fetchJson));
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`state-of-the-art: no ${selector} in #${section.id}`);
    return element;
  };
  const prior = (name) => ({ name, note: "active", value: bestPrior(ade.baselines, name).miou_pct, kind: "active" });
  drawBarRows(find(".ade20k .bars"), [
    prior("AME (MAE)"), prior("AdaGlimpse"), prior("AME (SETR)"),
    { name: "CanViT-B", note: "active, frozen, linear", value: macro(m, "adeBestMiou"), kind: "canvit" },
    { name: "DINOv3 ViT-B", note: "passive, frozen, linear", value: probe(ade.probe_table, "DINOv3 ViT-B/16"), kind: "passive" },
  ], { groups: ["active", "canvit", "passive"], mark: "dot", min: 20, max: 50, ticks: [20, 25, 30, 35, 40, 45, 50], label: "ADE20K mIoU: prior active models, CanViT-B and DINOv3 ViT-B" });
  const dinov3 = probes.probes.find((p) => p.model === "ViT-B/16");
  if (!dinov3) throw new Error(`state-of-the-art: no ViT-B/16 in ${dinov3Probes}`);
  drawBarRows(find(".in1k .bars"), [
    ...in1k.references.map((r) => ({ name: r.name, note: "active", value: r.value, kind: "active" })),
    { name: "CanViT-B", note: "active, frozen, linear", value: macro(m, "inkFrozenBest"), kind: "canvit" },
    { name: "CanViT-B", note: "active, fine-tuned", value: macro(m, "inkFinetunedBest"), kind: "canvit" },
    { name: "DINOv3 ViT-B", note: "passive, frozen, linear", value: dinov3.in1k_top1, kind: "passive" },
  ], { groups: ["active", "canvit", "passive"], mark: "dot", min: 74, max: 86, ticks: [74, 76, 78, 80, 82, 84, 86], label: "ImageNet-1k top-1: prior active models, CanViT-B and DINOv3 ViT-B" });
}
