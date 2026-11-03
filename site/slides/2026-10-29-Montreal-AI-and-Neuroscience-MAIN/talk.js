// What this talk's slides draw from data, by slide id. Paths resolve against this module, so the deck (index.html) and
// a page that shows one of its slides (../../js/talk-slide.js) draw the same thing from the same files.

import "../../js/canvit/index.js";
import "../../js/canvit/episode.js";
import "../../js/canvit/frontier.js";
import "../../js/canvit/policy-curves.js";
import "./foveate.js";
import "./glimpse-tour.js";
import "./title-teaser.js";
import { drawCanvit2 } from "./canvit2-glimpses.js";
import { drawTimeline } from "./timeline.js";
import { drawMapNames } from "./labeled-map.js";
import { drawUncertainty } from "./uncertainty.js";
import { drawDetail } from "./detail.js";
import { drawDistillation } from "./distillation.js";
import { drawVitOnGlimpse } from "./vit-glimpse.js";
import { drawCanvasUpdates } from "./canvas-updates.js";
import { drawSrRope } from "./sr-rope.js";
import { drawPcaCycle } from "./pca-cycle.js";
import { drawFoundationTasks } from "./foundation-tasks.js";
import { drawPretrainingGrid } from "./pretraining-grid.js";
import { drawPolicyStages } from "./policy-stages.js";
import { placeTableAnswers } from "./table.js";
import { drawGap, drawIn1kGap } from "./gap.js";
import { totalFlops, projectionRatio } from "./flops-chart.js";
import { fetchJson } from "./bar-rows.js";
import { POLICIES } from "../../js/canvit/policies.js";

const at = (path) => new URL(path, import.meta.url).href;
const one = (slide, selector) => {
  const element = slide.querySelector(selector);
  if (!element) throw new Error(`#${slide.id}: no ${selector}`);
  return element;
};

const ADE20K = at("../../assets/paper/ade20k_seg.json");
const PASSIVE_ADE20K = at("sources/ade20k-passive-references.json");
const VIT_EXAMPLE = at("../../data/talk/vit-glimpse-examples/ADE_val_00001517-lamp");
const TABLE_BOXES = at("../../data/talk/table/ADE_val_00001271-table/boxes.json");

/** Each slide's drawing, by id. A slide absent from the document is not drawn. */
export const DRAW = {
  "timeline": (slide) => drawTimeline(one(slide, ".timeline")),
  "foundation": async (slide) => {
    // Principal components 1 to 12, then over again.
    await drawPcaCycle(slide, at("../../data/talk/foundation/ADE_val_00001509"), "maps.json", { lastComponent: 12 });
    await drawFoundationTasks(slide, at("../../data/talk/foundation-tasks/ADE_val_00001509"));
  },
  "gap": (slide) => drawGap(one(slide, ".bar-chart .bars"), { paper: ADE20K, references: PASSIVE_ADE20K }),
  "vit-on-a-glimpse": async (slide) => {
    await drawVitOnGlimpse(slide, VIT_EXAMPLE);
    await drawPcaCycle(slide, VIT_EXAMPLE, "meta.json");
  },
  "integrating-information-across-viewpoints": (slide) => placeTableAnswers(slide, TABLE_BOXES),
  "sr-rope": (slide) => drawSrRope(slide, at("../../data/talk/vit-glimpse")),
  "canvas-attention": async (slide) => {
    const flops = await fetchJson(at("../../data/talk/canvas-attention/flops.json"));
    totalFlops(one(slide, ".b-only .total"), flops, { width: 1100, height: 540, labels: 280 });
    totalFlops(one(slide, ".with-c .total"), flops, { width: 640, height: 540, labels: 280 });
    projectionRatio(one(slide, ".with-c .ratio"), flops, { width: 500, height: 540, labels: 90 });
  },
  "pretraining-policies": (slide) => drawPretrainingGrid(one(slide, ".pretraining-grid"), at("../../data/talk/pretraining"), "truck", ["random", "full_then_random"]),
  "distillation": (slide) => drawDistillation(slide, at("../../data/talk/distillation/ADE_val_00000124/riid-s124-n21/steps.json")),
  "table-callback": (slide) => placeTableAnswers(slide, TABLE_BOXES),
  "c2f-and-f2c": (slide) => drawPolicyStages(one(slide, ".stages"), [at("../../data/street-c2f"), at("../../data/street-f2c")]),
  "eg-c2f": async (slide) => {
    await drawUncertainty(slide, at("../../data/talk/metacognition/street"));
    slide.style.setProperty("--canvit-glimpse", POLICIES.entropy_coarse_to_fine.color);
  },
  "state-of-the-art": async (slide) => {
    await drawGap(one(slide, ".ade20k .bars"), { paper: ADE20K, references: PASSIVE_ADE20K, canvit: true });
    await drawIn1kGap(one(slide, ".in1k .bars"), { macros: at("../../assets/paper/data_macros.json"), probes: at("../../assets/paper/dinov3_probe_data.json"), references: at("sources/in1k-active-references.json") });
  },
  "next": () => drawCanvit2(at("../../data/talk/canvit2-glimpses/")),
  // Alternatives to the paper's cat: data/talk/canvas-updates/.
  "canvas-updates": (slide) => drawCanvasUpdates(slide, at("../../data/talk/canvas-updates/cat")),
  "detail": (slide) => drawDetail(slide, at("../../data/talk/looking-closer/ADE_val_00001715-television")),
};

/** What any slide may hold: names over its maps, and class legends (each class in its color, from a plot step's
 * legend.json). */
async function drawCommon(slide) {
  await drawMapNames(slide);
  for (const legend of slide.querySelectorAll(".class-legend[data-src]")) {
    const entries = await (await fetch(legend.dataset.src)).json();
    legend.replaceChildren(...entries.map(({ class: name, color }) => {
      const chip = document.createElement("span");
      chip.textContent = name;
      chip.style.setProperty("--chip", color);
      return chip;
    }));
  }
}

/** Draws one slide of this talk, wherever it is shown. */
export async function drawSlide(slide) {
  await drawCommon(slide);
  await DRAW[slide.id]?.(slide);
}

/** Draws every slide of this talk present under root, in document order. */
export async function drawSlides(root = document) {
  for (const slide of root.querySelectorAll("section[id]")) await drawSlide(slide);
}
