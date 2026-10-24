// The layer vocabulary of the web bundle schema (canvit-pytorch/docs/viz.md) and
// how each layer is colored. Every coloring decision for bundle data lives here.

import { ADE20K_PALETTE } from "./ade20k.js";
import { COLORMAPS } from "./colormaps.js";

// manifest.pca.protocol (canvit-pytorch/docs/viz.md) -> how the canvas colors were set.
const PCA_NOTES = {
  "paper": "PCA of canvas tokens → RGB, color limits per glimpse",
  "fixed-limits": "PCA of canvas tokens → RGB, one color scale for all glimpses",
};

/**
 * kind: "rgb" (colored in Python; note(bundle) says how), "labels" (class index per cell) or
 * "scalar" (a fraction in [0, 1]; quantity(bundle) names it, value(f, bundle) converts it for display).
 * size: the raster's side and what it covers: "canvas" canvas_grid, the whole scene; "glimpse" glimpse_px, the glimpse's
 * box; "patches" glimpse_px / model.patch_px, the glimpse's box, one cell per patch.
 * Scalar domains: "unit" [0, 1]; "frame" min–max of this glimpse; "bundle" min–max over all glimpses.
 * Defaults keep one color scale across glimpses, so brightness compares across time: a per-frame
 * range would paint a small late change as brightly as the first glimpse's rewrite.
 */

const LAYERS = {
  crop: { kind: "rgb", size: "glimpse", title: "Glimpse", note: () => "what the model saw" },
  canvas: { kind: "rgb", size: "canvas", title: "Canvas", note: (bundle) => PCA_NOTES[bundle.manifest.pca.protocol] },
  labels: { kind: "labels", size: "canvas", title: "Segmentation" },
  entropy: {
    kind: "scalar", size: "canvas", title: "Uncertainty", colormap: "magma", domain: "unit",
    // Stored as entropy / log C; shown in bits, from 0 (one class) to log2 C (all C classes equally likely).
    low: "sure", high: "unsure", quantity: () => "entropy, bits",
    value: (f, bundle) => f * Math.log2(bundle.manifest.readout.num_classes),
  },
  change: {
    kind: "scalar", size: "canvas", title: "Change", colormap: "magma", domain: "bundle",
    low: "kept", high: "rewritten", quantity: () => "1 − cos", value: (f) => 2 * f,
  },
};

export const DOMAINS = ["unit", "frame", "bundle"];

/** The class value of annotation cells without a label (truth.png). */
export const UNLABELED = 255;

/** Whether the class decoded at a cell matches the annotation there: light and dark, so the two differ in
 * lightness as well as hue; unlabeled cells are neutral gray. */
export const CORRECTNESS = {
  correct: { label: "correct", rgb: [0x9e, 0xe6, 0xae] },
  wrong: { label: "wrong", rgb: [0xc2, 0x41, 0x0c] },
  unlabeled: { label: "unlabeled", rgb: [0x94, 0xa3, 0xb8] },
};

/** The spec of a layer name found in a manifest; unknown names are errors, so new layers force a decision. */
export function layerSpec(name) {
  if (Object.hasOwn(LAYERS, name)) return LAYERS[name];
  const write = /^write(\d+)$/.exec(name);
  if (write) return { kind: "rgb", size: "canvas", title: `Write ${write[1]}`, note: () => "PCA of one Canvas Attention Write" };
  const after = /^write(\d+)_canvas$/.exec(name);
  if (after) return { kind: "rgb", size: "canvas", title: `Canvas after Write ${after[1]}`, note: () => "PCA of the canvas, as in its features layer" };
  const source = /^write(\d+)_glimpse$/.exec(name);
  if (source) return { kind: "rgb", size: "patches", title: `Glimpse read by Write ${source[1]}`, note: () => "PCA of the glimpse's patch tokens" };
  throw new Error(`Unknown bundle layer "${name}": add it to LAYERS in js/canvit/layers.js`);
}

/** A scalar layer's fraction in [0, 1] per cell, from 8- or 16-bit samples. */
export const fractions = (raster) => {
  const max = raster.depth === 16 ? 65535 : 255;
  return Float32Array.from(raster.data, (v) => v / max);
};

export function scalarDomain(bundle, t, name, domain) {
  switch (domain) {
    case "unit": return [0, 1];
    case "frame": return bundle.glimpses[t].range[name];
    case "bundle": return bundle.range[name];
    default: throw new Error(`Unknown domain "${domain}"; expected one of ${DOMAINS.join(", ")}`);
  }
}

/**
 * RGBA ImageData for a layer at glimpse t.
 * highlight: a class index whose cells stay opaque while others fade (labels only).
 */
export function renderLayer(bundle, t, name, { domain, highlight = null } = {}) {
  const spec = layerSpec(name);
  const raster = bundle.glimpses[t].layers[name];
  const { width, height, channels, data } = raster;
  const out = new ImageData(width, height);
  const px = out.data;
  const n = width * height;
  switch (spec.kind) {
    case "rgb":
      for (let i = 0; i < n; i++) {
        px[4 * i] = data[channels * i];
        px[4 * i + 1] = data[channels * i + 1];
        px[4 * i + 2] = data[channels * i + 2];
        px[4 * i + 3] = 255;
      }
      break;
    case "labels":
      for (let i = 0; i < n; i++) {
        const c = data[i];
        px[4 * i] = ADE20K_PALETTE[3 * c];
        px[4 * i + 1] = ADE20K_PALETTE[3 * c + 1];
        px[4 * i + 2] = ADE20K_PALETTE[3 * c + 2];
        px[4 * i + 3] = highlight === null || c === highlight ? 255 : 56;
      }
      break;
    case "scalar": {
      const [lo, hi] = scalarDomain(bundle, t, name, domain ?? spec.domain);
      const lut = COLORMAPS[spec.colormap];
      const f = raster.fractions;
      const scale = hi > lo ? 255 / (hi - lo) : 0;
      for (let i = 0; i < n; i++) {
        const k = 3 * Math.round(Math.min(255, Math.max(0, (f[i] - lo) * scale)));
        px[4 * i] = lut[k];
        px[4 * i + 1] = lut[k + 1];
        px[4 * i + 2] = lut[k + 2];
        px[4 * i + 3] = 255;
      }
      break;
    }
  }
  return out;
}

const drawn = new WeakMap();

/** An offscreen canvas holding a layer at glimpse t in its default coloring, for drawImage. Cached. */
export function layerImage(bundle, t, name) {
  const raster = bundle.glimpses[t].layers[name];
  if (!drawn.has(raster)) {
    const image = renderLayer(bundle, t, name);
    const canvas = Object.assign(document.createElement("canvas"), { width: image.width, height: image.height });
    canvas.getContext("2d").putImageData(image, 0, 0);
    drawn.set(raster, canvas);
  }
  return drawn.get(raster);
}

/** An offscreen canvas coloring each cell by whether the decoded class at glimpse t matches the annotation. */
export function correctnessImage(bundle, t) {
  const labels = bundle.glimpses[t].layers.labels;
  if (!bundle.truth) throw new Error(`${bundle.url}: no annotation (truth.png) to check the segmentation against`);
  if (!drawnCorrectness.has(labels)) {
    const { width, height, data } = labels;
    const image = new ImageData(width, height);
    for (let i = 0; i < data.length; i++) {
      const truth = bundle.truth.data[i];
      const { rgb } = truth === UNLABELED ? CORRECTNESS.unlabeled : data[i] === truth ? CORRECTNESS.correct : CORRECTNESS.wrong;
      image.data.set([...rgb, 255], 4 * i);
    }
    const canvas = Object.assign(document.createElement("canvas"), { width, height });
    canvas.getContext("2d").putImageData(image, 0, 0);
    drawnCorrectness.set(labels, canvas);
  }
  return drawnCorrectness.get(labels);
}

const drawnCorrectness = new WeakMap();

/** CSS gradient of a colormap, for legends, from its low end toward `angle`. */
export function colormapGradient(name, { stops = 16, angle = "90deg" } = {}) {
  const lut = COLORMAPS[name];
  const colors = Array.from({ length: stops }, (_, i) => {
    const k = 3 * Math.round((i / (stops - 1)) * 255);
    return `rgb(${lut[k]} ${lut[k + 1]} ${lut[k + 2]})`;
  });
  return `linear-gradient(${angle}, ${colors.join(", ")})`;
}

export const paletteCss = (c) =>
  `rgb(${ADE20K_PALETTE[3 * c]} ${ADE20K_PALETTE[3 * c + 1]} ${ADE20K_PALETTE[3 * c + 2]})`;
