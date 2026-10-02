// Load a web bundle (schema canvit-web-bundle-70766501-..., canvit-pytorch/docs/viz.md): the manifest
// and every layer, decoded and checked once, shared by all elements showing it.

import { decodePng } from "./png.js";
import { layerSpec, fractions, UNLABELED } from "./layers.js";
import { ADE20K_PALETTE } from "./ade20k.js";

const SCHEMA = "canvit-web-bundle-70766501-fb7f-4856-8a88-bb16253c34c5";
const cache = new Map();

/** Resolve `src` (a bundle directory, relative to the page) and load it once per page. */
export function loadBundle(src) {
  const url = new URL(src.endsWith("/") ? src : `${src}/`, document.baseURI).href;
  if (!cache.has(url)) cache.set(url, load(url));
  return cache.get(url);
}

async function fetchOk(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    const hint = location.protocol === "file:" ? " (pages opened from file:// cannot fetch; serve site/ over HTTP)" : "";
    throw new Error(`Could not fetch ${url}${hint}: ${error.message}`);
  }
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
  return response;
}

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

async function load(url) {
  const manifest = await (await fetchOk(new URL("manifest.json", url))).json();
  const where = `${url}manifest.json`;
  check(manifest.schema === SCHEMA, `${where}: schema is ${JSON.stringify(manifest.schema)}, expected "${SCHEMA}"`);
  const { canvas_grid: grid, glimpse_px: glimpsePx, glimpses, readout } = manifest;
  check(Number.isInteger(grid) && grid > 0, `${where}: canvas_grid must be a positive integer`);
  const sides = { canvas: grid, glimpse: glimpsePx, patches: glimpsePx / manifest.model.patch_px };
  check(Number.isInteger(sides.patches), `${where}: glimpse_px ${glimpsePx} is not a whole number of ${manifest.model.patch_px} px patches`);
  check(Array.isArray(glimpses) && glimpses.length > 0, `${where}: no glimpses`);
  check(readout.class_names?.length === readout.num_classes,
        `${where}: readout.class_names has ${readout.class_names?.length} names for ${readout.num_classes} classes`);

  const layerNames = Object.keys(glimpses[0].layers);
  layerNames.forEach(layerSpec); // unknown layer names fail here
  if (layerNames.some((name) => layerSpec(name).kind === "labels")) {
    // The only class palette is ADE20K's; another readout needs its own.
    check(readout.kind === "ade20k-segmentation" && readout.num_classes * 3 === ADE20K_PALETTE.length,
          `${where}: class labels from readout "${readout.kind}" with ${readout.num_classes} classes have no palette`);
  }

  const sceneBlob = await (await fetchOk(new URL(manifest.scene.image, url))).blob();
  const sceneUrl = URL.createObjectURL(sceneBlob);
  const sceneImage = new Image();
  sceneImage.src = sceneUrl;
  await sceneImage.decode(); // decoded once here, so <img src=sceneUrl> shows it at once
  check(sceneImage.naturalWidth === manifest.scene.px && sceneImage.naturalHeight === manifest.scene.px,
        `${where}: scene is ${sceneImage.naturalWidth}×${sceneImage.naturalHeight}, manifest says ${manifest.scene.px}`);

  // truth.png: the annotated class at each canvas cell's center, UNLABELED where the annotation has none.
  let truth = null;
  if (manifest.scene.truth) {
    const file = new URL(manifest.scene.truth, url).href;
    truth = await decodePng(await (await fetchOk(file)).arrayBuffer(), file);
    check(truth.width === grid && truth.height === grid && truth.channels === 1,
          `${file}: ${truth.width}×${truth.height}×${truth.channels}, expected ${grid}×${grid}×1`);
    check(truth.data.every((c) => c < readout.num_classes || c === UNLABELED), `${file}: a class index out of range`);
  }

  const decoded = await Promise.all(glimpses.map(async (g, t) => {
    check(g.t === t, `${where}: glimpse ${t} has t=${g.t}`);
    check(JSON.stringify(Object.keys(g.layers)) === JSON.stringify(layerNames),
          `${where}: glimpse ${t} has layers ${Object.keys(g.layers)}, glimpse 0 has ${layerNames}`);
    const { top, left, size } = g.box;
    const eps = 1e-6;
    check(size > 0 && top >= -eps && left >= -eps && top + size <= 1 + eps && left + size <= 1 + eps,
          `${where}: glimpse ${t} box ${JSON.stringify(g.box)} is outside the scene`);
    const layers = {};
    await Promise.all(layerNames.map(async (name) => {
      const file = new URL(g.layers[name], url).href;
      const raster = await decodePng(await (await fetchOk(file)).arrayBuffer(), file);
      layers[name] = checkRaster(raster, name, file, { sides, numClasses: readout.num_classes });
    }));
    return { t, box: g.box, viewpoint: g.viewpoint, layers, pixelAccuracy: g.pixel_accuracy ?? null };
  }));

  const range = {};
  for (const g of decoded) g.range = {};
  for (const name of layerNames.filter((n) => layerSpec(n).kind === "scalar")) {
    for (const g of decoded) g.range[name] = extent(g.layers[name].fractions);
    range[name] = extent(decoded.flatMap((g) => g.range[name]));
  }

  return {
    url, manifest, grid, glimpsePx, sides, layerNames, range, sceneUrl, truth,
    classNames: readout.class_names,
    glimpses: decoded,
  };
}

/** A layer name from an element attribute, checked against the bundle. */
export function requireLayer(bundle, name) {
  if (!name) throw new Error("missing layer attribute");
  check(bundle.layerNames.includes(name),
        `layer "${name}" is not in ${bundle.url} (it has ${bundle.layerNames.join(", ")})`);
  return name;
}

function checkRaster(raster, name, file, { sides, numClasses }) {
  const spec = layerSpec(name);
  const side = sides[spec.size];
  check(raster.width === side && raster.height === side,
        `${file}: ${raster.width}×${raster.height}, expected ${side}×${side}`);
  const channels = spec.kind === "rgb" ? [3, 4] : [1];
  check(channels.includes(raster.channels), `${file}: ${raster.channels} channels, expected ${channels.join(" or ")}`);
  if (spec.kind === "labels") {
    const max = raster.data.reduce((a, b) => Math.max(a, b), 0);
    check(max < numClasses, `${file}: class index ${max} ≥ ${numClasses} classes`);
  }
  if (spec.kind === "scalar") raster.fractions = fractions(raster);
  return raster;
}
