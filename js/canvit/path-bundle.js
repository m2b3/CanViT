// A path bundle (canvit-pytorch/docs/viz.md, "Path bundle"): CanViT looking along a smooth path of viewpoints,
// one glimpse per sample. Layers are atlases with one tile per sample, row-major, `atlas_columns` tiles per row.

import { decodePng } from "./png.js";

const SCHEMA = "canvit-path-bundle-df96391b-61d4-49cd-a1e5-5a4af9a42e77";
const CONDITIONS = ["carried", "reset"];

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
  return response;
}

async function png(url) {
  return decodePng(await (await fetchOk(url)).arrayBuffer(), url.href);
}

async function image(url) {
  const img = new Image();
  img.src = url.href;
  await img.decode();
  return img;
}

/** The viewpoint (row, col, scale) at phase in [0, 1) on the closed curve, as the Python recorder samples it. */
export function pointOn(segments, phase) {
  const position = phase * segments.length;
  const index = Math.min(Math.floor(position), segments.length - 1);
  const fraction = position - index;
  let points = segments[index].map((p) => [...p]);
  while (points.length > 1) points = points.slice(0, -1).map((p, i) => p.map((v, j) => v + fraction * (points[i + 1][j] - v)));
  return points[0];
}

/** The top-left pixel (y, x) of a sample's tile in an atlas whose tiles are `size` pixels square. */
export function tileOrigin(manifest, sample, size) {
  const columns = manifest.atlas_columns;
  return [Math.floor(sample / columns) * size, (sample % columns) * size];
}

export async function loadPathBundle(src) {
  const base = new URL(src.endsWith("/") ? src : `${src}/`, document.baseURI);
  const manifest = await (await fetchOk(new URL("manifest.json", base))).json();
  if (manifest.schema !== SCHEMA) throw new Error(`${base}manifest.json: schema ${manifest.schema}, expected ${SCHEMA}`);
  const [scene, inputs, truth] = await Promise.all([
    image(new URL(manifest.scene.image, base)),
    image(new URL(manifest.inputs, base)),
    manifest.scene.truth ? png(new URL(manifest.scene.truth, base)) : null,
  ]);
  const conditions = {};
  for (const [name, entry] of Object.entries(manifest.conditions)) {
    if (!CONDITIONS.includes(name)) throw new Error(`${base}manifest.json: unknown condition "${name}"`);
    const layers = {};
    for (const [layer, file] of Object.entries(entry.layers)) layers[layer] = await png(new URL(file, base));
    conditions[name] = { layers, pixelAccuracy: entry.pixel_accuracy ?? null };
  }
  return { manifest, scene, inputs, truth, conditions, count: manifest.path.viewpoints.length };
}
