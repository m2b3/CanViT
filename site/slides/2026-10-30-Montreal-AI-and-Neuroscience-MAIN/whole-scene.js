// The rollout slide's contrast: DINOv3 ViT-B's segmentation of the episode's current scene, seen whole at 128 px
// (experiments/whole_scene_baseline), in the canvas's class colors, one cell per DINOv3 patch (nearest), with its pixel
// accuracy measured as the episode's meter measures CanViT's. It follows the episode's scene tabs (canvit-select).

import { ADE20K_PALETTE } from "../../js/canvit/ade20k.js";
import { decodePng } from "../../js/canvit/png.js";

async function fetched(url, as) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`whole-scene: ${url} answered ${response.status}`);
  return as === "json" ? response.json() : response.arrayBuffer();
}

export function followEpisode(figure, episode) {
  const canvas = figure.querySelector("canvas"), accuracy = figure.querySelector(".accuracy");
  let shown = null;
  const show = async (source) => {
    const name = source.split("/").filter(Boolean).at(-1);
    shown = name;
    const dir = `${figure.dataset.src}/${name}`;
    const [png, meta] = await Promise.all([fetched(`${dir}/labels.png`), fetched(`${dir}/meta.json`, "json")]);
    if (shown !== name) return;
    const labels = await decodePng(png, `${dir}/labels.png`);
    if (labels.channels !== 1 || labels.width !== meta.grid || labels.height !== meta.grid) {
      throw new Error(`whole-scene: ${dir}/labels.png is not a ${meta.grid} x ${meta.grid} label map`);
    }
    const image = new ImageData(labels.width, labels.height);
    labels.data.forEach((c, i) => image.data.set([...ADE20K_PALETTE.slice(3 * c, 3 * c + 3), 255], 4 * i));
    [canvas.width, canvas.height] = [labels.width, labels.height];
    canvas.getContext("2d").putImageData(image, 0, 0);
    accuracy.textContent = `${(100 * meta.pixel_accuracy).toFixed(1)}%`;
  };
  episode.addEventListener("canvit-select", (event) => show(event.detail.source));
  if (episode.source) show(episode.source);
}
