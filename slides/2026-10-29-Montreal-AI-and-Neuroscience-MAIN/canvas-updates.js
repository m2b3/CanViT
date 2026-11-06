// "Canvas updates within a glimpse": the scene with the first glimpse's box, the glimpse, and what each Canvas Attention
// Write of that glimpse adds to the canvas, from an example of experiments/canvas_updates/export.py (meta.json gives
// each glimpse's box as fractions of the scene). Each Write's figure shows its residual (the paper's
// g0_write<k>_residual.png, k counting from 0 where data-write counts from 1). The glimpse's box is drawn on the scene
// and on every map, which covers the whole scene.

export async function drawCanvasUpdates(section, src) {
  const response = await fetch(`${src}/meta.json`);
  if (!response.ok) throw new Error(`canvas-updates: ${src}/meta.json answered ${response.status}`);
  const { glimpses, writes_per_glimpse: writes } = await response.json();
  const box = glimpses?.[0]?.box;
  if (!box) throw new Error(`canvas-updates: ${src}/meta.json has no first glimpse; rerun the export`);
  const figures = section.querySelectorAll("figure.write");
  if (figures.length !== writes) throw new Error(`canvas-updates: #${section.id} shows ${figures.length} writes, the model makes ${writes}`);
  const one = (root, selector) => {
    const element = root.querySelector(selector);
    if (!element) throw new Error(`canvas-updates: no ${selector} in #${section.id}`);
    return element;
  };
  one(section, ".scene img").src = `${src}/scene.jpg`;
  one(section, ".glimpse img").src = `${src}/g0_glimpse.png`;
  for (const figure of figures) {
    const k = Number(figure.dataset.write) - 1;
    one(figure, "img").src = `${src}/g0_write${k}_residual.png`;
  }
  for (const element of section.querySelectorAll(".box")) {
    Object.assign(element.style, { top: `${100 * box.top}%`, left: `${100 * box.left}%`,
                                   width: `${100 * box.size}%`, height: `${100 * box.size}%` });
  }
}
