// "Scene-Relative Rotary Position Embeddings (SR-RoPE)": a glimpse's patches as the backbone receives them (an 8 x 8
// grid) and the positions SR-RoPE gives them in the scene, beside the canvas patches' fixed positions. The glimpses
// are experiments/vit_glimpse's, sampled from the scene as the model samples them: "vit" (#vit-on-a-glimpse's) and
// "zoom", a smaller one; meta.json gives their boxes, set here as --first-* and --zoom-*. talk.css moves the cells
// and dots between the two.

export async function drawSrRope(section, src) {
  const response = await fetch(`${src}/meta.json`);
  if (!response.ok) throw new Error(`sr-rope: ${src}/meta.json answered ${response.status}`);
  const { boxes, token_grid: grid } = await response.json();
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`sr-rope: no ${selector} in #${section.id}`);
    return element;
  };
  for (const [name, prefix] of [["vit", "first"], ["zoom", "zoom"]]) {
    if (!boxes[name]) throw new Error(`sr-rope: no box "${name}" in ${src}/meta.json`);
    for (const key of ["top", "left", "size"]) section.style.setProperty(`--${prefix}-${key}`, boxes[name][key]);
  }
  section.style.setProperty("--grid", grid);
  // Each glimpse's viewpoint as the paper writes it, (x, y, s): its center, x across and y down, in [-1, 1], and its scale.
  const signed = (v) => v.toFixed(2).replace("-", "−");
  for (const [name, cls] of [["vit", "first"], ["zoom", "zoom"]]) {
    const { top, left, size } = boxes[name];
    find(`.viewpoint .${cls}`).textContent = `(x, y, s) = (${signed(2 * (left + size / 2) - 1)}, ${signed(2 * (top + size / 2) - 1)}, ${size.toFixed(2)})`;
  }
  find(".scene img").src = `${src}/scene.png`;
  find(".glimpse img.first").src = `${src}/glimpse-vit.png`;
  find(".glimpse img.zoom").src = `${src}/glimpse-zoom.png`;
  // The same patches on both sides: at their place in the glimpse, and at their position in the scene.
  for (const container of [find(".glimpse .dots"), find(".scene .dots")]) {
    container.replaceChildren(...Array.from({ length: grid * grid }, (_, k) => {
      const dot = document.createElement("span");
      dot.style.setProperty("--row", Math.floor(k / grid));
      dot.style.setProperty("--col", k % grid);
      return dot;
    }));
  }
}
