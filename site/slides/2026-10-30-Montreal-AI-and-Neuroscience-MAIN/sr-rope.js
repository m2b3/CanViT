// "Scene-Relative Rotary Position Embeddings (SR-RoPE)": a glimpse's patches as the backbone receives them (an 8 x 8
// grid) and the positions SR-RoPE gives them in the scene, beside the canvas patches' fixed positions. The first
// glimpse is #vit-on-a-glimpse's (experiments/vit_glimpse, meta.json); the second, smaller one is set on the section
// (--zoom-top, --zoom-left, --zoom-size, fractions of the scene). talk.css moves the dots between the two.

export async function drawSrRope(section, src) {
  const response = await fetch(`${src}/meta.json`);
  if (!response.ok) throw new Error(`sr-rope: ${src}/meta.json answered ${response.status}`);
  const { box: { top, left, size }, token_grid: grid } = await response.json();
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`sr-rope: no ${selector} in #${section.id}`);
    return element;
  };
  section.style.setProperty("--first-top", top);
  section.style.setProperty("--first-left", left);
  section.style.setProperty("--first-size", size);
  section.style.setProperty("--grid", grid);
  find(".scene img").src = `${src}/scene.png`;
  find(".crop").style.backgroundImage = `url(${src}/scene.png)`;
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
