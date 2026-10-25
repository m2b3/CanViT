// "Processing a glimpse with a Vision Transformer": the scene with the glimpse's box, the glimpse, and its patch tokens
// as colors, from experiments/vit_glimpse (meta.json gives the glimpse's box, "vit", as fractions of the scene).

export async function drawVitOnGlimpse(section, src) {
  const response = await fetch(`${src}/meta.json`);
  if (!response.ok) throw new Error(`vit-glimpse: ${src}/meta.json answered ${response.status}`);
  const { boxes: { vit: { top, left, size } } } = await response.json();
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`vit-glimpse: no ${selector} in #${section.id}`);
    return element;
  };
  find(".scene img").src = `${src}/scene.png`;
  find(".glimpse img").src = `${src}/glimpse-vit.png`;
  Object.assign(find(".scene .box").style, { top: `${100 * top}%`, left: `${100 * left}%`, width: `${100 * size}%`, height: `${100 * size}%` });
}
