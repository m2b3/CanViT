// "Processing a glimpse with a Vision Transformer": the scene with the glimpse's box, the glimpse, and its patch tokens
// as colors, from an example of experiments/vit_glimpse/examples.py (meta.json gives the glimpse's box as fractions of
// the scene, and the object it is on).

export async function drawVitOnGlimpse(section, src) {
  const response = await fetch(`${src}/meta.json`);
  if (!response.ok) throw new Error(`vit-glimpse: ${src}/meta.json answered ${response.status}`);
  const { box, object } = await response.json();
  if (!box || !object) throw new Error(`vit-glimpse: ${src}/meta.json lacks the box or the object; rerun examples.py`);
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`vit-glimpse: no ${selector} in #${section.id}`);
    return element;
  };
  Object.assign(find(".scene img"), { src: `${src}/scene.png`, alt: `A scene, with the glimpse's box on the ${object}` });
  Object.assign(find(".glimpse img"), { src: `${src}/glimpse.png`, alt: `The glimpse: the ${object}, 128 by 128 pixels` });
  Object.assign(find(".scene .box").style, { top: `${100 * box.top}%`, left: `${100 * box.left}%`,
                                              width: `${100 * box.size}%`, height: `${100 * box.size}%` });
}
