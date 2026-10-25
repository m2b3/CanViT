// "Scene-Relative Rotary Position Embeddings (SR-RoPE)": a glimpse's patches as the backbone receives them (an 8 x 8
// grid) and the positions SR-RoPE gives them in the scene, beside the canvas patches' fixed positions. The glimpses are
// experiments/vit_glimpse's, sampled from the scene as the model samples them: meta.json's "path" runs from the glimpse
// of #vit-on-a-glimpse to a smaller one, each with its box. When the slide gains "shows-zoom", the path plays (the
// glimpse, its box, its patches' positions and its (x, y, s) together); when it loses it, the path plays back. Off the
// current slide, the end state is set at once.

const PLAY_MS = 1600;

export async function drawSrRope(section, src) {
  const response = await fetch(`${src}/meta.json`);
  if (!response.ok) throw new Error(`sr-rope: ${src}/meta.json answered ${response.status}`);
  const { path, token_grid: grid } = await response.json();
  if (!path?.length) throw new Error(`sr-rope: no path in ${src}/meta.json; rerun experiments.vit_glimpse`);
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`sr-rope: no ${selector} in #${section.id}`);
    return element;
  };
  section.style.setProperty("--grid", grid);
  find(".scene img").src = `${src}/scene.png`;
  // The same patches on both sides: at their place in the glimpse, and at their position in the scene.
  for (const container of [find(".glimpse .dots"), find(".scene .dots")]) {
    container.replaceChildren(...Array.from({ length: grid * grid }, (_, k) => {
      const dot = document.createElement("span");
      dot.style.setProperty("--row", Math.floor(k / grid));
      dot.style.setProperty("--col", k % grid);
      return dot;
    }));
  }
  const frames = path.map((step) => Object.assign(new Image(), { src: `${src}/${step.image}` }));
  const image = find(".crop img"), tag = find(".box .tag");
  const signed = (v) => v.toFixed(2).replace("-", "−");
  const show = (k) => {
    const { top, left, size } = path[k];
    section.style.setProperty("--top", top);
    section.style.setProperty("--left", left);
    section.style.setProperty("--size", size);
    image.src = frames[k].src;
    // The viewpoint as the paper's Figure 1A writes it: its center, x across and y down, in [-1, 1], and its scale s.
    tag.textContent = `x=${signed(2 * (left + size / 2) - 1)}, y=${signed(2 * (top + size / 2) - 1)}, s=${size.toFixed(2)}`;
  };
  let shown = 0, timer = null;
  const play = (target) => {
    cancelAnimationFrame(timer);
    if (!section.classList.contains("present")) { shown = target; return show(shown); }
    const from = shown, start = performance.now();
    const step = (now) => {
      const u = Math.min(1, (now - start) / PLAY_MS);
      const eased = u < .5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
      shown = Math.round(from + (target - from) * eased);
      show(shown);
      if (u < 1) timer = requestAnimationFrame(step);
    };
    timer = requestAnimationFrame(step);
  };
  let zoomed = false;
  new MutationObserver(() => {
    const now = section.classList.contains("shows-zoom");
    if (now !== zoomed) { zoomed = now; play(now ? path.length - 1 : 0); }
  }).observe(section, { attributes: true, attributeFilter: ["class"] });
  show(0);
}
