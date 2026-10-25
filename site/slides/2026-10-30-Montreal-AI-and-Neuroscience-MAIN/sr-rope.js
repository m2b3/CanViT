// "Scene-Relative Rotary Position Embeddings (SR-RoPE)": a glimpse's patches as the backbone receives them (an 8 x 8
// grid) and the positions SR-RoPE gives them in the scene, beside the canvas patches' fixed positions. meta.json
// (experiments/vit_glimpse) gives the glimpse of #vit-on-a-glimpse ("vit") and a smaller one ("zoom"). When the slide
// gains "shows-zoom", the viewpoint moves from one to the other, centers linearly and scales geometrically, and every
// frame draws its 128 px glimpse from the 512 px scene onto a canvas (the browser's resampling, close to the model's)
// and moves the box, the patches' scene positions and the (x, y, s) with it; losing "shows-zoom" plays it back. Off the
// current slide, the end state is set at once.

const PLAY_MS = 1400;

export async function drawSrRope(section, src) {
  const response = await fetch(`${src}/meta.json`);
  if (!response.ok) throw new Error(`sr-rope: ${src}/meta.json answered ${response.status}`);
  const { boxes, token_grid: grid, glimpse_px: glimpsePx } = await response.json();
  if (!boxes?.vit || !boxes?.zoom) throw new Error(`sr-rope: ${src}/meta.json lacks the "vit" and "zoom" boxes`);
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`sr-rope: no ${selector} in #${section.id}`);
    return element;
  };
  section.style.setProperty("--grid", grid);
  const scene = find(".scene img");
  scene.src = `${src}/scene.png`;
  await scene.decode();
  // The same patches on both sides: at their place in the glimpse, and at their position in the scene.
  for (const container of [find(".glimpse .dots"), find(".scene .dots")]) {
    container.replaceChildren(...Array.from({ length: grid * grid }, (_, k) => {
      const dot = document.createElement("span");
      dot.style.setProperty("--row", Math.floor(k / grid));
      dot.style.setProperty("--col", k % grid);
      return dot;
    }));
  }
  const canvas = find(".crop canvas"), context = canvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  const frame = find(".scene .frame"), tag = find(".box .tag");
  const side = scene.naturalWidth;
  const signed = (v) => v.toFixed(2).replace("-", "−");
  // u from 0 ("vit") to 1 ("zoom"): the box's center moves linearly, its size geometrically.
  const center = ({ top, left, size }) => [top + size / 2, left + size / 2];
  const show = (u) => {
    const { vit: a, zoom: b } = boxes;
    const [[ra, ca], [rb, cb]] = [center(a), center(b)];
    const size = a.size * (b.size / a.size) ** u;
    const top = ra + (rb - ra) * u - size / 2, left = ca + (cb - ca) * u - size / 2;
    frame.style.setProperty("--top", top);
    frame.style.setProperty("--left", left);
    frame.style.setProperty("--size", size);
    context.drawImage(scene, left * side, top * side, size * side, size * side, 0, 0, glimpsePx, glimpsePx);
    // The viewpoint as the paper's Figure 1A writes it: its center, x across and y down, in [-1, 1], and its scale s.
    tag.textContent = `x=${signed(2 * (left + size / 2) - 1)}, y=${signed(2 * (top + size / 2) - 1)}, s=${size.toFixed(2)}`;
  };
  let shown = 0, timer = null;
  const play = (target) => {
    cancelAnimationFrame(timer);
    if (!section.classList.contains("present")) { shown = target; return show(shown); }
    const from = shown, start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / PLAY_MS);
      shown = from + (target - from) * (t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
      show(shown);
      if (t < 1) timer = requestAnimationFrame(step);
    };
    timer = requestAnimationFrame(step);
  };
  let zoomed = false;
  new MutationObserver(() => {
    const now = section.classList.contains("shows-zoom");
    if (now !== zoomed) { zoomed = now; play(now ? 1 : 0); }
  }).observe(section, { attributes: true, attributeFilter: ["class"] });
  show(0);
}
