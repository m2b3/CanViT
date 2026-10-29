// SR-RoPE (#sr-rope): a glimpse's patches as the backbone receives them and the scene positions SR-RoPE gives them,
// beside the canvas patches' fixed positions. Viewpoints from meta.json (experiments/vit_glimpse): "vit", the glimpse
// of #vit-on-a-glimpse; "shows-zoom", the same center at the scale of "zoom", so only the patches' spacing changes;
// "shows-move", that scale slid to the cat's other eye, so the patches move together. Between states the center moves
// linearly and the scale geometrically; each frame resamples the glimpse from the scene in the browser (close to the
// model's sampling). A glimpse patch and a canvas patch are singled out with their scene coordinates.

const PLAY_MS = 1400;
// The glimpse patch singled out in both panels (row, column of the glimpse's grid), and a canvas patch (row, column of
// the canvas grid) chosen away from everywhere that glimpse patch goes, so their rings and labels never meet.
const PICK = { row: 2, col: 5 };
const CANVAS_PICK = { row: 27, col: 6 };
// Where the glimpse slides on "shows-move", horizontally only, as a fraction of the scene's side: the cat's other eye.
const MOVED_LEFT = .636;

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
  const frame = find(".scene .frame"), viewCenter = find(".viewpoint .center"), viewScale = find(".viewpoint .scale");
  const side = scene.naturalWidth;
  const signed = (v) => v.toFixed(2).replace("-", "−");
  // A position in [0, 1] of the scene's side, written in scene coordinates: x across and y down, in [-1, 1].
  const sceneXY = (fx, fy) => `(${signed(2 * fx - 1)}, ${signed(2 * fy - 1)})`;
  const patchInBox = ({ top, left, size }) => [left + size * (PICK.col + .5) / grid, top + size * (PICK.row + .5) / grid];
  const place = (element, fx, fy) => { element.style.left = `${100 * fx}%`; element.style.top = `${100 * fy}%`; };
  place(find(".glimpse .pick"), (PICK.col + .5) / grid, (PICK.row + .5) / grid);
  // Canvas patches tile the scene and stay where they are, whatever the viewpoint.
  const canvasGrid = Number(getComputedStyle(find(".canvas-cells")).getPropertyValue("--n"));
  if (!(canvasGrid > 0)) throw new Error("sr-rope: .canvas-cells has no --n");
  const [cx, cy] = [(CANVAS_PICK.col + .5) / canvasGrid, (CANVAS_PICK.row + .5) / canvasGrid];
  const canvasPick = find(".scene .canvas-pick"), glimpsePick = find(".scene .glimpse-pick");
  canvasPick.style.setProperty("--canvas-grid", canvasGrid);
  place(canvasPick, cx, cy);
  find(".canvas-pick .coord").textContent = sceneXY(cx, cy);
  // The three viewpoints as centers and scales; u runs from 0 to 2 through them, the center linearly, the scale
  // geometrically between consecutive views.
  const centerOf = ({ top, left, size }) => ({ top: top + size / 2, left: left + size / 2, size });
  const first = centerOf(boxes.vit);
  const views = [first, { ...first, size: boxes.zoom.size }, { ...first, left: MOVED_LEFT, size: boxes.zoom.size }];
  const show = (u) => {
    const k = Math.min(Math.floor(u), views.length - 2), v = u - k, [a, b] = [views[k], views[k + 1]];
    const size = a.size * (b.size / a.size) ** v;
    const top = a.top + (b.top - a.top) * v - size / 2, left = a.left + (b.left - a.left) * v - size / 2;
    frame.style.setProperty("--top", top);
    frame.style.setProperty("--left", left);
    frame.style.setProperty("--size", size);
    context.drawImage(scene, left * side, top * side, size * side, size * side, 0, 0, glimpsePx, glimpsePx);
    // The viewpoint, which is not what SR-RoPE encodes: the glimpse's center in scene coordinates and its scale.
    viewCenter.textContent = `center ${sceneXY(left + size / 2, top + size / 2)}`;
    viewScale.textContent = `scale ${size.toFixed(2)}`;
    // What SR-RoPE encodes: each patch's own position in scene coordinates, here for the singled-out glimpse patch.
    const [gx, gy] = patchInBox({ top, left, size });
    place(glimpsePick, gx, gy);
    glimpsePick.querySelector(".coord").textContent = sceneXY(gx, gy);
  };
  let shown = 0, timer = null;
  const play = (target) => {
    cancelAnimationFrame(timer);
    // Off the current slide, or while it is being entered (deck.js), the state is set at once.
    if (!section.classList.contains("present") || section.classList.contains("entering")) { shown = target; return show(shown); }
    const from = shown, start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / PLAY_MS);
      shown = from + (target - from) * (t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
      show(shown);
      if (t < 1) timer = requestAnimationFrame(step);
    };
    timer = requestAnimationFrame(step);
  };
  const stateOf = () => section.classList.contains("shows-move") ? 2 : section.classList.contains("shows-zoom") ? 1 : 0;
  let state = 0;
  new MutationObserver(() => {
    const now = stateOf();
    if (now !== state) { state = now; play(now); }
  }).observe(section, { attributes: true, attributeFilter: ["class"] });
  show(0);
}
