// The architecture slide's drawing, on a recorded rollout (a web bundle): the scene and glimpse t's viewpoint; the
// glimpse, cut into patches, through the Vision Transformer's blocks; the canvas before and after that glimpse; the
// writes and reads between the two streams after the blocks the bundle's manifest names; then glimpse t + 1, the
// canvas carried over. Positions are in the drawing's own pixels (SIZE); talk.css shows its parts on the slide's
// named states.

const NS = "http://www.w3.org/2000/svg";
const SIZE = { width: 840, height: 552 };
const IMAGE = 120;
const BOTTOM = 382; // top of the glimpse's and the canvas-before's images; the scene's bottom is aligned with theirs
const SCENE = { x: 0, size: 250 };
const GLIMPSE_X = 360, CANVAS_X = 650, CANVAS_AFTER_Y = 0;
const TOWER = { width: 86, top: 144, bottom: 360, gap: 5 };
const BLUE_X = GLIMPSE_X + IMAGE / 2, RED_X = CANVAS_X + IMAGE / 2;
const TOWER_RIGHT = BLUE_X + TOWER.width / 2;
const CARRY_X = CANVAS_X + IMAGE + 40;

const signed = (value) => value.toFixed(2).replace("-", "−");

function svg(name, attributes, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent.append(node);
  return node;
}

function image(className, src, x, y, size, parent) {
  const node = document.createElement("img");
  node.className = className;
  node.src = src;
  node.alt = "";
  Object.assign(node.style, { left: `${x}px`, top: `${y}px`, width: `${size}px`, height: `${size}px` });
  parent.append(node);
  return node;
}

function label(className, lines, x, y, parent) {
  const node = document.createElement("p");
  node.className = `label ${className}`;
  node.replaceChildren(...lines.map((text) => Object.assign(document.createElement("span"), { textContent: text })));
  Object.assign(node.style, { left: `${x}px`, top: `${y}px` });
  parent.append(node);
  return node;
}

// An arrow that draws itself: a line (pathLength 1, for the dash animation) and its head, shown once the line is drawn.
function wire(className, points, parent) {
  const group = svg("g", { class: `wire ${className}` }, parent);
  svg("path", { class: "line", d: `M${points.map((p) => p.join(",")).join(" L")}`, pathLength: 1 }, group);
  const [[x0, y0], [x1, y1]] = points.slice(-2);
  const angle = Math.atan2(y1 - y0, x1 - x0);
  const side = (turn) => [x1 - 12 * Math.cos(angle + turn), y1 - 12 * Math.sin(angle + turn)].join(",");
  svg("path", { class: "head", d: `M${side(-0.45)} L${x1},${y1} L${side(0.45)} Z` }, group);
  return group;
}

export async function drawArchitecture(container) {
  const src = container.dataset.src, t = Number(container.dataset.t);
  const response = await fetch(`${src}/manifest.json`);
  if (!response.ok) throw new Error(`architecture: ${src}/manifest.json answered ${response.status}`);
  const manifest = await response.json();
  const glimpse = (k) => {
    const found = manifest.glimpses[k];
    if (!found) throw new Error(`architecture: no glimpse ${k} in ${src}`);
    return found;
  };
  const [before, now, next] = [glimpse(t - 1), glimpse(t), glimpse(t + 1)];
  const { num_blocks: blocks, read_after_blocks: reads, write_after_blocks: writes } = manifest.model;
  const layer = (g, name) => `${src}/${g.layers[name]}`;

  container.replaceChildren();
  Object.assign(container.style, { width: `${SIZE.width}px`, height: `${SIZE.height}px` });
  const drawing = svg("svg", { class: "wires", viewBox: `0 0 ${SIZE.width} ${SIZE.height}` }, container);

  // The scene and the viewpoint of each glimpse, the arrow to the glimpse it gives.
  image("scene", `${src}/${manifest.scene.image}`, SCENE.x, BOTTOM + IMAGE - SCENE.size, SCENE.size, container);
  for (const [g, when] of [[now, "now"], [next, "next"]]) {
    const { top, left, size } = g.box;
    svg("rect", { class: `box ${when}`, x: SCENE.x + left * SCENE.size, y: BOTTOM + IMAGE - SCENE.size + top * SCENE.size,
                  width: size * SCENE.size, height: size * SCENE.size }, drawing);
  }
  wire("feed", [[SCENE.x + SCENE.size + 12, BOTTOM + IMAGE / 2], [GLIMPSE_X - 10, BOTTOM + IMAGE / 2]], drawing);

  // The glimpse stream: the glimpse, its patches, and the blocks it goes through.
  image("glimpse pixels now", layer(now, "crop"), GLIMPSE_X, BOTTOM, IMAGE, container);
  image("glimpse pixels next", layer(next, "crop"), GLIMPSE_X, BOTTOM, IMAGE, container);
  const patches = Math.round(manifest.glimpse_px / manifest.model.patch_px);
  const grid = svg("g", { class: "patches" }, drawing);
  for (let i = 1; i < patches; i++) {
    const offset = (i * IMAGE) / patches;
    svg("line", { x1: GLIMPSE_X + offset, x2: GLIMPSE_X + offset, y1: BOTTOM, y2: BOTTOM + IMAGE }, grid);
    svg("line", { x1: GLIMPSE_X, x2: GLIMPSE_X + IMAGE, y1: BOTTOM + offset, y2: BOTTOM + offset }, grid);
  }
  svg("line", { class: "stream glimpse-stream", x1: BLUE_X, x2: BLUE_X, y1: BOTTOM, y2: TOWER.top }, drawing);
  const height = (TOWER.bottom - TOWER.top - (blocks - 1) * TOWER.gap) / blocks;
  const blockTop = (k) => TOWER.bottom - (k + 1) * height - k * TOWER.gap;
  for (let k = 0; k < blocks; k++) {
    svg("rect", { class: "block", x: BLUE_X - TOWER.width / 2, y: blockTop(k), width: TOWER.width, height, rx: 3 }, drawing);
  }

  // The canvas stream: the canvas before this glimpse, below; after it, above.
  image("canvas pixels before now", layer(before, "canvas"), CANVAS_X, BOTTOM, IMAGE, container);
  image("canvas pixels before next", layer(now, "canvas"), CANVAS_X, BOTTOM, IMAGE, container);
  image("canvas pixels after now", layer(now, "canvas"), CANVAS_X, CANVAS_AFTER_Y, IMAGE, container);
  image("canvas pixels after next", layer(next, "canvas"), CANVAS_X, CANVAS_AFTER_Y, IMAGE, container);
  wire("stream canvas-stream", [[RED_X, BOTTOM], [RED_X, CANVAS_AFTER_Y + IMAGE + 2]], drawing);

  // Reads and writes after the blocks the manifest names, numbered in depth order for the build's timing.
  const exchanges = [...reads.map((k) => ({ k, kind: "read" })), ...writes.map((k) => ({ k, kind: "write" }))]
    .sort((a, b) => a.k - b.k);
  const seen = { read: 0, write: 0 };
  exchanges.forEach(({ k, kind }, depth) => {
    if (!(k >= 0 && k < blocks)) throw new Error(`architecture: ${kind} after block ${k} of ${blocks}`);
    const y = blockTop(k) - TOWER.gap / 2;
    const ends = kind === "write" ? [[TOWER_RIGHT + 2, y], [RED_X - 2, y]] : [[RED_X - 2, y], [TOWER_RIGHT + 2, y]];
    const arrow = wire(`exchange ${kind}`, ends, drawing);
    arrow.style.setProperty("--depth", depth);
    arrow.style.setProperty("--order", seen[kind]);
    if (seen[kind]++ === 0) {
      svg("text", { class: `exchange-label ${kind}`, x: (TOWER_RIGHT + RED_X) / 2, y: y - 8, "text-anchor": "middle" }, drawing)
        .textContent = kind;
    }
  });

  // The canvas after the glimpse becomes the canvas before the next one.
  wire("carry", [[CANVAS_X + IMAGE + 4, CANVAS_AFTER_Y + IMAGE / 2], [CARRY_X, CANVAS_AFTER_Y + IMAGE / 2],
                 [CARRY_X, BOTTOM + IMAGE / 2], [CANVAS_X + IMAGE + 6, BOTTOM + IMAGE / 2]], drawing);

  const labelY = BOTTOM + IMAGE + 8;
  label("scene-label", ["Scene"], SCENE.x + SCENE.size / 2, labelY, container);
  for (const [g, when] of [[now, "now"], [next, "next"]]) {
    const { row, col, scale } = g.viewpoint;
    label(`glimpse-label ${when}`, ["Glimpse", `(${[col, row, scale].map(signed).join(", ")})`], BLUE_X, labelY, container);
  }
  label("canvas-label", ["Canvas", `${manifest.canvas_grid} × ${manifest.canvas_grid}`], RED_X, labelY, container);
  label("tower-label", ["Vision Transformer"], BLUE_X - TOWER.width / 2 - 14, (TOWER.top + TOWER.bottom) / 2, container);
}
