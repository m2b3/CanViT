// The architecture slide's drawing, on a rollout recorded with each Write's canvas (a web bundle, --capture-writes):
// the scene and glimpse t's viewpoint; the glimpse, cut into patches, up through the Vision Transformer's blocks;
// beside them the canvas stream: the canvas before the glimpse (the initial canvas before the first), then the canvas
// after each Write, level with it, as in the paper's canvas-evolution figure; the writes into it and the reads out of
// it after the blocks the manifest names. Glimpse t + 1 then replaces glimpse t, the canvas after it starting the
// stream. Positions are in the drawing's own pixels (SIZE); talk.css shows its parts on the slide's named states.

const NS = "http://www.w3.org/2000/svg";
const SIZE = { width: 1000, height: 500 };
const SCENE = { x: 0, y: 185, size: 310 };
const TOWER = { x: 400, width: 100, top: 55, bottom: 365, gap: 6 };
const GLIMPSE = { size: 110, y: 385 };
const CANVAS = { x: 760, size: 96, beforeY: 399 };
const BLUE_X = TOWER.x + TOWER.width / 2, RED_X = CANVAS.x + CANVAS.size / 2;
const GLIMPSE_X = BLUE_X - GLIMPSE.size / 2;
const NOTE_X = CANVAS.x + CANVAS.size + 16; // labels right of the canvas stream

const signed = (value) => value.toFixed(2).replace("-", "−");

function svg(name, attributes, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  parent.append(node);
  return node;
}

function image(className, src, { x, y, size }, parent) {
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
  const [now, next] = [glimpse(t), glimpse(t + 1)];
  const { num_blocks: blocks, read_after_blocks: reads, write_after_blocks: writes } = manifest.model;
  const layer = (g, name) => {
    if (!(name in g.layers)) throw new Error(`architecture: glimpse ${g.t} of ${src} has no ${name} layer (record it with --capture-writes)`);
    return `${src}/${g.layers[name]}`;
  };

  container.replaceChildren();
  Object.assign(container.style, { width: `${SIZE.width}px`, height: `${SIZE.height}px` });
  // Two layers of lines: the canvas stream under the images, everything else over them.
  const under = svg("svg", { class: "wires under", viewBox: `0 0 ${SIZE.width} ${SIZE.height}` }, container);
  const drawing = svg("svg", { class: "wires", viewBox: `0 0 ${SIZE.width} ${SIZE.height}` }, container);
  const height = (TOWER.bottom - TOWER.top - (blocks - 1) * TOWER.gap) / blocks;
  const blockTop = (k) => TOWER.bottom - (k + 1) * height - k * TOWER.gap;
  const after = (k) => blockTop(k) - TOWER.gap / 2; // the height of an exchange after block k

  // The scene, the viewpoint of each glimpse, and the crop it gives.
  image("scene", `${src}/${manifest.scene.image}`, SCENE, container);
  for (const [g, when] of [[now, "now"], [next, "next"]]) {
    const { top, left, size } = g.box;
    svg("rect", { class: `box ${when}`, x: SCENE.x + left * SCENE.size, y: SCENE.y + top * SCENE.size,
                  width: size * SCENE.size, height: size * SCENE.size }, drawing);
  }
  const cropY = GLIMPSE.y + GLIMPSE.size / 2;
  wire("feed", [[SCENE.x + SCENE.size + 10, cropY], [GLIMPSE_X - 10, cropY]], drawing);

  // The glimpse stream: the glimpse in its patches, up through the blocks.
  const glimpseBox = { x: GLIMPSE_X, y: GLIMPSE.y, size: GLIMPSE.size };
  image("glimpse pixels now", layer(now, "crop"), glimpseBox, container);
  image("glimpse pixels next", layer(next, "crop"), glimpseBox, container);
  const patches = Math.round(manifest.glimpse_px / manifest.model.patch_px);
  const grid = svg("g", { class: "patches" }, drawing);
  for (let i = 1; i < patches; i++) {
    const offset = (i * GLIMPSE.size) / patches;
    svg("line", { x1: GLIMPSE_X + offset, x2: GLIMPSE_X + offset, y1: GLIMPSE.y, y2: GLIMPSE.y + GLIMPSE.size }, grid);
    svg("line", { x1: GLIMPSE_X, x2: GLIMPSE_X + GLIMPSE.size, y1: GLIMPSE.y + offset, y2: GLIMPSE.y + offset }, grid);
  }
  svg("line", { class: "glimpse-stream", x1: BLUE_X, x2: BLUE_X, y1: GLIMPSE.y, y2: TOWER.top - 12 }, drawing);
  for (let k = 0; k < blocks; k++) {
    svg("rect", { class: "block", x: TOWER.x, y: blockTop(k), width: TOWER.width, height, rx: 4 }, drawing);
  }

  // The canvas stream: the canvas before the glimpse at the bottom, then the canvas after each Write, level with it.
  const beforeBox = { x: CANVAS.x, y: CANVAS.beforeY, size: CANVAS.size };
  svg("line", { class: "canvas-stream", x1: RED_X, x2: RED_X, y1: CANVAS.beforeY, y2: after(writes.at(-1)) }, under);
  const canvasBefore = t === 0 ? `${src}/${manifest.initial_canvas}` : layer(glimpse(t - 1), "canvas");
  image("canvas pixels before now", canvasBefore, beforeBox, container);
  image("canvas pixels before next", layer(now, "canvas"), beforeBox, container);
  label("canvas-label", ["Canvas", `${manifest.canvas_grid} × ${manifest.canvas_grid}`], NOTE_X, CANVAS.beforeY + 26, container);

  // Reads and writes after the blocks the manifest names, numbered in depth order for the build's timing; after each
  // Write, the canvas it leaves.
  const exchanges = [...reads.map((k) => ({ k, kind: "read" })), ...writes.map((k) => ({ k, kind: "write" }))]
    .sort((a, b) => a.k - b.k);
  const seen = { read: 0, write: 0 };
  const towerRight = TOWER.x + TOWER.width + 3;
  exchanges.forEach(({ k, kind }, depth) => {
    if (!(k >= 0 && k < blocks)) throw new Error(`architecture: ${kind} after block ${k} of ${blocks}`);
    const y = after(k), order = seen[kind]++;
    const ends = kind === "write" ? [[towerRight, y], [CANVAS.x - 3, y]] : [[RED_X, y], [towerRight, y]];
    wire(`exchange ${kind}`, ends, drawing).style.cssText = `--depth: ${depth}; --order: ${order}`;
    if (order === 0) {
      svg("text", { class: `exchange-label ${kind}`, x: (towerRight + CANVAS.x) / 2, y: y - 8, "text-anchor": "middle" }, drawing)
        .textContent = kind;
    }
    if (kind === "write") {
      const box = { x: CANVAS.x, y: y - CANVAS.size / 2, size: CANVAS.size };
      for (const [g, when] of [[now, "now"], [next, "next"]]) {
        image(`canvas pixels snapshot ${when}`, layer(g, `write${order}_canvas`), box, container)
          .style.cssText += `--depth: ${depth}; --order: ${order}`;
      }
      label("snapshot-label", [`after write ${order + 1}`], NOTE_X, y - 12, container).style.cssText += `--depth: ${depth}; --order: ${order}`;
    }
  });

  label("heading scene-label", ["Scene"], SCENE.x + SCENE.size / 2, SCENE.y - 40, container);
  label("heading tower-label", ["Vision Transformer"], BLUE_X, 0, container);
  for (const [g, when] of [[now, "now"], [next, "next"]]) {
    const { row, col, scale } = g.viewpoint;
    label(`glimpse-label ${when}`, ["Glimpse", `(${[col, row, scale].map(signed).join(", ")})`],
          GLIMPSE_X + GLIMPSE.size + 18, GLIMPSE.y + GLIMPSE.size / 2 - 26, container);
  }
}
