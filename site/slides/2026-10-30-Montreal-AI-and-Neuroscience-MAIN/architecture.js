// The architecture slide's drawing, on a rollout recorded with each Write's canvas (a web bundle, --capture-writes),
// with depth running left to right: the scene and glimpse t's viewpoint at the left, the glimpse below it; the glimpse
// stream through the Vision Transformer's blocks along the bottom; the canvas stream along the top: the canvas before
// the glimpse (the initial canvas before the first), then the canvas after each Write, above it, as in the paper's
// canvas-evolution figure; writes up into the canvas and reads down out of it, after the blocks the manifest names.
// Glimpse t + 1 then replaces glimpse t, the canvas after it starting the stream. Positions are in the drawing's own
// pixels (SIZE); talk.css shows its parts on the slide's named states.

const NS = "http://www.w3.org/2000/svg";
const SIZE = { width: 1120, height: 515 };
const CAPTION_Y = 0;
const SCENE = { x: 0, y: 30, size: 230 };
const GLIMPSE = { x: 55, y: 330, size: 120 };
const BLOCKS = { x: 330, width: 38, pitch: 60, top: 330, bottom: 450 };
const CANVAS = { y: 30, size: 150 };
const STREAM_Y = (BLOCKS.top + BLOCKS.bottom) / 2; // the glimpse stream's line
const RED_Y = CANVAS.y + CANVAS.size / 2; // the canvas stream's line
const LABEL_Y = BLOCKS.bottom + 14;

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

/** Lines of text centered on x, from y down. */
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
  const blockX = (k) => BLOCKS.x + k * BLOCKS.pitch;
  const after = (k) => blockX(k) + BLOCKS.width + (BLOCKS.pitch - BLOCKS.width) / 2; // between block k and k + 1
  const firstRead = after(Math.min(...reads));

  container.replaceChildren();
  Object.assign(container.style, { width: `${SIZE.width}px`, height: `${SIZE.height}px` });
  // Two layers of lines: the streams under the images and blocks, everything else over them.
  const under = svg("svg", { class: "wires under", viewBox: `0 0 ${SIZE.width} ${SIZE.height}` }, container);
  const drawing = svg("svg", { class: "wires", viewBox: `0 0 ${SIZE.width} ${SIZE.height}` }, container);

  // The scene, the viewpoint of each glimpse, and the crop it gives, below it.
  image("scene", `${src}/${manifest.scene.image}`, SCENE, container);
  label("caption scene-label", ["Scene"], SCENE.x + SCENE.size / 2, CAPTION_Y, container);
  for (const [g, when] of [[now, "now"], [next, "next"]]) {
    const { top, left, size } = g.box;
    svg("rect", { class: `box ${when}`, x: SCENE.x + left * SCENE.size, y: SCENE.y + top * SCENE.size,
                  width: size * SCENE.size, height: size * SCENE.size }, drawing);
  }
  const glimpseCenter = GLIMPSE.x + GLIMPSE.size / 2;
  wire("feed", [[glimpseCenter, SCENE.y + SCENE.size + 10], [glimpseCenter, GLIMPSE.y - 10]], drawing);

  // The glimpse stream: the glimpse in its patches, then the blocks, left to right.
  image("glimpse pixels now", layer(now, "crop"), GLIMPSE, container);
  image("glimpse pixels next", layer(next, "crop"), GLIMPSE, container);
  const patches = Math.round(manifest.glimpse_px / manifest.model.patch_px);
  const grid = svg("g", { class: "patches" }, drawing);
  for (let i = 1; i < patches; i++) {
    const offset = (i * GLIMPSE.size) / patches;
    svg("line", { x1: GLIMPSE.x + offset, x2: GLIMPSE.x + offset, y1: GLIMPSE.y, y2: GLIMPSE.y + GLIMPSE.size }, grid);
    svg("line", { x1: GLIMPSE.x, x2: GLIMPSE.x + GLIMPSE.size, y1: GLIMPSE.y + offset, y2: GLIMPSE.y + offset }, grid);
  }
  svg("line", { class: "glimpse-stream", x1: GLIMPSE.x + GLIMPSE.size, x2: blockX(blocks - 1) + BLOCKS.width + 24,
                y1: STREAM_Y, y2: STREAM_Y }, under);
  for (let k = 0; k < blocks; k++) {
    svg("rect", { class: "block", x: blockX(k), y: BLOCKS.top, width: BLOCKS.width, height: BLOCKS.bottom - BLOCKS.top, rx: 5 }, under);
  }
  for (const [g, when] of [[now, "now"], [next, "next"]]) {
    const { row, col, scale } = g.viewpoint;
    label(`glimpse-label ${when}`, ["Glimpse", `(${[col, row, scale].map(signed).join(", ")})`], glimpseCenter, LABEL_Y, container);
  }
  label("tower-label", ["Vision Transformer"], (blockX(0) + blockX(blocks - 1) + BLOCKS.width) / 2, LABEL_Y, container);

  // The canvas stream: the canvas before the glimpse, left of the first read, then the canvas after each Write.
  const before = { x: firstRead - 16 - CANVAS.size, y: CANVAS.y, size: CANVAS.size };
  const canvasBefore = t === 0 ? `${src}/${manifest.initial_canvas}` : layer(glimpse(t - 1), "canvas");
  image("canvas pixels before now", canvasBefore, before, container);
  image("canvas pixels before next", layer(now, "canvas"), before, container);
  label("caption canvas-label", ["Canvas"], before.x + CANVAS.size / 2, CAPTION_Y, container);
  svg("line", { class: "canvas-stream", x1: before.x + CANVAS.size / 2, x2: after(writes.at(-1)), y1: RED_Y, y2: RED_Y }, under);

  // Reads and writes after the blocks the manifest names, numbered in depth order for the build's timing; above each
  // Write, the canvas it leaves.
  const exchanges = [...reads.map((k) => ({ k, kind: "read" })), ...writes.map((k) => ({ k, kind: "write" }))]
    .sort((a, b) => a.k - b.k);
  const seen = { read: 0, write: 0 };
  exchanges.forEach(({ k, kind }, depth) => {
    if (!(k >= 0 && k < blocks)) throw new Error(`architecture: ${kind} after block ${k} of ${blocks}`);
    const x = after(k), order = seen[kind]++;
    const ends = kind === "write" ? [[x, STREAM_Y - 4], [x, CANVAS.y + CANVAS.size + 4]] : [[x, RED_Y], [x, STREAM_Y - 6]];
    wire(`exchange ${kind}`, ends, drawing).style.cssText = `--depth: ${depth}; --order: ${order}`;
    if (order === 0) {
      svg("text", { class: `exchange-label ${kind}`, x: x + 10, y: (CANVAS.y + CANVAS.size + BLOCKS.top) / 2 + 6 }, drawing)
        .textContent = kind;
    }
    if (kind === "write") {
      const box = { x: x - CANVAS.size / 2, y: CANVAS.y, size: CANVAS.size };
      for (const [g, when] of [[now, "now"], [next, "next"]]) {
        image(`canvas pixels snapshot ${when}`, layer(g, `write${order}_canvas`), box, container)
          .style.cssText += `--depth: ${depth}; --order: ${order}`;
      }
      label("caption snapshot-label", [`after write ${order + 1}`], x, CAPTION_Y, container)
        .style.cssText += `--depth: ${depth}; --order: ${order}`;
    }
  });
}
