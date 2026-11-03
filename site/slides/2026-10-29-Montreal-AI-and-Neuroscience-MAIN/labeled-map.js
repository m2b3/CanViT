// Class names written over an annotation map: each <div class="names" data-src="names.json"> gets one span per name,
// at the position the file gives as fractions of the image (the center of the largest disk inside the class's
// region), with --i its rank, so a slide can stagger their appearance. Once laid out, the names are placed in the
// file's order (largest regions first): a name too close to the map's side moves inward until it fits, and a name that
// would touch one placed before it moves to that one's right (below it when there is no room), until it touches none.

const GAP_PX = 4; // the least horizontal space between two names on the same line; more moves names off their regions

function layOut(container) {
  const width = container.offsetWidth, height = container.offsetHeight;
  if (!width || !height) return;
  const placed = [];
  for (const span of container.children) {
    const w = span.offsetWidth, h = span.offsetHeight;
    if (!w) continue;
    const inside = (x) => Math.min(Math.max(x, w / 2), width - w / 2);
    let x = inside(Number(span.dataset.x) * width), y = Number(span.dataset.y) * height;
    const touched = () => placed.find((o) => Math.abs(x - o.x) < (w + o.w) / 2 + GAP_PX && Math.abs(y - o.y) < (h + o.h) / 2);
    for (let other = touched(), moves = 0; other && moves < placed.length * 2; other = touched(), moves++) {
      const right = other.x + (w + other.w) / 2 + GAP_PX;
      if (inside(right) === right) x = right;
      else y = other.y + (h + other.h) / 2;
    }
    span.style.left = `${x}px`;
    span.style.top = `${y}px`;
    placed.push({ x, y, w, h });
  }
}

const resized = new ResizeObserver((entries) => {
  for (const container of new Set(entries.map(({ target }) => target.closest(".names")))) layOut(container);
});

export async function drawMapNames() {
  for (const container of document.querySelectorAll(".names[data-src]")) {
    const response = await fetch(container.dataset.src);
    if (!response.ok) throw new Error(`map names: ${container.dataset.src} answered ${response.status}`);
    const { names } = await response.json();
    container.replaceChildren(...names.map(({ name, x, y }, i) => {
      const span = document.createElement("span");
      span.textContent = name;
      span.style.left = `${100 * x}%`;
      span.style.top = `${100 * y}%`;
      span.style.setProperty("--i", i);
      Object.assign(span.dataset, { x, y });
      return span;
    }));
    for (const node of [container, ...container.children]) resized.observe(node);
  }
}
