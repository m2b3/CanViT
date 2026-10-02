// Class names written over an annotation map: each <div class="names" data-src="names.json"> gets one span per name,
// at the position the file gives as fractions of the image (the center of the largest disk inside the class's
// region), with --i its rank, so a slide can stagger their appearance. A name whose center is too close to the map's
// side for its width moves inward until it fits, once it is laid out.

const keepInside = new ResizeObserver((entries) => {
  for (const { target: span } of entries) {
    const width = span.parentElement.offsetWidth, half = span.offsetWidth / 2;
    if (!width || !half) continue;
    span.style.left = `${Math.min(Math.max(Number(span.dataset.x) * width, half), width - half)}px`;
  }
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
      span.dataset.x = x;
      keepInside.observe(span);
      return span;
    }));
  }
}
