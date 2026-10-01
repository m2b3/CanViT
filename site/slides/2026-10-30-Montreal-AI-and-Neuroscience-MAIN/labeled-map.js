// Class names written over an annotation map: each <div class="names" data-src="names.json"> gets one span per name,
// at the position the file gives as fractions of the image (the center of the largest disk inside the class's
// region), with --i its rank, so a slide can stagger their appearance.

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
      return span;
    }));
  }
}
