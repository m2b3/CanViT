// The detail slide's geometry, from the export's meta.json (experiments/looking_closer/plot.py): the zoomed viewpoint's
// box and the object's box, as CSS variables in fractions of the scene, and the class name in its captions.

export async function drawDetail(section, dir) {
  const response = await fetch(`${dir}/meta.json`);
  if (!response.ok) throw new Error(`detail: ${dir}/meta.json answered ${response.status}`);
  const { class: name, zoom: [row, col, scale], object_box: [top, left, bottom, right] } = await response.json();
  const style = section.style;
  // The zoom box (scale is its half side in [-1, 1] coordinates, so its side is `scale` of the scene's side).
  style.setProperty("--zoom-left", (col - scale + 1) / 2);
  style.setProperty("--zoom-top", (row - scale + 1) / 2);
  style.setProperty("--zoom-side", scale);
  style.setProperty("--object-top", top);
  style.setProperty("--object-left", left);
  style.setProperty("--object-height", bottom - top);
  style.setProperty("--object-width", right - left);
  // data-class-caption names whose probability the panel shows, on a line above it: "CanViT", then "p(television)".
  for (const caption of section.querySelectorAll("[data-class-caption]")) {
    caption.textContent = caption.dataset.classCaption ? `${caption.dataset.classCaption}\np(${name})` : `p(${name})`;
  }
}
