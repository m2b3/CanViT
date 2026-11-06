// Other tasks on #foundation's scene with released DINOv3 task heads: fills each `svg.detections[data-min-score]`
// inside `section` with the ViT-7B detector's COCO boxes from detections.json scoring at least that, each with its
// class name, in the scene's pixel coordinates (the SVG's viewBox). XXX: no script in experiments/ produces
// detections.json or the depth map.

const SVG = "http://www.w3.org/2000/svg";
// One color per box, distinct on a photograph and clear of the talk's meanings (glimpse blue, canvas red, policy teal,
// prior models' amber).
const BOX_COLORS = ["#ff2bd6", "#a3e635", "#ffffff", "#a78bfa", "#fde047", "#38bdf8"];
const LABEL_PX = 58;

export async function drawFoundationTasks(section, src) {
  const response = await fetch(`${src}/detections.json`);
  if (!response.ok) throw new Error(`foundation-tasks: ${src}/detections.json answered ${response.status}`);
  const { input_px: [width, height], detections } = await response.json();
  for (const svg of section.querySelectorAll("svg.detections[data-min-score]")) {
    const minScore = Number(svg.dataset.minScore);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren(...detections.filter((d) => d.score >= minScore).map(({ class: name, box: [x0, y0, x1, y1] }, i) => {
      const color = BOX_COLORS[i % BOX_COLORS.length];
      const group = document.createElementNS(SVG, "g");
      group.style.setProperty("--box", color);
      const rect = document.createElementNS(SVG, "rect");
      Object.entries({ x: x0, y: y0, width: x1 - x0, height: y1 - y0, class: "box" }).forEach(([k, v]) => rect.setAttribute(k, v));
      // The class name on a tag in the box's color: inside the box's top edge when the box reaches the top of the
      // scene, above it otherwise.
      const tagTop = y0 < LABEL_PX * 1.4 ? y0 : y0 - LABEL_PX * 1.3;
      const tag = document.createElementNS(SVG, "rect");
      // As wide as its label, estimated from the label's length (a hidden slide cannot be measured).
      const tagWidth = name.length * LABEL_PX * .64 + LABEL_PX * .6;
      Object.entries({ x: x0, y: tagTop, width: tagWidth, height: LABEL_PX * 1.3, class: "tag" }).forEach(([k, v]) => tag.setAttribute(k, v));
      const label = document.createElementNS(SVG, "text");
      Object.entries({ x: x0 + LABEL_PX * .3, y: tagTop + LABEL_PX * 1.02 }).forEach(([k, v]) => label.setAttribute(k, v));
      label.style.fontSize = `${LABEL_PX}px`;
      label.textContent = name;
      group.append(rect, tag, label);
      return group;
    }));
  }
}
