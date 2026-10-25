// The EG-C2F slide's data, drawn from the export of experiments/metacognition/export.py (a directory holding
// cells.json and tiles.json): rings on the scene at a sure and an unsure canvas cell (solid and dashed, without
// words), the class probabilities read out there as bars under a glyph of the same ring, and the four quadrants with
// the one EG-C2F visits next.

const ringLabels = { certain: "sure", uncertain: "unsure" };
// Entropy in bits, as the map's legend shows it: 0 for one class, log2(150) ≈ 7.2 for all 150 equally likely.
const bits = (nats) => (nats / Math.LN2).toFixed(nats / Math.LN2 < 1 ? 2 : 1);
// A probability as a percentage that never reads as certainty it is not: 0.997 is "99.7%", not "100%".
const percent = (p) => {
  const value = 100 * p;
  if (value > 99.5 && value < 100) return `${value.toFixed(1)}%`;
  if (value > 0 && value < 0.5) return "<1%";
  return `${Math.round(value)}%`;
};

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`uncertainty: ${url} answered ${response.status}`);
  return response.json();
}

/** Fill section's .rings, .bars.certain, .bars.uncertain and .quadrants from the export in dir. */
export async function drawUncertainty(section, dir) {
  const [{ cells }, { after }] = await Promise.all([json(`${dir}/cells.json`), json(`${dir}/tiles.json`)]);
  const SVG = "http://www.w3.org/2000/svg";
  const rings = section.querySelector(".rings");
  for (const cell of cells) {
    const [cx, cy] = [(cell.col + 0.5) / cell.grid * 100, (cell.row + 0.5) / cell.grid * 100];
    const g = document.createElementNS(SVG, "g");
    g.setAttribute("class", `ring-${cell.kind}`);
    g.innerHTML = `<circle class="halo" cx="${cx}" cy="${cy}" r="5"/><circle class="ring" cx="${cx}" cy="${cy}" r="5"/>`;
    rings.append(g);

    const bars = section.querySelector(`.bars.${cell.kind}`);
    const rows = [...cell.top.slice(0, 4).map(({ class: name, p }) => [name, p]), ["other", cell.top.slice(4).reduce((s, { p }) => s + p, cell.p_rest)]];
    const glyph = `<svg class="glyph ring-${cell.kind}" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/></svg>`;
    bars.innerHTML = `<p class="head">${glyph}${ringLabels[cell.kind]}</p>` + rows.map(([name, p]) =>
      // Each bar at its printed percentage, so a 0% class draws no hairline.
      `<div class="row"><span class="name">${name}</span><span class="bar" style="--p: ${Math.round(100 * p) / 100}"></span><span class="p">${percent(p)}</span></div>`).join("")
      + `<p class="entropy">this cell's entropy: ${bits(cell.entropy_nats)} bits</p>`;
  }

  // The quadrants after the first glimpse, the one with the highest mean entropy marked: where EG-C2F looks next.
  const { quadrants } = after[0];
  const highest = quadrants.reduce((best, q) => (q.mean_entropy_nats > best.mean_entropy_nats ? q : best));
  const next = after[0].next;
  if (Math.abs(highest.row - next.row) > 1e-6 || Math.abs(highest.col - next.col) > 1e-6) {
    throw new Error("uncertainty: the quadrant of highest mean entropy is not the one EG-C2F visited next");
  }
  section.querySelector(".quadrants").innerHTML = quadrants.map((q) =>
    `<span class="${q === highest ? "next" : ""}" style="left: ${(q.col - q.scale + 1) * 50}%; top: ${(q.row - q.scale + 1) * 50}%;`
    + ` width: ${q.scale * 100}%; height: ${q.scale * 100}%"><small>mean&nbsp;</small>${bits(q.mean_entropy_nats)}<small> bits</small></span>`).join("");
}
