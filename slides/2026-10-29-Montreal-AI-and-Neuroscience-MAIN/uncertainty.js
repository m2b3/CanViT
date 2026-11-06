// #eg-c2f's data, from experiments/metacognition/export.py (cells.json, tiles.json): a sure and an unsure canvas patch
// after the first glimpse, outlined at their true size inside a ring and linked by a leader line to the class
// probabilities read out there and their entropy; then the quadrants with their mean entropy and rank. talk.css
// (#eg-c2f) sets the geometry the leader lines follow: FRAME_PX and the panels' left edge and tops.

const FRAME_PX = 480, PANEL_X = 590, PANEL_TOP = { certain: 10, uncertain: 270 }, HEAD_Y = 14;
const RING = 2.4; // the ring's radius, in canvas patches
const HEADS = { certain: "Class probabilities at a sure patch", uncertain: "Class probabilities at an unsure patch" };
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

export async function drawUncertainty(section, dir) {
  const [{ cells }, { after }] = await Promise.all([json(`${dir}/cells.json`), json(`${dir}/tiles.json`)]);
  const find = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`uncertainty: no ${selector} in #${section.id}`);
    return element;
  };
  const patches = find(".patches"), leaders = find(".leaders");
  const marks = [], lines = [];
  for (const cell of cells) {
    if (!(cell.kind in PANEL_TOP)) throw new Error(`uncertainty: a cell of kind "${cell.kind}"`);
    const grid = cell.grid;
    patches.setAttribute("viewBox", `0 0 ${grid} ${grid}`);
    for (const cls of ["halo", "outline"]) {
      marks.push(`<rect class="${cls} ${cell.kind}-mark" x="${cell.col}" y="${cell.row}" width="1" height="1"/>`);
      marks.push(`<circle class="${cls} ring ${cell.kind}-mark" cx="${cell.col + 0.5}" cy="${cell.row + 0.5}" r="${RING}"/>`);
    }
    // From the ring's right edge, out to the panel's heading.
    const [px, py] = [(cell.col + 0.5 + RING) / grid * FRAME_PX + 2, (cell.row + 0.5) / grid * FRAME_PX];
    const [qx, qy] = [PANEL_X - 12, PANEL_TOP[cell.kind] + HEAD_Y];
    lines.push(`<path class="${cell.kind}-mark" d="M${px},${py} C${px + 60},${py} ${qx - 60},${qy} ${qx},${qy}"/>`);
    const rows = [...cell.top.slice(0, 4).map(({ class: name, p }) => [name, p]), ["other", cell.top.slice(4).reduce((s, { p }) => s + p, cell.p_rest)]];
    find(`.patch.${cell.kind}`).innerHTML = `<p class="head">${HEADS[cell.kind]}</p>` + rows.map(([name, p]) =>
      // Each bar at its printed percentage, so a 0% class draws no hairline.
      `<div class="row"><span class="name">${name}</span><span class="bar" style="--p: ${Math.round(100 * p) / 100}"></span><span class="p">${percent(p)}</span></div>`).join("")
      + `<p class="entropy">Entropy: ${bits(cell.entropy_nats)} bits</p>`;
  }
  patches.innerHTML = marks.join("");
  leaders.innerHTML = lines.join("");

  // The quadrants after the first glimpse, each with its mean entropy, and their rank by it. EG-C2F re-ranks the
  // quadrants it has not visited after every glimpse; the slide shows one ranking, visited in order, so the order of
  // its visits (each the highest unvisited quadrant after the glimpse before) must be that first ranking.
  const { quadrants } = after[0];
  const same = (a, b) => Math.abs(a.row - b.row) < 1e-6 && Math.abs(a.col - b.col) < 1e-6;
  const ranking = [...quadrants].sort((a, b) => b.mean_entropy_nats - a.mean_entropy_nats);
  const visited = [];
  for (const step of after.slice(0, quadrants.length)) {
    const open = step.quadrants.filter((q) => !visited.some((v) => same(v, q)));
    const highest = open.reduce((best, q) => (q.mean_entropy_nats > best.mean_entropy_nats ? q : best));
    if (!same(highest, step.next)) throw new Error(`uncertainty: after glimpse ${step.after_glimpse + 1}, EG-C2F did not visit the highest unvisited quadrant`);
    visited.push(step.next);
  }
  if (!ranking.every((q, i) => same(q, visited[i]))) {
    throw new Error("uncertainty: EG-C2F's visits do not follow the first ranking of the quadrants; the ranks would mislead");
  }
  find(".quadrants").innerHTML = quadrants.map((q) => {
    const rank = 1 + ranking.indexOf(q);
    return `<span class="rank-${rank}" style="left: ${(q.col - q.scale + 1) * 50}%; top: ${(q.row - q.scale + 1) * 50}%; width: ${q.scale * 100}%; height: ${q.scale * 100}%">`
      + `<em class="bits"><small>mean&nbsp;</small>${bits(q.mean_entropy_nats)}<small> bits</small></em><b class="rank">${rank}</b></span>`;
  }).join("");
}
