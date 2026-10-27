// The paper's figures and tables on a talk's slides, from the talk's sources/paper-figures.json, which paper_figures.py
// (beside this file) writes and documents. A talk's page calls this once, before startDeck():
//
//   import { insertPaperFigures } from "../paper-figures.js";
//   await insertPaperFigures("sources/paper-figures.json");
//
// Every enabled entry becomes a slide after the slide its `where` names, or at the end of the Backup slides for
// "backup"; entries with the same `where` keep the manifest's order. An accepted entry shows its tracked render; any
// other its trial render in the talk's .paper-figures/, which is never published (deploy.sh refuses an enabled entry
// that is not accepted) and is marked as a trial unless ?present. The published deck has no Backup slides
// (published_deck.py), so an entry placed in Backup, or after a Backup slide, is left out of it.

const BACKUP = "backup";
const TRIALS = ".paper-figures";
const params = new URLSearchParams(location.search);
const presenting = params.has("present") || params.has("receiver");

/** An entry's file name and slide id: its id (an export name or a LaTeX label) with ":" replaced, as paper_figures.py
 * names its renders. */
const slug = (id) => id.replaceAll(":", "-");

// The render fills the slide's content area under its title, if any, at its own aspect ratio.
const STYLE = `
.reveal .paper-figure .paper-figure-layout { display: flex; flex-direction: column; height: 100%; }
.reveal .paper-figure figure { flex: 1; min-height: 0; margin: 0; display: flex; align-items: center; justify-content: center; }
.reveal .paper-figure figure img { display: block; width: 100%; height: 100%; max-width: none; max-height: none; object-fit: contain; }
.reveal .paper-figure .paper-figure-missing { font-size: 22px; color: var(--muted); }
.reveal .paper-figure .paper-figure-trial { position: absolute; top: 10px; left: 12px; z-index: 20; padding: 4px 10px;
  border-radius: 6px; background: #fde68a; color: #713f12; font: 600 13px/1.3 var(--sans); pointer-events: none; }
`;

/** A trial render's record, or null when the entry has not been rendered. */
async function trialRecord(entry) {
  const response = await fetch(`${TRIALS}/${slug(entry.id)}.json`);
  return response.ok ? response.json() : null;
}

async function slideFor(entry) {
  const section = document.createElement("section");
  section.id = `paper-${slug(entry.id)}`;
  section.className = "paper-figure";
  section.dataset.status = entry.status ?? "draft";
  const layout = section.appendChild(document.createElement("div"));
  layout.className = "paper-figure-layout";
  if (entry.title) layout.appendChild(document.createElement("h2")).innerHTML = entry.title;
  const figure = layout.appendChild(document.createElement("figure"));
  const trial = entry.accepted ? null : await trialRecord(entry);
  if (entry.accepted || trial) {
    const image = figure.appendChild(document.createElement("img"));
    image.src = entry.accepted ? entry.accepted.file : `${TRIALS}/${slug(entry.id)}.svg`;
    image.alt = entry.title ? layout.querySelector("h2").textContent : entry.id;
  } else {
    const missing = figure.appendChild(document.createElement("p"));
    missing.className = "paper-figure-missing";
    missing.textContent = `${entry.id} has no render: uv run site/slides/paper_figures.py try ${entry.id} --talk … --paper …`;
  }
  if (!entry.accepted && !presenting) {
    const tag = section.appendChild(document.createElement("p"));
    tag.className = "paper-figure-trial";
    tag.textContent = trial ? `Trial of ${entry.id} at ${trial.commit.slice(0, 12)}: not accepted` : `${entry.id}: not rendered`;
  }
  if (entry.notes?.length) {
    const list = section.appendChild(document.createElement("aside")).appendChild(document.createElement("ul"));
    list.parentElement.className = "notes";
    for (const note of entry.notes) list.appendChild(document.createElement("li")).innerHTML = note;
  }
  return section;
}

export async function insertPaperFigures(manifestUrl) {
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`${manifestUrl}: HTTP ${response.status}`);
  const enabled = (await response.json()).figures.filter((entry) => entry.enabled);
  if (enabled.length === 0) return;
  const slides = document.querySelector(".reveal .slides");
  const hasBackup = Boolean(slides.querySelector(":scope > section#backup"));
  const placed = new Map(); // where -> the slide last placed there
  for (const entry of enabled) {
    let after;
    if (entry.where === BACKUP) {
      if (!hasBackup) continue;
      after = placed.get(BACKUP) ?? slides.querySelector(":scope > section:last-of-type");
    } else {
      const named = slides.querySelector(`:scope > section#${CSS.escape(entry.where)}`);
      if (!named) {
        if (hasBackup) throw new Error(`${manifestUrl}: ${entry.id} has where="${entry.where}", which names no slide`);
        continue;
      }
      after = placed.get(entry.where) ?? named;
    }
    const section = await slideFor(entry);
    after.after(section);
    placed.set(entry.where, section);
  }
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.append(style);
}
