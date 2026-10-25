// The canvas introduced as the answer to integrating information across glimpses, on a rollout's recorded data: the
// scene beside the canvas at the same size, so that a glimpse's box on the scene and its footprint on the canvas sit at
// the same place. Named states: the canvas before any glimpse (the learned initial canvas); "first", glimpse 1 written;
// "second", glimpse 2 reading the canvas, then writing; "more", the remaining glimpses played one after another (the
// player has the deck's data-play contract: play, pause, restart). Canvas features in the bundle's three principal
// components, as everywhere in the talk.

const STEP_MS = 650;

function place(element, { top, left, size }) {
  Object.assign(element.style, { top: `${100 * top}%`, left: `${100 * left}%`, width: `${100 * size}%`, height: `${100 * size}%` });
}

export async function drawCanvasConcept(section, src) {
  const response = await fetch(`${src}/manifest.json`);
  if (!response.ok) throw new Error(`canvas-concept: ${src}/manifest.json answered ${response.status}`);
  const { scene, initial_canvas: initial, glimpses } = await response.json();
  if (glimpses.length < 3) throw new Error(`canvas-concept: ${src} has ${glimpses.length} glimpses; the slide needs at least 3`);
  const $ = (selector) => {
    const element = section.querySelector(selector);
    if (!element) throw new Error(`canvas-concept: no ${selector} in #${section.id}`);
    return element;
  };
  $(".scene img").src = `${src}/${scene.image}`;
  $(".canvas .initial").src = `${src}/${initial}`;
  for (const t of [0, 1]) {
    const { box, layers } = glimpses[t];
    place($(`.scene .box.g${t}`), box);
    place($(`.canvas .footprint.g${t}`), box);
    $(`.glimpse img.g${t}`).src = `${src}/${layers.crop}`;
    $(`.canvas img.g${t}`).src = `${src}/${layers.canvas}`;
  }

  // The remaining glimpses, from glimpse 3 to the last, each shown for STEP_MS, then the last held.
  const current = { box: $(".scene .box.now"), footprint: $(".canvas .footprint.now"), crop: $(".glimpse img.now"), canvas: $(".canvas img.now") };
  const label = $(".glimpse .count");
  const show = (t) => {
    const { box, layers } = glimpses[t];
    place(current.box, box);
    place(current.footprint, box);
    current.crop.src = `${src}/${layers.crop}`;
    current.canvas.src = `${src}/${layers.canvas}`;
    label.textContent = `glimpse ${t + 1}`;
  };
  for (const { layers } of glimpses.slice(2)) for (const name of ["crop", "canvas"]) new Image().src = `${src}/${layers[name]}`;
  const player = $(".player");
  let timer = null, t = 2;
  const tick = () => {
    show(t);
    if (t < glimpses.length - 1) { t += 1; timer = setTimeout(tick, STEP_MS); } else timer = null;
  };
  player.pause = () => { clearTimeout(timer); timer = null; };
  player.restart = () => { player.pause(); t = 2; tick(); };
  player.play = () => { if (timer === null && t < glimpses.length - 1) tick(); };
  show(2);
}
