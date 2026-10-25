// The DINOv3 slide's feature map as several views of the same features: the plot step's projections on consecutive
// triplets of principal components (maps.json's pca_cycle), shown one after another with a label naming the
// components and the features' dimension. The player has the deck's data-play contract (play, pause, restart): it
// cycles for as long as the slide is shown, through its clicks, and starts over from the first view whenever the
// slide is entered anew.

const STEP_MS = 1400;

export async function drawPcaCycle(section, src) {
  const response = await fetch(`${src}/maps.json`);
  if (!response.ok) throw new Error(`pca-cycle: ${src}/maps.json answered ${response.status}`);
  const { pca_cycle: views, feature_dim: dim } = await response.json();
  if (!views?.length || !dim) throw new Error(`pca-cycle: ${src}/maps.json has no pca_cycle or feature_dim; rerun experiments.foundation.plot`);
  const image = section.querySelector("img.features");
  const label = section.querySelector(".pca-label");
  const player = section.querySelector(".pca-player");
  if (!image || !label || !player) throw new Error(`pca-cycle: #${section.id} lacks img.features, .pca-label or .pca-player`);
  for (const view of views) new Image().src = `${src}/${view.image}`;
  const show = (i) => {
    const { components, image: file } = views[i];
    image.src = `${src}/${file}`;
    label.textContent = `Principal components ${components[0]}–${components.at(-1)} of ${dim}`;
  };
  let timer = null, i = 0;
  const tick = () => {
    show(i);
    i = (i + 1) % views.length;
    timer = setTimeout(tick, STEP_MS);
  };
  player.pause = () => { clearTimeout(timer); timer = null; };
  player.restart = () => { player.pause(); i = 0; tick(); };
  player.play = () => { if (timer === null) tick(); };
  show(0);
}
