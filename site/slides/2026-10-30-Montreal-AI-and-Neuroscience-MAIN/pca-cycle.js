// A feature map as several views of the same features, on the DINOv3 slide and on "Processing a glimpse with a ViT":
// the plot step's projections on consecutive triplets of principal components (its manifest's pca_cycle, written by
// experiments/foundation/pca_colors.pca_cycle), shown one after another in the section's img.features, with
// .pca-label naming the components and the features' dimension. The player has the deck's data-play contract (play, pause, restart): it
// cycles for as long as the slide is shown, through its clicks, and starts over from the first view whenever the
// slide is entered anew.

const STEP_MS = 1400;

export async function drawPcaCycle(section, src, manifest) {
  const response = await fetch(`${src}/${manifest}`);
  if (!response.ok) throw new Error(`pca-cycle: ${src}/${manifest} answered ${response.status}`);
  const { pca_cycle: views, feature_dim: dim } = await response.json();
  if (!views?.length || !dim) throw new Error(`pca-cycle: ${src}/${manifest} has no pca_cycle or feature_dim; rerun its plot step`);
  const image = section.querySelector("img.features");
  const label = section.querySelector(".pca-label");
  const player = section.querySelector(".pca-player");
  if (!image || !label || !player) throw new Error(`pca-cycle: #${section.id} lacks img.features, .pca-label or .pca-player`);
  for (const view of views) new Image().src = `${src}/${view.image}`;
  const show = (i) => {
    const { components, image: file } = views[i];
    image.src = `${src}/${file}`;
    label.textContent = `PCs ${components[0]}–${components.at(-1)} of ${dim}`;
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
