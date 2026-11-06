// A feature map as successive views on triplets of principal components (the manifest's pca_cycle, written by
// experiments/foundation/pca_colors.pca_cycle), shown in the section's img.features with .pca-label naming them. Follows
// the deck's data-play contract: it cycles while the slide is shown and starts over when the slide is entered anew.

const STEP_MS = 1400;

/** lastComponent: the cycle stops at the view ending on that principal component, then starts over (all views if absent). */
export async function drawPcaCycle(section, src, manifest, { lastComponent = Infinity } = {}) {
  const response = await fetch(`${src}/${manifest}`);
  if (!response.ok) throw new Error(`pca-cycle: ${src}/${manifest} answered ${response.status}`);
  const { pca_cycle: all, feature_dim: dim } = await response.json();
  if (!all?.length || !dim) throw new Error(`pca-cycle: ${src}/${manifest} has no pca_cycle or feature_dim; rerun its plot step`);
  const views = all.filter((view) => view.components.at(-1) <= lastComponent);
  if (!views.length) throw new Error(`pca-cycle: ${src}/${manifest} has no view within components 1-${lastComponent}`);
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
