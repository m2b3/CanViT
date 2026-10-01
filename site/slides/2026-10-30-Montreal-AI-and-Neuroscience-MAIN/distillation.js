// The distillation slide's numbers, from throwaway/distillation/training_step.py's steps.json: under each glimpse its
// viewpoint (x, y, scale), and between prediction and target the pretraining loss (mean squared error) after it.

const signed = (value) => value.toFixed(2).replace("-", "−");

export async function drawDistillation(section, src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`distillation: ${src} answered ${response.status}`);
  const { viewpoints_x_y_scale: viewpoints, patch_loss: losses } = await response.json();
  for (const cell of section.querySelectorAll("[data-t]")) {
    const t = Number(cell.dataset.t);
    if (!(t in viewpoints)) throw new Error(`distillation: no glimpse ${t} in ${src}`);
    if (cell.classList.contains("viewpoint")) cell.textContent = `(${viewpoints[t].map(signed).join(", ")})`;
    else if (cell.classList.contains("score")) cell.textContent = losses[t].toFixed(2);
    else throw new Error(`distillation: data-t on an element that is neither .viewpoint nor .score`);
  }
}
