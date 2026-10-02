// The distillation slide's numbers, from experiments/distillation/training_step.py's steps.json: under each glimpse its
// viewpoint (x, y, scale), and between prediction and target the pretraining loss (mean squared error) after it,
// colored from orange (the loss of predicting the average target everywhere) to green (no error).

const signed = (value) => value.toFixed(2).replace("-", "−");
const WORST = [194, 65, 12], BEST = [21, 128, 61];  // orange-700, green-700
const lossColor = (loss, averageLoss) => {
  const quality = Math.min(Math.max(1 - loss / averageLoss, 0), 1);
  return `rgb(${WORST.map((w, i) => Math.round(w + (BEST[i] - w) * quality)).join(", ")})`;
};

export async function drawDistillation(section, src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`distillation: ${src} answered ${response.status}`);
  const { viewpoints_x_y_scale: viewpoints, patch_loss: losses, zeros_patch_loss: averageLoss } = await response.json();
  for (const cell of section.querySelectorAll("[data-t]")) {
    const t = Number(cell.dataset.t);
    if (!(t in viewpoints)) throw new Error(`distillation: no glimpse ${t} in ${src}`);
    if (cell.classList.contains("viewpoint")) cell.textContent = `(${viewpoints[t].map(signed).join(", ")})`;
    else if (cell.classList.contains("score")) {
      cell.textContent = losses[t].toFixed(2);
      cell.style.color = lossColor(losses[t], averageLoss);
    }
    else throw new Error(`distillation: data-t on an element that is neither .viewpoint nor .score`);
  }
}
