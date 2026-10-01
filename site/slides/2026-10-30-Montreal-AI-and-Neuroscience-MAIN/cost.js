// The cost slide's latencies, from the paper's inference benchmark export (hw_bench.json, its appendix on inference
// latency): CanViT-B's minimum latency for one glimpse on a 64 × 64 canvas, as the paper reports it, per device.

const CANVAS_GRID = 64;

export async function drawCost(section, src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`cost: ${src} answered ${response.status}`);
  const { configs } = await response.json();
  for (const cell of section.querySelectorAll("[data-latency]")) {
    const [device, dtype, threads] = cell.dataset.latency.split(" ");
    const matches = configs.filter((c) => c.model === "canvit" && c.canvas_grid === CANVAS_GRID && c.device === device
                                          && c.dtype === dtype && c.num_threads_actual === Number(threads));
    if (matches.length !== 1) throw new Error(`cost: ${matches.length} configs for ${cell.dataset.latency} in ${src}`);
    const ms = matches[0].min_ms;
    cell.textContent = ms < 10 ? ms.toFixed(1) : String(Math.round(ms));
  }
}
