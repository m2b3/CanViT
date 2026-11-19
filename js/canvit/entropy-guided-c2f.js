// Entropy-Guided C2F (EG-C2F) for one scene, as canvit_pytorch.policies.EntropyGuidedC2F: the quadtree levels
// in order, and within a level the unvisited tile whose canvas cells have the highest mean predictive entropy.
// The tiles come from the live model's manifest (policy.levels, (row, col, scale) each); a canvas cell belongs
// to a tile when its center lies in the tile, as in tile_masks.

export class EntropyGuidedC2F {
  #levels;
  #cells;
  #visited = null;
  t = 0;

  constructor({ levels, grid }) {
    this.#levels = levels;
    const center = (i) => ((i + 0.5) / grid) * 2 - 1;
    this.#cells = levels.map((tiles) => tiles.map(([row, col, scale]) => {
      const cells = [];
      for (let i = 0; i < grid; i++) {
        for (let j = 0; j < grid; j++) {
          if (Math.abs(center(i) - row) <= scale && Math.abs(center(j) - col) <= scale) cells.push(i * grid + j);
        }
      }
      return cells;
    }));
    this.length = levels.reduce((n, tiles) => n + tiles.length, 0);
  }

  get done() { return this.t >= this.length; }

  /** The next viewpoint {row, col, scale}, given the entropy of the current canvas's cells (ignored at t = 0). */
  next(entropy) {
    if (this.done) throw new Error(`EG-C2F takes ${this.length} glimpses; this episode is over`);
    let level = 0, start = 0;
    while (this.t >= start + this.#levels[level].length) start += this.#levels[level++].length;
    const tiles = this.#levels[level];
    if (this.t === start) this.#visited = tiles.map(() => false);
    this.t++;
    let chosen = 0;
    if (tiles.length > 1) {
      let best = -Infinity;
      this.#cells[level].forEach((cells, k) => {
        if (this.#visited[k]) return;
        let sum = 0;
        for (const i of cells) sum += entropy[i];
        const score = sum / Math.max(cells.length, 1);
        if (score > best) { best = score; chosen = k; } // the first maximum, as torch.argmax
      });
    }
    this.#visited[chosen] = true;
    const [row, col, scale] = tiles[chosen];
    return { row, col, scale };
  }
}
