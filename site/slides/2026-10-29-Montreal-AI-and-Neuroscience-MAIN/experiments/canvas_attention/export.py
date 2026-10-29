"""The cost panels of the paper's Canvas Attention figure (fig:canvas-attention, panels B and C), for #canvas-attention,
computed by the functions that draw them, paper/exporter's flops/canvas_attn_rhs.py, so the slide's chart and the paper's
plot the same numbers.

Writes <deck data>/canvas-attention/flops.json: panel B's total inference FLOPs against the output patch grid (every grid
from 8 to 128, as the paper's curves are drawn) for DINOv3 ViT-B/16 and ViT-S/16 on the whole scene and for CanViT-B
after 1 and 21 glimpses; panel C's FLOP ratio of canvas-side QKVO projections to the asymmetric ones, per Canvas
Attention Read/Write pair, by canvas and glimpse patch grid.

    uv run --project ../../../paper/exporter python -m experiments.canvas_attention.export   # from the deck
"""

import json

from canvit_paper_exporter.flops import canvas_attn_rhs as rhs

from experiments.outputs import DECK_DATA


def main() -> None:
    grids = list(range(rhs.SPATIAL_GRIDS[0], rhs.SPATIAL_GRIDS[-1] + 1))
    early, late = rhs.T_EARLY + 1, rhs.T_LATE + 1
    total = [
        {"label": "DINOv3 ViT-B", "family": "dinov3", "weight": "thick",
         "flops": [rhs._dinov3("DINOv3 ViT-B/16", g) for g in grids]},
        {"label": "DINOv3 ViT-S", "family": "dinov3", "weight": "thin",
         "flops": [rhs._dinov3("DINOv3 ViT-S/16", g) for g in grids]},
        {"label": f"CanViT-B, {late} glimpses", "family": "canvit", "weight": "thick",
         "flops": [rhs._canvit_forward(g) * late for g in grids]},
        {"label": f"CanViT-B, {early} glimpse", "family": "canvit", "weight": "thin",
         "flops": [rhs._canvit_forward(g) * early for g in grids]},
    ]
    ratio = [
        {"glimpse_grid": lg,
         "ratio": [rhs._rw_pair_flops(lg, cg, full_proj=True) / rhs._rw_pair_flops(lg, cg, full_proj=False)
                   for cg in rhs.CANVAS_GRIDS]}
        for lg in rhs.GLIMPSE_GRIDS
    ]
    out = DECK_DATA / "canvas-attention"
    out.mkdir(parents=True, exist_ok=True)
    (out / "flops.json").write_text(json.dumps({
        "_about": __doc__.split("\n\n")[1].replace("\n", " "),
        "total_flops": {"grids": grids, "ticks": list(rhs.SPATIAL_GRIDS), "series": total},
        "projection_ratio": {"canvas_grids": list(rhs.CANVAS_GRIDS), "series": ratio},
    }, indent=1) + "\n")
    print(f"wrote {out / 'flops.json'}")


if __name__ == "__main__":
    main()
