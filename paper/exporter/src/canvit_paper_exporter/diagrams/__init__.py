"""Diagram registry.

Two-stage pipeline for each diagram:
    canvit_paper_exporter.diagrams.<name>.main()   # emits PNGs to diagrams/outputs/<name>/
    typst compile diagrams/<name>.typ     # composes PNGs -> exports/<name>.svg

Register as Diagram(name, prep, typst_source) so `run.py` can dispatch them
alongside Dataset and Figure; every producer lands its output at
`exports/<name>.<ext>`.

Run one:  uv run python -m canvit_paper_exporter.run arch_overview
Diagrams are not part of the no-argument run; `run list` names them.
"""

from canvit_paper_exporter.core import Diagram, run_figure
from canvit_paper_exporter.diagrams import (
    _common,
    arch_overview,
    canvas_attn,
    canvas_write_evolution,
    canvit_overview,
)
from canvit_paper_exporter.flops import canvas_attn_rhs


def _mk(name: str, module, typst_name: str) -> Diagram:
    """Wrap module.main(module.Config()) as the no-arg prep callable `run_diagram`
    expects. CLI overrides (`python -m canvit_paper_exporter.diagrams.<mod>`) still work
    via each module's own `if __name__ == '__main__'` tyro entrypoint."""
    return Diagram(
        name=name,
        prep=lambda m=module: m.main(m.Config()),
        typst_source=_common.DIAGRAMS_DIR / f"{typst_name}.typ",
    )


def _canvas_attention_combined_prep() -> None:
    """Prep for the combined Fig 3: refresh the LHS PCA PNGs and the RHS SVG.

    The combined wrapper imports `canvas_attention_content.typ` directly and
    reads the PCA PNGs from `diagrams/outputs/canvas_attn/` — no separate
    canvas_attention.svg intermediate. So prep just runs the PNG generator
    and the matplotlib RHS figure; the wrapper's own typst compile composes
    them.
    """
    canvas_attn.main(canvas_attn.Config())
    run_figure(canvas_attn_rhs.figure)


DIAGRAMS: list[Diagram] = [
    _mk("arch_overview",    arch_overview,          "arch_overview"),
    _mk("canvas_evolution", canvas_write_evolution, "canvas_evolution"),
    _mk("canvit_overview",  canvit_overview,        "canvit_overview_standalone"),
    Diagram(
        name="canvas_attention_combined",
        prep=_canvas_attention_combined_prep,
        typst_source=_common.DIAGRAMS_DIR / "canvas_attention_combined_standalone.typ",
    ),
]
