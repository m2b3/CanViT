// Standalone compilation of the Canvas Attention figure (mechanism + dual-panel RHS):
// one .typ in, one SVG and one PDF out, which the manuscript includes as a single figure.
//
// LHS uses the imported `canvas-attention-diagram()` function so the four
// corner PCA PNGs land at one base64 nesting depth in the output SVG. RHS
// image()s the matplotlib `canvas_attention_rhs.svg`, which contains only
// vector paths.
//
// The `../../exports/` reference needs `typst compile --root <paper/>`, which
// `core.py::run_diagram` passes.
//
// Letter A is placed absolutely so it does not claim layout height inside
// the LHS cell. Letters B / C are baked into the matplotlib RHS SVG.
//
// Flip DEBUG to true to fill grid cells with translucent color so column
// boundaries and content positioning become visible.

#let DEBUG = false

#set page(width: 5.5in, height: 3.0in, margin: (left: 0.4em, top: 0em, right: 0em, bottom: 0em))
#set text(font: ("Helvetica", "Liberation Sans", "Nimbus Sans"), size: 11pt)

#import "canvas_attention_content.typ": canvas-attention-diagram

#let lhs-fill = if DEBUG { red.transparentize(85%) } else { none }
#let rhs-fill = if DEBUG { blue.transparentize(85%) } else { none }

#grid(
  columns: (5fr, 3fr),
  gutter: 0em,
  block(width: 100%, height: 100%, fill: lhs-fill)[
    #place(top + left, dx: -0.3em, dy: 0.7em, text(weight: "bold", size: 12pt)[A])
    #text(size: 8.8pt, canvas-attention-diagram())
  ],
  block(
    width: 100%, height: 100%, fill: rhs-fill,
    align(top + right, image("../../exports/canvas_attention_rhs.svg")),
  ),
)
