// Standalone compilation of the CanViT overview diagram (Figure 1).
// Usage: typst compile canvit_overview_standalone.typ canvit_overview.svg
//
// This wraps canvit_overview_content.typ in a minimal document; the manuscript includes the PDF.

// width must be concrete (not auto) because the diagram uses layout(size => ...).
// 5.5in = manuscript text width. height: auto so it shrink-wraps vertically.
// Small margin to prevent clipping of panel labels ("A", "B") and arrows
// that use negative dx/dy offsets in the diagram code.
// height: 6.3cm trims the automatic height (~6.7cm).
#set page(width: 5.5in, height: 6.3cm, margin: (left: 0.035cm, right: 0.07cm, top: 0cm, bottom: 0cm))
// Match font used by other standalone diagrams (arch_overview, canvas_attention).
#set text(font: ("Helvetica", "Liberation Sans", "Nimbus Sans"), size: 11pt)
#import "canvit_overview_content.typ": canvit-overview-diagram

#canvit-overview-diagram(
  scene-path: "outputs/canvas_evolution/Places365/scene.png",
  glimpse-width: 1.42cm,
)
