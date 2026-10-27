// Standalone canvas evolution diagram - compiles to SVG/PDF
#import "canvas_evolution_content.typ": canvas-evolution-diagram

// No creation date, so identical inputs give identical files.
#set document(date: none)
#set page(width: auto, height: auto, margin: 0.5em)
#set text(font: ("Liberation Sans", "Helvetica", "Nimbus Sans"), size: 11pt)

#canvas-evolution-diagram()
