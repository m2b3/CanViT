// Canvas write evolution diagram: the canvas after each residual Write within a glimpse.
// Built by `uv run python -m canvit_paper_exporter.run canvas_evolution`, which renders the
// PCA snapshots (diagrams/canvas_write_evolution.py) and then compiles this diagram.

// -------------------------
// Residual Row Template
// -------------------------

#let residual-row(
  g-idx, 
  coords: "", 
  img-dir: "outputs/canvas_evolution/cat", 
  width: 56.69pt,
) = {
  
  // Helper for consistent path generation
  let g-str = "g" + str(g-idx)
  let img(name) = image(img-dir + "/" + name + ".png", width: width)
  let sym-style(txt) = box(
    fill: white,
    inset: 2pt,
    radius: 3pt,
    text(size: 10pt, weight: "bold", fill: gray.darken(70%))[#txt]
  )

  // Adding glimpse contour
  // Example: "s0.20_c-0.5_-0.5" -> scale 0.2, x -0.5, y -0.5
  let parts = coords.split("_")
  let s = float(parts.at(0).slice(1)) // Extracts 0.20
  let cy = float(parts.at(1).slice(1))  // "c-0.5"  → -0.5
  let cx = float(parts.at(2))           // "-0.5"   → -0.5  
  
  let full_with_contour = box(width: width, height: width, {         
    image(img-dir + "/scene.png", width: 100%)                       
    place(                                                           
      top + left,                                                    
      dx: (cx - s + 1) / 2 * width,                                  
      dy: (cy - s + 1) / 2 * width,                                  
      rect(       
        width:  s * width,                                           
        height: s * width,
        stroke: (dash: "dashed", paint: white),                  
      )           
    )
  })
  // Timestamp index
  let time-label = rotate(-90deg, reflow: true)[
    #text(size: 7pt, weight: "bold", fill: gray.darken(30%))[t=#g-idx]
  ]
  
  // Creating the row
  (
    align(center + horizon, time-label),
    full_with_contour,
    // Visual separator to Glimpse
    img(g-str + "_glimpse_" + coords),
    sym-style("→"),                 // Visual separator to initial canvas
    img(g-str + "_initial"), 
    sym-style("+"), 
    img(g-str + "_write0_residual"), 
    sym-style("="), 
    img(g-str + "_write0_updated"),
    sym-style("+"),
    img(g-str + "_write1_residual"),
    sym-style("="),
    img(g-str + "_write1_updated"),
    sym-style("+"),
    img(g-str + "_write2_residual"),
    sym-style("="),
    img(g-str + "_write2_updated")
  )
}

// -------------------------
// Helpers
// -------------------------
#let sub-head(txt) = text(size: 8pt, weight: "light", fill: gray.darken(20%))[#txt]

#let residual-box(x-offset) = place(
  dx: x-offset,
  dy: 0pt, 
  rect(
    width: 61pt, 
    height: 388pt, 
    fill: gray.lighten(75%),
    stroke: (dash: "dashed", paint: gray.lighten(40%))
  )
)

// -------------------------
// Main Evolution Diagram
// -------------------------
// Coords loaded from metadata.json written by canvas_write_evolution.py.
// No hardcoded viewpoint coords — single source of truth is the Python script.
#let _base = "outputs/canvas_evolution"
#let _meta_cat = json(_base + "/Cat03/metadata.json")
#let _meta_places = json(_base + "/Places365/metadata.json")

#let canvas-evolution-diagram(
  img-dir: _base,
  glimpse-coords-cat: _meta_cat.glimpse_coords,
  glimpse-coords-places365: _meta_places.glimpse_coords,
  col-spacing: 2.27pt,
  row-spacing: 56.69pt,
  img-width: 39.69pt,
) = {
  let residual-x0   = 165pt  // x-offset aligned to Write 0 residual column
  let residual-step = 114pt  // pitch between successive residual columns

  block(
    stack(
      ..range(3).map(i => residual-box(residual-x0 + i * residual-step)),

      grid(
        columns: (5.5pt, 46.2pt) + (auto,) * 15,
        column-gutter: col-spacing,
        rows: (16.5pt, 11pt, 55pt) + (row-spacing,) * 2 + (22pt,) + (row-spacing,) * 2,
        align: center + horizon,

        // Header labels
        [], [*Scene*], [*Glimpse*], [], [*Canvas*], [], [*Residual*], [], [*Canvas*], [], [*Residual*], [], [*Canvas*], [], [*Residual*], [], [*Canvas*],

        // Sub-headings (Write 0/1/2 — 0-indexed, matching Python hook labels)
        ..([], [], [], [], sub-head[Initial], [], sub-head[Write 0], [], sub-head[Updated], [], sub-head[Write 1], [], sub-head[Updated], [], sub-head[Write 2], [], sub-head[Final])
          .map(item => {
            if item == [] { [] }
            else { stack(dir: ttb, v(-3.3pt), item) }
          }),

        // Cat rows
        ..glimpse-coords-cat.enumerate().map(((i, c)) =>
          residual-row(i, coords: c, img-dir: img-dir + "/" + _meta_cat.dir_name, width: img-width),
        ).flatten(),

        grid.cell(colspan: 17)[#v(21.26pt)],

        // Places365 rows
        ..glimpse-coords-places365.enumerate().map(((i, c)) =>
          residual-row(i, coords: c, img-dir: img-dir + "/" + _meta_places.dir_name, width: img-width),
        ).flatten()
      )
    )
  )
}