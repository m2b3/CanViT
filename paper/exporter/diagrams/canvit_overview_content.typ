// Canvit overview diagram: help people understand what is a scene, glimpse, canvas and how they interact with each other and through time
#import "@preview/tiptoe:0.4.0"

// Default aesthetic
// Font fallback: Helvetica (macOS) → Liberation Sans / Nimbus Sans (Linux).
#let default_font = ("Helvetica", "Liberation Sans", "Nimbus Sans")
#let C = (
  spatial: rgb("#e05050"),
  patches: rgb("#4080d0"),
  stroke: rgb("#202020"),
)
#let arrow(paint, thickness: 0.5pt, width: 800%) = tiptoe.straight.with(length: 6pt, width: width, stroke: (thickness: thickness, paint: paint))
#let arrow_line_thickness = 0.5pt
#let panel-label(letter) = text(font: default_font, size: 16pt, weight: "bold")[#letter]

// SCENE HELPERS
#let compute_centers(glimpses, p-width, p-height) = {
  let centers = glimpses.map(g => {
        let (scale, cy, cx) = g
        let px = (cx + 1) / 2 * p-width
        let py = (cy + 1) / 2 * p-height
        (px, py, scale)
      })
  centers
}

#let draw_g_boxes(glimpses, p-width, p-height) = {
  let grid_transp = 50%
  for g in glimpses {
    let (scale, cy, cx) = g
    let box-w = scale * p-width
    let box-h = scale * p-height
    let px = (cx + 1) / 2 * p-width  - box-w / 2
    let py = (cy + 1) / 2 * p-height - box-h / 2
    let grid_size = 8

    // Box
    place(top + left, dx: px, dy: py,
      rect(
        width: box-w,
        height: box-h,
        stroke: (paint: C.patches, thickness: 1.5pt),
        fill: C.patches.transparentize(70%),
      )
    )

    // Grid
    for i in range(1, grid_size) {
      let x = px + box-w * i / grid_size
      let y = py + box-h * i / grid_size
      
      place(top + left, dx: x, dy: py,
        line(angle: 90deg, length: box-h, stroke: arrow_line_thickness + white.transparentize(grid_transp)))

      place(top + left, dx: px, dy: y,
        line(length: box-w, stroke: arrow_line_thickness + white.transparentize(grid_transp)))
    }
    // Dots at cell centers
    for row in range(grid_size) {
      for col in range(grid_size) {
        let dot-x = px + box-w * (col + 0.5) / grid_size
        let dot-y = py + box-h * (row + 0.5) / grid_size
        let dot-r = 0.5pt
        place(top + left, dx: dot-x - dot-r, dy: dot-y - dot-r,
          circle(radius: dot-r, fill: white.transparentize(grid_transp), stroke: none)
        )
      }
    }
    let annotation_size = 3.2pt
    place(
    top + left, dx: px - 0.75pt, dy: py - annotation_size,
    text(font: default_font, weight: "bold", size: annotation_size, fill: white.transparentize(30%))[#highlight(fill: C.patches)[(x=#cx, y=#cy, s=#scale)]]
  )
  }
  // Viewpoint text
}

#let draw-saccades(centers) = {
  let circle-r = 6pt

  // Numbered circles
  for (i, center) in centers.enumerate() {
    let (px, py, s) = center
    place(top + left, dx: px - circle-r, dy: py - circle-r,
      circle(radius: circle-r, fill: gray.darken(20%), stroke: 0.8pt + white)
    )
    place(top + left, dx: px - circle-r, dy: py - circle-r,
      box(width: 2 * circle-r, height: 2 * circle-r,
        inset: (top: 4pt, left: 4.2pt),
        text(font: default_font, size: 6pt, fill: white)[$#str(i)$]
      )
    )
  }

  // Arrows between centers using tiptoe
  for i in range(centers.len() - 1) {
    let (x0, y0, s) = centers.at(i)
    let (x1, y1, s) = centers.at(i + 1)
    let offset = if x1 < x0 { -circle-r } else { circle-r }
    place(top + left,
      tiptoe.line(
        start: (x0 + offset, y0),
        end: (x1 - offset, y1),
        tip: arrow(white, thickness: 0.75pt, width: 500%), 
        stroke: 1.2pt + white.darken(7.5%),
      )
    )
  }
}
  
#let scene-panel(path, panel-height, panel-width, glimpses: ()) = {
  box(
    width: panel-width,
    height: panel-height,
    clip: true,
    {
      let centers = compute_centers(glimpses, panel-width, panel-height)
      image(path, width: panel-width, height: panel-height, fit: "cover")
      draw_g_boxes(glimpses, panel-width, panel-height)
      if centers.len() > 0 { draw-saccades(centers) }
    }
  )
}

// Sequence helpers
#let timeline(
  glimpse-coords,
  arrow-w,
  coord_box_width: 45pt,
  label-dist: 10pt,
) = {
  let n = glimpse-coords.len()
  let arrow-y = 10pt
  let slot-w = arrow-w / n
  let arrow_w_offset = 6pt

  box(width: arrow-w, height: 34pt, {
    // Arrow
    place(top + left, dx: 0pt, dy: arrow-y,
      tiptoe.line(
        start: (label-dist, 0pt),
        end: (arrow-w - arrow_w_offset, 0pt),
        tip: arrow(C.stroke),
        stroke: arrow_line_thickness + C.stroke,
      )
    )
    // "t" label
    place(left + top, dx: arrow-w - arrow_w_offset + 2pt, dy: arrow-y - 6pt,
      text(font: default_font, size: 11pt, fill: C.stroke)[t])

    for i in range(n) {
      let cx = (i + 0.6) * slot-w  // center of each slot
      let (s, y, x) = glimpse-coords.at(i)

      // Tick
      place(left + top, dx: cx, dy: arrow-y - 2pt,
        line(angle: 90deg, length: 5pt, stroke: arrow_line_thickness + C.stroke))

      // Timestep index
      place(left + top, dx: cx - 2pt, dy: arrow-y - 10pt,
        text(font: default_font, size: 6pt, fill: C.stroke)[#str(i)])

      // Coord box
      let coords_pos = i * (slot-w + 0.375pt) + label-dist
      place(top + left, dx: coords_pos,  dy: arrow-y + 4pt,
        box(
          fill: C.patches,
          width: coord_box_width,
          height: 16pt,
          inset: (x: 2pt, y: 3pt),
          align(center,
            text(font: default_font, size: 4.6pt, fill: white)[
              *$v_#str(i) = (x_#str(i), y_#str(i), s_#str(i))$
              #v(-0.06cm)
              $(#str(x), #str(y), #str(s))$*
            ]
          )
        )
      )
    }
  })
}

// SEQUENCE
#let sequence-row(
  label, 
  color, 
  n-steps, 
  img-dir, 
  stage, 
  glimpse-width, 
  coords-list: (),
  arrows: false,
  border_width: 3.5pt,
) = {
  let items = ()
  let label-color = if color == none { C.stroke } else { color }
  let img-stroke = if color == none { none } else { border_width + color }

  // Labels
  items.push(rotate(-90deg, reflow: true)[
      #text(font: default_font, weight: "bold", fill: label-color, size: 10pt)[#label]
  ])
  
  // Images
  for i in range(n-steps) {
    let suffix = if stage == "glimpse" and coords-list.len() > i { "_" + coords-list.at(i) } else { "" }
    let img-path = img-dir + "g" + str(i) + "_" + stage + suffix + ".png"
    
    items.push(
      box(
        stroke: img-stroke,
        image(img-path, width: glimpse-width)
      )
    )
  }
  
  return items
}

#let parse-coords(coords-list) = {
  coords-list.map(s => {
    let parts = s.split("_")
    // parts: ("s0.50", "c-0.4", "-0.4")
    let scale = float(parts.at(0).slice(1))   // strip "s"
    let cy    = float(parts.at(1).slice(1))   // strip "c"
    let cx    = float(parts.at(2))
    (scale, cy, cx)
  })
}

// Coords loaded from per-image metadata.json written by canvit_overview.py.
// No hardcoded viewpoint coords — single source of truth is the Python script.
// The intro figure uses Places365. Change this path to use a different scene.
#let _overview_scene = "Places365_IMG_9600"
#let _overview_meta = json("outputs/canvit_overview/" + _overview_scene + "/metadata.json")

// DIAGRAM
#let canvit-overview-diagram(
  scene-path: "outputs/canvas_evolution/Places365/scene.png",
  img-dir: "outputs/canvit_overview/" + _overview_scene + "/",
  coords-list: _overview_meta.glimpse_coords,
  glimpse-width: 1.8cm,
  show-pixel-arrow: false,
) = {
  let n = coords-list.len()
  let glimpse_coords = parse-coords(coords-list)
  let label_dist = 5pt
  let img_col_gutter = 15.59pt
  let img_row_gutter = 0.8pt
  let img_border_width = 3.5pt
  let scene_width = 3 * (glimpse-width + 2*img_border_width) + 2 * img_row_gutter
  
  // Wrap in a layout to ensure we don't overflow the slide/page
  layout(size => {
    let available-width = size.width
    
    // Panel labels: placed at block level for consistent absolute y position.
    // Grid: columns (1fr, 1.5fr), gutter 2em. Total fr = 2.5.
    // Distributable width = available-width - gutter. Col1 width = distributable * 1/2.5.
    // Col2 starts at col1_width + gutter.
    let label-dy = 0.00cm
    let gutter = 2em
    let col1-width = (available-width - gutter) * (1 / 2.5)
    let col2-x = col1-width + gutter
    block(width: available-width)[
      #place(top + left, dx: 0.02cm, dy: label-dy, panel-label("A"))
      #place(top + left, dx: col2-x - 0.07cm, dy: label-dy, panel-label("B"))
      #grid(
        columns: (1fr, 1.5fr),
        column-gutter: 2em,
        align: horizon,
        
        // Row 1, Col 1: Pixel and [-1, +1] extents + panel label A
        // offset_x aligns arrows with the scene image in Row 2, which is
        // preceded by the rotated "Scene" label (~14pt) + vertical axis box (14pt)
        // + stack spacing (-3pt) = ~25pt from the grid cell's left edge.
        box(height: 27pt, {
          let arrow1_height = 15.5pt
          let dist_btw_arrows = 9pt
          let tick-positions = (0pt, scene_width)
          let tick-labels = ("-1", "1")

          // Measurement line with end caps (1024px). Both elements are `place`d,
          // so gating them does not shift anything else — the parent box has a
          // fixed height and everything below is either placed or separately anchored.
          let offset_x = 17pt
          if show-pixel-arrow {
            place(top + left, dx: offset_x + scene_width / 2 - 8pt, dy: arrow1_height - dist_btw_arrows - 7pt,
              text(font: default_font, size: 6pt, fill: C.stroke)[1024px])

            place(top + left, dx: offset_x, dy: arrow1_height - dist_btw_arrows,
              tiptoe.line(
                start: (0pt, 0pt),
                end: (scene_width, 0pt),
                tip: tiptoe.combine(tiptoe.bar, arrow(C.stroke)),
                toe: tiptoe.combine(tiptoe.bar, arrow(C.stroke)),
                stroke: arrow_line_thickness + C.stroke,
              )
            )
          }
          // Coordinate system [-1, +1]
          place(top + left, dx: offset_x, dy: arrow1_height,
            tiptoe.line(
              start: (0pt, 3pt),
              end: (scene_width, 3pt),
              tip: arrow(C.stroke),
              stroke: arrow_line_thickness + C.stroke,
            )
          )
          for (i, tx) in tick-positions.enumerate() {
            place(left + top, dx: offset_x + tx - 1.5pt, dy: arrow1_height - 3.5pt,
              text(font: default_font, size: 6pt, fill: C.stroke)[#tick-labels.at(i)])
          }
        }),

        // Row 1, Col 2: Timeline + panel label B
        box({
          timeline(
            glimpse_coords,
            n * (glimpse-width + img_border_width) + (n - 1) * img_col_gutter,
            coord_box_width: glimpse-width + img_border_width,
            label-dist: label_dist + 6.8pt
          )
        }),

        // Row 2, Col 1: Full Scene
        move(
          dy: -4pt,
          stack(dir: ltr, spacing: -3pt,
            rotate(-90deg, reflow: true,
              text(font: default_font, size: 12pt, fill: C.stroke, weight: "bold")[*Scene*]
            ),
            // Vertical axis arrow
            box(width: 14pt, height: scene_width, {
              // Arrow
              place(top + left, dx: 7pt, dy: 0pt,
                tiptoe.line(
                  start: (0pt, 0pt),
                  end: (0pt, scene_width),
                  tip: arrow(C.stroke),
                  stroke: arrow_line_thickness + C.stroke,
                )
              )
              // Ticks and labels
              for (i, (label, ty)) in (("-1", 0pt), ("1", scene_width - 6pt)).enumerate() {
                place(top + left, dx: -2pt, dy: ty - 3pt,
                  text(font: default_font, size: 6pt, fill: C.stroke)[#label])
              }
            }),
            scene-panel(
              scene-path, scene_width, scene_width,
              glimpses: glimpse_coords
            )
          )
        ),
        
        // Row 2, Col 2: Glimpse Rows
        grid(
          columns: (0.7em,) + (glimpse-width,) * n,
          column-gutter: (label_dist, img_col_gutter, img_col_gutter, img_col_gutter, img_col_gutter),
          // Per-gap row gutter: (Glimpse→Canvas, Canvas→Canvas Δ)
          row-gutter: (0.6em, 0.3em),
          align: center + horizon,
          
          // Rows 1-3: Change, Canvas, Glimpse (top to bottom)
          ..sequence-row("Glimpse", C.patches, n, img-dir, "glimpse", glimpse-width, coords-list: coords-list, border_width: img_border_width),
          ..sequence-row("Canvas", C.spatial, n, img-dir, "canvas", glimpse-width, border_width: img_border_width),
          ..sequence-row([Canvas $Delta$], C.stroke, n, img-dir, "change", glimpse-width, border_width: img_border_width),

        // Canvas arrows
        grid.cell(colspan: n + 1)[
          #layout(size => {
            box(width: size.width, {
              for i in range(n - 1) {
                let cx = (i + 1) * (glimpse-width + img_col_gutter) - 1.5pt
                place(left + horizon, dx: cx, dy: -2*(img_row_gutter + glimpse-width),
                  text(size: 12pt, fill: C.spatial)[→]
                )
              }
            })
          })
        ],
        )
      )
    ]
  })
}
