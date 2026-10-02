// Architecture overview diagram content - importable by other documents
// Usage: #import "arch_overview_content.typ": arch-overview-diagram
//        #arch-overview-diagram()

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let arch-overview-diagram(img-dir: "outputs/arch_overview") = {
  // --- CONFIG ---
  let debug = false
  let meta = json(img-dir + "/metadata.json")
  let n-timesteps = meta.n_timesteps
  let viewpoints = meta.viewpoints
  let n-pairs = 6
  let n-rw = 3
  // Format viewpoint as (x, y, s)
  let fmt-vp(vp) = $(#vp.x, #vp.y, #vp.scale)$

  // --- IMAGES ---
  let img(name, t) = img-dir + "/" + name + str(t) + ".png"
  let glimpse-in-imgs = range(n-timesteps).map(t => img("glimpse_t", t))
  let glimpse-out-imgs = range(n-timesteps).map(t => img("local_out_t", t))
  let canvas-in-imgs = range(n-timesteps).map(t => if t == 0 { none } else { img("canvas_out_t", t - 1) })
  let canvas-out-imgs = range(n-timesteps).map(t => img("canvas_out_t", t))

  // --- COLORS ---
  let C = (
    spatial: rgb("#e05050"),
    patches: rgb("#4080d0"),
    cls: rgb("#50b050"),
    reg: rgb("#b0b0b0"),
    vpe: rgb("#f0a030"),
    block: rgb("#d0a8e0"),
    stroke: rgb("#202020"),
  )

  // --- SIZES ---
  let grid-size = 3em
  let g-n = 16
  let l-n = 4
  let sz = (
    g-n: g-n, g-cell: grid-size / g-n,
    l-n: l-n, l-cell: grid-size / l-n,
    vit: (w: 1.6em, h: 1.5em),
    r: 2pt, sw: 0.5pt, gs: 0.25pt,
    cls-sw: 1.0pt,
    glimpse-sw: 1.8pt,
    canvas-sw: 1.8pt,
    font-sm: 5pt, font-md: 7pt, font-lg: 9pt,
  )

  let arrows = (
    glimpse: "-stealth",
    canvas: "-stealth",
    cls: "->",
    rw: "->",
  )

  // --- LAYOUT ---
  let col-spacing = 1.3
  let X-local = 0
  let X-canvas = 0.5
  let y-viewpoint-label = 0.75
  let y-timestep-label = 1.15
  let y-legend = 1.8
  let step = 0.45
  let tok-gap = 1.4
  let recur-below = 1.5

  let y-tok-out = 0
  let y-base = y-tok-out + tok-gap
  let y-pair(i) = y-base + (n-pairs - 1 - i) * 2 * step
  let y-read(i) = y-pair(2 * i) - step
  let y-write(i) = y-pair(2 * i + 1) - step
  let y-tok-in = y-pair(0) + tok-gap
  let y-recur-arrow = (y-tok-out + y-tok-in) / 2  // mid-path arrows for recurrence edges

  // --- COMPONENTS ---
  let D = (g: sz.g-n * sz.g-cell, l: sz.l-n * sz.l-cell)
  let tok-size = sz.l-cell
  let n-canvas-reg = 3
  let n-local-reg = 1

  let Y-patch = 0
  let Y-reg = -0.65
  let Y-vpe = -0.85
  let Y-cls = -1.05

  let tokgrid(n, cell, fill, stroke-w: sz.gs) = box(inset: 0pt, grid(
    columns: (cell,) * n, rows: (cell,) * n,
    ..range(n * n).map(_ => rect(width: cell, height: cell, fill: fill, stroke: stroke-w + C.stroke, inset: 0pt))
  ))

  let toksquare(fill) = rect(
    width: tok-size, height: tok-size,
    fill: fill, stroke: sz.sw + C.stroke, inset: 0pt,
  )

  let reg-row(n, fill) = stack(dir: ltr, spacing: 1pt,
    ..range(n).map(_ => toksquare(fill))
  )

  let canvas-gs = 0.03pt

  let clstok() = toksquare(C.cls)
  let vpetok() = toksquare(C.vpe)
  let local-regtok() = reg-row(n-local-reg, C.reg)
  let patchtok() = tokgrid(sz.l-n, sz.l-cell, C.patches)

  let local-aux-row() = stack(dir: ltr, spacing: 1pt,
    toksquare(C.cls),
    toksquare(C.vpe),
    ..range(n-local-reg).map(_ => toksquare(C.reg)),
  )

  let ctok() = stack(dir: ttb, spacing: 2pt,
    reg-row(n-canvas-reg, C.reg),
    tokgrid(sz.g-n, sz.g-cell, C.spatial, stroke-w: canvas-gs),
  )

  let vitpair(pair-idx) = {
    let lo = 2 * pair-idx
    let hi = 2 * pair-idx + 1
    let cell-h = sz.vit.h / 2
    rect(
      width: sz.vit.w, height: sz.vit.h,
      fill: C.block, stroke: sz.sw + C.stroke, inset: 0pt,
      grid(
        columns: (sz.vit.w,), rows: (cell-h, cell-h),
        rect(width: 100%, height: 100%, fill: C.block, stroke: (bottom: sz.sw + C.stroke), inset: 0pt,
          align(center + horizon, text(size: 4pt, weight: "bold", str(hi)))),
        rect(width: 100%, height: 100%, fill: C.block, stroke: none, inset: 0pt,
          align(center + horizon, text(size: 4pt, weight: "bold", str(lo)))),
      )
    )
  }

  let debug-fill = rgb("#ff000030")
  let sized(pos, content, name, w, h) = node(
    pos,
    if debug {
      box(fill: debug-fill, width: w, height: h, align(center + horizon, content))
    } else { content },
    name: name, width: w, height: h,
  )

  let canvas-regnode(pos, name) = sized(pos, reg-row(n-canvas-reg, C.reg), name, n-canvas-reg * tok-size, tok-size)
  let canvas-spatialnode(pos, name, img: none) = sized(
    pos,
    if img != none { image(img, width: D.g, height: D.g) } else { tokgrid(sz.g-n, sz.g-cell, C.spatial, stroke-w: canvas-gs) },
    name, D.g, D.g
  )

  let clsnode(pos, name) = sized(pos, clstok(), name, tok-size, tok-size)
  let vpenode(pos, name) = sized(pos, vpetok(), name, tok-size, tok-size)
  let local-regnode(pos, name) = sized(pos, local-regtok(), name, n-local-reg * tok-size, tok-size)
  let local-auxnode(pos, name) = sized(pos, local-aux-row(), name, (2 + n-local-reg) * tok-size + 2pt, tok-size)
  let patchnode(pos, name, img: none) = sized(
    pos,
    if img != none { image(img, width: D.l, height: D.l) } else { patchtok() },
    name, D.l, D.l
  )
  let vpairnode(pos, name, pair-idx) = sized(pos, vitpair(pair-idx), name, sz.vit.w, sz.vit.h)

  let spine(..a) = edge(..a, stroke: sz.glimpse-sw + C.patches)
  let canvas-spine(..a) = edge(..a, stroke: sz.canvas-sw + C.spatial)
  let xfer(..a) = edge(..a, arrows.rw, stroke: sz.sw + C.stroke)
  let canvas-recur(..a) = edge(..a, arrows.canvas, stroke: sz.canvas-sw + C.spatial, corner-radius: 0pt)
  let cls-recur(..a) = edge(..a, arrows.cls, stroke: sz.cls-sw + C.cls, corner-radius: 0pt)
  let lbl(t) = text(size: sz.font-md, [*#t*])

  // Mid-path direction indicator for recurrence edges
  let recur-arrow-size = 18pt
  let recur-arrow-content(color) = text(
    size: recur-arrow-size,
    fill: color,
    // stroke: 0.5pt + C.stroke,
    top-edge: "bounds",
    bottom-edge: "bounds",
    sym.triangle.filled.b,
  )
  let recur-arrow(pos, color) = node(
    pos,
    if debug {
      box(fill: debug-fill, recur-arrow-content(color))
    } else {
      recur-arrow-content(color)
    },
    inset: 0pt,
    outset: 0pt,
    width: 1em,
    height: 0.5em,
  )

  let enc(nodes, name, color: C.stroke, width: auto) = node(enclose: nodes, stroke: (dash: "dashed", paint: color), inset: 3pt, name: name, width: width)

  let timestep(t) = {
    let x-base = t * col-spacing
    let xl = x-base + X-local
    let xg = x-base + X-canvas
    let lb(name) = label(name + str(t))

    (
      node((xl, y-tok-in + y-viewpoint-label),
        text(size: sz.font-sm, $bold(v)_#t = #fmt-vp(viewpoints.at(t))$)),
      node(((xl + xg) / 2, y-tok-in + y-timestep-label), text(size: sz.font-lg, $t = #t$)),
      local-auxnode((xl, y-tok-in + Y-reg), lb("lai")),
      patchnode((xl, y-tok-in + Y-patch), lb("pi"), img: glimpse-in-imgs.at(t)),
      enc((lb("lai"), lb("pi")), lb("li"), color: C.patches),
      clsnode((xl, y-tok-out + Y-cls), lb("clo")),
      vpenode((xl, y-tok-out + Y-vpe), lb("vpeo")),
      local-regnode((xl, y-tok-out + Y-reg), lb("lro")),
      patchnode((xl, y-tok-out + Y-patch), lb("po"), img: glimpse-out-imgs.at(t)),
      enc((lb("clo"), lb("vpeo"), lb("lro"), lb("po")), lb("lo"), color: C.patches),
      canvas-regnode((xg, y-tok-in + Y-reg), lb("cri")),
      canvas-spatialnode((xg, y-tok-in + Y-patch), lb("csi"), img: canvas-in-imgs.at(t)),
      enc((lb("cri"), lb("csi")), lb("ci"), color: C.spatial),
      canvas-regnode((xg, y-tok-out + Y-reg), lb("cro")),
      canvas-spatialnode((xg, y-tok-out + Y-patch), lb("cso"), img: canvas-out-imgs.at(t)),
      enc((lb("cro"), lb("cso")), lb("co"), color: C.spatial),
      ..range(n-pairs).map(i => vpairnode((xl, y-pair(i)), lb("p" + str(i)), i)),
      enc(range(n-pairs).map(i => lb("p" + str(i))), lb("backbone"), color: C.block, width: D.l),
      ..range(n-rw).map(i => (
        node((xl, y-read(i)), none, name: lb("pr" + str(i))),
        node((xg, y-read(i)), none, name: lb("cr" + str(i))),
        xfer(lb("cr" + str(i)), lb("pr" + str(i)), label: text(size: 5pt, "R"), label-side: center),
        node((xl, y-write(i)), none, name: lb("lw" + str(i))),
        node((xg, y-write(i)), none, name: lb("pw" + str(i))),
        xfer(lb("lw" + str(i)), lb("pw" + str(i)), label: text(size: 5pt, "W"), label-side: center),
      )).flatten(),
      spine(lb("li"), lb("p0"), "-"),
      ..range(n-rw).map(i => (
        spine(lb("p" + str(2*i)), lb("pr" + str(i)), "-"),
        spine(lb("pr" + str(i)), lb("p" + str(2*i + 1)), "-"),
        spine(lb("p" + str(2*i + 1)), lb("lw" + str(i)), "-"),
        if i < n-rw - 1 {
          spine(lb("lw" + str(i)), lb("p" + str(2*i + 2)), "-")
        } else {
          edge(lb("lw" + str(i)), lb("lo"), arrows.glimpse, stroke: sz.glimpse-sw + C.patches, mark-scale: 60%)
        },
      )).flatten(),
      canvas-spine(lb("ci"), lb("cr0"), "-"),
      ..range(n-rw - 1).map(i => (
        canvas-spine(lb("cr" + str(i)), lb("pw" + str(i)), "-"),
        canvas-spine(lb("pw" + str(i)), lb("cr" + str(i + 1)), "-"),
      )).flatten(),
      canvas-spine(lb("cr" + str(n-rw - 1)), lb("pw" + str(n-rw - 1)), "-"),
      edge(lb("pw" + str(n-rw - 1)), lb("co"), arrows.canvas, stroke: sz.canvas-sw + C.spatial, mark-scale: 60%),
    )
  }

  let legend-item(color, label) = stack(dir: ltr, spacing: 2pt,
    rect(width: 6pt, height: 6pt, fill: color, stroke: 0.4pt + C.stroke),
    text(size: 5pt, label),
  )

  let legend = box(
    inset: (x: 4pt, y: 3pt),
    stack(dir: ltr, spacing: 6pt,
      legend-item(C.block, "ViT Blocks"),
      legend-item(C.patches, "Glimpse"),
      legend-item(C.spatial, "Canvas"),
      legend-item(C.cls, "CLS"),
      legend-item(C.vpe, "VPE"),
      legend-item(C.reg, "Registers"),
    )
  )

  diagram(
    spacing: (3em, 1.5em),
    node-stroke: none, node-inset: 0pt, edge-stroke: sz.sw,
    ..range(n-timesteps).map(t => timestep(t)).flatten(),
    ..range(n-timesteps - 1).map(t => {
      let clo-name = label("clo" + str(t))
      let lai-next = label("lai" + str(t + 1))
      let gap-start = t * col-spacing + X-canvas
      let gap-end = (t + 1) * col-spacing + X-local
      let x-between = gap-start + (gap-end - gap-start) * 0.4
      let y-cls-out = y-tok-out + Y-cls
      let y-aux-in = y-tok-in + Y-reg
            edge(
        clo-name,
        (x-between, y-cls-out),
        (x-between, y-recur-arrow),
        "-",
        stroke: sz.cls-sw + C.cls,
        corner-radius: 0pt,
      )
      recur-arrow((x-between, y-recur-arrow), C.cls)
      edge(
        (x-between, y-recur-arrow),
        (x-between, y-aux-in),
        lai-next,
        "->",
        stroke: sz.cls-sw + C.cls,
        corner-radius: 0pt,
        mark-scale: 60%,
      )
    }).flatten(),
    ..range(n-timesteps - 1).map(t => {
      let co-name = label("co" + str(t))
      let ci-next = label("ci" + str(t + 1))
      let gap-start = t * col-spacing + X-canvas
      let gap-end = (t + 1) * col-spacing + X-local
      let x-between = gap-start + (gap-end - gap-start) * 0.6
      let y-canvas-out = y-tok-out + (Y-patch + Y-reg) / 2
      let y-canvas-in = y-tok-in + (Y-patch + Y-reg) / 2
      let y-below = y-tok-in + recur-below
      let x-ci = (t + 1) * col-spacing + X-canvas
            edge(
        co-name,
        (x-between, y-canvas-out),
        (x-between, y-recur-arrow),
        "-",
        stroke: sz.canvas-sw + C.spatial,
        corner-radius: 0pt,
      )
      recur-arrow((x-between, y-recur-arrow), C.spatial)
      edge(
        (x-between, y-recur-arrow),
        (x-between, y-below),
        (x-ci, y-below),
        ci-next,
        "-stealth",
        stroke: sz.canvas-sw + C.spatial,
        corner-radius: 0pt,
        mark-scale: 60%,
      )
    }).flatten(),
    node(((n-timesteps - 1) * col-spacing / 2 + X-canvas / 2, y-tok-in + y-legend), legend),
  )
}
