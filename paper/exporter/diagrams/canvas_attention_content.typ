// Canvas attention diagram content - importable by other documents
// Usage: #import "canvas_attention_content.typ": canvas-attention-diagram
//        #canvas-attention-diagram()

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let canvas-attention-diagram(img-dir: "outputs/canvas_attn") = {
  // --- COLORS ---
  let C = (
    canvas: rgb("#e05050"), glimpse: rgb("#4080d0"),
    prefix: rgb("#d0d0d0"),
    ln: rgb("#b8e0b8"),
    rope: rgb("#b8c8d8"),
    proj: rgb("#e0c080"), sdpa: rgb("#e89060"), block: rgb("#d0a8e0"),
    stroke: rgb("#202020"), bound: rgb("#707070"),
    read-bg: rgb("#f4f8fc"), write-bg: rgb("#fcf8f4"),
  )

  // --- IMAGES ---
  let imgs = (
    glimpse-in: img-dir + "/glimpse_before_read.png",
    glimpse-out: img-dir + "/glimpse_before_write.png",
    canvas-in: img-dir + "/canvas_in.png",
    canvas-out: img-dir + "/canvas_after_write.png",
  )

  // --- SIZES ---
  let sz = (
    c-n: 16, c-cell: 0.18em,  // 16 × 0.18 = 2.88em (matches glimpse)
    g-n: 4,  g-cell: 0.72em, //  4 × 0.72 = 2.88em
    box: (w: 2.58em, h: 1.16em),
    sm: (w: 1.55em, h: 1.03em),
    sdpa: (w: 3.35em, h: 2.78em),
    blk: (w: 3.10em, h: 2.32em),
    rope-w: 3.1em,
    plus-r: 0.57em,
    r: 3pt, sw: 0.6pt, main-sw: 1.2pt, gs: 0.3pt,
    f: 6pt, ft: 12pt, fb: 8pt, fg: 10pt,
    adapter-inset: 0.8em,
    layer-inset: 8pt,
  )

  // --- LAYOUT ---
  let x-g = 0
  let x-ln-g = 0.16
  let x-proj = x-ln-g + 0.12
  let x-rope-g = x-proj + 0.12
  let x-rope-c = x-rope-g + 0.32
  let x-ln-c = x-rope-c + 0.14
  let x-c = x-ln-c + 0.16
  let x-sdpa = (x-rope-g + x-rope-c) / 2
  let X = (g: x-g, ln-g: x-ln-g, proj: x-proj, rope-g: x-rope-g, sdpa: x-sdpa, rope-c: x-rope-c, ln-c: x-ln-c, c: x-c)

  let Y = (
    tok-out: 0.35,
    write-out: 1.3,
    write: 2.3,
    blk: 2.35,
    read-out: 3.4,
    read: 4.5,
    tok-in: 5.2,
  )

  let off = (
    label-dx: 0.16,
    group-label-dx: 0.02,
    branch-dx: 0.05,
    out-dy: -0.15,
    sdpa-y: -0.5,
    k-y: -0.35,
    v-y: -0.65,
    prefix-dy-out: -0.51,  // output (top) prefix offset above grid
    prefix-dy-in: -0.42,   // input (bottom) prefix offset above grid — larger to visually match
  )

  let ry = (r: Y.read + off.sdpa-y, ro: Y.read-out + off.out-dy, rk: Y.read + off.k-y, rv: Y.read + off.v-y)
  let wy = (w: Y.write + off.sdpa-y, wo: Y.write-out + off.out-dy, wk: Y.write + off.k-y, wv: Y.write + off.v-y)

  // --- COMPONENTS ---
  let D = (
    c: sz.c-n * sz.c-cell,
    g: sz.g-n * sz.g-cell,
    pd: sz.plus-r * 2,
  )

  let mkbox(txt, fill, w, h, tfill: C.stroke, wt: "regular") = rect(
    width: w, height: h, radius: sz.r, fill: fill, stroke: sz.sw + C.stroke, inset: 0pt,
    align(center + horizon, text(size: sz.f, fill: tfill, weight: wt, txt))
  )
  let bigbox(txt, fill, w, h, tfill: C.stroke) = rect(
    width: w, height: h, radius: sz.r, fill: fill, stroke: sz.sw + C.stroke, inset: 0pt,
    align(center + horizon, text(size: sz.fb, fill: tfill, weight: "bold", txt))
  )

  let smbox(txt, fill) = mkbox(txt, fill, sz.sm.w, sz.sm.h)
  let ropebox(txt, fill) = mkbox(txt, fill, sz.rope-w, sz.sm.h)

  let sdpa() = bigbox("SDPA", C.sdpa, sz.sdpa.w, sz.sdpa.h)
  let blk() = bigbox([ViT\ Blocks], C.block, sz.blk.w, sz.blk.h)
  let ln() = smbox("LN", C.ln)
  let W(s) = smbox($W_#s$, C.proj)
  let srope() = ropebox("SR-RoPE", C.rope)
  let plus() = circle(radius: sz.plus-r, fill: white, stroke: sz.sw + C.stroke, inset: 0pt,
    align(center + horizon, text(size: sz.f, weight: "bold", "+")))

  let tokgrid(n, cell, fill) = box(inset: 0pt, grid(
    columns: (cell,) * n, rows: (cell,) * n,
    ..range(n * n).map(_ => rect(width: cell, height: cell, fill: fill, stroke: sz.gs + C.stroke, inset: 0pt))
  ))
  let prefixtok(w) = rect(
    width: w, height: sz.g-cell, radius: sz.r, fill: C.prefix, stroke: sz.sw + C.stroke, inset: 0pt,
    align(center + horizon, text(size: sz.f, weight: "bold", "Prefix"))
  )

  let cgrid(img: none) = if img != none { image(img, width: D.c, height: D.c) } else { tokgrid(sz.c-n, sz.c-cell, C.canvas) }
  let ggrid(img: none) = if img != none { image(img, width: D.g, height: D.g) } else { tokgrid(sz.g-n, sz.g-cell, C.glimpse) }

  let sized(pos, content, name, w, h) = node(pos, content, name: name, width: w, height: h)
  let cprefixnode(pos, name) = sized(pos, prefixtok(D.c), name, D.c, sz.g-cell)
  let gprefixnode(pos, name) = sized(pos, prefixtok(D.g), name, D.g, sz.g-cell)
  let cgridnode(pos, name, img: none) = sized(pos, cgrid(img: img), name, D.c, D.c)
  let ggridnode(pos, name, img: none) = sized(pos, ggrid(img: img), name, D.g, D.g)
  let snode(pos, name) = sized(pos, sdpa(), name, sz.sdpa.w, sz.sdpa.h)
  let bnode(pos, name) = sized(pos, blk(), name, sz.blk.w, sz.blk.h)
  let pnode(pos, name) = sized(pos, plus(), name, D.pd, D.pd)

  let enc(nodes, fill, name: none, dash: none, inset: sz.adapter-inset) = node(
    enclose: nodes,
    stroke: if dash == none { sz.sw + C.bound } else { (dash: dash, paint: C.bound) },
    fill: fill, inset: inset, corner-radius: sz.r, name: name,
  )

  let spine(..a) = edge(..a, stroke: sz.main-sw + C.stroke)

  // --- DIAGRAM ---
  diagram(
    spacing: (20em, 2.32em),
    node-stroke: none, node-inset: 0pt, edge-stroke: sz.sw,

    // INPUT TOKENS
    gprefixnode((X.g, Y.tok-in + off.prefix-dy-in), <gi-prefix>),
    ggridnode((X.g, Y.tok-in), <gi>, img: imgs.glimpse-in),
    cprefixnode((X.c, Y.tok-in + off.prefix-dy-in), <ci-prefix>),
    cgridnode((X.c, Y.tok-in), <ci>, img: imgs.canvas-in),

    // STREAM LABELS — centered between prefix and grid for each row.
    // Y midpoint = tok + prefix-dy/2
    node((X.ln-g + 0.05, Y.tok-out + off.prefix-dy-out / 3), text(size: sz.fg, weight: "bold", fill: C.glimpse, [Updated \ glimpse])),
    node((X.ln-c - 0.05, Y.tok-out + off.prefix-dy-out / 3), text(size: sz.fg, weight: "bold", fill: C.canvas, [Updated \ canvas])),
    node((X.ln-g + 0.05, Y.tok-in + off.prefix-dy-in / 3), text(size: sz.fg, weight: "bold", fill: C.glimpse, "Glimpse")),
    node((X.ln-c - 0.05, Y.tok-in + off.prefix-dy-in / 3), text(size: sz.fg, weight: "bold", fill: C.canvas, "Canvas")),

    // READ ADAPTER
    node((X.ln-g, ry.r), ln(), name: <lng-r>),
    node((X.proj, ry.r), W("Q"), name: <wq>),
    node((X.rope-g, ry.r), srope(), name: <rope-gr>),
    snode((X.sdpa, ry.r), <rd>),
    node((X.rope-c, ry.rk), srope(), name: <rope-cr>),
    node((X.ln-c, ry.r), ln(), name: <lnc-r>),
    node((X.sdpa, ry.ro), W("O"), name: <wo>),
    pnode((X.g, ry.ro), <pr>),

    // VIT BLOCKS
    bnode((X.g, Y.blk), <blk>),

    // WRITE ADAPTER
    node((X.ln-g, wy.w), ln(), name: <lng-w>),
    node((X.proj, wy.wk), W("K"), name: <wk>),
    node((X.proj, wy.wv), W("V"), name: <wv>),
    node((X.rope-g, wy.wk), srope(), name: <rope-gw>),
    snode((X.sdpa, wy.w), <wr>),
    node((X.rope-c, wy.w), srope(), name: <rope-cw>),
    node((X.ln-c, wy.w), ln(), name: <lnc-w>),
    node((X.sdpa, wy.wo), none, name: <wwo>),
    node((X.proj, wy.wo), none, name: <wsp>),
    pnode((X.c, wy.wo), <pw>),

    // OUTPUT TOKENS
    gprefixnode((X.g, Y.tok-out + off.prefix-dy-out), <go-prefix>),
    ggridnode((X.g, Y.tok-out), <go>, img: imgs.glimpse-out),
    cprefixnode((X.c, Y.tok-out + off.prefix-dy-out), <co-prefix>),
    cgridnode((X.c, Y.tok-out), <co>, img: imgs.canvas-out),

    // EDGES - READ
    edge(<gi>, (X.g, ry.r), <lng-r>, "->", corner: right),
    edge(<lng-r>, <wq>, "->"), edge(<wq>, <rope-gr>, "->"), edge(<rope-gr>, <rd>, "->"),
    edge(<ci>, (X.c, ry.r), <lnc-r>, "->", corner: left),
    edge(<lnc-r>, (X.ln-c - off.branch-dx, ry.r), (X.ln-c - off.branch-dx, ry.rk), <rope-cr>, "->"),
    edge(<rope-cr>, (X.sdpa, ry.rk), "->"),
    edge(<lnc-r>, (X.ln-c - off.branch-dx, ry.r), (X.ln-c - off.branch-dx, ry.rv), (X.sdpa, ry.rv), "->"),
    edge(<rd>, <wo>, "->"), edge(<wo>, <pr>, "->"),
    spine(<gi>, <pr>, "->"), spine(<pr>, <blk>, "->"),

    // EDGES - WRITE
    edge(<ci>, (X.c, wy.w), <lnc-w>, "->", corner: left),
    edge(<lnc-w>, <rope-cw>, "->"), edge(<rope-cw>, <wr>, "->"),
    edge(<blk>, (X.g, wy.w), <lng-w>, "->", corner: right),
    edge(<lng-w>, (X.ln-g + off.branch-dx, wy.w), (X.ln-g + off.branch-dx, wy.wk), <wk>, "->"),
    edge(<wk>, <rope-gw>, "->"), edge(<rope-gw>, (X.sdpa, wy.wk), "->"),
    edge(<lng-w>, (X.ln-g + off.branch-dx, wy.w), (X.ln-g + off.branch-dx, wy.wv), <wv>, "->"),
    edge(<wv>, (X.sdpa, wy.wv), "->"),
    edge(<wr>, (X.sdpa, wy.wo), <pw>, "->"),
    spine(<ci>, <pw>, "->"), spine(<pw>, <co>, "->"), spine(<blk>, <go>, "->"),

    // ENCLOSURES
    enc((<lng-r>, <wq>, <rope-gr>, <rd>, <rope-cr>, <lnc-r>, <wo>), C.read-bg, name: <rb>),
    node((X.ln-c - off.group-label-dx, ry.ro), text(size: sz.ft, weight: "bold", "Read")),

    enc((<lng-w>, <wk>, <wv>, <rope-gw>, <wr>, <rope-cw>, <lnc-w>, <wwo>, <wsp>), C.write-bg, name: <wb>),
    node((X.ln-g + off.group-label-dx, wy.wo), text(size: sz.ft, weight: "bold", "Write")),

    enc((<rb>, <wb>, <blk>, <pr>, <pw>), none, dash: "dashed", inset: sz.layer-inset),
  )
}
