# Visualizations, demos and the web bundle

One pipeline serves the project page, slides and papers:

```
released model + policy + image
  → canvit_pytorch.viz.record   a Rollout: per glimpse, viewpoint, crop, canvas tokens,
                                readout logits, per-Write canvas residuals
  → canvit_pytorch.viz.pca      the paper's PCA protocol (and a fixed-limit variant)
  → canvit_pytorch.viz.web      a web bundle: manifest.json + PNG layers per glimpse
  → site/js/canvit/             ES-module components, e.g. <canvit-rollout src=…>,
                                used unchanged by the site and by slides
```

## Principles

- **Paper version first.** Policies, readouts and the PCA protocol come from
  the code behind the paper (`canvit_pytorch.policies`, EG-C2F included,
  and the ADE20K probe). Anything else is an
  extension and says so in the manifest (`"origin": "extension"`).
- **Record once, render anywhere.** The browser never recomputes model
  outputs; it colors and animates recorded ones. A bundle is generated, never
  edited by hand, and names the checkpoint revision, code commit, torch
  version and device that produced it.
- **Geometry is computed once, in Python.** Viewpoints are `(row, col, scale)`
  in `[-1, 1]` (see the root `AGENTS.md`); the bundle also stores each
  glimpse's box as `top, left, size` fractions of the scene, so no renderer
  converts coordinates.
- **Data layers are lossless and uncolored.** Class labels and scalar maps
  (entropy, confidence, canvas change) are 8-bit grayscale PNGs; the browser
  applies palettes and colormaps, so themes and legends live in one place.
  Entropy is divided by `log(num_classes)`; change is cosine dissimilarity
  divided by 2.
- **Maps sit beside the photograph**, never over it, on a fixed scale across
  glimpses.

## Web bundle, schema `canvit-rollout/1`

```json
{
  "schema": "canvit-rollout/1",
  "title": "…",
  "scene": {"image": "scene.png", "px": 512, "source": "…", "attribution": "…", "license": "…"},
  "model": {"repo": "canvit/…", "revision": "<hub commit>"},
  "readout": {"kind": "ade20k-segmentation", "repo": "canvit/probe-…", "revision": "…", "num_classes": 150},
  "policy": {"name": "entropy_coarse_to_fine", "origin": "paper", "description": "…"},
  "canvas_grid": 64,
  "glimpse_px": 128,
  "pca": {"protocol": "fixed-limits", "basis": "last-glimpse"},
  "glimpses": [
    {"t": 0,
     "viewpoint": {"row": 0.0, "col": 0.0, "scale": 1.0},
     "box": {"top": 0.0, "left": 0.0, "size": 1.0},
     "layers": {"crop": "t00/crop.png", "canvas": "t00/canvas.png", "labels": "t00/labels.png",
                "entropy": "t00/entropy.png", "change": "t00/change.png"}}
  ],
  "provenance": {"code_commit": "…", "torch": "…", "device": "mps", "precision": "float32", "created": "…"}
}
```

`pca.protocol` is `"paper"` (basis fitted on the last glimpse's canvas,
min-max per frame, as in the paper's Figure 1) or `"fixed-limits"` (same
basis, one set of color limits for every glimpse, so animations do not
flicker; an extension).

## Live in-browser inference (next phase)

[Author request, 2026-09-26: people should be able to play with the model in
the browser, fast.] Plan: export CanViT and the ADE20K probe to ONNX (the
package already tests an export path, `tests/test_export.py`) and run them with
ONNX Runtime Web on WebGPU, WebAssembly as fallback, behind the same component
interface as recorded bundles. Constraints to design for: download size
(about 100M parameters, roughly 200 MB in fp16; load on demand, consider
quantization), WebGPU availability per browser, and numerical parity with
PyTorch, checked like MPS parity before anything is shown as model output.
