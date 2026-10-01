# MAIN 2026 talk: open work

A 30-minute talk at MAIN 2026 (Montreal AI and Neuroscience), October 30, 2026, for an audience of
neuroscientists, cognitive scientists and machine-learning researchers; presented by Yohaï-Eliel Berreby, possibly
with Sabrina Du. The NeurIPS camera-ready is due November 6, 2026 (`camera_ready/REQUIREMENTS.md` in the paper's
repository), so work that serves both is worth more.

Each slide's status is its `data-status` in `index.html`. This file ranks what is left. Difficulty: S (hours),
M (a day), L (days). Leverage: what the talk loses without it. Status: `todo`, `doing`, `done`, `decide` (needs the
authors). "Also" names what else the work serves: the project page (`site/index.html`) or the camera-ready.

## Visuals and experiments

| # | Item | Slide | Diff. | Leverage | Cool | Also | Status |
|---|---|---|---|---|---|---|---|
| 1 | `<canvit-foveate>`: a scene through a retina, sharp at the fixation, blurred with eccentricity, fixations along a recorded scanpath; the pointer can take over | `human-vision` | M | high: the opening hook | high | page | todo |
| 2 | DINOv3 feature explorer: teacher-feature export (`canvit_pytorch.viz`) and `<canvit-features>`: similarity to the hovered patch, PCA colors, linear-probe segmentation | `dinov3` | M–L | high: what a feature map is | very high | page | todo |
| 3 | Distillation, shown: bundles recording the pretrained readout's prediction of the teacher's features, in the teacher's PCA colors, glimpse by glimpse | `distillation` | M | high: the training objective | high | page, paper figure | todo |
| 4 | Architecture built click by click: tokens through reads and writes, the canvas filling | `architecture` | M | high | high | page | todo |
| 5 | SR-RoPE geometry: glimpse patch centers over the canvas grid in scene coordinates, packed or spread with zoom | `sr-rope` | S–M | medium: the retinotopic/spatiotopic binding | high | page | todo |
| 6 | History timeline, two lanes, with thumbnails and citations (being verified) | `history` | M | high for this audience | medium | camera-ready related work | doing |
| 7 | `<canvit-frontier>` that can show only the prior active models and passive DINOv3 (probe_table) first, then CanViT | `gap`, `results` | S–M | high | medium | page | todo |
| 8 | The page's "What's in an active-vision model?" as one component used by the page and the talk, with the cycling policy slot | `three-axes` | S | medium | medium | page | todo |
| 9 | Canvas resolution dial: one rollout decoded at 16², 32², 64²; DINOv3's input-matched number read from the export | `decoupling` | S | medium | medium | page | todo |
| 10 | Pretraining's random rollouts, sampled live with the paper's distribution | `policy-agnosticism` | S | medium | medium | | todo |
| 11 | Architecture control (rebuttal Table R1) and ablations as charts; numbers into the paper's macros at camera-ready | `architecture-vs-pretraining` | S | high: architecture vs pretraining | medium | camera-ready | todo |
| 12 | Pretraining economics as a chart: teacher export, pretraining, the baseline architecture's training | `pretraining` | S | high | medium | camera-ready compute table | todo |
| 13 | Filling in: two glimpses at the two ends of an object, the segmentation filling the space between (recording with chosen viewpoints) | `extrapolation` | S–M | high for neuroscientists (amodal completion) | high | page | todo |
| 14 | Policy curves (C2F, F2C, RFS) from `ade20k_seg.json`, as a component | `policies` | S–M | medium | medium | page (Figure 3B) | todo |
| 15 | Free-form entropy guidance: next glimpse at the most uncertain region, any position and zoom; live in the browser with the priority map and scanpath; measured on ADE20K against EG-C2F | `entropy-guided` | M–L | high: novel, interactive | very high | page | todo |
| 16 | Live canvas similarity: hover the live canvas, see what the model groups together | `live` | S–M | medium | high | page | todo |
| 17 | `<canvit-live>` laid out for a slide: large panels, little chrome; one model session shared by every live slide | `live` | M | high on the day | | | todo |
| 18 | Learned viewing policy on a frozen observer (CanViT-PyTorch-RL): results chart from its saved evaluations | `learned-policy` | S | medium | medium | | decide: unpublished |
| 19 | The learned policy's scanpaths, recorded (needs the RL code migrated to canvit-pytorch 0.2) | `learned-policy` | L | low | high | | decide |
| 20 | CanViT in JAX and MLX: clean up the ports so the "runs everywhere" slide is true | `code` | L each | low for the talk | medium | | decide |

## Content and logistics

| # | Item | Diff. | Status |
|---|---|---|---|
| 21 | Logos: McGill, Mila, Université Laval, funders, PyTorch, JAX, MLX, ONNX Runtime | S | doing |
| 22 | QR code to the project page, generated (not a downloaded image) | S | todo |
| 23 | Collaborators and directions slide: names and wording from the authors | S | decide |
| 24 | Funding slide from the paper's acknowledgments: the authors confirm the list | S | decide |
| 25 | Third-party figures for the primer (Yarbus's scanpaths, acuity falloff, RAM), cited on the slide | S | todo |
| 26 | Offline on the day: ONNX Runtime Web loads from jsDelivr (`js/canvit/live-model.js`); install it locally like reveal.js | S–M | todo |
| 27 | Canvases sized from their displayed size, so components stay sharp when the deck scales (episode, mosaic) | S | todo |
| 28 | Rehearse, and shape the talk around what is correct, clear and meaningful for this audience | — | decide |
| 29 | Who presents what, if Sabrina co-presents | — | decide |

## Ideas worth considering

- Human visual working memory holds a few objects (Luck & Vogel 1997); the canvas holds 4,096 × 1,024 numbers.
  The contrast invites the question of what a capacity-limited canvas would learn to keep.
- Change blindness, in the model: change part of a scene between glimpses; the canvas keeps the old content
  wherever it does not look again. A demonstration of what a static-scene observer assumes, and a natural
  bridge to video.
- Inhibition of return: a purely greedy uncertainty-driven policy may keep returning to the same confusing
  region; suppressing visited regions is the textbook fix. Test it (item 15) before claiming it.
- Priority maps (LIP, FEF, superior colliculus) as the frame for the uncertainty map that drives the free-form
  policy (item 15).
- Coarse-to-fine perception (global precedence; low spatial frequencies first) as the frame for C2F beating F2C
  (item 14).
