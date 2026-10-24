# Checked facts

Facts the talk relies on, each with where it lives, dated when checked. Numbers the paper generates stay in its
macros; this note points to them. Paths are relative to this talk's directory.

Paper: `../../../paper/latex/CanViT_Toward_AVFMs.tex`, cited by section; its numbers are macros (`../../../paper/latex/data.tex`), which the site
reads through a link (`../../assets/paper/data_macros.json`).

The paper [read 2026-10-01]:
- §1: deep networks as models of biological vision cite Yamins 2014, Yamins & DiCarlo 2016, Schrimpf 2018, Zhuang
  2021, Bakhtiari 2021, Mehrer 2021, Raugel 2025. Human vision: gaze shifts toward regions of interest; sequential,
  with strategic planning (Yarbus 1967; Hoppe & Rothkopf 2019), integration across time in visual working memory
  (Baddeley & Hitch 1974; Melcher 2001), top-down recurrent feedback (Rao & Ballard 1999; Gilbert & Li 2013; Kar et
  al. 2019). Active models "have struggled to match the accuracy, efficiency, flexibility and representational
  richness of their passive counterparts", most strikingly on dense prediction. The three axes and "no policy can
  make up for a poor observer" (given perfect instantaneous vision and memory, naive and strategic policies reach
  the same accuracy after enough glimpses of a static scene). The goal: "a task- and policy-agnostic AVFM".
- §2: deep active vision starts with RAM (2014), confined to digits until Saccader (2019, 75% ImageNet-1k with an
  intermediate pretraining step); GFNet and AdaptiveNN: efficiency on real-world classification, fixed zoom; AME and
  AdaGlimpse: dense outputs by an MAE-style decoder over the full grid at every step (intractable at high
  resolution). Distillation "transfers across problem settings rather than across model sizes" (vs Proteus).
- §3: viewpoint (x, y, s), s the half side, a glimpse covers s² of the scene, resized to a fixed resolution; s trades
  coverage for detail. The paper writes (x, y); the code and the page use (row, col) (`../../../AGENTS.md`).
- §4: the canvas = 16 canvas registers + an H × W grid of canvas patches, broadcast from one learnable patch, so its
  resolution is chosen at inference; canvas tokens never see an MLP, self-attention or a projection; "cognitive map"
  cites Tolman 1948. SR-RoPE: positions are patch centers in scene coordinates; glimpse-patch spacing carries the
  zoom. Canvas Attention: Read = glimpse queries canvas, Write = canvas queries glimpse; stride 2 in CanViT-B, so 3
  reads and 3 writes per glimpse over 12 blocks (App. E). VPE token: (x/s, y/s, log s) through random Fourier
  features (App. B); the ablation shows it matters least.
- §5: the target is DINOv3 ViT-B's patch and CLS features of the 512 px scene (32 × 32 patches), per-position
  z-scored; readout = LayerNorm then a linear map per token; loss = squared error, patches + CLS, averaged over
  time. Dual rollouts R-IID and F-IID (F-IID starts at (0, 0, 1)). Viewpoints: A ~ U[0, (1 − 0.05)²], s = 1 − √A,
  p(s) ∝ 1 − s, s ≥ 0.05 (0.25% of the scene). TBPTT chunks K = 2, stop with p = 0.5 after each, mean length 4.
- §6: 13.2 M ImageNet-21k scenes at 512 px, about 1 B glimpses, 166 h on one H100, 32² canvas in pretraining;
  ADE20K linear probes on canvas patches, one per canvas grid; ImageNet-1k probe on the recurrent CLS; LP-FT for
  fine-tuning. Policies C2F, F2C, EG-C2F (entropy of the probe's class distribution, per quadtree level), RFS; up
  to 21 glimpses. Results (macros): `adeSingleGlimpseMiou` (one full-scene glimpse), `adeBestMiou` (C2F, 64²
  canvas), `inkFinetunedBest`, `adeBestPriorMiou` (AME), `adeCheapestBeatFlopRatio`. C2F beats F2C with the same
  viewpoints; RFS improves then declines; a 64² canvas helps although pretraining used 32².
- §7: limitations: static scenes; forgetting may need gating; RL policies left to future work; depth expected to
  transfer; a passive teacher is required (active-to-active self-distillation proposed); one model size, a dataset
  over 100× smaller than DINOv3's LVD-1689M.
- App. C: teacher features computed once, about 8 H100-equivalent hours, about 19 TiB in float16.
- App. G (canvas evolution figure): Write 0's residual follows the glimpse's 8 × 8 patch grid; Write 1's, the
  objects in the glimpse; Write 2's extrapolates beyond the glimpse.
- App. H: about 2500 H100-equivalent hours for the whole project; the ImageNet-1k fine-tuning experiments together
  (failed runs, preliminary experiments and sweeps included) under 800 USD; the reported fine-tuning run under 15
  wall-clock hours on a TPU v6e-4.
- Figure exports: `../../../paper/exports/` (figures and their JSON data); `../../copy_paper_figures.sh` copies the
  diagrams the site shows into `../../assets/paper/`; the JSON is linked.

Accounting computed for the talk (`experiments/training_cost/`) and limitations [2026-10-01]:
- The absolute contribution of DINOv3 is not isolated; this stays a stated limitation, as do static scenes.
- Training scale: CanViT-B saw 1 B glimpses; AdaptiveNN about 1.9 B (estimated from its released parameters and
  code, 300 ImageNet-1k epochs); AdaGlimpse 38 to 150 B (600 epochs); AME about 16 M in its active-vision adaptation,
  on top of pretrained MAE or SETR backbones. CanViT, AdaptiveNN and AdaGlimpse start from random initialization.
- Training compute: 166 H100-hours of pretraining plus 8 H100-equivalent hours of teacher features (174); computing
  the teacher online instead of caching would make it about 246 and remove the 19 TiB. AdaGlimpse, estimated from its
  repository's "around 1 week on 4x A100 GPUs": 210–410 H100-equivalent hours.
- Efficiency claims are scoped to dense prediction: CanViT is not expected to lead on accuracy-efficiency in
  classification. On ImageNet-1k (recomputed from released code): CanViT-B 95.2 M active parameters, 15.48 GFLOPs per
  glimpse; AdaptiveNN (DeiT-S) 2.15 GFLOPs for its first glimpse, 83.2 M parameters in all; GFNet (EfficientNet-B3)
  0.86 GFLOPs, 39.1 M in all.
- Passive efficient segmenters for context (single-scale, their own settings): EfficientViT-L2 51 M, 90 GFLOPs,
  50.7 mIoU; FocalNet-B 126 M, 2384 GFLOPs, 50.5; CanViT-B 15.8 GFLOPs for 38.5 at t = 0, 464.1 GFLOPs for 45.89 at
  t = 20 (64² canvas).
- Training FLOPs, each system's own training only (`experiments/training_cost/accounting.py`; 1 MAC = 2 FLOPs,
  backward = 2 × forward, measured 1.996; inherited pretrained weights and teachers are not counted on any side, so
  DINOv3's own pretraining is not counted for CanViT either): CanViT-B 52.5 EFLOPs of pretraining plus 2.84 of
  teacher features (55.3), from random initialization; AdaGlimpse 62.9 to 144.6 (its 600-epoch reconstruction
  pretraining by its code or by its paper, plus 100 epochs of classification), from random initialization with a
  DeiT-III-21k teacher; AME 1.4 to 1.8 for its segmentation training, on top of MAE-L or SETR-ADE20K weights whose
  training is not counted. AdaptiveNN discloses no training cost; its ImageNet-1k run is from scratch, 300 epochs,
  four 112 px fixations plus a glance and a regularization pass per image per step. Reading: own-training compute is
  comparable between CanViT-B and AdaGlimpse's ImageNet-1k pipeline; the defensible differences are wall time as
  disclosed (AdaGlimpse's README about 672 A100-hours; CanViT 166 H100-hours plus 8), AME's inherited weights, and
  that CanViT-B is one task- and policy-agnostic model where the others are trained per task. For a scientist the
  training cost that follows is adapting it: a linear probe on frozen features, or LP-FT (the reported ImageNet-1k run
  under 15 wall-clock hours on a TPU v6e-4); never claim CanViT-B was cheaper to pretrain than every prior model.
- CanViT-B's adaptation, counted with the same formulas (`experiments/training_cost/export.py`, run 2026-10-01):
  ImageNet-1k LP-FT 4.76 EFLOPs (100,080 steps × 256 scenes × 4 glimpses, full BPTT, 32² canvas); the ADE20K probe
  at the 64² canvas 0.16 EFLOPs (40,000 steps × 16 × 10 glimpses, CanViT-B frozen). The pretraining run's average rate:
  52.5 EFLOPs over 166 h is 87.9 TFLOP/s, 8.9% of an H100 SXM's dense bf16 peak (11.6% for a PCIe H100; which variant
  ran is not recorded here).
- The prior active models' own training compute, counted from their code, papers and released checkpoints under the
  same conventions [a subagent, 2026-10-01, `sources/baseline-training-compute.md`, script
  `experiments/training_cost/baselines.py`]: AdaptiveNN-DeiT-S on ImageNet-1k 22.6 EFLOPs per training run (1.53 B
  fixations of 112 px; its count reproduces the paper's inference-cost curve within 1.3%, whose "GFLOPs" are
  multiply-adds; whether its 82.2 ± 0.12 averages five training runs is not determined); AdaGlimpse on ADE20K 83.7 to
  166.6 EFLOPs over pretraining, reconstruction and segmentation training (32.4 to 33.5 without the pretraining, which
  may come from "Beyond Grids", arXiv:2309.13353, by the same first author). Its segmentation glimpses of 48 px are 9
  patches.

Measurements for the talk:
- The uncertainty read out from the canvas tracks the segmentation's errors [measured 2026-10-01,
  `experiments/metacognition/calibration.py`, `../../data/talk/metacognition/calibration.json`]: over the 2,000 ADE20K
  validation images (7.5 M labeled canvas cells, 64 × 64 canvas, each cell's decoded class against the annotation at
  its center), after the full-scene glimpse cell accuracy 77.9%, entropy tells wrong cells from right ones with an
  AUROC of 0.836 (mean 1.96 bits against 0.79), and accuracy falls from 99% in the surest tenth of cells to 37% in the
  least sure; after C2F's first five glimpses 80.1%, AUROC 0.841, 99% to 41%.
- The VPE token helps at long horizons and fine canvases (+1.28 mIoU at t = 20, 64² canvas); EG-C2F's viewpoint
  selection costs about 1–5 MFLOP, under 0.03% of a glimpse.
- Memory reset [measured 2026-10-01, summary in `../../data/talk/memory/overall-k1.json`]: every "thing" object
  covering 1–15% of an ADE20K validation scene (2,781 objects); one glimpse on the object (its box about half the
  glimpse), then three glimpses as far as possible, none containing the object's class; mean p(class) over the
  object's pixels after the last glimpse: 0.462 ± 0.006 with the canvas carried, 0.031 reset before each glimpse,
  0.038 when the object's own glimpse is skipped. Among objects recognized after their glimpse (p ≥ 0.5, 1,579):
  0.683 against 0.038.
- Memory across three glimpses [measured 2026-10-01 by a subagent; its sweep reproduced exactly the same day by
  `experiments/memory/sweep.py`; the numbers beyond the means were read from its report]: 6,218 sequences of three
  objects of distinct classes in 442 ADE20K validation images, one glimpse per object (box about half the glimpse),
  left to right; after the third glimpse, mean p over the first object's pixels 0.410 kept against 0.041 reset, the
  second's 0.451 against 0.042. Good examples sit above the 95th percentile of the weakest kept object (median 0.073).
  The left-to-right order flatters the chosen examples (other orders keep the first object at 0.52 to 0.75, reset
  stays at most 0.08); after two earlier glimpses the current object scores 0.048 lower with the canvas kept than
  alone (unexplained). The slide's example, ADE_val_00000836 (shop sign 0.91 / 0.90 / 0.04, person 0.76 / 0.68 / 0.06
  after its glimpse / kept / reset), is below the strict bar for person; ADE_val_00001182 (a bedroom) meets it.
- DINOv3 ViT-B on the whole scene at 128 px [measured 2026-10-01, `experiments/whole_scene_baseline`, pixel accuracy
  as the bundles measure it]: 72.1, 69.9 and 40.7% on the street, shop and ferry of the recorded R-IID bundles, against
  CanViT-B's 87.9, 77.8 and 71.4% after their 21 glimpses.

The timeline's models, what each introduced, checked against the papers [a subagent, 2026-10-01,
`sources/active-vision-novelty.md`, pages there]: AdaptiveNN continues GFNet (same Tsinghua group: Yulin Wang, Shiji
Song, Gao Huang; glance, fixed-size fixations, stopping, PPO), not AdaGlimpse, which it never cites; its human
comparison uses SALICON, collected by mouse tracking, not eye tracking. Saccader's 75.03% ImageNet-1k is its six
locations read by a separate NASNet classifier on 331 px images; Saccader alone at 224 px is 70.31%; its attention
network sees the whole image. Of the seven, all but AME learn where to look by reinforcement learning; Larochelle &
Hinton 2010 and Ranzato 2014 did not, and RAM is the usual start of the deep-learning line, not its first glimpse
model. Random against learned glimpse choice, in the papers' own ablations: GFNet at 5 steps 74.46% against 75.93%
(ImageNet-1k, Table 1, p. 9); AME 72.4% against 73.4% (SUN360 classification, 8 glimpses of 32² px, Table 5, p. 7).

The history data [assembled 2026-10-01]: `sources/sota-history.json` (every point read in its paper; `_about`
defines the fields and series) and `sources/sota-history.md` (sources, leads, caveats). Sequential active models top
out at 82.2 on ImageNet-1k (AdaptiveNN, 2025) and 27.6 mIoU on ADE20K (AME, 2023); passive models reach 91.1 and
63.0, and a frozen DINOv2 linear probe alone 49.0 mIoU (2023). STAM (CVPR 2022, 80.78) is missing from the paper's
comparison table.

DINOv3 [read 2026-10-01 in arXiv:2508.10104v1]: LVD-1689M, 1,689 million images curated from about 17 billion web
images (§3.1); the ViT-7B teacher, 6,716M parameters (Fig. 16a), trained for 1M iterations at 4,096 images per batch,
61,440 H100-hours for its pretraining (Table 20); about 9 million GPU-hours for the whole project (its carbon
footprint section); the smaller ViTs, ViT-B included, distilled from the 7B (§5.2); ADE20K with a linear layer: ViT-B
51.8 mIoU (Table 14), ViT-7B 55.9 (the text of its dense evaluation).

The code (`canvit-pytorch/`, by module and symbol) [read 2026-10-01]:
- The released segmenter: `viz.released_model.load_released_segmenter(scene_size_px=512, canvas_grid_size=64,
  device=...)` = `hub.repos.FLAGSHIP` + the ADE20K probe for that canvas grid.
- The distillation readout: `model.pretraining.CanViTForPretraining.from_pretrained(hub.repos.FLAGSHIP)`;
  `predict_teacher_patches(canvas)` is standardized and needs a 32² canvas (the standardizer has 32² positions);
  `teacher_patch_standardizer.destandardize(...)` gives DINOv3 space.
- The teacher: `teacher.load_teacher(teacher.TEACHER_REPO, device)`; `.patches` are after DINOv3's final LayerNorm
  (5 prefix tokens dropped: CLS and 4 registers).
- DINOv3 ADE20K probes: `hub.repos.released_dinov3_ade20k_probe("dv3b" | "dv3s", input_size_px=128 … 512)`;
  `probes.SegmentationProbe` takes [B, H, W, D].
- Rollouts: `episode.run_episode(canvit=, images=, policy=, num_glimpses=, glimpse_size_px=, initial_state=)`;
  `policies.FixedSequence`, `policies.random.random_viewpoints` (the paper's distribution),
  `policies.quadtree.coarse_to_fine`; `viewpoint.Viewpoint(centers=[B, 2] (row, col), scales=[B])`,
  `Viewpoint.full_scene`, `viewpoint.sample_at_viewpoint`.
- PCA as in the paper: `viz.pca` (`layernorm`, `fit_pca`, `project`, `color_limits`, `to_rgb`).
- `uv run --no-sync --project canvit-pytorch python -m canvit_pytorch.viz` records bundles (`Rollout`: policy,
  `--capture-writes` for write residuals; `SmoothPath`).
- URLs (paper, code, Hub, page): `canvit_pytorch.project`; `../../make_qr_codes.py` reads them.


Experiments behind slides: `experiments/`, a package per experiment, each step run from this directory as
`uv run --no-sync --project ../../../canvit-pytorch python -u -m experiments.<experiment>.<step>` (`training_cost` in
`../../../paper/exporter`'s); the sweeps write to `../../.experiments/` (ignored by git), the exports the deck loads to
`../../data/talk/` (deployed, never committed). `experiments/build_deck_data.sh` reruns them all, or the named ones,
with the examples the slides show.

## Claims and their scope

- 38.5% mIoU "in a single glimpse" is a single low-resolution glimpse of the *full scene*; say so.
- "Under 800 USD" (App. H) is everything the ImageNet-1k fine-tuning experiments cost together, failed runs and
  sweeps included, never the cost of a fine-tuning run [Yohaï, 2026-10-01]. The run itself: under 15 wall-clock hours
  on a TPU v6e-4.
- "Perception, not viewpoint selection, was the bottleneck" is the paper's sentence; the evidence is CanViT beating
  the prior models even with F2C.
- "Any policy" means the tested policies, horizons and resolutions.
- The canvas resolution result uses a separately trained linear probe per resolution.
- The canvas's geometry is given (each glimpse's position and scale): CanViT addresses what to keep and how to
  integrate across views, not how the brain computes remapping.
- The 45.9% headline is C2F at a 64² canvas; a chart showing it must show that curve.
- Never say CanViT saw "an order of magnitude more" data than previous active models [Yohaï, 2026-10-01]: it saw ten
  times more scenes than ImageNet-1k-trained models, but AdaGlimpse's pipeline saw 38 to 150 billion glimpses against
  CanViT's 1 billion (Checked facts). Training compute is "comparable or lower" against AdaGlimpse only, and higher
  than AME's own task training.
- Table corners [measured 2026-10-01, `experiments/table_corners/sweep.py`]: over 1,436 large
  objects of ADE20K validation, after glimpses at both ends the canvas labels on average 65% of the unseen middle as
  the object (53% after one end), and it also labels as the object 46% of the other pixels in the band between the
  glimpses (mean; median 41%). The extrapolation is real and imprecise. The conference-table example was selected
  for few false positives (8%); say so when giving the average.
- Compare like with like: never against far larger models [Yohaï, 2026-10-01]. The history chart's passive lines are
  restricted to models of the active models' size (base size, ViT-B or smaller backbones), each point with its
  parameter count and test resolution known (`sources/sota-history.md`); the all-sizes frontier (up to billions of
  parameters, 475–800 px, billions of extra images) is not drawn. The sizes of the active incumbents are recorded as
  precisely as the passive ones, every network they run at test time (`params_detail` in `sources/sota-history.json`)
  [Yohaï, 2026-10-01].
  Most active models start from passive networks or teachers. LookWhere (NeurIPS 2025; 83.0 ImageNet-1k, 44.6 ADE20K)
  selects patches once from a low-resolution view: not a sequential glimpse model under the paper's definition, but
  close to CanViT-B's numbers and not cited by the paper; expect the question.
