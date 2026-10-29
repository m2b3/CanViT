# Checked facts

Facts about CanViT, its paper, its code and the talk's own measurements, each with where it lives and when it was
checked. Facts only: positions and plans are in `../TALK.md`. Paths are relative to this talk's directory. Numbers the paper generates stay in its macros (`../../../paper/latex/data.tex`, read by the
site through `../../assets/paper/data_macros.json`); this file points to them. The paper is
`../../../paper/latex/CanViT_Toward_AVFMs.tex`, cited by section.

## Essential

What the talk's current argument rests on. Each line was checked on the date given.

- The observer/policy framing is the paper's (§1): an active-vision model has three axes, instantaneous vision, memory
  and action selection (where and at what zoom to look next); the first two make the observer, the last the sensory
  policy. "They do not play comparable roles: given perfect instantaneous vision and memory, naive and strategic
  policies should reach the same accuracy after enough glimpses of a static scene, but no policy can make up for a
  poor observer." Prior work "has often focused on action selection". The goal: a task- and policy-agnostic AVFM,
  which "disentangles 'how to see in an active-vision setting' from 'where to look,' thus freeing active-vision
  pretraining from the complexity of RL" [read 2026-10-24].
- §6 says "perception, not viewpoint selection, was the bottleneck in this task" (line 762 of the source)
  [2026-10-24].
- The gap on ADE20K, paper protocol (`../../assets/paper/ade20k_seg.json`, read 2026-10-24): frozen DINOv3 ViT-B with
  a linear probe at 512 px 47.19 mIoU (`probe_table`); AME 27.6 at 8 glimpses, 311.5 GFLOPs; AdaGlimpse 25.7 at 8
  glimpses, 835.0 GFLOPs (22.7 at 4); CanViT-B frozen, 64² canvas: 39.59 after the full-scene glimpse; at 8 glimpses
  (176.8 GFLOPs) EG-C2F 45.36, C2F 44.94, full-then-random 43.82, random 39.36, F2C 33.22; best 45.89 (C2F, 21
  glimpses, macro `adeBestMiou`). At the second glimpse, random 27.07 and EG-C2F 42.23.
- Of the seven prior models on the timeline, all but AME learn where to look by reinforcement learning
  (`active-vision-timeline.json`; each read in its paper).
- Architecture (§4): the canvas is a grid of canvas patches tiling the scene's [−1, 1]² plus 16 canvas registers (the
  App. pretraining table and `canvit_core.config`, checked 2026-10-24), broadcast from one learnable patch so its
  resolution is set at inference; each canvas patch maps onto a fixed scene region whatever the viewpoint, which
  allows token-wise decoding and unbroken gradient flow across time; canvas tokens get no MLP, self-attention or
  projection and change only through Canvas Attention writes. Glimpse patch grids do not align across time, so glimpse
  tokens are not carried forward. SR-RoPE: positions are patch centers in scene coordinates; glimpse-patch spacing
  carries the zoom. Canvas Attention: Read = glimpse queries canvas, Write = canvas queries glimpse, residual into the
  querying stream; QKVO only on the glimpse side; stride 2 in CanViT-B, so 3 reads and 3 writes per glimpse over 12
  blocks (App. E). VPE token: (x/s, y/s, log s) through random Fourier features (App. B), "a non-essential
  architectural affordance, meant to facilitate future end-to-end policy learning".
- Pretraining (§5): after every glimpse the target is DINOv3 ViT-B's patch and CLS features of the whole 512 px scene
  (32 × 32 patches of 768 dimensions), per-position z-scored; readout = LayerNorm then a linear map per token; loss =
  squared error, patches plus CLS, averaged over time. Two rollouts per scene from a fresh canvas: R-IID (random from
  t = 0) and F-IID (full scene (0, 0, 1) at t = 0); 1 F-IID + 1 R-IID converges faster than 2 R-IID even on R-IID
  evaluation (App. ablations). Viewpoints: A ~ U[0, (1 − 0.05)²], s = 1 − √A, p(s) ∝ 1 − s, s ≥ 0.05 (0.25% of the
  scene). Truncated BPTT over chunks of K = 2, stop with p = 0.5 after each, mean length 4, constant memory.
- Scale and cost: 13.2 M ImageNet-21k scenes at 512 px, about 1 B glimpses, 166 h on one H100, 32² canvas in
  pretraining (§6); teacher features computed once, about 8 H100-equivalent hours, about 19 TiB in float16 (App. C);
  about 2500 H100-equivalent hours before submission (App. H; the camera-ready's compute reporting adds the review
  period's runs separately). DINOv3's 7B teacher: 61,440 H100-hours for its
  pretraining (Siméoni et al., arXiv:2508.10104v1, Table 20); about 9 million GPU-hours for the whole DINOv3 project
  (its carbon-footprint section); ViT-B and the other small models distilled from the 7B (§5.2); 1,689 M curated
  images (LVD-1689M, §3.1) [read 2026-10-01].
- Precision: mixed precision, with the canvas and every coordinate-related component (grid coordinates, RoPE, the VPE
  token's projection) in float32; SDPA and learned projections in bfloat16. Subtle correctness issues caused
  regressions only after several hundred thousand steps (paper App. pretraining, "Numerical precision considerations")
  [2026-10-24].
- The learned policy (`learned-policy.json`, its `_about` says how it was made): the camera-ready's appendix "Learned
  viewing policies" (`supp:learned-policies`) trains viewing policies with reinforcement learning, Q-learning and REINFORCE, on the frozen, released
  CanViT-B and its frozen ADE20K probe (64² canvas); the reward is the relative reduction in segmentation
  cross-entropy after a glimpse, with the discount factor at zero; the policy chooses each later glimpse among 512
  candidate viewpoints; every rollout starts with the full-scene glimpse. 8 training runs per objective, each 8,000
  steps of 16 images with 5 glimpses (640,000 glimpses), 74 to 91 minutes per run on one RTX 4090, validation
  included; each run's best and last checkpoints are evaluated with the paper's protocol. The deck shows Q-learning's
  best checkpoints: their means and 95% intervals are in `learned-policy.json`; C2F and EG-C2F come from the paper's
  ADE20K export [read in the camera-ready's export, 2026-10-26].

## Bank

Facts checked for earlier versions of the talk, kept because they are costly to find again.

The paper [read 2026-10-01]:
- §1: deep networks as models of biological vision cite Yamins 2014, Yamins & DiCarlo 2016, Schrimpf 2018, Zhuang
  2021, Bakhtiari 2021, Mehrer 2021, Raugel 2025. Human vision: gaze shifts toward regions of interest; sequential,
  with strategic planning (Yarbus 1967; Hoppe & Rothkopf 2019), integration across time in visual working memory
  (Baddeley & Hitch 1974; Melcher 2001), top-down recurrent feedback (Rao & Ballard 1999; Gilbert & Li 2013; Kar et
  al. 2019). Active models "have struggled to match the accuracy, efficiency, flexibility and representational
  richness of their passive counterparts", most strikingly on dense prediction.
- §2: deep active vision starts with RAM (2014), confined to digits until Saccader (2019, 75% ImageNet-1k with an
  intermediate pretraining step "to stabilize learning"); GFNet and AdaptiveNN: efficiency on real-world
  classification, fixed zoom; AME and AdaGlimpse: dense outputs by an MAE-style decoder over the full grid at every
  step (intractable at high resolution). Distillation "transfers across problem settings rather than across model
  sizes" (vs Proteus).
- §3: viewpoint (x, y, s), s the half side, a glimpse covers s² of the scene, resized to a fixed resolution; s trades
  coverage for detail. The paper writes (x, y); the code and the page use (row, col) (`../../../AGENTS.md`).
- §6: ADE20K linear probes on canvas patches, one per canvas grid; ImageNet-1k probe on the recurrent CLS; LP-FT for
  fine-tuning. Policies C2F, F2C, EG-C2F (entropy of the probe's class distribution, per quadtree level), RFS; up to
  21 glimpses. Macros: `adeSingleGlimpseMiou` (one full-scene glimpse), `adeBestMiou` (C2F, 64² canvas),
  `inkFinetunedBest`, `adeBestPriorMiou` (AME), `adeCheapestBeatFlopRatio`. C2F beats F2C with the same viewpoints;
  RFS improves then declines; a 64² canvas helps although pretraining used 32², with a separately trained linear probe
  per canvas resolution.
- §7: limitations: static scenes; forgetting may need gating; RL policies left to future work; depth expected to
  transfer; a passive teacher is required (active-to-active self-distillation proposed); one model size, a dataset
  over 100× smaller than DINOv3's LVD-1689M.
- App. G (canvas evolution figure): Write 0's residual follows the glimpse's 8 × 8 patch grid; Write 1's, the objects
  in the glimpse; Write 2's extrapolates beyond the glimpse.
- App. H: the ImageNet-1k fine-tuning experiments together (failed runs, preliminary experiments and sweeps included)
  cost under 800 USD; the reported fine-tuning run took under 15 wall-clock hours on a TPU v6e-4.
- Figure exports: `../../../paper/exports/` (figures and their JSON data); `../../copy_paper_figures.sh` copies the
  diagrams the site shows into `../../assets/paper/`; the JSON is linked.
- ADE20K training split: 20,210 images (ADEChallengeData2016). The paper scores ADE20K with images and masks resized
  to 512 × 512 without keeping the aspect ratio (App. "Task: ADE20K segmentation").

DINOv3 [read 2026-10-01 in arXiv:2508.10104v1]: the ViT-7B teacher has 6,716 M parameters (Fig. 16a) and trained for
1 M iterations at 4,096 images per batch; ADE20K with a linear layer: ViT-B 51.8 mIoU (Table 14), ViT-7B 55.9 (its
dense evaluation). The CanViT paper's own probes of DINOv3 ViT-B at 512 px: 47.19 mIoU on ADE20K, 85.0% on
ImageNet-1k.

Training and inference cost, computed for the talk by an experiment since removed from the repository [2026-10-01]:
- Training scale: CanViT-B saw 1 B glimpses; AdaptiveNN about 1.9 B (estimated from its released parameters and code,
  300 ImageNet-1k epochs); AdaGlimpse 38 to 150 B (600 epochs); AME about 16 M in its active-vision adaptation, on top
  of pretrained MAE or SETR backbones. CanViT, AdaptiveNN and AdaGlimpse start from random initialization. CanViT saw
  ten times more scenes than ImageNet-1k-trained models, but fewer glimpses than AdaGlimpse's pipeline.
- Training compute, each system's own training only (1 MAC = 2 FLOPs,
  backward = 2 × forward, measured 1.996; inherited weights and teachers not counted on any side, so DINOv3's
  pretraining is not counted for CanViT either): CanViT-B 52.5 EFLOPs of pretraining plus 2.84 of teacher features
  (55.3); computing the teacher online instead of caching would make it about 246 H100-hours and remove the 19 TiB;
  AdaGlimpse 62.9 to 144.6 (its 600-epoch reconstruction pretraining by its code or by its paper, plus 100 epochs of
  classification), from random initialization with a DeiT-III-21k teacher; AME 1.4 to 1.8 for its segmentation
  training, on top of MAE-L or SETR-ADE20K weights; AdaptiveNN-DeiT-S 22.6 EFLOPs per ImageNet-1k run; AdaGlimpse on
  ADE20K 83.7 to 166.6 over pretraining, reconstruction and segmentation (32.4 to 33.5 without the pretraining). Wall
  time as disclosed: AdaGlimpse's README about 672 A100-hours (210–410 H100-equivalent hours); CanViT 166 H100-hours
  plus 8. Own-training compute is comparable between CanViT-B and AdaGlimpse's pipeline and higher than AME's own task
  training.
- CanViT-B's adaptation: ImageNet-1k LP-FT 4.76 EFLOPs (100,080 steps × 256 scenes × 4 glimpses, full BPTT, 32²
  canvas); the ADE20K probe at the 64² canvas 0.16 EFLOPs (40,000 steps × 16 × 10 glimpses, CanViT-B frozen). The
  pretraining run averaged 87.9 TFLOP/s, 8.9% of an H100 SXM's dense bf16 peak (11.6% for a PCIe H100; which variant
  ran is not recorded).
- Inference: CanViT-B 15.48 GFLOPs per glimpse on ImageNet-1k; AdaptiveNN (DeiT-S) 2.15 GFLOPs for its first
  glimpse; GFNet (EfficientNet-B3) 0.86. Parameters as the camera-ready's ImageNet-1k cost table counts them
  (`supp:in1k-efficiency`: every module a model runs up to the glimpse evaluated, classifiers included; read
  2026-10-26): CanViT-B 95.9 M at every glimpse (95.2 M without its classifier); AdaptiveNN 22.0 M at its first glimpse,
  82.4 M at its fifth; GFNet (EfficientNet-B3) 12.2 M at its first, 37.4 M at its fourth. CanViT is not expected to
  lead on accuracy-efficiency in classification. Passive efficient segmenters for context
  (single-scale, their own settings): EfficientViT-L2 51 M, 90 GFLOPs, 50.7 mIoU; FocalNet-B 126 M, 2384 GFLOPs, 50.5;
  CanViT-B 15.8 GFLOPs for 38.5 at t = 0, 464.1 GFLOPs for 45.89 at t = 20 (64² canvas). EG-C2F's viewpoint selection
  costs about 1–5 MFLOP, under 0.03% of a glimpse.
- The absolute contribution of DINOv3 to CanViT is not isolated.

Measurements for the talk:
- The uncertainty read out from the canvas tracks the segmentation's errors [2026-10-01,
  measured by a talk experiment since removed from the repository]: over the 2,000 ADE20K
  validation images (7.5 M labeled canvas cells, 64 × 64 canvas, each cell's decoded class against the annotation at
  its center), after the full-scene glimpse cell accuracy 77.9%, entropy tells wrong cells from right ones with an
  AUROC of 0.836 (mean 1.96 bits against 0.79), and accuracy falls from 99% in the surest tenth of cells to 37% in the
  least sure; after C2F's first five glimpses 80.1%, AUROC 0.841, 99% to 41%.
- The VPE token helps at long horizons and fine canvases (+1.28 mIoU at t = 20 under EG-C2F, 64² canvas on 1024 px
  scenes; the camera-ready's `supp:vpe-long-horizon`, read 2026-10-26).
- Memory reset [2026-10-01, a talk experiment since removed]: every "thing" object covering 1–15% of an
  ADE20K validation scene (2,781 objects); one glimpse on the object (its box about half the glimpse), then three
  glimpses as far as possible, none containing the object's class; mean p(class) over the object's pixels after the
  last glimpse: 0.462 ± 0.006 with the canvas carried, 0.031 reset before each glimpse, 0.038 when the object's own
  glimpse is skipped. Among objects recognized after their glimpse (p ≥ 0.5, 1,579): 0.683 against 0.038.
- Memory across three glimpses [2026-10-01, a talk experiment since removed]: 6,218 sequences of three objects of
  distinct classes in 442 ADE20K validation images, one glimpse per object, left to right; after the third glimpse,
  mean p over the first object's pixels 0.410 kept against 0.041 reset, the second's 0.451 against 0.042. The
  left-to-right order flatters the chosen examples (other orders keep the first object at 0.52 to 0.75, reset stays at
  most 0.08); after two earlier glimpses the current object scores 0.048 lower with the canvas kept than alone
  (unexplained). ADE_val_00000836 (shop sign 0.91 / 0.90 / 0.04, person 0.76 / 0.68 / 0.06 after its glimpse / kept /
  reset) is below the strict bar for person; ADE_val_00001182 (a bedroom) meets it.
- Table corners [2026-10-01, `experiments/table_corners/sweep.py`]: over 1,436 large objects of ADE20K validation,
  after glimpses at both ends the canvas labels on average 65% of the unseen middle as the object (53% after one end),
  and also labels as the object 46% of the other pixels in the band between the glimpses (mean; median 41%). The
  conference-table example was selected for few false positives (8%).
- DINOv3 ViT-B on the whole scene at 128 px [2026-10-01, a talk experiment since removed, pixel accuracy as the
  bundles measure it]: 72.1, 69.9 and 40.7% on the street, shop and ferry of the recorded R-IID bundles, against
  CanViT-B's 87.9, 77.8 and 71.4% after their 21 glimpses. The paper's matched comparison: DINOv3 ViT-B at 128 px 28.8
  mIoU against CanViT-B's 29.3 to 39.6 from one full-scene glimpse (`passive_comparison_rows`).
- Policies on ImageNet-1k: after one glimpse C2F 76.8 against F2C 32.2 top-1 (ADE20K 39.6 against 11.0 mIoU); after 21
  glimpses the ImageNet-1k policies end within 1.3 points of each other (RFS aside), C2F 81.13 and F-IID 81.14, while
  the ADE20K ones end 3 points apart and still rising.

Scope of the paper's claims:
- 38.5% mIoU "in a single glimpse" is a single low-resolution glimpse of the full scene.
- "Any policy" means the tested policies, horizons and resolutions.
- The 45.9% headline is C2F at a 64² canvas.
- The canvas's geometry is given (each glimpse's position and scale): CanViT addresses what to keep and how to
  integrate across views, not how the brain computes remapping.
- "By a wide margin" on ImageNet-1k is +2.3 fine-tuned; frozen CanViT-B is below AdaptiveNN.
- Saccader's 75.03% ImageNet-1k is its six locations read by a separate NASNet classifier on 331 px images; Saccader
  alone at 224 px is 70.31%; its attention network sees the whole image. AdaptiveNN continues GFNet (same group), not
  AdaGlimpse; its human comparison uses SALICON, collected by mouse tracking.
- LookWhere (NeurIPS 2025; 83.0 ImageNet-1k, 44.6 ADE20K) selects patches once from a low-resolution view: not a
  sequential glimpse model under the paper's definition, close to CanViT-B's numbers, not cited by the paper.

The code (`canvit-pytorch/`, by module and symbol) [read 2026-10-01]:
- The released segmenter: `viz.released_model.load_released_segmenter(scene_size_px=512, canvas_grid_size=64,
  device=...)` = `hub.repos.FLAGSHIP` + the ADE20K probe for that canvas grid.
- The distillation readout: `model.pretraining.CanViTForPretraining.from_pretrained(hub.repos.FLAGSHIP)`;
  `predict_teacher_patches(canvas)` is standardized and needs a 32² canvas (the standardizer has 32² positions);
  `teacher_patch_standardizer.destandardize(...)` gives DINOv3 space.
- The teacher: `teacher.load_teacher(teacher.TEACHER_REPO, device)`; `.patches` are after DINOv3's final LayerNorm (5
  prefix tokens dropped: CLS and 4 registers).
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

Published accuracy over time, each point read in its paper [2026-10-01]: sequential active models top out at 82.2 on
ImageNet-1k (AdaptiveNN, 2025) and 27.6 mIoU on ADE20K (AME, 2023); passive models reach 91.1 and 63.0, and a frozen
DINOv2 linear probe alone 49.0 mIoU (2023). STAM (CVPR 2022, 80.78) is missing from the paper's comparison table.
