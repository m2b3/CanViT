# The MAIN 2026 talk

This talk's guide: the story, Yohaï's own wording for it, the claims to state carefully, the facts a session had to
look up, and the authors' decisions. `../AGENTS.md` (how slides are written and checked) applies. `OUTLINE.md` is
the talk slide by slide, `PLAN.md` the open work, `sources/` the material the slides draw on.

A talk of about 30 minutes at MAIN 2026 (Montreal AI and Neuroscience; October 29–30, 2026, HEC Montréal; organized by
Centre UNIQUE), on October 30, given at the program committee's invitation. Presented by Yohaï-Eliel Berreby,
possibly with Sabrina Du. The audience mixes neuroscientists, cognitive scientists and machine-learning researchers.

## The story

[Yohaï, 2026-10-01: "the logical ordering, proving important propositions, etc, are truly the most important
things".] Each slide states or shows one step; a slide that does neither goes. The opening order is Yohaï's: the
neuro-AI tradition, then vision is active, then why knowledge of the world matters (cognition first, then foundation
models), then what active computer vision tried and why it stalled, and only then CanViT.

- Deep networks became models of the visual system; the ones used today see each image whole, once.
- Human vision is active: sharp central vision, blurry periphery, the eyes moving to what matters, and what was
  seen carried from one fixation to the next.
- Seeing two corners of a table, you know the table runs between them: you understand objects, you know the
  world. Pasting a passive model's answers for each view in place leaves the space in between empty.
- Vision foundation models are where machines get that knowledge, without labels.
- Active computer vision was not nearly as smart as passive computer vision, and never had that knowledge.
- It was stuck because it focused on choosing where to look: no policy can make up for a poor observer.
- CanViT is that observer: a Vision Transformer for the glimpse, a canvas for the scene, memory dumb but not too
  dumb.
- It learns by passive-to-active distillation: wherever it looks, its best guess about the whole scene.
- It works: it paints the scene glimpse by glimpse, keeps what it saw, extends what it did not see, and beats the
  prior active models with any of the tested policies, even a worse-than-random one.
- What this opens for neuro-AI.

Efficiency is a link in this chain, never a selling point [Yohaï, 2026-10-01: "they dont [care] about
efficiency unless you make them understand why you need efficiency (because you need scale. why do you need scale?
etc)"]: knowledge of the world comes with scale; scale needs cheap training and cheap reading of the memory; active
models had neither. Cost numbers appear only in that role.

## Yohaï's words

The slides are his talk, so his wording comes first for titles, captions and notes (`../AGENTS.md`, "Writing
slides"). His spoken explanation of CanViT (September 2026; an edited reconstruction of the recording, read in
full on 2026-10-01; check the audio before quoting it as exact speech in print):

- How he talks: plain words, concrete everyday examples (a cup thrown at you, closing your eyes in a room, the two
  corners of a table, a drone photographing cows), short sentences, first person ("I realized", "I wanted"), a
  question answered bluntly ("What about the space in between? No clue."), dry irony ("Great proposition."), no
  jargon without its meaning. Notes are written the way he would say them.
- Active vision: "We have sharp central vision and blurry peripheral vision." CanViT sees only part of the image,
  and "there's a world of difference between seeing something blurry and seeing nothing at all."
- The history: "Active computer vision was not nearly as smart as passive computer vision. Even with lots of
  training, lots of data, and lots of inference compute, it was terrible. The whole point was supposed to be
  efficiency ... Instead, there was worse peak accuracy and worse efficiency at every level." Why: "people were
  focusing on the wrong thing ... I made the same mistake as everybody else: I focused on choosing where to look."
- The observer: "it doesn't matter what policy I have if I don't have a good observer." "What's an observer? It's
  the thing that makes sense of the world given some inputs. That process involves not just instantaneous
  vision—what I'm looking at right now—but also memory. That's where the canvas comes from." "We had to build a
  good observer because there wasn't one. There were good computer-vision models, but passive ones. They didn't
  have this concept of integrating information into a global scene understanding." "No policy can make up for a
  poor observer, and a general-purpose observer lets you use any policy."
- Prior observers borrowed from passive models: "the geometry was wrong. That model understood individual frames;
  it didn't understand how to relate them to one another."
- The table: "Suppose I see one corner of the table, then another corner. The naïve approach is to paste the
  understanding from here and paste the understanding from there. What about the space in between? No clue. A
  smart observer should infer from those observations that the table extends between them, because it
  understands objectness and has knowledge of the world."
- The canvas: "a fine-grained, scene-wide memory that would let me ask, at any point in time and for any
  position: what does the model think is there?" Prior dense models "would have a cheap encoder and a massive
  decoder ... decoding could be ten times the budget." "I put the intelligence on the Vision Transformer side, and
  made the memory dumb—but not too dumb. Just dumb enough that I could interact with it cheaply, while leaving it
  flexible enough to reconstruct unseen things from world-knowledge priors, like the table example." The name:
  "It's as though it paints its understanding on a canvas." What it stores: "what you saw and what you
  understood. Not just pixels ... my best guess—my best understanding at any given time."
- Distillation: "separating the architecture problem from the problem of learning good representations." "We
  already know what good visual representations look like." "Distillation across settings: passive-to-active
  distillation." "Wherever the model was looking, and at whatever zoom, it should be able to produce its best
  guess about the entire scene, not only the region it had looked at." "Distillation is the sincerest form of
  flattery." Pixel reconstruction is distillation too, from the identity teacher [Yohaï, 2026-10-01]: both map
  partial views to a prediction of g(whole scene); pixels choose g = identity, CanViT a pretrained feature
  extractor. The teacher sets what to preserve and infer; the student only ever sees glimpses.
- The rollout: "It takes glimpses and gradually paints its understanding into the latent space." On a person
  readout: "If it sees the top of a person, it predicts that there's probably a body as well." Memory reset:
  "unless you get a good look at something, the representation is bad."
- The results: "Even our worst policy—a policy worse than random, going from fine detail to coarse ... got better
  performance at lower cost" than the best prior model.
- Understanding improves two ways: "more evidence and more processing" (a briefly flashed image is understood over
  time, as psychophysics has measured).
- Colors: "blue for the glimpses and red for the canvas."

His written framings (`sources/unique-travel-award.md`; the 2025 UNIQUE, FRQNT and NSERC applications; the paper):

- "Human vision is fundamentally active: we shift our gaze multiple times per second, orienting our
  high-resolution fovea toward regions of interest, and integrate information from multiple viewpoints into a
  coherent understanding of our surroundings' spatial and semantic structure."
- "Leading vision foundation models used to model brain function lack the active sensing and top-down recurrent
  feedback of biological vision. This limits their value as models of human vision, which fundamentally relies on
  eye movements."
- Active models "do not scale to large networks and datasets as effectively as their passive counterparts. This has
  limited their potential as AI tools and as models of human vision." "Rich semantic visual priors that can only
  emerge at scale."
- CanViT "pairs retinotopic glimpse processing with spatiotopic memory"; "the canvas can function as a cognitive map
  of the scene, providing a stable spatial reference frame and top-down recurrent feedback to the backbone's visual
  processing." "We disentangle 'learning to see through glimpses' from 'learning where to look'."
- "This rich learning signal allows CanViT to quickly inherit much of its teacher's world knowledge, while developing
  the ability to iteratively refine its understanding of a scene as it ingests glimpses at arbitrary locations and
  zoom levels."
- "A foundation for action-first deep models of human vision" (citing Rothkopf et al. 2023); "the first of many
  Active-Vision Foundation Models"; "a major step toward action-aware neuro-AI research on vision".

## Claims to state with their scope

From an independent review of the draft (Codex, model gpt-6-astra, 2026-10-01; its report is not committed), the
authors' data and the history data:

- 38.5% mIoU "in a single glimpse" is a single low-resolution glimpse of the *full scene*; say so.
- "Perception, not viewpoint selection, was the bottleneck" is the paper's sentence; the evidence is CanViT beating
  the prior models even with F2C.
- Transfer to "any policy" is empirical: across the tested policies, horizons and resolutions.
- The architecture control (Checked facts, the rebuttal) compares two complete designs; it does not isolate the
  canvas, nor measure the teacher's share of absolute performance.
- The canvas resolution result uses a separately trained linear probe per resolution.
- Predicting unobserved regions is not amodal completion; "cognitive map", "workspace" and "belief state" are
  functional analogies, not claims about hippocampus, consciousness or calibrated posteriors.
- The canvas's geometry is given (each glimpse's position and scale), so CanViT does not model how the brain
  computes remapping; it addresses what to keep and how to integrate across views.
- The 45.9% headline is C2F at a 64² canvas; a chart showing it must show that curve.
- Never say CanViT saw "an order of magnitude more" data than previous active models [Yohaï, 2026-10-01: "SHOULD
  BE REMOVED"]: it saw ten times more scenes than ImageNet-1k-trained models, but AdaGlimpse's pipeline saw 37 to
  150 billion glimpses against CanViT's 1 billion (rebuttal tables, section 10). Training compute is "comparable or
  lower" against AdaGlimpse only, and higher than AME's own task training.
- Table corners [measured 2026-10-01, `throwaway/table_corners/run.log` and `examples.json`]: over 1,436 large
  objects of ADE20K validation, after glimpses at both ends the canvas labels on average 65% of the unseen middle as
  the object (53% after one end), and it also labels as the object 46% of the other pixels in the band between the
  glimpses (mean; median 41%). The extrapolation is real and imprecise. The conference-table example was selected
  for few false positives (8%); say so when giving the average.
- The history chart (`sources/sota-history.md`, "Caveats"): the passive frontier after 2019 uses far larger models,
  higher resolutions and up to billions of extra images; show the ImageNet-1k-only and frozen-probe lines beside it.
  Most active models start from passive networks or teachers. LookWhere (NeurIPS 2025; 83.0 ImageNet-1k, 44.6 ADE20K)
  selects patches once from a low-resolution view: not a sequential glimpse model under the paper's definition, but
  close to CanViT-B's numbers and not cited by the paper; expect the question.

## Visual decisions

- [Yohaï, 2026-10-01] Light mode; the project page's identity; no color scheme borrowed from elsewhere.
- [Yohaï, 2026-10-01, on the conference-table example] For an object, show the canvas's probability of its class,
  not PCA colors: the shape reads directly. Probability, not the class logit (the logit gives the floor around the
  table mid-level evidence and blurs the shape); inferno, linear from 0 to 1 (red-on-white and gamma 0.5 were
  rejected). Show A alone, B alone and both: from the near corner alone, the canvas already runs the table back
  toward the far end, along the room's perspective.
- [Yohaï, 2026-10-01] Viewing policies always wear the paper's per-policy colors (`../../js/canvit/policies.js`,
  read at runtime): their labels and their glimpse boxes. Their name is the caption; what each does is said.
- [Yohaï, 2026-10-01] The authors' faces on the title slide, round; QR codes, large, with the GitHub and Hugging
  Face logos, at the end.

## Checked facts

What a session had to look up, with where it lives (`../AGENTS.md`). Paper: `~/code/CanViT-Toward-AVFMs/latex/
CanViT_Toward_AVFMs.tex`, cited by section; its numbers are macros (`data.tex`, copied to
`../../assets/paper/data_macros.json`).

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
- App. H: about 2500 H100-equivalent hours for the whole project; ImageNet-1k fine-tuning under 800 USD, under 15 h
  on a TPU v6e-4.
- Figure exports (SVG, PDF and their JSON data): `~/code/CanViT-Toward-AVFMs/latex/figures/exported/`;
  `../../copy_paper_figures.sh` copies the ones the site shows into `../../assets/paper/`.

The rebuttal (`~/code/CanViT-Toward-AVFMs/rebuttal/DOSSIER.md`; results promised for the camera-ready, not yet in
the paper's macros) [read 2026-10-01]:
- "THE DINOv3-DISTILLED PAIR": CanViT and an AdaGlimpse-derived design, both pretrained 26,875 steps on the same
  precomputed DINOv3 ViT-B features, data, glimpse budget, optimizer and schedule, then probed alike (ADE20K, EG-C2F,
  T = 4): mIoU at t = 3 is .3369 against .1561 (+18.1 points), and the baseline costs 4.1× more FLOPs per glimpse.
  Two complete designs, each at its own paper's input configuration (the baseline sees four 16 px patches), so input
  fidelity travels with the architecture: claim the design, never the canvas alone. Short runs at about a tenth of
  the flagship's glimpse budget: not comparable to 45.9.
- IN1k-only pretraining (`hub.repos.PRETRAINED["in1k"]`, same recipe, 1.28 M instead of 13.2 M scenes; tables in
  `rebuttal/tables_for_responses_2026-07-26_17-01-EDT.md`, section 2, at t = 20): ADE20K, frozen, C2F, 64² canvas,
  43.43 against the flagship's 45.89 (−2.46, n = 10); 1.8 to 3.4 points lower across policies and canvases; frozen
  ImageNet-1k 0.2 to 0.9 points higher; after LP-FT 84.19 against 84.44 (C2F).
- A teacher-free control (RGB reconstruction for both designs) separates them less than the DINOv3 pair does.

The history data [assembled 2026-10-01]: `sources/sota-history.json` (every point read in its paper; `_about`
defines the fields and series) and `sources/sota-history.md` (sources, leads, caveats). Sequential active models top
out at 82.2 on ImageNet-1k (AdaptiveNN, 2025) and 27.6 mIoU on ADE20K (AME, 2023); passive models reach 91.1 and
63.0, and a frozen DINOv2 linear probe alone 49.0 mIoU (2023). STAM (CVPR 2022, 80.78) is missing from the paper's
comparison table.

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
- `python -m canvit_pytorch.viz` records bundles (`Rollout`: policy, `--capture-writes` for write residuals;
  `SmoothPath`).
- URLs (paper, code, Hub, page): `canvit_pytorch.project`; `../../make_qr_codes.py` reads them.

The machine [checked 2026-10-01]: `ADE20K_ROOT=/Users/yberreby/datasets/ADEChallengeData2016` is not set in the
shell, pass it; the Hugging Face cache holds every released checkpoint and probe and DINOv3 ViT-S/B; MPS inference
matches CPU (relative L2 5e-6, identical argmax; `throwaway/bench_episode.log`); the disk is nearly full (under
1 GB free on 2026-10-01).

Experiments behind slides (gitignored `throwaway/`, outputs under `../../data/talk/`): `table_corners/` (two
glimpses at the ends of large ADE20K objects; `run.log` has the averages), `distillation/` (DINOv3 features of a
scene and CanViT's prediction of them after each glimpse; `export.log` has the cosine similarity per glimpse),
`looking_closer/` (small objects found by a zoomed second glimpse).

## Decisions

- [Yohaï, 2026-10-01] Titles and text follow `../AGENTS.md`, "Writing slides": his wording first.
- [Yohaï, 2026-10-01] The "active models have struggled" chart is year against accuracy, with no CanViT result; CanViT
  appears on the results slide.
- [Yohaï, 2026-10-01] Slide count and duration follow from correctness and clarity, not a rule.
- [Yohaï, 2026-10-01] Third-party figures are welcome (fair use in an academic talk), cited on the slide.
- [Yohaï, 2026-10-01] The live model is a bonus, in Backup, titled by the science it shows (any policy on the same
  observer), never by the technology ("in the browser"): the audience cares about the science and what they can do
  with it.
- [Yohaï, 2026-10-01] Control experiments (the rebuttal's architecture and data-scale controls) are not slides: a bar
  chart of short-run numbers under two different conditions was "junk" to this audience. They stay in "Checked
  facts" for questions; the idea they support (the architecture and the learning signal are separate problems) is
  said on the distillation slide.
- [Yohaï, 2026-10-01] His Journal of Vision work on remapping is not part of this talk.
- [Yohaï, 2026-10-01] Unpublished learned-policy results (CanViT-PyTorch-RL): their use is still to be decided.
- [Yohaï, 2026-10-01] The paper's viewing policies get their own slide or slides: what each is, why it was chosen,
  shown animated on one scene.
- [Yohaï, 2026-10-01] Slides that make sense but whose place in the story is not settled go in the deck's Backup
  section, after the last slide, to be reordered later: "it's easier to reorder than to build a slide that's missing
  with genuinely nice animations".
