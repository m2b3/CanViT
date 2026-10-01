# The MAIN 2026 talk

This talk's own guide, kept current as the talk takes shape: the argument, the framings and sentences worth keeping,
the claims to state carefully, and the authors' decisions. `../AGENTS.md` (how slides are written and checked)
applies; `PLAN.md` ranks the open work; `sources/` holds the material the slides draw on.

A talk of about 30 minutes at MAIN 2026 (Montreal AI and Neuroscience; October 29–30, 2026, HEC Montréal; organized by
Centre UNIQUE), on October 30. Presented by Yohaï-Eliel Berreby, possibly with Sabrina Du. The audience mixes
neuroscientists, cognitive scientists and machine-learning researchers.

## The argument

[Yohaï, 2026-10-01: "the logical ordering, proving important propositions, etc, are truly the most important
things".] The talk is a chain of propositions, each shown by evidence; `OUTLINE.md` holds the chain and every slide.
A slide exists to state or prove one step; a slide that does neither goes. The opening order was set by Yohaï: the
vision/neuro-AI tradition, then vision is active, then why world knowledge matters (cognition first, then foundation
models), and a strong introduction to the context, the problems, what exists and the opportunities before CanViT
itself appears.

Yohaï's classic example, the spine of the "why world knowledge" step: looking at two opposite corners of a table, you
need not look at its center to know there is more table there, because you know tables. CanViT does the same; a
DINOv3 run on each glimpse alone has no prediction outside the glimpses, which makes the contrast visible.
For objects, show the canvas's probability of the object's class rather than PCA colors: the shape reads directly.

Efficiency is a link in this chain, not a selling point [Yohaï, 2026-10-01: "they dont [care] about efficiency
unless you make them understand why you need efficiency (because you need scale. why do you need scale? etc)"]:
understanding emerges at scale; reaching scale needs efficient training and inference; active models never had it.
Cost numbers appear only in that role.

## Framings and sentences worth keeping

From the authors (`sources/unique-travel-award.md`, the UNIQUE, FRQNT and NSERC applications of 2025, the paper):

- "Human vision is fundamentally active: we shift our gaze multiple times per second, orienting our
  high-resolution fovea toward regions of interest, and integrate information from multiple viewpoints into a
  coherent understanding of our surroundings' spatial and semantic structure."
- "Leading vision foundation models used to model brain function lack the active sensing and top-down recurrent
  feedback of biological vision. This limits their value as models of human vision, which fundamentally relies on
  eye movements."
- "Image information is unevenly distributed within and across scenes"; passive models "process images at uniform
  resolution, in a single-shot manner".
- Active models "do not scale to large networks and datasets as effectively as their passive counterparts. This has
  limited their potential as AI tools and as models of human vision."
- "Rich semantic visual priors that can only emerge at scale."
- CanViT "pairs retinotopic glimpse processing with spatiotopic memory"; "decouples sensory processing from working
  memory"; "the canvas can function as a cognitive map of the scene, providing a stable spatial reference frame and
  top-down recurrent feedback to the backbone's visual processing."
- "We disentangle 'learning to see through glimpses' from 'learning where to look'."
- "This rich learning signal allows CanViT to quickly inherit much of its teacher's world knowledge, while developing
  the ability to iteratively refine its understanding of a scene as it ingests glimpses at arbitrary locations and
  zoom levels."
- "No policy can make up for a poor observer" (paper §1, with its condition: given perfect instantaneous vision and
  memory, naive and strategic policies reach the same accuracy after enough glimpses of a static scene).
- "A foundation for action-first deep models of human vision" (applications, citing Rothkopf et al. 2023).
- "The first of many Active-Vision Foundation Models"; "a major step toward action-aware neuro-AI research on vision".

From Yohaï's spoken explanation (September 2026): there is "a world of difference between seeing something blurry
and seeing nothing at all" (CanViT sees nothing outside the glimpse); the question the canvas answers is "at any
point in time and for any position: what does the model think is there?"; seeing two corners of a table, a good
observer infers the table between them instead of pasting two views; the memory is "dumb, but not too dumb": cheap
to interact with, rich enough to reconstruct from world knowledge; prior dense models had a cheap encoder and a
decoder costing far more at every step.

Yohaï's framing of distillation: pixel reconstruction is distillation from the identity teacher. Both map partial
observations to a prediction of g(whole scene); pixels choose g = identity, CanViT a pretrained feature extractor.
The teacher tells the observer what information to preserve and infer; the student only ever sees glimpses.

## Claims to state with their scope

From an independent review of the draft (Codex, model gpt-6-astra, 2026-10-01; its report is not committed):

- 38.5% mIoU "in a single glimpse" is a single low-resolution glimpse of the *full scene*; say so.
- "Perception, not viewpoint selection, was the bottleneck" is the paper's sentence; the comparison behind it shows
  CanViT beating prior models even with F2C, not a single isolated cause.
- Transfer to "any policy" is empirical: across the tested policies, horizons and resolutions.
- The architecture control does not fully quantify the teacher's contribution to absolute performance (the
  rebuttal's own follow-up says so).
- The canvas resolution result uses a separately trained linear probe per resolution.
- Predicting unobserved regions is not amodal completion; "cognitive map", "workspace" and "belief state" are
  functional analogies, not claims about hippocampus, consciousness or calibrated posteriors.
- The canvas's geometry is given (each glimpse's position and scale), so CanViT does not model how the brain
  computes remapping; it addresses what to keep and how to integrate across views.
- The 45.9% headline is C2F at a 64² canvas; a chart showing it must show that curve.

## Decisions

- [Yohaï, 2026-10-01] Light mode; the project page's identity; no color scheme borrowed from elsewhere.
- [Yohaï, 2026-10-01] Titles follow `../AGENTS.md`, "Writing slides".
- [Yohaï, 2026-10-01] The "active models have struggled" slide shows no CanViT result, and is a year-by-performance
  chart, not the accuracy–compute frontier.
- [Yohaï, 2026-10-01] Slide count and duration follow from correctness and clarity, not a rule.
- [Yohaï, 2026-10-01] Third-party figures are welcome (fair use in an academic talk), cited on the slide.
- [Yohaï, 2026-10-01] His Journal of Vision work on remapping is not part of this talk.
- [Yohaï, 2026-10-01] Unpublished learned-policy results (CanViT-PyTorch-RL): their use is still to be decided.
