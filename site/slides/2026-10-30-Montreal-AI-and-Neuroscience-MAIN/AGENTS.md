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
models had neither. Cost numbers appear only in that role. The accuracy-against-cost chart (`#frontier`) makes that
case before it shows a cost axis [Yohaï, 2026-10-01: "you WILL have to properly make the case of why anyone would
[care] about efficiency (it's about money, making experiments cheap, etc)"]: compute is money and time, so a
cheaper model makes every experiment cheaper (more scenes, more glimpses, more policies tried, policy learning by
trial and error) and puts it within reach of a lab without a GPU cluster. Every cost number said comes from the
paper, its exports or a measurement noted in "Checked facts". For this audience the cost that matters most is
training: what it takes to train or adapt a model for their own experiment; inference cost matters too, second
[Yohaï, 2026-10-01: "for cost, for scientists, remember that TRAINING cost is main concern; inference can also be
relevant tho"].

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

## Key ideas

The words the audience should leave with, each with the slide that carries it [CC synthesis from the paper, the
transcript and Yohaï's requests, 2026-10-01; to be confirmed by the authors]:

- **Active vision**: glimpses at chosen viewpoints (position and zoom), one after another (`#human-vision`,
  `#glimpses`).
- **Observer and policy**: "No policy can make up for a poor observer"; "a general-purpose observer lets you use any
  policy" (`#active-vision-model`, `#policies`).
- **Trans-saccadic integration**: the canvas accumulates what each glimpse brings into one understanding of the scene,
  the machine counterpart of integrating information across fixations (the paper cites Melcher 2001 for integration
  across time in visual working memory). Established terms from the audience's own fields that name what the model
  does are the wording to aim for [Yohaï, 2026-10-01: "the idea of trans-saccadic integration is great, see, this is
  more the kind of wording and analogy we want to go toward"]; `sources/concepts.md` collects them.
- **The canvas**: a scene-wide memory in scene coordinates (spatiotopic) beside a backbone that sees glimpses
  (retinotopic), bound by SR-RoPE; memory "dumb, but not too dumb", cheap to read and write (`#architecture`,
  `#canvas-attention`). It holds; it never acts.
- **Passive-to-active distillation**: wherever it looks, its best guess about the whole scene, in a passive teacher's
  feature space; pixels would be the identity teacher (`#distillation`).
- **Inferring what was not seen**: completing the scene from what was seen and from knowledge of the world (the
  paper: "extrapolates to unobserved regions"); the table between two glimpses (`#table`, `#extrapolation`). The
  model guesses, infers, completes; it never "sees" what it did not sample [Yohaï, 2026-10-01: "it's not SEEING
  it's guessing, inferring, completing"]. The audience's words for it include amodal completion and boundary
  extension (`sources/cognition.md`; more such links, each read in its primary source, in `sources/concepts.md`)
  [Yohaï, 2026-10-01: "'amodal completion' can be a great keyword no? think of great keywords and concepts and things
  to link to that I MIGHT NOT HAVE IN MIND or EVEN KNOW"].
- **Metacognition**: the model's own uncertainty about what is where, which a policy (EG-C2F) uses to choose where to
  look (`#policies`) [Yohaï, 2026-10-01: "a keyword i really want to see emphasized on the policy stuff, particularly
  EG-C2F, is the idea of metacognition"].
- **Order matters**: C2F against F2C, same views; what was seen changes how the next view is processed (top-down
  feedback) (`#policies`) [Yohaï, 2026-10-01: "that IS a good point"].
- **Foundation model, cheap to use**: trained once, adapted with a linear layer; what it costs to train and to run
  (`#frontier`, `#quickstart`).

## Questions to anticipate

Questions an informed member of this audience would ask, in the words they would use: from the reviews and the
meta-review, from Yohaï's own objections, or from a literature the audience knows (name it), and that this audience
cares about: benchmark minutiae a NeurIPS reviewer raised (COCO detection, classification efficiency against
AdaptiveNN) are not [Yohaï, 2026-10-01: "no one [cares]"]. A question nobody competent would ask in that form
is a strawman and does not belong here [Yohaï, 2026-10-01, on "Is the canvas a brain area?": "WHAT ARE
THESE [...] QUESTIONS"]. Each answer points to its facts (below) [CC,
2026-10-01; the authors decide what to prepare as backup slides]:

- "Isn't it just DINOv3?" The architecture control (same teacher and data, an AdaGlimpse-derived design: 15.6 against
  33.7 mIoU at t = 3, a quarter of the cost per glimpse); the frozen teacher probed alike reaches 47.2, CanViT-B 45.9,
  and CanViT-B beats its input- or FLOP-matched teacher at low budgets. The teacher's absolute share is not isolated;
  say so.
- "Is learning from a model that sees everything cheating?" Pixels are the identity teacher; the student only ever
  gets glimpses.
- "Change blindness says people keep little detail across saccades (the change-blindness literature; Rensink,
  O'Regan and colleagues: to be read before answering). Why a dense, scene-wide memory?" To prepare with the
  authors: the canvas holds features, a best guess rather than a picture, and it is dense so that any position can be
  read out; what it keeps across views is measurable, and comparing that with trans-saccadic memory is the kind of
  question the closing slide proposes. Not yet read: do not answer from memory.
- "Where does the viewpoint come from? The brain has to compute it (corollary discharge, remapping)." CanViT is given
  each glimpse's position and scale; it does not model how they are computed.
- "LookWhere?" Select-once, not sequential; its numbers (Claims, below).
- "AME already chose glimpses by uncertainty." Yes: by the entropy of its decoder's attention maps (2023, the timeline
  slide). EG-C2F uses the entropy of the class probabilities read out from the canvas, with no policy training; the
  paper's point is that what the canvas holds can guide where to look.
- "Video, moving objects, forgetting?" Static scenes only; the rebuttal's answer (gating, remapping-inspired updates,
  a 3D canvas) is future work.
- "Learned policies? Reinforcement learning?" Left to future work; the VPE token is there for it; unpublished
  learned-policy results are the authors' call (Decisions).
- "Can my lab use it? On what hardware?" Released checkpoints, a linear probe or LP-FT; inference on a laptop
  (`#quickstart`, run on an M4 Pro on 2026-10-01).
- "Is it ImageNet-21k doing the work?" The ImageNet-1k-only control: 43.4 against 45.9 mIoU, still above every prior
  active model.

## Claims to state with their scope

From an independent review of the draft (Codex, model gpt-6-astra, 2026-10-01; its report is not committed), the
authors' data and the history data:

- 38.5% mIoU "in a single glimpse" is a single low-resolution glimpse of the *full scene*; say so.
- "Perception, not viewpoint selection, was the bottleneck" is the paper's sentence; the evidence is CanViT beating
  the prior models even with F2C.
- "Any policy" means the tested policies, horizons and resolutions.
- The architecture control (Checked facts, the rebuttal) compares two complete designs; it does not isolate the
  canvas, nor measure the teacher's share of absolute performance.
- The canvas resolution result uses a separately trained linear probe per resolution.
- The canvas's geometry is given (each glimpse's position and scale): CanViT addresses what to keep and how to
  integrate across views, not how the brain computes remapping.
- The 45.9% headline is C2F at a 64² canvas; a chart showing it must show that curve.
- Never say CanViT saw "an order of magnitude more" data than previous active models [Yohaï, 2026-10-01: "SHOULD
  BE REMOVED"]: it saw ten times more scenes than ImageNet-1k-trained models, but AdaGlimpse's pipeline saw 38 to
  150 billion glimpses against CanViT's 1 billion (the posted responses; Checked facts). Training compute is "comparable or
  lower" against AdaGlimpse only, and higher than AME's own task training.
- Table corners [measured 2026-10-01, `throwaway/table_corners/run.log` and `examples.json`]: over 1,436 large
  objects of ADE20K validation, after glimpses at both ends the canvas labels on average 65% of the unseen middle as
  the object (53% after one end), and it also labels as the object 46% of the other pixels in the band between the
  glimpses (mean; median 41%). The extrapolation is real and imprecise. The conference-table example was selected
  for few false positives (8%); say so when giving the average.
- Compare like with like: never against far larger models [Yohaï, 2026-10-01, on a gap drawn between a frozen 7B
  model and the best active model: "comparing to nonsensically large models is indeed stupid and counterproductive"].
  The history chart's passive lines are restricted to models of the active models' size (base size, ViT-B or
  smaller backbones), each point with its parameter count and test resolution known (`sources/sota-history.md`); the
  all-sizes frontier (up to billions of parameters, 475–800 px, billions of extra images) is not drawn. The sizes of
  the active incumbents are recorded as precisely as the passive ones, every network they run at test time
  (`params_detail` in `sources/sota-history.json`) [Yohaï, 2026-10-01: "keep track of the SIZES of the active-vision
  incumbents precisely as well"].
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
- [Yohaï, 2026-10-01] ImageNet-1k and ADE20K one after the other, never side by side from the start ("don't try to
  have in1k and ade20k side by side together from get go, one after the other"), and the talk explains the key
  differences between classification and segmentation and why segmentation is harder before showing a number on
  either. The paper's words: "non-spatial, global prediction tasks like object classification or spatially-grounded,
  dense tasks like semantic segmentation" (§3); dense prediction "is unsupported by most existing active vision
  models; the few exceptions lag dramatically behind passive models" (§1); dense outputs "require explicit
  architectural handling, as the active vision setting breaks the direct, connectivity-based mapping between input
  and output feature maps" (§3).
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

The posted responses and the decision (`rebuttal/response_drafts_2026-07-26/response_*.md`,
`follow_up_drafts_2026_07_31/response_DjZB.md`, `ac_comment_2026-08-03/response_AC.md`,
`../camera_ready/decision_2026-09-24.md`, all under `~/code/CanViT-Toward-AVFMs/`) [read 2026-10-01]; what reviewers
received, so the talk may say it:
- The reviewers' and the AC's central concerns: attribution (architecture against the DINOv3 teacher against the
  pretraining scale) and one-time offline cost (the 19 TiB feature cache, ImageNet-21k pretraining) against
  deployment cost. Accepted as a poster on 2026-09-24; DjZB raised to Accept; "the absolute contribution of DINOv3
  is not fully isolated" stays a stated limitation, as do static scenes.
- Training scale, as posted: CanViT-B saw 1 B glimpses; AdaptiveNN about 1.9 B (estimated from its released
  parameters and code, 300 ImageNet-1k epochs); AdaGlimpse 38 to 150 B (600 epochs); AME about 16 M in its
  active-vision adaptation, on top of pretrained MAE or SETR backbones. CanViT, AdaptiveNN and AdaGlimpse start from
  random initialization.
- Training compute, as posted: 166 H100-hours of pretraining plus 8 H100-equivalent hours of teacher features (174);
  computing the teacher online instead of caching would make it about 246 and remove the 19 TiB. AdaGlimpse,
  estimated from its repository's "around 1 week on 4x A100 GPUs": 210–410 H100-equivalent hours. ImageNet-1k
  fine-tuning: under 800 USD (App. H).
- The architecture control, as posted: CanViT-B 96.5 M parameters against an AdaGlimpse baseline of 112.3 M, both
  from random initialization on the same DINOv3 harness (ImageNet-1k scenes, about 110 M glimpses); 22 against 212
  TPU v6e chip-hours to pretrain; per glimpse 15.8 against 70.2 GFLOPs, so four CanViT glimpses cost less than the
  baseline's first. COCO detection on these checkpoints (CenterNet head, about 35 epochs, preliminary): 19.02 against
  8.12 AP at t = 4; the flagship reaches 32.42 AP at t = 20 (C2F).
- Efficiency claims are scoped to dense prediction: "We do not expect CanViT to lead on accuracy-efficiency in
  classification". On ImageNet-1k (Tables R5–R6, recomputed from released code): CanViT-B 95.2 M active parameters,
  15.48 GFLOPs per glimpse; AdaptiveNN (DeiT-S) 2.15 GFLOPs for its first glimpse, 83.2 M parameters in all;
  GFNet (EfficientNet-B3) 0.86 GFLOPs, 39.1 M in all.
- Passive efficient segmenters for context (Table R2 of DjZB, single-scale, their own settings): EfficientViT-L2 51 M,
  90 GFLOPs, 50.7 mIoU; FocalNet-B 126 M, 2384 GFLOPs, 50.5; CanViT-B 15.8 GFLOPs for 38.5 at t = 0, 464.1 GFLOPs
  for 45.89 at t = 20 (64² canvas).
- Training FLOPs, each system's own training only (`rebuttal/training_flops.py`, run 2026-10-01 into the session
  scratchpad; 1 MAC = 2 FLOPs, backward = 2 × forward, measured 1.996; inherited pretrained weights and teachers
  are not counted on any side, so DINOv3's own pretraining is not counted for CanViT either): CanViT-B 52.5 EFLOPs of
  pretraining plus 2.84 of teacher features (55.3), from random initialization; AdaGlimpse 62.9 to 144.6 (its
  600-epoch reconstruction pretraining by its code or by its paper, plus 100 epochs of classification), from random
  initialization with a DeiT-III-21k teacher; AME 1.4 to 1.8 for its segmentation training, on top of MAE-L or
  SETR-ADE20K weights whose training is not counted. AdaptiveNN discloses no training cost; its ImageNet-1k run is
  from scratch, 300 epochs, four 112 px fixations plus a glance and a regularization pass per image per step
  (`rebuttal/adaptivenn_training_facts.md`). The dossier's own reading ("Training-FLOP accounting vs baselines"):
  own-training compute is comparable between CanViT-B and AdaGlimpse's ImageNet-1k pipeline; the defensible
  differences are wall time as disclosed (AdaGlimpse's README about 672 A100-hours; CanViT 166 H100-hours plus 8),
  AME's inherited weights, and that CanViT-B is one task- and policy-agnostic model where the others are trained per
  task. For a scientist the training cost that follows is adapting it: a linear probe on frozen features, or LP-FT
  (ImageNet-1k fine-tuning under 800 USD, under 15 h on a TPU v6e-4); never claim CanViT-B was cheaper to pretrain
  than every prior model.
- The VPE token helps at long horizons and fine canvases (+1.28 mIoU at t = 20, 64² canvas); EG-C2F's viewpoint
  selection costs about 1–5 MFLOP, under 0.03% of a glimpse.

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

Memory reset [measured 2026-10-01, `throwaway/memory_reset/`, summary in `../../data/talk/memory/overall-k1.json`,
read]: every "thing" object covering 1–15% of an ADE20K validation scene (2,781 objects); one glimpse on the object
(its box about half the glimpse), then three glimpses as far as possible, none containing the object's class; mean
p(class) over the object's pixels after the last glimpse: 0.462 ± 0.006 with the canvas carried, 0.031 reset before
each glimpse, 0.038 when the object's own glimpse is skipped. Among objects recognized after their glimpse (p ≥ 0.5,
1,579): 0.683 against 0.038.

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
