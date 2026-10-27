# The MAIN 2026 talk: positions, open questions, the deck, open work

The talk's strategy, which changes as the authors and agents iterate. How to work is in `AGENTS.md`; facts are in
`sources/checked-facts.md` and `sources/literature.md`.

Markers:
- **(Y)**: a position Yohaï stated, in clean words, with its date. Where he gave a formulation for the talk, it is
  reproduced as given.
- **(A)**: an agent's inference or proposal. It is not the authors' position until they confirm it.
- Nothing in the deck is approved unless the authors discussed it explicitly. A position is re-confirmed before a
  slide depends on it, and again when a related position changes.

## Setting

"Toward Active-Vision Foundation Models", MAIN 2026 (Montreal AI and Neuroscience), 2026-10-30: 25 minutes and 5 of
questions. An AI talk in a neuro-AI session; it is not about videogames. Part I by Sabrina Du, Part II and the
outro by Yohaï (Y, 2026-10-25: Sabrina likely presents Part I only, with no outro). The
paper (arXiv:2603.22570) is accepted at NeurIPS 2026; its camera-ready is due 2026-11-06.

## Positions

Stated 2026-10-24:
- (Y) Practicality, a principle of the project, runs through the talk: what works, not beautiful ideas that do not
  work. This will not change.
- (Y) "Before knowing where to look, we should know how to see." Prior active models did not yet see well, so it
  makes sense to fix the observer before trying to give them a great policy.
- (Y) The distinction between observer and policy is one of the work's most important conceptual moves and must be
  made explicit. Before it, the audience must understand what active vision is, what it has been about, and why they
  might care.
- (Y) "The observer is the bottleneck, not the policy" goes in, and the nuances of the argument must be preserved.
- (Y) Training observer and policy together, done naively, works badly, for reasons that stay separate (wording
  corrected 2026-10-24 late: co-adaptation itself is not what works badly):
  - Specificity. Observer and policy co-adapt. That can in principle give peak performance with that policy; the
    cost is an observer specific to its policy, and so inflexible (detailed below).
  - Instability. It comes from policy gradients, that is from training with reinforcement learning: gradient
    variance, instability, possibly collapse. Reinforcement learning should be avoided whenever possible.
  - Value. Why pay for policy-specific pretraining when learned policies are probably going to be rather basic
    (AdaGlimpse's Fig. 8, possibly shown whole on a slide). Reconsidered 2026-10-25: may not be relevant.
- (Y) Joint training remains possible later; a frozen observer with a policy on top is what this work did; a
  policy-agnostic observer can be an initialization for joint training ("qui peut le plus peut le moins").
- (Y) Stability and signal are fundamental. Transferring knowledge about the world into the model is what training
  does in general: every optimizer step teaches something.
- (Y) The work has two major contributions, both to be understood and conveyed: the architecture (what it solves
  and why) and the pretraining (where policy agnosticism and the learning signal matter).
- (Y) On cost, a principle of the project: one should not waste enormous resources on the wrong problem. The point
  is not that CanViT needs DINOv3: models like DINOv3 give task-agnostic representations, cost a great deal to train,
  are at the heart of much current research and have been applied in neuro-AI (Raugel et al.), and it is smart to
  use them. To discuss further.
- (Y) CanViT at 21 glimpses beside prior models at 8 is fair: they stopped at 8 because of diminishing returns; had
  they performed well, they would have shown it.
- (Y) A gain of 0.4 mIoU over EG-C2F is a big difference against a heuristic policy that is already excellent.
- (Y) CanViT's glimpses are uniform-resolution crops, not foveated; eccentricity-dependent acuity is part of what
  CanViT 2 covers. Reference frames are important and are discussed, as a functional argument: CanViT's backbone
  works in glimpse (retinotopic) coordinates and its canvas in scene (spatiotopic) coordinates. What is set aside for
  the time being is the neuroscientific plausibility of a spatiotopic representation (for instance against the
  predominantly eye-centered reference frame Xiao et al. 2024 found in ventral neurons). Clarified 2026-10-24 late;
  it replaces an agent's broader "avoid reference frames".

Stated 2026-10-24, reviewing the deck:
- (Y) The distinction between object classification and semantic segmentation, image-level (global) against
  pixel-level (dense) tasks, is important to get clear.
- (Y) The title slide names the speakers only, Yohaï and Sabrina. The paper gets its own slide, where the audience
  first hears "CanViT" and "Canvas Vision Transformer": the whole team, a QR code, and NeurIPS 2026. NeurIPS may also
  appear at the very end, with a screenshot and "see you in Paris!".
- (Y) Integrating visual information across viewpoints into a coherent understanding is part of the problem the architecture solves:
  part of understanding the key components of an active-vision model. It comes before the architecture.
- (Y) Where and how each result appears is still open; nothing is repeated. Peak performance and efficiency can be
  separated: efficiency at inference is a bonus, efficiency at training is what matters scientifically.
- (Y) The viewing policies are introduced before any result that uses them: no graph that means F2C or C2F before
  then. Pretraining policies and evaluation policies are introduced separately, on slides titled "Pretraining
  policies" and "Evaluation policies".
- (Y) A dedicated slide for the learned policy, as a closing touch: the talk argues against RL, then uses it at the
  end, efficiently. A possible title: "Learning where to look".
- (Y) "How to see" apart from "where to look" is an important distinction: policy-agnostic pretraining against policy
  learning.
- (Y) Pretraining's practical details: precomputing the teacher's features can be a divisive point; short training
  rollouts (truncated backpropagation through time) may confuse or be too technical, or may belong as part of the
  recipe, and the model generalizing to long sequences is a nice result; the paper holds the technical details.
- (Y) The learning signal is an important idea, but `#learning-signals` conveyed it badly and fit nowhere in the
  story; how and where to convey it is open (see information density, below).
- (Y) Showing how few glimpses learning where to look took, beside the pretraining's, as a reveal; RTX 4090 time can go
  with it. Built on `#learned-policy` 2026-10-24.
- (Y) `#pretraining`'s hours chart is not wrong: doing what DINOv3 did would take a long time. Its framing against the
  cost principle is a minor question for later.
- (Y) Whether the model is presented with custom animations or with the paper's figures, which may be better, is open.

Stated 2026-10-25 evening, reviewing the deck's structure:
- (Y) Policy-agnostic pretraining is an important point to get across; which arguments for it belong in the talk
  needs careful thought ("learned policies tend to be simple" may not be relevant).
- (Y) The parallel with LLMs (pretraining without RL, post-training with RL) may get a dedicated slide, after the
  learned-policy results.
- (Y) The results have two distinct angles, kept apart: the differences between viewing policies, and peak
  performance against existing models. The second, CanViT's peak ADE20K and ImageNet-1k results against prior
  active models, is missing from the deck; showing the narrowing gap with passive models, without a whole
  dedicated slide, possibly where the new active-vision state of the art is said, would be valuable. FLOP efficiency
  is not dealt with either; to think about.
- (Y) Canvas Attention stays. SR-RoPE must be explained or at least mentioned: it is how the model makes sense of
  space, fundamental to the work.
- (Y) Policy agnosticism makes replaying recorded scanpaths possible: worth getting across, even in a few seconds,
  plainly, without over-claiming or anchoring on papers and datasets the authors do not know well. Not settled.
  Uniform-acuity glimpses are not a limitation of policy agnosticism; no caveat about eye-centered reference frames.
- (Y) On the learned policy, "cheap" is always qualified: cheap to train, and it works well.
- (Y) `#compare`'s "Rich world knowledge" ✗ for prior active models can be softened to an intermediate mark
  (AdaGlimpse starts from MAE weights); not critical.

Stated 2026-10-25 late, the pretraining section's logic (supersedes earlier agent framings of it):
- (Y) The question: with the architecture in hand, how to pretrain it into a foundation model. It is first a
  representation learning problem, not an RL problem; RL brings noisy, high-variance gradients, a sparse learning
  signal, and observer and policy coupled.
- (Y) Problem 1: how to make the model robust to any policy. Answer: policy-agnostic pretraining. Behind it: task
  agnosticism, since where people look depends on the goal (Yarbus, a callback); a policy-specific observer co-adapts
  with its policy (even with a stop-gradient), with no guarantee under others; and, independently, joint training
  through policy gradients is unstable. Whether instability is said is open; the nuance is kept.
- (Y) Problem 2: how to give the model an understanding of the visual world, rich world knowledge, without an
  enormous amount of computation, costly and ecologically wasteful. Answer: dense latent distillation from DINOv3,
  which supplies that knowledge for a scene seen whole; what CanViT must learn is building it from glimpses.
- (Y) The label-free angle stays, at both levels: DINOv3 learned without labels, and CanViT learns from its features without
  labels (a callback to Part I's supervised against self-supervised).
- (Y) Payoffs come later: cheap policy training on the frozen observer (the learned-policy results); replaying
  recorded scanpaths (likely the ending).

Stated 2026-10-25 late, on the open threads:
- (Y) The pretraining section, so that people understand: the two problems to solve and the two answers; then answer
  1 in detail, policy-agnostic pretraining; then answer 2, dense latent distillation (where the learning-signal argument
  belongs); then the two together, the policy-agnostic dense latent distillation slide.
- (Y) The canvas's own moment may already be the SR-RoPE slide; not sure.
- (Y) The LLM parallel: possibly a quick animation on "Learning where to look", after the result. Built, then set aside the same day (Y: too verbose); said aloud while the policy-learning budgets are shown.
- (Y) "Extrapolation to unobserved regions" stays in Backup for now (the title slide's teaser shows extrapolation; it
  may be said aloud); a direct, streamlined callback to the table slide would be good: quick, not several rows and
  columns.
- (Y) No softening of "Rich world knowledge" for prior models for now. CanViT's red bars: low priority. AME's two
  names: does not matter. Segmentation palettes: unify if it can be done meaningfully.

Stated 2026-10-27, reviewing the pass of that day:
- (Y) "Before knowing where to look, we should know how to see" is the philosophy of policy-agnostic pretraining as a
  whole; it is not a lead-in to the learned policy.
- (Y) Part II answers two problems, the architecture's and the pretraining's; neither is "the" problem.
- (Y) The paper's Figure 1 goes at the start of the CanViT part, before the architecture, without a title (a title
  starting with "A" read as the figure's panel A). It builds up from its own code (the scene, then each glimpse and its
  column); nothing is drawn over a render of the figure.
- (Y) Signposts stay a few words: the reasoning they mark is said, not written on them.
- (Y) A review thinks through the argument; rearranging slides is not a review.

Stated 2026-10-24, in a discussion between the speakers:
- (Y) Say "observer", never "vision encoder": an encoder covers instantaneous vision only and leaves out memory.
- (Y) Use the terms policy-specific and policy-agnostic pretraining. Training observer and policy at the same time
  (co-training) differs from training them jointly through shared gradients; a stop-gradient allows the first without
  the second. They raise different problems:
  - Policy-specific pretraining, even with a stop-gradient, makes observer and policy co-adapt: what the observer
    sees in pretraining is shaped by the policy, and the policy reads features of what the observer saw. The observer
    may then work well with its own policy and fail or underperform badly with others. It is inflexible: a new policy
    (scanpaths recorded from humans or monkeys, or a better policy found later) comes with no guarantee, and getting
    a new policy can take unnecessarily long when its training is coupled to the model's.
  - Instability and gradient variance come from policy gradients; with a stop-gradient, the observer does not
    receive them.
- (Y) Replaying recorded scanpaths through the observer is a strong argument for policy-agnostic pretraining: the
  model is built for any sequence of fixations, so human or monkey scanpaths can be tried with it meaningfully.
- (Y) Task-dependent scanpaths motivate policy agnosticism (Sabrina's suggestion): a foundation model should be
  task-agnostic, and Yarbus showed that viewing patterns depend on the viewer's goal, not only on the image; so the
  model should handle any sequence of fixations. Put Yarbus on screen; it is taught in introductory neuroscience and
  psychology, so much of the audience will know it.
- (Y) A flow for the talk, proposed in the discussion: what a foundation model is (much training, task-agnostic, a
  general-purpose understanding); passive vision has them, active vision does not; what active vision needs that
  passive vision does not (memory; retinotopic against spatiotopic coordinates); the architecture provides those
  components, as the paper lays out; then how to make it a foundation model: task agnosticism argues against pretraining
  the observer with a policy learned by RL, since scanpaths depend on the goal (Yarbus), which motivates policy agnosticism. Beyond that argument,
  sub-arguments: replaying scanpaths and not overfitting to a policy; and complexity removed, since pretraining comes
  first and policy training afterwards can be far cheaper. Show how quick post-hoc policy training is: the learned
  policy took roughly 600 thousand to a million glimpses, pretraining about a billion (exact count: the RL repository's
  results notes, private until it goes on a slide).
- (Y) The nuances and counterarguments stay recorded even when the talk does not say them all.
- (Y) The parallel with LLMs belongs in the talk: pretraining without RL, post-training with RL.
- (Y) The deeper point is information density and what the model learns, and policy agnosticism is a practical answer
  to it. RL serves to learn where to look, but knowing whether a look was good says little about the world; predicting
  what one will see teaches about the world. Active-vision papers mix the two objectives, one for the policy and one
  for the observer. RL is information-scarce compared with supervised or self-supervised learning; possibly an
  information-funnel graphic, possibly with studies of how much information RL post-training gives LLMs. RL is a last
  resort, not useless. Provocation, if any, is spoken; not necessarily a quote on screen.
- (Y) Be prepared for another meaning of "active-vision foundation model": a model given a task in language that
  selects where to look, as I-Perceive does (`sources/literature.md`); a possible future direction, from a different
  tradition (camera poses in 3D scenes).

Stated 2026-10-24 evening, in two discussions between the speakers:
- (Y) Vision foundation models must be introduced before Active-Vision Foundation Models; the talk introduces DINOv3
  specifically, since it returns later (Raugel et al.; CanViT's teacher). The point: image in, semantic features out,
  directly meaningful without adapting the model; that abstraction boundary is what makes its output usable as a
  training signal. No need to contrast with MAE or fine-tuning.
- (Y) PCA is signposted on the DINOv3 slide as how the talk shows features throughout.
- (Y) The key distinction on the models-of-vision slide is supervised against self-supervised: learning purely from
  visual data, without imposing a human understanding of the world through labels (CLIP is flexible but supervised by
  human pairings, so "flexibility" is not the argument). "General", not "flexible".
- (Y) Raugel et al.'s Fig. 5E, framed as a requirement, not a bonus: modeling higher regions appears to require more
  training (the half time grows with distance from V1). The question of how this fits the debate that bigger, better
  models are not more brain-like is open and worth reading on.
- (Y) What's missing from these models, said in one breath: gaze shifts and a partial, high-acuity view (the fovea);
  this ties back to CanViT 2's eccentricity-dependent acuity.
- (Y) The history of active vision in two acts: a few deep active computer vision models tracing back to RAM, then,
  more recently, neuro-AI active models; "drawing inspiration from this process".
- (Y) Titles might be removed from many slides in the end; to decide later.

Stated 2026-10-23 (Montreal time; from 2026-10-24 01:00 UTC on, this session's earliest messages):
- (Y) Ideas the talk needs, to think about critically: active vision; foundation models; why neuro-AI cares about
  foundation models; how to train them without breaking the bank; supervised against self-supervised learning against
  RL; practical issues such as instability. A point raised is not necessarily a slide; every important point is kept
  track of. Yes/no comparison tables are welcome for storytelling.
- (Y) The audience does not necessarily know that vision is active better than we do; going quickly over the logic of
  deep networks as models of the visual system is worth it, to show the logic of the motivation.
- (Y) The Yarbus figure likely will not speak as it is.

- (Y) "Action" means action selection: where and at what zoom to look next. Nothing broader is claimed.
- (Y) An AI talk about getting good foundation models for action-aware research: what belongs in the policy and what
  in the observer, policy-agnostic pretraining, avoiding RL when you can. The audience should leave with a recipe
  they can use.
- (Y) The learned-policy results (unpublished) are included.
- (Y) CanViT 2 appears in broad strokes only (eccentricity-dependent acuity, dynamic scenes); no details anywhere in
  this repository.
- (Y) Free-viewing neural datasets: a quick slide with figures only, which may be dropped.
- (Y) Lamarck Labs joins the funders on the last slide (the paper acknowledges its compute support).
- (Y) Timings are guesses until rehearsal, and agents' estimates are likely to be far off.

Stated 2026-10-01, for an earlier design of the talk; to re-confirm before relying on them:
- (Y) The logical order and the propositions each step proves matter most. How CanViT was built serves what it does;
  overall architecture and policy-agnostic pretraining first, Canvas Attention in Backup. (The order itself lives in
  one place, "The deck as it stands"; it can change at any time.)
- (Y) Efficiency only as a link (scale needs cheap training); for scientists, training and adaptation cost first,
  inference second.
- (Y) Never say CanViT saw "an order of magnitude more" data than previous active models (the paper's abstract, the
  README and the project page still say it; remove it for the camera-ready).
- (Y) The authors' faces
  round (on the title then; 2026-10-24: the speakers on the title, the whole team on `#paper`); QR codes large at the
  end; the live demo in Backup (removed 2026-10-24); control experiments not as slides; no
  remapping work from the Journal of Vision; unsettled slides go to Backup.

Reviewed by Yohaï 2026-10-24 and dropped from the list above: "compare like with like, never against far larger
models" (he does not recall saying it); "under 800 USD" for the ImageNet-1k fine-tuning (true, in
`sources/checked-facts.md`, but the talk will not mention it); a list of keywords to highlight (trans-saccadic
integration, boundary extension, metacognition, order matters, pixels as the identity teacher: mostly out of place or
not worth highlighting); a fixed opening order.

## Agent inferences and proposals

- (A) The pass of 2026-10-27 over the whole deck, after a full read of the camera-ready: its reasoning, what it built
  and what it leaves to the authors are in "The pass of 2026-10-27" below. It absorbed the proposals of 2026-10-26
  that followed from the camera-ready (`PROPOSALS.md`, removed).
- (A) A summary of the talk, which Yohaï judged a fair start (2026-10-24): active vision has long promised vision
  that is more efficient and closer to biology but lagged far behind passive models; separating seeing from choosing
  where to look, and building a general observer the way foundation models are built, is a way forward; CanViT is a
  first step, which makes active-vision questions practical to study, including for neuro-AI.
- (A) How the two contributions fit, from paper §4–5 (Yohaï skimmed this account on 2026-10-24 and found it much
  better than before; not checked line by line): glimpses from arbitrary viewpoints cannot be carried forward as a
  grid, so CanViT keeps memory in a canvas fixed in scene coordinates; that makes dense outputs a token-wise readout
  and lets gradients flow across time. The canvas is large, so it has no self-attention or MLP and Canvas Attention
  projects only on the glimpse side. Pretraining asks, after every glimpse, for the teacher's features of the whole
  scene: spatially dense (unseen regions too, which demands integration and extrapolation) and temporally dense (a
  loss every step, so short truncated BPTT chunks with constant memory). A canvas fixed in scene coordinates matches
  the teacher's scene-wide patch grid token for token, which makes that target directly usable.
- (A) Prior work's own ablations (`sources/literature.md`) suggest that choosing where to look added about 0 to 3
  points on natural images with a low-resolution glance, many glimpses or dense segmentation, and 5 to 70 points
  where information is small, sparse and off-centre; and that within a paper, changes to the observer often
  outweighed changes to the policy. Yohaï found this too zoomed in for now; how to scope the claim on stage is open.
- (A) Scanpath replay, on `#neuro-ai` until that slide was removed on 2026-10-24 ("human fixations, given a scale, become viewpoints: a
  policy on the same observer"): because CanViT is policy-agnostic, recorded human or monkey scanpaths can be replayed
  through it and its memory read after every fixation, to compare with fixation-locked recordings such as the
  free-viewing datasets. (Y) 2026-10-24: worth keeping in mind, and later that day a strong argument for policy-agnostic
  pretraining (Positions); caveats as above (uniform crops, not foveated; the Xiao et al. contrast above).
- (A) What the dense learning signal buys the observer has not been worked out with the authors. An earlier agent
  claim, that density is why CanViT trains without RL, is wrong: it could be trained without RL without a dense
  signal.

## The paper in the talk

What the paper says, and where each point stands in the deck (A, 2026-10-25; read against the paper in full). "Said"
means in the notes only, which the authors have not reviewed. Each entry ends with an agent's judgment, for the
authors to confirm.

- Why active vision: efficient and biologically plausible perception (abstract). Biology on screen (foveation, Yarbus);
  efficiency nowhere. Needed only if the FLOP result is shown.
- Viewpoint = center and scale; a glimpse has fixed pixels, so scale trades coverage against detail (Preliminaries).
  Said on `#vit-on-a-glimpse`. Should be visible: every policy, and "where and at what zoom", depend on it.
- Glimpses do not align with the scene, so patch tokens cannot be carried across time (Preliminaries, §4). Shown by
  `#integrating-information-across-viewpoints`.
- Canvas: persistent, scene coordinates, token-wise readout, gradients across time; no MLP or self-attention on it
  (§4). On screen in the paper's figure only; the figure is small on `#architecture`.
- Canvas resolution set at inference (grid broadcast from one learned patch, §4); trained at 32², better at 64² (§6).
  Nowhere. A distinctive property; candidate for the results or Q&A.
- Recurrent CLS token: carried across glimpses, read out for classification (§4, §6). Said only. Yohaï (2026-10-25):
  likely explained quickly with the architecture diagram.
- SR-RoPE: positions in scene coordinates bind the retinotopic backbone and the spatiotopic canvas; zoom is carried by
  the spacing of patch positions (§4). Said only. Must be explained (Y, 2026-10-25).
- Canvas Attention, asymmetric (§4). On screen; stays (Y).
- VPE token: a non-essential affordance for future policy learning (§4, App.). Set aside; Q&A if the learned policy
  prompts it.
- Distillation in DINOv3 latent space rather than pixels (§5); teacher features precomputed. On screen
  (`#distillation`); pixels against latents only in Q&A.
- CLS target in the loss (§5) and its ablation: dense supervision also improves CLS reconstruction (App.). Not shown;
  goes with the recurrent CLS token if anywhere.
- R-IID and F-IID, scale density favoring small glimpses (§5). On screen. F-IID speeds convergence (App.): set aside.
- Truncated BPTT over 2-glimpse chunks, random rollout length averaging 4; generalizes to 21 (§5, §6). Said on
  `#distillation`. Yohaï (2026-10-24): the generalization is a nice result; the mechanism may be too technical.
- 166 H100-hours, 13.2 M scenes, 1 billion glimpses (§6). On screen.
- Frozen weights with linear probes, "to assess feature decodability"; fine-tuning only for ImageNet-1k peak (§6). Said
  in the notes; whether it needs a visible moment is open. (An agent's claim that it is what makes the results a
  foundation-model claim was wrong, Y 2026-10-25.)
- Peak against prior active models (Table 1: 45.9 against 27.6 mIoU; 84.5 fine-tuned against 82.2 top-1). On
  `#state-of-the-art` since 2026-10-27, as bars on `#gap`'s chart; ImageNet-1k said at `#policy-accuracy`'s fine-tuned click.
- One glimpse: 38.5 mIoU, above every prior active model's best, with 20× fewer FLOPs than AME at 27.6; also above its
  input- or FLOP-matched DINOv3 teacher (abstract, §6, Fig. 4). The single glimpse and its cost on `#frontier` since
  2026-10-27; the teacher comparison in Backup (Fig. 4).
- C2F far above R-IID and F-IID on ADE20K; EG-C2F more efficient; ImageNet-1k plateaus early (§6). On screen.
- F2C still beats prior models: perception, not viewpoint selection, was the bottleneck (§6). On `#frontier` since
  2026-10-27, where F2C's curves pass above the prior models, with Part I's "No policy can make up for a poor observer".
- C2F above F2C on identical viewpoints: the paper reads it as "the importance of contextually informed processing"
  (§6). On screen as a fact; say what the paper says.
- RFS: recurrence alone improves accuracy, then declines without new viewpoints (§6). Removed from the deck; Q&A.
- Fig. 4 (canvas resolution, mask size, frontier against DINOv3). Backup (the paper's figure), since 2026-10-27.
- Ablations (App.): reads matter, BPTT matters, ViT-S, VPE least. Backup since 2026-10-27: the camera-ready's table of
  the ablations' ADE20K and ImageNet-1k accuracy.
- Canvas evolution within a glimpse: writes go from local to scene-wide (App. figure). Backup since 2026-10-27 (the
  paper's figure); proposed for the main flow ("The pass of 2026-10-27").
- Limitations and future work (§7): static scenes (`#next`); forgetting; RL policy learning (now done, `#learned-policy`);
  depth expected; a passive teacher is needed (cost at high resolution, 19 TiB of features), active-to-active
  self-distillation; one model size, a dataset 100× smaller than DINOv3's. Only static scenes is on screen.

## Open questions for the authors

- Kept open until addressed or decided (Y, 2026-10-25):
  - Part I read-through (2026-10-25 evening), flagged, not to change yet (Y):
    - `#task-dependent-scanpaths` asks "Can we train a general-purpose policy?" and ends on the right viewing policy
      depending on the task; the conclusion the argument needs (an observer meant for any task must work with any
      policy, hence the "Policy-agnostic" row of `#compare`) is not said there or on `#compare`. Yarbus shows that no
      single fixed policy suits every task, not that no general-purpose policy can exist.
      (Y, 2026-10-25, in the speakers' run-through) The question is left open on purpose, without a conclusion; its
      payoff belongs on `#compare`. A general-purpose policy does exist in one sense (free viewing), and a human's
      policy is conditioned on the goal: a model conditioned on any goal would need something like a future
      vision-language model, so a task-agnostic model is made policy-agnostic instead. This is for questions, which
      Sabrina may hand to Yohaï.
    - "Foundation model" is not defined in Part I; assumed known for now (Y).
    - The scale point of `#dinov3-and-the-brain` is not picked up again in Part I; to keep in mind for Part II's
      pretraining (rich world knowledge without enormous compute). Not critical (Y).
  - The architecture: the canvas has no moment of its own (it appears in the paper's figure on `#architecture`, now
    600 px wide). Built 2026-10-25, for the authors to review: `#sr-rope` after `#architecture` (a glimpse's 8 x 8
    cells and their scene positions, the canvas's 32 x 32 cells, a smaller glimpse whose cells pack together), and a
    click on `#architecture` ringing the recurrent CLS token where it leaves one glimpse and enters the next (removed
    2026-10-27: the CLS token is said, with the second glimpse).
  - The pretraining: why not pretrain the observer with its own policy is not argued on screen; the notes of
    `#pretraining-desiderata` carry it (AI draft). Rebuilt 2026-10-25 after Yohaï's review, then `#pretraining-desiderata` moved to Backup (Y: it says little; said aloud instead, a model that understands the world and works with any policy): was `#pretraining-desiderata`
    (robust to any viewing policy; rich knowledge of the visual world without enormous compute; the answers on a click),
    then `#distillation` ("Policy-agnostic dense latent distillation"), `#pretraining-policies`, `#pretraining`, the
    table callback. `#how-to-see-first` is in Backup; where it belongs is open. The problem signposts and
    `#dense-latent-distillation` were removed (rejected).
  - The results (Y, 2026-10-25): Table 1's numbers side by side, as bars on a dedicated slide, could be interesting;
    each concept needs a clear home without messy repetition, and covering the same results from different angles is
    fine when the talk says so. Since 2026-10-25 the ImageNet-1k panel of "Accuracy by viewing policy" shows the
    fine-tuned curves (solid) beside the frozen ones (dashed), as the paper's Fig. 3C does. (A, 2026-10-27) Built as a
    callback to `#gap`: `#state-of-the-art` ("The pass of 2026-10-27").
  - The results: peak performance against prior active models is partly there (the ADE20K panel widening to AME and
    AdaGlimpse); the narrowing of the gap with passive models is not. Yohaï is thinking about how to present it.
    Candidates from the paper (A): Fig. 3A (accuracy against cumulative FLOPs, CanViT's curves far above the prior
    models' points; `#frontier` in Backup) and Fig. 4C (the Pareto frontier against DINOv3 ViT-B at input resolutions
    up to 512 px: CanViT above it at low FLOPs, close to its 47.2 at the top). Fig. 4C compares CanViT with its own
    teacher's architecture, which is the comparison the gap is about. (A, 2026-10-27) `#state-of-the-art` now shows
    CanViT-B's 45.9 on `#gap`'s chart, beside DINOv3 ViT-B's 47.2 read out the same way; Fig. 4 is in Backup.
- Prior models on "Accuracy by viewing policy" (A, 2026-10-25): the ADE20K panel has AME and AdaGlimpse as lines
  (agreed 2026-10-24), the ImageNet-1k panel none, because only the ADE20K export carries baselines (ImageNet-1k's are
  typed in the paper's Table 1). Adding them there would mislead: AdaptiveNN's 82.2 sits above CanViT's frozen 81.1,
  and CanViT's 84.5 is fine-tuned and off that panel. The lines also span every glimpse count though prior results
  are at 8 glimpses. Proposal: the policy slide compares policies only; prior models, on both tasks with each
  number's protocol, go where peak performance is shown (Y, 2026-10-25: two angles kept apart). (A, 2026-10-27) Done:
  `#policy-accuracy`'s notes compare policies only, and `#frontier` shows the prior models against the policies, so
  the agreed reference lines may no longer be needed.
- How Part I presents active vision as a field (what it has been about, why one might care) before the observer and
  policy distinction.
- How to word the functional argument for neuro-AI.
- The cost slide's framing (reuse of an investment, not a cheaper pretraining), to discuss.
- `#free-viewing` ("Toward AVFMs for Neuro-AI") is in Backup (Y, 2026-10-25): its precise argument must be thought
  through before it returns, without mistakes; one option is the Active Visual Semantics (AVS) dataset alone, named on
  screen with its figure. Whether to address LookWhere in the talk or only if asked.
- Slides with value but undecided (Yohaï, 2026-10-24): `#human-vision` (likely redundant, since the speaker can say
  that we move our eyes, and it shows foveation); `#extrapolation` (may be too heavy, may be reworked); `#gap-canvit`
  (repeats `#gap`'s title; putting results in context is wanted, but where is open); `#timeline` (has issues and its
  place is open; not to be deleted); a callback to `#compare` with CanViT's column animated in (not convinced it is worth it).
- A David Silver remark about AlphaFold (that the team should stop treating it as an RL problem), which Yohaï found:
  its source is not yet verified (a web search on 2026-10-24 did not find it). Needed before it is quoted anywhere.
- On-screen wordings an agent wrote that state more than the evidence may support: `#active-vision-model`'s "prior
  work focused here" (the paper: "often focused") and "A general-purpose observer lets you use any policy" (tested
  policies); `#compare`'s rows "Any viewing policy, no RL" and "Rich world knowledge" (Y 2026-10-24; was "Accuracy close to passive models") (45.9 against 47.2,
  at 21 glimpses).
- From 2026-10-01, still undecided: `#architecture` as the click-by-click drawing (the paper's full figure is in
  Backup as `#architecture-paper`); one segmentation
  palette for the talk; policy colors: settled 2026-10-25, the paper's (Y); "runs everywhere" (JAX and MLX ports) or PyTorch and the browser only.

## The deck: a design (agent's proposal, 2026-10-25; not approved)

Rethought from the positions above and the paper (read in full on 2026-10-25), after Yohaï's review found the deck
sloppy: the demo came before the architecture and pretraining were understood, signposting was missing, and
concepts were smuggled in. Principles: every move of the argument is marked, by a short signposting slide where
needed (a few words, 5 to 10 seconds); nothing the argument relies on is smuggled; each result appears once; the
demo comes last, when everything in it has been explained. Kinds: **S** a signposting slide, **C** a content slide.
Built on 2026-10-25 in this order (slide ids: `#passive-models`, `#how-to-see-first`, `#outline-architecture` with a
click that highlights the architecture, `#glimpses-do-not-line-up`, `#outline-pretraining`,
`#how-to-see-apart-from-where-to-look`, `#classification-and-segmentation`); the optional signposts 19 and 27 were
not built, their points said on `#architecture` and in the results' notes; `#gap-canvit` was removed, its numbers
said on `#cost`. Status words below are those of the proposal. Durations are guesses (Part I about 8 minutes, Part II
about 16, 1 of slack).

### Part I (Sabrina): why Active-Vision Foundation Models

1. C `#title` · 0:15 · exists. The title, the speakers and their affiliations on the left, on the brand gradient; on
   the right a silent teaser, CanViT running in a loop (Y, 2026-10-25), five glimpses per scene, each chosen to change
   the canvas a lot (Y). Scenes ADE_val_00000419 (a poplar in fields), 00001882 (tires and a treeline), 00000203 (a
   skyscraper), 00001860 (a street with a parked car), chosen by eye (A, 2026-10-25) among the 10 of 67 screened scenes
   whose glimpses 2 to 5 change the canvas most (`experiments/title_teaser`). Excluded: ADE_val_00001980, a paper
   machine (Y, 2026-10-25: looks odd).
2. C "Computational models of biological visual processing" · 1:15 · changed 2026-10-25. Yamins 2014 under
   "Supervised", then Zhuang et al. 2021 (Fig. 2B, IT) under "Unsupervised / self-supervised", each word large (Y:
   the key distinction of the slide). Raugel's DINOv3 result is said on `#foundation`.
   (Y) Approved for now, 2026-10-25.
3. C "DINOv3" · 1:00 · exists. Image and features side by side from the start (Sabrina, 2026-10-25), then the linear
   decoding.
3a. C "DINOv3 and the brain" (`#dinov3-and-the-brain`; its own slide since 2026-10-25, Sabrina: it was `#foundation`'s
   last click): Raugel et al.'s
   Fig. 5E (arXiv numbering), each cortical region's training half time against its distance from V1. (Y) The scaling
   argument: representations that match higher cortex take training at scale; that is what a foundation model is,
   why neuro-AI cares, and why the pretraining builds on DINOv3 instead of repeating its training. Says: trained once,
   without labels, at great cost; reused across tasks; also what neuro-AI now compares with the brain.
   (Y, 2026-10-25) The slide does not open on scale: it opens on the study, the paper's OpenReview header above its
   Fig. 1 (what Raugel et al. did: 7-Tesla fMRI and MEG to disentangle the factors of convergence between brains and
   DINOv3), and Fig. 5E follows on a click.
4. S "What's missing?" · 0:10. The implicit answer is eye movements, answered by Yarbus next (Y, 2026-10-25: the
   answer is not that passive models see the whole image, which feeding them a crop would disprove).
4b. C `#human-vision`, no title · 0:20 (from Backup, Y 2026-10-25): (Y, 2026-10-25) opens on the foveated glimpses, a white
   fixation cross jumping between fixations, then a click stops on the street at full resolution, the cross at its
   center, as the models see it; the foveation animation on the street,
   the answer to "What's missing?".
6. C "A brief history of deep active computer vision" · 1:00 · exists. Opens on RAM alone on a short axis, beside its Fig. 1A (glimpse sensor) and a Fig. 3 glimpse path; a click fades them, grows the axis and brings the later models in turn, without further clicks; then the neuro-AI models (Sabrina, 2026-10-25). Models since 2014; how each chose where to
   look (mostly RL); neuro-AI active models, each built for a task or a way of looking.
7. C "The wide gap between passive and active computer vision" · 0:45 · exists. ADE20K segmentation, then the bars. (Y, 2026-10-25) The image and
   its mask start large and centered, then shrink to the left to make room for the bars.
8. C "What's in an active-vision model?" · 1:30 · exists. Clicks (Sabrina, 2026-10-25): the claim; then a teal contour around action selection (the focus of most prior active vision work), then, on the next click, a blue one around instantaneous vision (the focus of passive vision foundation models), in place of labels. The prior models' ablations left the main slide for its Backup copy, `#active-vision-model-ablations` (Sabrina, 2026-10-25). The three axes, observer and policy; "prior work often
   focused here"; prior papers' own ablations; "No policy can make up for a poor observer." The gap is the question,
   this slide a first answer.
9b. C "Where people look depends on their goal" · 1:00 (a click goes from the painting straight to all seven records, Sabrina, 2026-10-25) (moved 2026-10-25, Y: before the requirements of an
   AVFM). Yarbus: a model for any task must handle any sequence of fixations.
10. C "Toward Active-Vision Foundation Models" · 1:15 · exists. Passive foundation models and prior active
    models side by side, then the goal column on the only click (Y, 2026-10-25: no row by row). The row "Rich world knowledge"
    is now "Visuospatial intelligence" (Y, 2026-10-25). Rows (Y, 2026-10-25): Instantaneous vision, Memory,
    Visuospatial intelligence, Task-agnostic, Policy-agnostic; "Dense outputs" removed. Passive models have
    instantaneous vision (the paper's introduction), so that row is ✓ for them.

### Part II (Yohaï): how, and what it does

In deck order since the pass of 2026-10-27 (below). For each slide, the notes checklist of `AGENTS.md` answered by an
agent (A, 2026-10-25; A, 2026-10-27 for the slides that pass built or moved): what the slide must make the listener
understand, the terms it introduces, the paper's passage it reuses, the claims checked, and the spoken length of its
notes at Part I's measured rate (129 words per minute). The notes themselves are in `index.html`, still marked AI
drafts until Yohaï words them.

- `#paper`: hands over; names CanViT as the first task- and policy-agnostic AVFM (abstract), the team, NeurIPS 2026.
  About 15 s.
- `#canvit-rollout` (A, 2026-10-27, at Yohaï's request): the paper's Figure 1, built from its own code a glimpse at a
  time: the scene alone, then each glimpse's viewpoint on the scene with its column (glimpse, canvas, canvas change).
  No title. Glimpses counted from 1.
  - Must understand: what CanViT does, before how: each glimpse, wherever it is taken, updates one canvas for the whole
    scene. The first bullet says what is shown.
  - Knows: glimpses from Part I (`#timeline`), PCA (`#foundation`). Introduces viewpoint and canvas.
  - Paper: Figure 1's caption ("integrates broad context and fine detail from variable-scale glimpses"); the paper
    opens on this figure too. Its "extrapolates to unobserved regions" is left to `#table-callback`.
  - Checked on the figure: after the first glimpse, each canvas change concentrates where the glimpse looked.
  - About 70 words, 35 s.
- `#outline-architecture`: the two problems that building an Active-Vision Foundation Model raises, said over the
  three names: an architecture that gives the observer instantaneous vision and memory, and a pretraining that gives it
  visuospatial intelligence (the paper's Related work) for any task and any viewing policy; then experiments. About 20 s.
- `#vit-on-a-glimpse`: the architecture's problem, first half: a glimpse (a crop at a viewpoint, 128 px whatever the
  zoom) turned by a ViT into patch tokens that describe it alone. About 30 s. Its example, a lamp as white as its
  ceiling, and how it was chosen: "Open work", 2026-10-27.
- `#integrating-information-across-viewpoints`: the architecture's problem, second half: a passive model on separate
  glimpses cannot fill the gap between them and does not know where each glimpse was; what is needed is integration
  across viewpoints (Preliminaries, "glimpses need not align with the scene"). About 45 s.
- `#architecture`: the answer to the integration problem: backbone (instantaneous vision) and canvas (memory), reads
  and writes (§4: "pull information from the canvas and send updates to it"), the canvas persisting while glimpse tokens
  are discarded, the recurrent CLS token, dense decoding at every glimpse. Introduces: backbone, Canvas Attention, CLS
  token. About 60 s.
- `#sr-rope`: scene-relative positions bind the retinotopic backbone and the spatiotopic canvas (§4, SR-RoPE); the
  spacing of positions encodes zoom; it answers the problem slide's "where was each glimpse taken?". About 40 s.
- `#canvas-attention`: asymmetric cross-attention, projections on the glimpse side only, so the canvas can be large
  without the quadratic cost of self-attention (§4); Fig. 3B. About 35 s.
- `#outline-pretraining`. A few seconds.
- `#how-to-see-first` (A, 2026-10-27, after Yohaï's correction): the pretraining's problem and its principle. How to
  pretrain the observer, when prior models often learned how to see together with where to look, the policy through
  RL: "Before knowing where to look, we should know how to see" (Y, 2026-10-24), so the observer is pretrained on its
  own, without a policy, as representation learning (paper, Introduction: it "disentangles 'how to see in an
  active-vision setting' from 'where to look'"). Reasons said: an observer trained with one policy adapts to it, ours
  must work with any since where we look depends on the task (Yarbus, a callback); RL's noisy gradients. About 35 s.
- `#pretraining-policies`: the principle in practice, answer 1 of Yohaï's plan (2026-10-25): R-IID and F-IID (§5.2),
  static, each policy's sequence of boxes then its glimpses; the policies tested later never seen in pretraining.
  About 35 s.
- `#distillation`: answer 2 and both together. What to learn from random glimpses: predict the whole scene after every
  glimpse, unseen regions included, which requires understanding its spatial and semantic structure (§5.1); in DINOv3's
  features, not pixels, a passive teacher's knowledge of the visual world, learned without labels at great cost
  (Related work; `#dinov3-and-the-brain`, a callback); label-free at both levels (Y). About 70 s.
- `#pretraining`: the run (§6) and DINOv3 7B's cost, to scale; building on DINOv3 rather than repeating it (Y,
  2026-10-24: the hours chart is not wrong). Since 2026-10-27 the notes add the camera-ready's "Precomputation of
  teacher features is entirely optional, and is a tradeoff between storage and compute cost" (`subsec:distillation`),
  which answers the divisive point (Positions). About 30 s.
- `#table-callback`: CanViT after the same two glimpses fills the middle ("extrapolates to unobserved regions", Fig.
  1 caption). About 15 s.
- `#outline-experiments`: one pretrained model, linear readouts, frozen; ImageNet-1k also fine-tuned (§6). About 15 s.
- `#state-of-the-art` (A, 2026-10-27, rebuilt and moved first among the results): `#gap`'s chart as it ended, then
  CanViT-B's bar between the active and passive models.
  - Must understand: on the benchmark where Part I showed the gap, CanViT-B leaves the prior active models far behind
    and comes close to its passive teacher read out the same way. The first bullet returns to the gap; the click gives
    the number.
  - Knows: `#gap`'s chart and mIoU (Part I), frozen linear readouts (`#outline-experiments`).
  - Paper: §6, "CanViT-B outperforms prior active vision models ... achieving up to 45.9% ADE20K mIoU (up from
    27.6%)"; the abstract, "Our work narrows the wide gap between passive and active computer vision".
  - Checked: the bar is the export's best point (C2F, 64 x 64 canvas, glimpse 21), and `gap.js` fails if it is not the
    export's claim; DINOv3 ViT-B's 47.2 is the paper's probe at 512 px, the protocol CanViT-B's readout follows.
  - Title: the paper's claim; "Narrowing the gap between passive and active computer vision" is the alternative, if it
    fits one line.
  - About 55 words, 25 s.
- `#c2f-and-f2c`: static, in stages (Y, 2026-10-25); the same viewpoints in opposite orders isolate order (§6,
  Policies). About 30 s.
- `#eg-c2f`: within a level, the most uncertain tile first; the canvas guides viewpoint selection without RL (§6).
  About 40 s.
- `#policy-accuracy`: Figs. 3B and 3C as one view; curves keep their style across steps (Y, 2026-10-25). Since
  2026-10-27 its notes compare policies only (C2F ends three points above F2C on the same viewpoints), and give
  ImageNet-1k's fine-tuned peak against the best prior active model's 82.2 at the click that shows it. About 35 s.
- `#frontier` (A, 2026-10-27, promoted from Backup; its build is Yohaï's of 2026-10-01): F2C at 32 x 32 beside the
  prior models from the start (since 2026-10-27 evening), then F2C at 64 x 64, then EG-C2F.
  - Must understand: even a worse-than-random viewing order beats the prior models, for less compute, so perception,
    not viewpoint selection, was the bottleneck; and a single glimpse costs 20 times less than AME's best for a higher
    score. Part I's "No policy can make up for a poor observer" returns here as evidence.
  - Knows: F2C (`#c2f-and-f2c`) and its rank among policies (`#policy-accuracy`), EG-C2F (`#eg-c2f`), the prior models
    (`#gap`, `#state-of-the-art`). "Compute" is said for the axis's GFLOPs.
  - Paper: §6, "CanViT's advantage holds even with the F2C policy's worse-than-random viewing order, showing that
    perception, not viewpoint selection, was the bottleneck in this task"; "in a single glimpse, CanViT-B reaches 38.5%
    mIoU using 20x fewer FLOPs than AME does at 27.6% mIoU".
  - Checked (`ade20k_seg.json`): F2C passes AME (SETR)'s 27.6 at its fifth glimpse, 79 GFLOPs with a 32 x 32 canvas and
    110 with 64 x 64, against AME's 312 at eight glimpses.
  - About 95 words, 45 s.
- `#learned-policy`: the camera-ready's values since 2026-10-27 (Q-learning, best checkpoint of each of 8 runs, 95%
  intervals); a policy learned with RL on the frozen observer beats EG-C2F, at 640 thousand glimpses of policy learning
  against a billion of pretraining (Y, 2026-10-24); the LLM parallel said aloud (Y, 2026-10-25); "they'll be in the
  camera-ready version". Its first line calls back the principle: now that CanViT knows how to see, it can learn where
  to look. About 40 s.
- `#quickstart`: the code walked through, a bus read as car, then bus. About 30 s.
- `#next`: CanViT 2 in broad strokes only. About 10 s.
- `#links`: stays up during questions.

Part II's notes total about 12 minutes at Part I's rate; Part I's run-through took about 6:15 (rehearsal timings,
below).

Backup since 2026-10-27, each slide for a question the room may ask: `#free-viewing` (neuro-AI datasets; its argument
is open), `#rollout` (the live model, for a demonstration), `#extrapolation` (how far the canvas extrapolates, with
its measured overshoot), `#detail` (why zoom: a television only a zoomed glimpse finds); then the paper's figures and
tables, each with the question it answers in its notes (`sources/paper-figures.json`): Fig. 4, the ablations'
downstream accuracy, the canvas updates within a glimpse, the pretraining dataset, ImageNet-1k against inference cost,
the architecture control, COCO detection, the learned policies. Removed on 2026-10-27 (A, at Yohaï's request to
eliminate weak AI-written backups; git keeps them): `#architecture-paper` (the figure `#architecture` shows),
`#history` (the earlier form of `#gap`), `#results` (its point, CanViT-B against passive and active models, is now
`#state-of-the-art`'s), `#cost` (rejected as drawn; the architecture control answers training cost under matched
conditions, and the accounting stays in `experiments/training_cost/`), `#canvas` (`#canvit-rollout` shows the canvas,
from the paper), `#active-vision-model-ablations` (a second copy of a Part I slide; the prior models' ablations stay
in `sources/policy-ablations.json` and `sources/literature.md`), `#adaglimpse-policies` (its argument was reconsidered,
Y 2026-10-25), `#classification-and-segmentation` (introductory, no use in questions), `#pretraining-desiderata` (says
little, Y 2026-10-25). Still open from 2026-10-25: how to convey information density (the learning-signal slide was
rejected as drawn); whether `#free-viewing` returns; replaying recorded scanpaths as an ending.

### Not yet on screen, from the positions

Now said in the notes but with no visual of their own: replaying recorded scanpaths; the two contributions named as
such (end of `#architecture`); the LLM pretraining/post-training parallel; why neuro-AI cares about foundation models
(`#foundation`); the complexity that policy-agnostic pretraining removes. Not anywhere yet: a slide that introduces
classification (tied to where the ImageNet-1k results go).

Removed on 2026-10-24 after Yohaï's review (git history has them): `#compare-canvit`, `#neuro-ai`, `#avfm`,
`#teacher`, `#calibration`, `#live`, `#learning-signals` (labels against rewards against distillation targets),
`#memory` (memory kept against reset on a street; true, but it served nothing); the scale facts under
`#foundation`; the practical-details panel of `#pretraining`; the setup list beside `#learned-policy`'s chart.

## The pass of 2026-10-27 (A)

At Yohaï's request: a full pass over the deck after a full read of the camera-ready (its source at commit
`8924d417`), with the whole argument in mind. A first version (`bf09c295`) rearranged slides without re-deriving the
argument; Yohaï's corrections that day (Positions) rebuilt it as below. Every slide was rendered and read on screen.

**Part II's argument.** Part I ends on what an Active-Vision Foundation Model must have; `#paper` names CanViT as the
first. `#canvit-rollout` shows what it does, as the paper's Figure 1 does at its start: each glimpse updates one canvas
for the whole scene. Building such a model raises two problems, which the paper's two contributions answer
(`#outline-architecture`, said over the three names).
- The architecture's problem: a standard backbone turns a glimpse into tokens about that glimpse alone
  (`#vit-on-a-glimpse`), so separate glimpses leave the scene unintegrated and unplaced
  (`#integrating-information-across-viewpoints`). The answer: the backbone and the canvas (`#architecture`), placed in
  one frame (`#sr-rope`, which answers "where was each glimpse taken?"), affordable at fine canvases
  (`#canvas-attention`).
- The pretraining's problem: how to train the observer at all, when prior models learned how to see together with
  where to look. The principle (`#how-to-see-first`): before knowing where to look, we should know how to see, so the
  observer is pretrained without a policy. In practice, random glimpses (`#pretraining-policies`); as the target, the
  whole scene in DINOv3's features (`#distillation`); its cost against DINOv3's own training (`#pretraining`); its payoff
  on the problem slide's table (`#table-callback`).
- The experiments: the gap of Part I, closed on its own chart (`#state-of-the-art`); the policies (`#c2f-and-f2c`,
  `#eg-c2f`, `#policy-accuracy`); what they show about the observer, that even a worse-than-random order beats the prior
  models for less compute (`#frontier`: Part I's "No policy can make up for a poor observer" as evidence); then, the
  observer able to see, learning where to look (`#learned-policy`, whose first line calls back the principle).

Each result appears once: the peak on `#state-of-the-art`, the policy differences on `#policy-accuracy`, the single
glimpse and the bottleneck on `#frontier`, the learned policy on `#learned-policy`.

**Built.** `#canvit-rollout` from the paper's Figure 1 code (`experiments/canvit_rollout/`, built by
`build_deck_data.sh canvit_rollout`; the paper's composition gained parameters for the glimpses shown, the panel
letters, the first index and the time label, whose defaults leave the paper's figure byte-identical); the outline's
notes posing the two problems; `#how-to-see-first` opening the pretraining; `#pretraining-policies` before
`#distillation`, whose notes now separate the pretext task (the whole scene) from the target (DINOv3's features);
`#state-of-the-art` rebuilt as the callback to `#gap` (the paper's Table 1 as bars, Y 2026-10-25) and first among the
results; `#frontier` promoted from Backup; `#learned-policy` on the camera-ready's values (merged from
`talk-part2-camera-ready`); eight of the paper's figures and tables as Backup slides through `paper_figures.py` (merged
from `talk-paper-figures`); nine Backup slides removed (above); dangling stylesheet fragments from an earlier edit
removed.

**Open for the authors.**
- The canvas updates within a glimpse (the paper's emergent progression from the glimpse's patch grid to scene-wide
  content): about 20 s after `#canvas-attention` would speak to the neuroscientists. In Backup now.
- The input/output dissociation (Fig. 4A: from the same 128 px glimpse, a finer canvas beats the input-matched teacher):
  Backup only, or a last click on `#frontier`.
- `#learned-policy` shows Q-learning alone; the paper shows Q-learning and REINFORCE, both above EG-C2F. Both on the
  slide would make the point about the observer rather than one algorithm (needs `<canvit-policy-curves>` to take
  several added curves); the paper's figure is in Backup. The learned curve's color: the deck's policy teal or the
  paper's Q-learning purple.
- Lamarck Labs among `#paper`'s affiliations, as in the camera-ready and on `#title`.
- Part I (Sabrina's): `#active-vision-model`'s second line, "A general-purpose observer lets you use any policy", says
  more than the paper, whose claim is that with perfect instantaneous vision and memory, naive and strategic policies
  should reach the same accuracy after enough glimpses of a static scene.
- The Backup tables are the camera-ready's at `8924d417`; the dataset-scale table's values are still being decided for
  the paper. `paper_figures.py try` then `accept` refreshes them.

## Open work

- (A, 2026-10-27 evening) Done, for review: `#integrating-information-across-viewpoints` is titled as a question,
  "Integrating visual information across viewpoints?", the problem that `#table-callback` answers under the same title;
  `#architecture` lost its CLS rings and their click; `#sr-rope` shows the canvas patches from the start (one click,
  the zoom); CanViT's bars on `#state-of-the-art`, `#pretraining` and `#learned-policy` are in the brand's gradient
  (red to blue along each bar; a frozen bar's hatching is now a stripe mask, so the gradient shows through it);
  `#frontier` shows the prior models from the start, and a curve's label moves above the curve where another crosses
  its place at the end (F2C, 32²); on `#policy-accuracy` the first set (frozen; 32²) recedes to 35% opacity once the
  second joins, and close labels center on their heads instead of stacking below the top one (C2F and F-IID on
  ImageNet-1k).

- (A, 2026-10-27) `#canvas-updates` ("An emergent hierarchy within each glimpse"; the first draft's "Within a glimpse,
  from patches to objects to the scene" rejected, Y), after `#learned-policy` and before `#quickstart` (Y: the paper's
  canvas-updates figure, toward the end, briefly, called emergent and compared with nothing in the brain; wording
  open): the paper's figure "Canvas updates within and across glimpses" rebuilt from its generator
  (`experiments/canvas_updates/export.py` reuses paper/exporter's `WriteCapture` and PCA colors). In one row, the
  paper's cat with its first glimpse's box, the glimpse, an arrow, then what each of the glimpse's three Writes adds to
  the canvas (its residual, PCA per map as in the paper), labelled "Canvas Attention Write residuals" (Y), one per
  click, the glimpse's box on every map; the arrows here and on `#vit-on-a-glimpse` are one SVG style (Y: fix their
  rendering). Writes count from 1 on the slide, from 0 in the paper (open). Tried and set aside (renders in the
  session's scratchpad): the scene above the glimpse with the maps beside both (the arrow came from neither), the five
  pictures in one row at 196 px (maps too small), one large map replaced at each click (the progression needs the maps side by side), and the canvas
  after each Write instead of the residual (at a first glimpse it is the first residual exactly, then mostly the last
  residual). Alternatives exported beside the cat (`canvasUpdatesExample` in `index.html`): the paper's food truck, a
  bedroom window (`ADE_val_00000124`) and a seafront bus (`ADE_val_00000996`), chosen by eye from the export's sweep of
  the deck's scenes (`--sweep`, its contact sheet in `site/.experiments/canvas_updates/`). The notes name the cat.
- (A, 2026-10-27) `#vit-on-a-glimpse` shows a lamp hanging from a white ceiling (ADE20K validation
  `ADE_val_00001517`, a glimpse of scale 0.29 at the top of the room) instead of the cat, whose glimpse covered most of
  its scene; the authors asked for a zoomed glimpse on an object that its pixels barely set apart and its patch tokens
  do, and for the slide's alignment fixed. `experiments/vit_glimpse/sweep.py` frames every object of ADE20K validation
  that a glimpse of scale 0.2 to 0.4 can center on (its side 1.7 times the object's) and measures, over the glimpse's
  8 x 8 patches, how well the patches' mean colors separate object from surroundings and how well the patch tokens'
  colors as shown (PCs 1-3) do. From its best candidates, eleven were chosen by eye (`examples.py`, `EXAMPLES`, the
  slide's first): the lamp has the ceiling's mean color (`measures.delta_e` in its `meta.json`, far below a visible
  difference), and its tokens set it apart from every patch of ceiling. The alternatives, in that order: a white plush
  dog on white crib bedding, a white settee against a white wall, a gold chandelier against gold paneling, a wooden
  door in a wooden wall, a white hatch on a white wall, wooden cabinet doors in wooden paneling, cream cushions on a
  cream bed, a white frame on a white wall, a bicycle in shade, a white nightstand; `vitExample` in `index.html` picks
  one. The pictures now share one size and one grid, so they sit on one center line with the captions on one baseline.
  Open: `#sr-rope` still shows the cat (`data/talk/vit-glimpse`), though its first glimpse was meant as
  `#vit-on-a-glimpse`'s; whether it follows the lamp is for the authors.
- (A, 2026-10-27) From a co-author's review of Part I, checked against the sources (Yamins et al. 2014; Khaligh-Razavi
  & Kriegeskorte 2014; Raugel et al., whose ICLR version scopes its claims to DINOv3; Yarbus's later replications):
  two notes fixes, each marked as an AI draft for the speakers to reword. `#tradition`: the deep networks that first
  predicted the ventral stream well were supervised (earlier models were not, so "originally explored in the context of
  supervised learning" was wrong), and Yamins et al.'s link holds "among the networks they tested".
  `#dinov3-and-the-brain`: the scaling point framed as what Raugel et al.'s finding suggests for self-supervised
  foundation models like DINOv3 (Y: scoped to DINOv3 alone it lost the point); the authors themselves conjecture
  that their results generalize. A change to `#task-dependent-scanpaths` was reverted (Y: no one had flagged it; the
  original stays). On `#gap` and `#state-of-the-art`, the bars show names and values only (Y: the line under each
  was too much detail). `#state-of-the-art` then shows ImageNet-1k in the ADE20K chart's place (Y: after ADE20K, the
  most important; the gaps must show although accuracies saturate near the top): top-1 accuracy, never error (Y), as
  bars on an axis from 70% (dots rejected, Y), the prior active models as the paper's Table 1 prints them
  (`site/check_paper_numbers.py` holds them to it) and DINOv3 ViT-S and ViT-B read by the paper's probes, then
  CanViT-B frozen and fine-tuned. Both of the slide's charts bracket "Active" and "Passive" models and hatch frozen
  backbones in slanted lines under a key reading "End-to-end" and "Frozen" (Y: delineate active from passive, and
  frozen from fine-tuned; the hollow style rejected; less text). Every bar chart (`#gap` too) now has left and bottom
  spines with its axis title under the tick labels (Y). `#quickstart` has three clicks (the code with the model and
  scene, the first glimpse, the second) and hides its scene and maps by default: the section's class `code-only`
  switches them off (Y: less overloaded, easy to turn back on).
  `#paper`'s notes: the work was supervised by Audrey Durand and Suresh Krishna (Y), not "joint work".
- (Y, 2026-10-27) A full pass over the deck after a full read of the camera-ready: logical flow, order, the support
  on screen for each point; what is missing, with when, where and at what length; weak backup slides may go, new
  backups and additions to Part II may come. Then, during the pass: eliminate weak AI-written backups and reexamine
  every backup critically; the paper's Figure 1 at the start of CanViT, before the architecture; where
  `#integrating-information-across-viewpoints` belongs, with the whole flow in mind. Done (A, 2026-10-27), for
  review: "The pass of 2026-10-27"; its proposals there stay open.
- Rehearsal timings (run-throughs; agents' estimates are guesses until these exist):
  - 2026-10-25, Sabrina with Yohaï: `#title` 45 s, more than the notes say; can come down if needed.
  - 2026-10-25: `#tradition` 50 s (retitled "Deep neural networks as models of vision", Y).
  - 2026-10-25: `#foundation` 45 s.
  - 2026-10-25: `#dinov3-and-the-brain` 35 s.
  - 2026-10-25: "What's missing?" (`#passive-models`) 10 s.
  - 2026-10-25: `#human-vision` 15 s.
  - 2026-10-25: `#timeline` 35 s.
  - 2026-10-25: `#gap` 45 s.
  - 2026-10-25: `#active-vision-model` 45 s.
  - 2026-10-25: `#task-dependent-scanpaths` about 35 s.
  - 2026-10-25: `#compare` 45–50 s.
  - 2026-10-25: first full run-through of Part I, from `#title` to `#compare`: about 6:15.
- Yohaï's feedback on Sabrina's run-through of Part I (2026-10-25), not yet applied (to be worked through with her):
  - Title: the contrast with passive computer vision models is said too loosely.
  - `#tradition`: "a new class of models" is questionable (new?); spell out IT; say where the better categorization
    performance predicts neural responses (which recordings).
  - Self-supervision is a relaxation of constraints, not "more biologically plausible constraints".
  - `#foundation`: spell out principal components, three at a time, before "PCs"; say who "we" is.
  - Scale is not a reason to care about DINOv3 in itself: it is the means to the semantic representation, which
    requires it.
  - `#dinov3-and-the-brain`: too quick; say what Raugel et al. did and what the paper is about.
  - "What's missing?" and `#human-vision`: the non-uniform front end (foveation, eccentricity-dependent acuity) was
    glossed over.
  - `#timeline`: the neuro-AI active models are passed over too quickly.
  - `#gap`: introduce ADE20K segmentation first: what the task is, why it is interesting (harder than classification,
    both spatial and semantic).
  - `#active-vision-model`: too much text, partly redundant; remove the descriptions under each component.
  - "No policy can make up for a poor observer" needs its reasoning said, not just asserted.
- `#rollout`'s scene and R-IID seed chosen so that its mIoU never drops (Y), by `experiments/rollout_demo`, if
  `#rollout` returns from Backup (parked 2026-10-25, Y).
- The meter on `#rollout` shows a scene's mIoU (A, 2026-10-25): about 18 to 30% on the street, against the benchmark's
  45.9, because one scene's mIoU counts every class it predicts there, rare ones included, while the benchmark pools
  over the validation set. Whether that number on screen helps or confuses is for the authors.
- `#compare`'s "Rich world knowledge" puts ✗ for prior active models (A, 2026-10-25), though AdaGlimpse starts from
  MAE weights pretrained on ImageNet; an expert may object.
- The pretraining chain (Y 2026-10-24: the AdaGlimpse slide leading to grayed-out "How to see" / "Where to look"
  boxes was odd): reworked by an agent as the principle (`#how-to-see-first`), then `#pretraining-policies` (AdaGlimpse's
  learned policies went to Backup, 2026-10-25), whose notes say the disentangling; the boxes slide removed. For the authors to review.
- Agreed 2026-10-24, not yet applied: AME (27.6) and AdaGlimpse (25.7) as reference lines on the ADE20K panel of
  "Accuracy by viewing policy".
- `#rollout`'s meter shows pixel accuracy, never introduced; mIoU instead (Y, 2026-10-24). The DINOv3 whole-scene
  panel is off `#rollout` for now (Y, 2026-10-24); `experiments/whole_scene_baseline` still computes its data.
- RFS is off "Accuracy by viewing policy" (A, 2026-10-24) because no slide introduces it; the authors decide whether it
  comes back with an introduction.
- Asked 2026-10-24 and never run (the discussion moved on): how much AdaGlimpse's released reconstruction model loses
  when it gets glimpses its policy would not choose, with confidence intervals. Its encoder ignores glimpse order
  (`sources/literature.md`), so swapping the quadrants' order alone would change nothing after the last glimpse; an
  agent proposed shifted quadrants, random positions at the same scale, and the classification policy's glimpses.
- From an agent's audit of the whole deck (2026-10-24 late), not yet decided or fixed:
  - Classification is introduced where ImageNet-1k first appears, on `#cost` (a dog with its label beside the chart
    heading); whether that is where the ImageNet-1k results belong stays open (Y).
  - "Toward Active-Vision Foundation Models" is on screen three times: `#title`, `#compare`, `#paper`.
  - Glimpses count from 1 throughout the talk ("glimpse 1"), as on its charts; the components show the paper's t
    elsewhere (`data-glimpse-label` on `<html>`).
  - Segmentation palettes differ between slides; on `#extrapolation` floor and ceiling now take a brown and a pale
    yellow, apart from the glimpse boxes' blue (experiments/table_corners/plot.py).
  - GPN: Sabrina's outline cites "Thorat et al. 2026, Science Advances"; a search found only arXiv:2511.12715 (2025),
    which the deck cites.
- `#free-viewing` says replaying recorded scanpaths through the observer (notes, 2026-10-25) but does not show it:
  showing it needs a recorded scanpath turned into viewpoints and run through CanViT-B (a scanpath from one of the
  free-viewing datasets, or Yarbus's records digitized); none is in the repository.
- From an independent review (a fresh agent, 2026-10-25), for the authors to decide:
  - The demo stays last, after `#learned-policy`: Yohaï suggested it could come at the very end (2026-10-24).
  - "Pretrained in 166 hours on a single H100" is the paper's claim as a title; "CanViT 2" and "Accuracy against
    training compute" replaced a colon title and an understated one (2026-10-25).
  - `#cost`'s on-screen conditions line ("Not counted: …"), which the rules would send to the notes.
  - `#active-vision-model`'s ablations now show the conditions they were measured in (after a whole-image glance;
    8 glimpses), and the notes give the larger gain where what matters is small and off-centre (GFNet on traffic
    signs, 69 to 91.5%). Whether to show more rows is the authors' call (2026-10-25).
  - `#classification-and-segmentation` shows `#gap`'s living room again beside the dog.
  - `#distillation` shows the released model over 21 glimpses (training averages 4), and its first glimpse is a
    nearly blank wall.
  - Length: about 35 main slides, several of them signposts of 5 to 10 seconds. Cuts wait for the rehearsal,
    since timings are guesses until then (Y); candidates: `#timeline`'s neuro-AI step, `#extrapolation`.
- Sabrina reworks Part I over the weekend of 2026-10-24; merge her version.
- Sabrina, 2026-10-25: write the keywords and spoken text for every slide before `#paper`, slide by slide in
  discussion with her, she writing the text and the slide following it.
  - Her idea (2026-10-25), undecided: before `#tradition`, a short slide on the first deep networks used as models
    (Yamins et al. 2014's) and the images they were tested on.
- Missing material: a moving-scene clip for `#next`.
- Known defects: segmentation colors differ between Backup slides (the components, on `#rollout`, use the ADE20K
  palette, `#extrapolation` canvit-pytorch's `LABEL_COLORS`); the main deck's maps share experiments/segmentation_colors.py
  (checked 2026-10-25); amber means every prior model on some
  slides and AME alone on `#timeline`.
- For the paper's camera-ready (2026-11-06), not the talk: prepared on the local branch `paper-camera-ready-fixes`
  (from `main`, not pushed, paper builds with no Type 3 fonts): AME's authors with their diacritics, AdaGlimpse dated
  2024, Raugel et al. dated 2026 (its ICLR edition), and the "order of magnitude more" clause removed from the
  abstract of the paper, the README and the project page. Left open: GAE's pages (16137–16146 in the bib; IEEE's
  record gives 16117–16126, possibly a CVF/IEEE pagination difference) and DINOv3's month (`feb` on OpenReview, the
  TMLR PDF header says 04/2026).
