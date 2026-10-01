[A Claude Code subagent's cold read of the deck as of commit c7439d8, 2026-10-01, as a MAIN 2026 audience member. Its corrections to the notes were checked against the data and applied the same day (see git log); its open points are in PLAN.md. Not reviewed by the authors.]

# Cold review: CanViT talk, MAIN 2026

Reviewer stance: a member of the MAIN audience (neuroscientists, cognitive scientists, ML researchers) reading the deck cold.

What was reviewed, on 2026-10-01:
- Slide text and speaker notes: `/Users/yberreby/code/CanViT/site/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/index.html` at commit c7439d8 (18:53). The deck changed while I read it: c7439d8 added the GFNet and AME ablation panel to "What's in an active-vision model?". I judged that panel from its notes only.
- Screenshots: `/Users/yberreby/code/CanViT/site/.screens/all/*.png`, taken 18:44 to 18:46, before commits a69e2df, cffd71c and c7439d8. They lag the HTML in places; for example, the dashed history line reads "frozen + linear decoding" in the screenshots and "frozen self-supervised" in the code. For "The wide gap between passive and active computer vision" and "Benchmark results on ADE20K and ImageNet-1k", the screenshots show only the last build (ADE20K), so I could not see the ImageNet half in the current version. The "Canvas Vision Transformer architecture" screenshot was captured mid-animation (two of the three writes).
- Paper: `~/code/CanViT-Toward-AVFMs/latex/CanViT_Toward_AVFMs.tex` (abstract, introduction, experiments and results, conclusion, the pretraining ablations appendix), plus `/Users/yberreby/code/CanViT/site/assets/paper/data_macros.json` and `in1k_clf_frozen.json` to check the numbers that are spoken.
- The talk's guide: "The story", "Key ideas", "Questions to anticipate" in `.../AGENTS.md`.

Timing, computed from the notes: the main deck's notes total about 3,770 words, about 24 to 27 minutes at 140 to 160 words per minute, before animation pauses. About 1,300 of those words (9 to 10 minutes) come before CanViT appears in "CanViT in action". The closing neuro-AI slide gets 174 words (about 1 minute 15 seconds). If the 30 minutes include questions, the talk is about 5 minutes too long.

---

## Ranked summary of the most important findings

In order of importance:

- **The neuroscience payoff is thin, and the bridge to it is never stated.** The talk opens with a field that evaluates models against the brain ("Better at recognizing objects, better at predicting IT", Brain-Score). It never puts CanViT next to neural or behavioral data, and it ends with three questions and no result, in about a minute. One argument would carry the audience: active models were too weak to serve as candidate models of vision, so neuro-AI used passive ones; CanViT is the first active model good enough to compare. The pieces exist ("These models are among today's best models of visual cortex. But they're passive"; "Active computer vision was not nearly as smart"), but the talk never says that sentence. The audience's own vocabulary is also missing: "retinotopic" and "spatiotopic" (the paper's own words for the two streams), "amodal completion", "boundary extension", "filling-in", human coarse-to-fine findings, and uncertainty-driven fixation in humans. `sources/concepts.md` has these prepared and the talk uses none of them.
- **The notes contradict themselves on which policies were seen in training.** "Policy agnosticism" says "none of the policies I'll show you later was ever seen in training". Three slides later, "Viewing policies" introduces R-IID as "exactly how it was pretrained" and F-IID as "the other half of pretraining".
- **The policy message has a gap the audience will feel.** "What's in an active-vision model?" says "it doesn't matter what policy I have if I don't have a good observer" and shows random viewpoints doing nearly as well as learned ones. "Accuracy by viewing policy" then shows that the policy is worth about 29 ImageNet points at one glimpse (C2F 76.8 against R-IID 47.7) and keeps ADE20K results 3 points apart after 21 glimpses. The missing step: once the observer is good, the policy becomes measurable, and it matters most at small budgets, which is where active vision is meant to pay off. The talk also reads the GFNet and AME ablations as proof of poor observers without naming the alternatives (weak RL training; classification of centered objects needing little search).
- **"Training cost" argues one thing and plots another.** The notes argue that "a model that is cheap to train and adapt ... puts it within reach of a lab without a cluster". The chart plots pretraining compute, where CanViT-B (55 EFLOPs) sits above AdaptiveNN (about 23) and far above AME (under 2). The cost a lab actually pays to adapt CanViT (0.16 EFLOPs for the ADE20K probe, 4.76 for ImageNet LP-FT, per AGENTS.md "Checked facts") is not shown. The teacher accounting is also inconsistent within the slide; see the claims section.
- **"The wide gap" overclaims.** It says "it was terrible" and "worse accuracy and worse efficiency at every level". Later, "Active ADE20K segmentation" concedes "(We make no efficiency claim on ImageNet classification.)", and "Training cost" shows AdaptiveNN's 82.2 above CanViT-B frozen at 81.1.
- **"Metacognition: it knows what it doesn't know" is asserted, not shown.** The talk shows no calibration (whether high entropy marks where the segmentation is wrong). The entropy also comes from a linear probe trained on ADE20K labels. Cognitive scientists in the room will press on the term.
- **The top-down feedback evidence on the closing slide is confounded, and the chart stresses the wrong part.** C2F against F2C differs in write order as well as in reads. The paper's no-reads ablation, which drops patch reconstruction by 6.6% and CLS by 8.0% relative cosine similarity, is the more direct evidence, and the talk does not use it. On a 0 to 50 axis, the chart's most visible feature is the early gap, which comes from different inputs. The order effect is the 3-point difference at 21 glimpses.
- **Several spoken numbers are mis-scoped.** "38.5" on "Active ADE20K segmentation" is the 32×32 canvas, while "Accuracy by viewing policy" shows 39.6 at one glimpse (64×64). "Our best policy" (EG-C2F) ends below C2F (45.7 against 45.9). "The random policies ... end about three points lower" is wrong for F-IID (1.0 lower). "81.1 percent frozen ... above every active model" can be heard as including 81.1.
- **The talk's coordinates disagree with the code it shows.** The talk says viewpoints are "a position, x and y". The Quickstart code writes `centers=torch.tensor([[0.0, -0.4]])` for "zoomed in on the left of the street", which is (row, col).

---

## What I would remember, and which key ideas did not come through

What I would remember, in my own words:

- **Two corners of a table.** A passive model sees two patches with a question mark between them; CanViT, after both glimpses, paints a continuous table it never looked at. This is the most memorable image in the talk, because "Integrating multiple viewpoints into a coherent understanding" sets it up and "Extrapolation to unobserved regions" pays it off.
- **The model keeps a map of the scene and fills it in glimpse by glimpse.** Wipe the map before each glimpse and only the last object survives (sign, people, bicycle against bicycle alone).
- **"No policy can make up for a poor observer."** The authors built the observer first, trained it on random glimpses to reproduce DINOv3's view of the whole scene, and let anyone bring a policy.
- **A big jump on segmentation.** About 46% mIoU against about 28% for the best earlier active model, close to the passive teacher it learned from (if I caught the parenthetical "47.2" on the results slide; the chart does not show it).
- **Zooming lets it see a small TV** that a coarse full view misses, and looking at the full view twice does not help ("it's the zoom, not the extra step").

Intended key ideas (AGENTS.md "Key ideas") that did not come through, or came through weakly:

- **Trans-saccadic integration.** The term appears only as a card title on the closing slide. On "A persistent, evolving understanding of the scene" the notes say "That's integration across glimpses, what the eye does across saccades", which assigns integration to the eye.
- **The canvas as spatiotopic memory beside a retinotopic backbone.** These words never appear; the notes paraphrase spatiotopy ("each staying at its place in the scene whatever the glimpse"). For this audience, the established terms are the fastest route to understanding, and the paper's abstract uses them.
- **Inferring what was not seen, in the audience's words.** "Amodal completion", "boundary extension" and "filling-in" never appear. The "Integrating multiple viewpoints" slide still carries an `XXX` for its cognitive-science citation.
- **Order matters.** It is said on "Viewing policies" ("any difference comes from the order"), but "Accuracy by viewing policy", where the data is, never states the C2F against F2C comparison at 21 glimpses. The point reaches the audience only on the closing slide, as a question.
- **Foundation model, cheap to use.** "Quickstart" shows "easy to use". "Training cost" muddies "cheap", because CanViT is not the cheapest model on that chart.
- **Metacognition.** It is named, but without the scope `sources/concepts.md` asks for ("the slide should pick one [definition] aloud").

---

## Skipped steps, assumed knowledge, missing "why"

By slide title:

- **"Computational models of biological visual processing"**: the notes point to the Brain-Score inset ("past about 70 percent on ImageNet ... the relation flattens") and drop it. The audience will wonder what follows: if accuracy no longer buys brain predictivity, why chase active-vision accuracy? Either tie it forward (passive accuracy saturated as a route to the brain; what is missing may be how the model samples) or cut the sentence.
- **"Integrating multiple viewpoints into a coherent understanding"**: the panel shows a decoded probability map, "DINOv3, each glimpse alone", "probability of table", before the next slide explains linear decoding. "DINOv3, more on it in a minute" is a forward reference that costs attention on the slide meant to hook the cognitive scientists.
- **"DINOv3 feature maps"**: "These models are among today's best models of visual cortex" has no citation, and it is stronger than the opening slide's "among the models used to model brain function".
- **"A brief history of deep active computer vision"**: seven ML models, each with venue and mechanism, in 249 words and 8 clicks. The slide shows "where to look, learned by reinforcement learning" without saying why RL matters to the argument. The notes come closest with "Most of these papers put their novelty into choosing where to look", but the reason RL is a problem (hard to train, ties the observer to one policy) is said only in the paper's introduction ("freeing active-vision pretraining from the complexity of RL").
- **"The wide gap between passive and active computer vision"**: "The whole point was supposed to be efficiency" names efficiency without saying why efficiency matters. AGENTS.md calls efficiency "a link in this chain": knowledge comes with scale, scale needs cheap training, and active models had neither. That chain is never spoken anywhere in the deck, so "Training cost" later has no setup.
- **"What's in an active-vision model?"**: the causal diagnosis, "Why was it stuck? I think people were focusing on the wrong thing", rests on two ablations where random viewpoints cost about a point. Other explanations fit the same numbers: the learned policies were weak, or the classification tasks (centered objects, SUN360 scenes) need little search. The slide also never says what changes once the observer is good; see the ranked summary.
- **"CanViT in action"**: "Outside it there's nothing, not even a blur, and there's a world of difference between seeing something blurry and seeing nothing at all." The previous section opened with "sharp central vision and blurry peripheral vision". The audience needs the reason: is no periphery a harder setting chosen on purpose, a simplification, or something the full-scene glimpse stands in for? The notes never say. Scale (zoom) also has no human counterpart, and the talk does not mention its biological reading.
- **"Canvas Vision Transformer architecture"**: "It is 32 by 32 in training, and I can make it finer, 64 by 64, at test time, without retraining." The ADE20K probe is trained separately per canvas resolution (paper §6), so "without retraining" applies to CanViT only.
- **"Passive-to-active dense latent distillation"**: the "why a teacher" question is answered for ML ("the identity teacher"). A neuroscientist would ask what plays the teacher's role in the brain, and the talk has no answer. `sources/concepts.md` lists trans-saccadic feature prediction (learning how a peripheral object will look in the fovea) as the human counterpart.
- **"A persistent, evolving understanding of the scene"**: "keeps a probability around 0.4 to 0.45 with the memory, against 0.04 when it is reset" has no reference value for the object right after its own glimpse, so the audience cannot tell how much the memory loses. AGENTS.md has the number (0.683 kept against 0.038 among objects recognized after their glimpse). Also unsaid, and strong: resetting is as bad as never having looked (0.038 when the object's glimpse is skipped).
- **"Extrapolation to unobserved regions"**: "65 percent of the unseen middle" has no baseline (for example, how often the middle shares the end's class by scene statistics alone, or a naive fill between the two glimpses). In the single-glimpse columns ("CanViT, far end", "CanViT, near end"), the screenshot draws both glimpse boxes identically, so on screen you cannot tell which glimpse was taken.
- **"Spatial coverage and perception of detail"**: "how much of the object the model gets right, from 18 to 32 percent" is pixel recall (OUTLINE.md), which ignores false positives. An ML researcher will ask about precision or IoU.
- **"Viewing policies"**: "RFS ... separates thinking longer from seeing more", and then no slide interprets what RFS showed: on ADE20K at 64×64, it gains 1.5 points by the second glimpse (39.6 to 41.1), then falls to 39.3. The neuroscience reading ("more evidence and more processing", AGENTS.md "Yohaï's words") is never made.
- **"Uncertainty-based viewpoint selection"**: the four quadrant numbers on screen (0.33, 0.34, 0.57, 0.68) have no unit; the legend says bits. "AME, in the timeline, did something similar with its attention maps" concedes similarity without the distinction AGENTS.md prepared (the entropy of predicted classes against the entropy of attention).
- **"Benchmark results on ADE20K and ImageNet-1k"**: the chart puts CanViT-B at 45.9, below a dashed passive line at 51.8. The fair comparison, the teacher at 47.2 under the same probe, is spoken only, in parentheses. What the audience sees (6 points short) and what they hear (1.3 points short) disagree.
- **"Training cost"**: exaflops are meaningless to most of this room. The notes do give the conversion for CanViT ("166 hours on one H100"), but not for the others.
- **"Toward action-aware neuro-AI research on vision"**: each card is a question. None says what a first experiment would measure or what human result it would be compared with.

---

## What convinces, what does not, where each part of the audience drifts or objects

Convincing:

- **"Integrating multiple viewpoints" with "Extrapolation to unobserved regions"**: same scene, setup then payoff. The notes state the selection ("This one is a clean example") and the spill ("46 percent of the other pixels in between get the object's label too"). That candor strengthens the claim.
- **"A persistent, evolving understanding of the scene"**: same glimpses, memory kept against wiped. A clean manipulation that both halves of the audience will read as causal.
- **"Spatial coverage and perception of detail"**: it carries its own control ("showing the whole scene a second time barely helps, 19 percent: it's the zoom, not the extra step").
- **"Passive-to-active dense latent distillation"**: the unrolled step (glimpse, prediction, MSE, target) makes the objective concrete, and the closing line ("Reconstructing pixels from partial views would be distillation too, from the identity teacher") pre-empts the "cheating" objection.
- **"CanViT in action"**: the gradual build (scene, viewpoint, glimpse, model, canvas) is well paced.
- **"Active ADE20K segmentation"** convinces the ML researchers: "Even our worst policy gets better segmentation at lower cost than the best of them" is visible at a glance.

Less convincing:

- **"What's in an active-vision model?"**: a slogan backed by two small ablations, with no alternatives considered (see above).
- **"Uncertainty-based viewpoint selection"**: "metacognition" without calibration.
- **"Training cost"**: the chart does not show what the notes argue.
- **"Toward action-aware neuro-AI research on vision"**: no data. The top-down card is confounded. The scanpath card sets a dense Yarbus record beside a single quadtree box at "t = 5" (the screenshot shows one box and the quadrant grid, no path), which invites the reaction that the policy looks nothing like a scanpath.

Where a neuroscientist would lose interest:

- **"A brief history of deep active computer vision"** and the ImageNet half of **"The wide gap"**: about 4 minutes of ML model names, venues and benchmark lines.
- **"Benchmark results"**, **"Accuracy by viewing policy"**, **"Training cost"**, **"Active ADE20K segmentation"**: four benchmark slides in a row, two of them in EFLOPs and GFLOPs, right before the slide meant for them. The talk spends about 4 minutes on results and cost and about 1 minute on the neuro-AI payoff.

Where an ML researcher would object:

- **"Training cost"**: the teacher's own training (DINOv3) is excluded, while AME's label alone carries "+ pretrained ViT-L". CanViT-B is not labeled "+ DINOv3 teacher", and AdaGlimpse is not labeled "+ DeiT III teacher".
- **"The wide gap"**: "worse efficiency at every level" ignores GFNet's and AdaptiveNN's efficiency on ImageNet, which the authors concede elsewhere.
- **"Benchmark results"**: the history chart's 51.8 uses DINOv3's own probing protocol while CanViT-B's 45.9 uses CanViT's, so the chart compares two protocols.
- **"Passive-to-active dense latent distillation"**: "Here is a training step, unrolled in time" runs to 21 glimpses, while the next slide says training uses "four on average". Under the stated process (chunks of 2, stop with probability 0.5 after each), a rollout reaches 21 glimpses about once in a thousand; and with truncated backpropagation through time (K = 2), the gradient does not flow across 21 glimpses.
- **"Spatial coverage and perception of detail"**: DINOv3 at 128 px is a handicapped passive baseline. The point stands as a matched-input comparison, but a passive model's answer is more pixels; the backup "CanViT-B and its DINOv3 teacher" covers that only if asked.
- **LookWhere** (NeurIPS 2025, 83.0 on ImageNet-1k and 44.6 on ADE20K, per AGENTS.md) is absent from the timeline and the charts, and an ML researcher who knows it will raise it.
- **"A general-purpose observer lets you use any policy"**: six hand-designed policies were tested, none learned and no human scanpath (AGENTS.md: "'Any policy' means the tested policies").

---

## Questions I would ask at the end

Marked **[answered]** where a backup slide or the notes answer it, **[partly]** where they gesture at it, **[unanswered]** where nothing in the deck does.

- "Isn't this mostly DINOv3? How much of the 45.9 is the teacher?" **[partly]**: the backup "CanViT-B and its DINOv3 teacher" shows CanViT-B ahead at small FLOP budgets ("38.5 ... where DINOv3 at the same compute gets 33.2") and the teacher ahead at 512 px. The teacher's share is not isolated (AGENTS.md says so), and the deck never says it.
- "Have you compared CanViT's representations, or its memory, with neural recordings or human behavior?" **[unanswered]**. The opening frames the field by this test.
- "Change blindness says people keep little across saccades. Why a dense, scene-wide memory?" **[unanswered]**. AGENTS.md: "Not yet read: do not answer from memory."
- "The model is told each glimpse's position and zoom. The brain has to compute it (corollary discharge, remapping). What does CanViT say about that?" **[unanswered]** on slides; AGENTS.md has the verbal answer.
- "Why no periphery, when you opened with blurry peripheral vision? And what is 'zoom' for a human?" **[unanswered]**.
- "Is the entropy calibrated? Do high-entropy regions predict errors?" **[unanswered]**. The rollout widget already has "Uncertainty" and "Correctness" readouts, so "CanViT in action" or "Live demo" could answer it on the spot.
- "How do you know the C2F against F2C difference is top-down feedback and not later writes overwriting earlier ones?" **[unanswered]** in the deck; the paper's no-reads ablation (appendix, "Frequency and directionality of canvas–backbone interaction") answers part of it.
- "Have you put human scanpaths through CanViT?" **[unanswered]**. The closing slide proposes it; whether it has been tried is not said.
- "Learned policies? RL on top?" **[unanswered]** on slides; AGENTS.md lists it as future work, with unpublished results pending the authors' decision.
- "Video, moving objects, forgetting?" **[partly]**: "It has limits: static scenes" on the closing slide.
- "How does it compare with LookWhere?" **[unanswered]** on slides; numbers in AGENTS.md.
- "Is ImageNet-21k doing the work?" **[unanswered]** on slides; AGENTS.md has the ImageNet-1k-only control (43.4 against 45.9).
- "What does it cost me to use it, and on what hardware?" **[answered]** in part by "Quickstart" ("it runs on a laptop"). The adaptation cost is not given.
- "Is learning from a model that sees everything cheating?" **[answered]** on "Passive-to-active dense latent distillation".
- "How long does the memory last?" **[partly]**: three glimpses on the memory slide; up to 21 on "Accuracy by viewing policy"; forgetting is not discussed.
- "Why does accuracy fall with the repeated full scene?" **[unanswered]**: the curve is shown and never interpreted.

---

## One change per section that would most improve the talk

- **Vision, brains and machines.** On "Human vision is fundamentally active", put the components the notes already list ("sequential", "working memory", "top-down feedback", "planning") on screen as a build, in the three colors that "What's in an active-vision model?" and "Toward action-aware neuro-AI research on vision" use later. The audience gets a map that the closing slide then completes.
- **Knowledge of the world.** On "Integrating multiple viewpoints into a coherent understanding", name the concept in the audience's terms with its human result, resolving the slide's own `XXX`. One candidate from `sources/concepts.md`: foveal information extrapolated to the periphery only within an object's boundary, which can then be set against the table's measured spill on "Extrapolation to unobserved regions".
- **Active computer vision.** On "The wide gap", replace "worse accuracy and worse efficiency at every level" and "it was terrible" with a scoped statement: dramatically worse on dense prediction, behind passive models on classification. Add the missing link: "and no active model was good enough to stand in for the visual system". That link carries both the efficiency chain and the neuroscience case.
- **CanViT.** On "Canvas Vision Transformer architecture", label the two streams "retinotopic" and "spatiotopic" on screen, and keep "top-down feedback" for the reads. In the same section, fix "none of the policies I'll show you later was ever seen in training" on "Policy agnosticism" to name the four inference-only policies (C2F, F2C, EG-C2F, RFS).
- **What it does.** On "Uncertainty-based viewpoint selection", put the correctness map from the same rollout beside the entropy map, so "it knows what it doesn't know" is shown rather than asserted. Pick one definition of metacognition aloud, or say "its own uncertainty about what is where".
- **Results.** Rebuild "Training cost" around what a lab pays: CanViT-B's adaptation cost (0.16 EFLOPs for the ADE20K probe, 4.76 for ImageNet LP-FT) against each prior model's full training for that task, with the pretraining as a one-time shared cost. Alternatively, cut the slide and fold one sentence into "Active ADE20K segmentation". In either case, label every teacher or inherited model the same way. A smaller fix in the same section: draw the teacher's 47.2 on "Benchmark results".
- **Closing.** On "Toward action-aware neuro-AI research on vision", give each card a line naming the human finding it would be compared with (human coarse-to-fine against fine-to-coarse categorization, uncertainty-driven fixation in a shape-learning task, both in `sources/concepts.md`). For the top-down card, cite the no-reads ablation, or show the C2F and F2C endpoints at 21 glimpses, where the viewpoint sets are identical, in place of the whole curves.

---

## Claims that look wrong, overstated or inconsistent

Between slides or notes:

- "Policy agnosticism" notes: **"none of the policies I'll show you later was ever seen in training"**. This contradicts "Viewing policies": **"R-IID: random places and zoom levels, exactly how it was pretrained"** and **"F-IID ... that's the other half of pretraining"**.
- "The wide gap" notes: **"worse accuracy and worse efficiency at every level"** and **"it was terrible"**. "Active ADE20K segmentation" says **"(We make no efficiency claim on ImageNet classification.)"**; AGENTS.md records the authors' own concession ("We do not expect CanViT to lead on accuracy-efficiency in classification"); and on ImageNet, AdaptiveNN's 82.2 is above CanViT-B frozen (81.1, on "Training cost").
- "Benchmark results" notes: **"81.1 percent frozen, with a linear layer, and 84.5 after fine-tuning ...: above every active model."** Only 84.5 is above every active model; 81.1 is below AdaptiveNN's 82.2.
- "Training cost" notes: **"55 exaflops, that is 166 hours on one H100 plus 8 hours to compute the teacher's features"**, then **"None of this counts the teachers"**, and on screen **"Teachers and pretrained weights are not counted, on any side."** The teacher's inference is counted; its training is not. Say "the teachers' own training is not counted". On screen, only **"AME 27.6 + pretrained ViT-L"** carries an inherited-model label.
- "Active ADE20K segmentation" notes: **"And our best policy, guided by its own uncertainty"**. EG-C2F ends at 45.7 at 64×64 against C2F's 45.9 (macros `adeECTFCSixFourTTwenty`, `adeCTFCSixFourTTwenty`). It is best at small budgets, and the 45.9 headline is C2F. Say "our most efficient policy".
- "Active ADE20K segmentation" notes: **"From the very first glimpse of the whole scene, 38.5 percent"**. 38.5 is the 32×32 canvas (`adeCTFTZero`). At 64×64 it is 39.6 (`adeCTFCSixFourTZero`), the value a viewer just saw at one glimpse on "Accuracy by viewing policy". The slide plots both canvases, so state the canvas.
- "Accuracy by viewing policy" notes: **"give them a few glimpses and they all end up together: by the end, within about a point of each other"**. F2C reaches 79.8 only at 21 glimpses, 1.3 below C2F's 81.1. At 10 glimpses, F2C is at 75.9, 5 points behind (`in1k_clf_frozen.json`).
- "Accuracy by viewing policy" notes: **"the random policies and fine-to-coarse end about three points lower"**. R-IID ends at 43.1 and F2C at 42.9 (about 3 lower), but F-IID ends at 44.9, 1.0 lower.
- "CanViT in action" notes: **"a position, x and y"**, and "Passive-to-active dense latent distillation" labels **"Glimpse (x, y, scale)"**. "Quickstart" shows `Viewpoint(centers=torch.tensor([[0.0, -0.4]]), ...)` under **"# A second glimpse: zoomed in on the left of the street"**, which only reads correctly as (row, col), the package's convention. A listener who learned "x, y" from the talk would misuse the API.
- "Computational models of biological visual processing" notes: **"vision foundation models are among the models used to model brain function"**. "DINOv3 feature maps" notes: **"These models are among today's best models of visual cortex."** The second is a stronger claim, with no citation on screen.

Overstated or loosely worded:

- "What's in an active-vision model?" on screen: **"A general-purpose observer lets you use any policy."** "Live demo" notes: **"Same observer, any policy."** Six hand-designed policies were tested. AGENTS.md: "'Any policy' means the tested policies, horizons and resolutions."
- "What's in an active-vision model?" notes: **"it doesn't matter what policy I have if I don't have a good observer"**. The literal reading is defensible (the observer bounds accuracy). Followed by the random-against-learned ablations, the audience will hear "policies don't matter", which "Accuracy by viewing policy" then contradicts ("for segmentation, it keeps mattering").
- "Uncertainty-based viewpoint selection" notes: **"A kind of metacognition: it knows what it doesn't know."** No calibration is shown, and the entropy is that of a supervised linear probe's class distribution.
- "Toward action-aware neuro-AI research on vision" notes: **"Top-down feedback: ... Coarse-to-fine against fine-to-coarse: by the end, the same views, in another order, and a different result."** The paper words this result as "the importance of contextually informed processing". Order effects can also come from the write sequence. The no-reads ablation is the evidence specific to top-down reads.
- "Passive-to-active dense latent distillation" notes: **"Here is a training step, unrolled in time"** for a 21-glimpse rollout, against "four on average" on the next slide. Say "what the training objective scores, glimpse by glimpse, on a long rollout".
- "Canvas Vision Transformer architecture" notes: **"the canvas never goes through a learned layer"**, and "Canvas Attention" (backup): **"The canvas never goes through a learned layer: no MLP, no self-attention, no projection."** The canvas-side LayerNorms (`CanvasAttention.ln_q` and `ln_kv` in `/Users/yberreby/code/CanViT/canvit-pytorch/canvit_pytorch/model/attention/base.py`) are `nn.LayerNorm` with the default learnable gain and bias. The paper's wording ("never see an MLP, self-attention or a projection") is the accurate one.
- "Canvas Vision Transformer architecture" notes: **"I can make it finer, 64 by 64, at test time, without retraining"**. The linear probe is retrained per canvas resolution.
- "A persistent, evolving understanding of the scene" notes: **"what the eye does across saccades"**. The brain does the integrating.
- "Spatial coverage and perception of detail" notes: **"how much of the object the model gets right"** is recall of the object's pixels (OUTLINE.md), which a zoomed glimpse could raise by over-predicting. Name the metric, or report IoU.

Checked and consistent (so they need no change): 18 points above AME (45.9 − 27.6 = 18.3); "20 times fewer FLOPs" (311.5 / 15.8 = 19.7); "3.2 bits, as unsure as choosing among about nine equally likely classes" (2^3.2 = 9.2); "at most 7.2 bits for all 150" (log2 150 = 7.23); the distillation slide's MSE "to about a third" (0.31 / 0.87 = 0.36); "a quarter of a percent" for the smallest glimpse (0.05² = 0.0025); the cost figures against AGENTS.md "Checked facts"; and the distillation slide's displayed viewpoints, which read correctly as (x, y) against the glimpses shown.
