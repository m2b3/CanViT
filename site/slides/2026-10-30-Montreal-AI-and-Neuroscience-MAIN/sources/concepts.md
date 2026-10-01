# Concepts the audience already uses: sources for the talk

[Compiled 2026-10-01 by a Claude Code research subagent; read by Claude Code, not by the authors.] Established terms
from cognitive science, psychology and neuroscience (and a few from machine learning) that name what CanViT does, for
titles, on-slide keywords and notes. `cognition.md` covers boundary extension, scene schemas, contextual guidance,
amodal completion and perception as inference; this file extends it and does not repeat it.

How to read the entries. "Read:" says what was read this session: **full** (the whole text), **part** (the named
sections), **abstract**, or **metadata** (title, authors, venue only). Metadata come from Crossref unless stated;
abstracts from Europe PMC (PubMed), Crossref or OpenAlex as stated; arXiv records from the arXiv API. "Zotero" is a
PDF in the user's library. Quotes are verbatim. A work marked UNVERIFIED was not checked in its own text or record.
CanViT facts come from the paper (introduction, sections 3 to 7 and the appendix on viewing policies, read
2026-10-01), the code (`canvit_pytorch.policies.entropy.EntropyGuidedC2F`) and `AGENTS.md` ("Checked facts").

Wording used throughout [Yohaï, 2026-10-01]: CanViT sees only its glimpses; it infers, guesses or completes what it did
not sample. The canvas holds; a policy chooses (EG-C2F chooses where the segmentation read out from the canvas is least
certain).

## Recommendation

Ranked, most useful to this audience first. The reason is what the concept lets the talk say that it cannot say now.

- **Trans-saccadic integration and memory.** The audience's own name for the canvas's job, and the field's history
  (pixel-level fusion rejected, abstract accumulation accepted) places the canvas precisely: abstract like human
  trans-saccadic memory, dense and spatially exact unlike it. Carries "Integrating multiple viewpoints into a coherent
  understanding", "The Canvas Vision Transformer" and "Resetting the memory".
- **Coarse-to-fine, in perception and in eye movements.** Human experiments show the same filtered scene images
  categorized faster coarse-to-fine than fine-to-coarse (Musel et al. 2012; Kauffmann et al. 2015): the human
  counterpart of C2F against F2C with the same viewpoints. "Order matters" then rests on a human finding as well as on
  the model's.
- **Looking where you are uncertain (local uncertainty, active sensing).** In a shape-learning task, Renninger,
  Verghese & Coughlan 2007 found people's next fixations predicted at least as well by a local rule (fixate where
  uncertainty is highest) as by maximizing total information: EG-C2F's rule, in humans. Parietal neurons encode the
  uncertainty reduction a saccade is expected to bring (Foley et al. 2017).
- **Metacognition: monitoring and control, and where the term stops.** The author's keyword. The field's own
  definitions (Fleming 2024; Pouget, Drugowitsch & Kepecs 2016) decide which wording survives an expert's question:
  EG-C2F uses the model's uncertainty about the world (certainty, in their terms) to control sampling; whether that is
  metacognition depends on the definition, and the slide should pick one aloud.
- **Spatiotopic memory, which the brain may not have.** The paper's own words. Current reviews hold that visual maps
  are retinotopic and that the brain approximates spatiotopy by remapping a few attended targets (Cavanagh & Melcher
  2026); CanViT builds the dense scene-coordinate map instead, with the geometry given. Spatial view cells and the
  entorhinal map of visual space anchor "cognitive map" without claiming a brain area.
- **Scene memory as objects bound to locations; change blindness.** Needed to answer the anticipated change-blindness
  question from the literature, read: change blindness rules out a picture-like memory and leaves open an
  accumulating abstract one (Hollingworth 2008).
- **Perceptual filling-in and completion.** The precise vocabulary for "completes", with a human result to compare
  against the table's spill: foveal information is extrapolated to the periphery only within an object's boundary
  (Toscani, Gegenfurtner & Valsecchi 2017).
- **Trans-saccadic feature prediction.** People learn from experience how a peripheral object will look in the fovea
  (Herwig & Schneider 2014): the human counterpart of learning to predict a high-resolution teacher from glimpses.
  Serves the distillation slide and the paper's proposed active-to-active self-distillation.
- **Vision at a glance, then recurrent refinement.** Gist for the single full-scene glimpse; feedforward sweep and
  recurrent processing for RFS. Notes.
- **Priority map, inhibition of return, Bayesian surprise, active inference.** Precise names for parts of EG-C2F and
  for policies not yet tried. Notes.
- **Words the paper uses that this audience hears differently.** "Cognitive map", "workspace", "mental image",
  "working memory": what each evokes and the scope to state.

Recommended on-slide keywords: "trans-saccadic integration" (canvas), "coarse to fine" (policies, with the human
CtF/FtC result), "the model's own uncertainty about what's where" (EG-C2F; "metacognition" only with the scope below),
"spatiotopic / retinotopic" (architecture), "filling-in" (extrapolation). Everything else belongs in the notes.

## Trans-saccadic integration and memory

- **What it is:** retaining what was seen before an eye movement and combining it with what is seen after, so that
  processing of the scene continues instead of starting anew at each fixation.
- **Sources:**
  - The rejected idea, an "integrative visual buffer" fusing successive snapshots in spatiotopic coordinates: Jonides
    J., Irwin D.E., Yantis S. 1982. Integrating visual information from successive fixations. *Science*
    215(4529):192–194. doi:10.1126/science.7053571. Read: abstract (Europe PMC). Its effect was later traced to
    phosphor persistence on the display (as reviewed by Stewart et al. 2020, read; the 1983 correction itself not
    read). O'Regan J.K., Lévy-Schoen A. 1983. Integrating visual information from successive fixations: does
    trans-saccadic fusion exist? *Vision Research* 23(8):765–768. doi:10.1016/0042-6989(83)90198-0. Read: abstract.
    Irwin D.E., Yantis S., Jonides J. 1983. Evidence against visual integration across saccadic eye movements.
    *Perception & Psychophysics* 34(1):49–57. doi:10.3758/BF03205895. Read: its experiment as described in Irwin 1991
    (pp. 421–422); own text not read.
  - The replacement: Irwin D.E. 1991. Information integration across saccadic eye movements. *Cognitive Psychology*
    23(3):420–456. doi:10.1016/0010-0285(91)90015-G. Read: part (introduction and general discussion, Zotero). Irwin
    D.E. 1992. Memory for position and identity across eye movements. *J. Exp. Psychol. Learn. Mem. Cogn.*
    18(2):307–317. doi:10.1037/0278-7393.18.2.307. Read: abstract (OpenAlex). Hayhoe M., Lachter J., Feldman J. 1991.
    Integration of form across saccadic eye movements. *Perception* 20(3):393–402. doi:10.1068/p200393. Read: abstract.
  - Accumulation over a scene: Melcher D. 2001. Persistence of visual memory for scenes. *Nature* 412(6845):401.
    doi:10.1038/35086646. Read: full (one page, Zotero); the paper cites it for integration across time in visual
    working memory. Melcher D. 2006. Accumulation and persistence of memory for natural scenes. *Journal of Vision*
    6(1):2. doi:10.1167/6.1.2. Read: abstract.
  - Reliability-weighted integration of peripheral and foveal views: Ganmor E., Landy M.S., Simoncelli E.P. 2015.
    Near-optimal integration of orientation information across saccades. *J. Vis.* 15(16):8. doi:10.1167/15.16.8.
    Wolf C., Schütz A.C. 2015. Trans-saccadic integration of peripheral and foveal feature information is close to
    optimal. *J. Vis.* 15(16):1. doi:10.1167/15.16.1. Oostwoud Wijdenes L., Marshall L., Bays P.M. 2015. Evidence for
    optimal integration of visual feature representations across saccades. *J. Neurosci.* 35(28):10146–10153.
    doi:10.1523/JNEUROSCI.1040-15.2015. Read: abstracts.
  - Limits: Kong G., Kroell L.M., Schneegans S., Aagten-Murphy D., Bays P.M. 2021. Transsaccadic integration relies on
    a limited memory resource. *J. Vis.* 21(5):24. doi:10.1167/jov.21.5.24. Stewart E.E.M., Schütz A.C. 2018. Optimal
    trans-saccadic integration relies on visual working memory. *Vision Research* 153:70–81.
    doi:10.1016/j.visres.2018.10.002. Kleene N.J., Michel M.M. 2018. The capacity of trans-saccadic memory in visual
    search. *Psychological Review* 125(3):391–408. doi:10.1037/rev0000099. Read: abstracts.
  - Reviews, oldest first: Melcher D., Colby C.L. 2008. Trans-saccadic perception. *Trends Cogn. Sci.*
    12(12):466–473. doi:10.1016/j.tics.2008.09.003. Read: abstract only (its "five principles" not reachable). Prime
    S.L., Vesia M., Crawford J.D. 2011. Cortical mechanisms for trans-saccadic memory and integration of multiple
    object features. *Phil. Trans. R. Soc. B* 366(1564):540–553. doi:10.1098/rstb.2010.0184. Read: abstract.
    Aagten-Murphy D., Bays P.M. 2019. Functions of memory across saccadic eye movements. *Curr. Top. Behav. Neurosci.*
    41:155–183 (online 2018). doi:10.1007/7854_2018_66. Read: abstract. Stewart E.E.M., Valsecchi M., Schütz A.C. 2020.
    A review of interactions between peripheral and foveal vision. *J. Vis.* 20(12):2. doi:10.1167/jov.20.12.2. Read:
    part (extrapolation and trans-saccadic integration sections, PMC full text). Baltaretu B.R., Crawford J.D. 2026.
    Cortical mechanisms for transsaccadic vision: extrinsic and intrinsic feature updating. *Neurosci. Biobehav.
    Rev.* 185:106644. doi:10.1016/j.neubiorev.2026.106644. Read: abstract. Higgins & Rayner 2015 is in `cognition.md`.
  - Naturalistic scenes: Choi Y.M., Chiu T.-Y., Ferreira J., Golomb J.D. 2025. Maintaining visual stability in
    naturalistic scenes: the roles of trans-saccadic memory and default assumptions. *Cognition* 262:106165.
    doi:10.1016/j.cognition.2025.106165. Read: abstract.
- **Quotes:** the rejected view, "the visible contents of successive fixations are superimposed in memory according to
  their spatiotopic or environmental coordinates ... analogous to overlaying transparencies on an overhead projector"
  (Irwin 1991, p. 421). The finding: "transsaccadic memory is an undetailed, limited-capacity, long-lasting memory
  that is not strictly tied to absolute spatial position" (Irwin 1991, abstract). Accumulation: "Memory of each scene
  continued to accumulate over repeated viewings as though the scene had never been out of sight" (Melcher 2001).
  Function: memory contributes to "bridging the gap in input so that visual processing does not have to start anew"
  (Aagten-Murphy & Bays 2019, abstract).
- **CanViT, analogous:** the canvas carries what each glimpse brought into the processing of the next and accumulates
  it over the rollout. It stores features ("what you saw and what you understood", in Yohaï's words), the kind of
  abstract code Irwin found, not fused pixels. Reads condition the next glimpse's processing on what was seen, as a
  presaccadic preview conditions postsaccadic processing (next entry). The memory-reset measurement (`AGENTS.md`,
  "Checked facts", "Memory reset") is the model's version of "does not have to start anew".
- **CanViT, differs:** a glimpse is a crop at a chosen zoom, with no periphery and no eye; each glimpse's position and
  scale are given, as if corollary discharge were exact and free. The canvas is dense and tied to absolute scene
  position (a grid of 1024-dimensional tokens, 32² in pretraining and up to 64² in the paper's evaluations), where
  human trans-saccadic memory holds three to four items and is "not strictly tied to absolute spatial position" (Irwin
  1991; Prime et al. 2011), though precise world-centred positions can be kept when the task requires it (Hayhoe et
  al. 1991). Human integration weights each view by its reliability; CanViT's write rule is learned and its weighting
  has not been measured. Scenes are static, so CanViT
  never has to decide whether the world changed between views (Stewart et al. 2020, "Integration's limits: causal
  inference"; Choi et al. 2025).
- **Slide:** "Integrating multiple viewpoints into a coherent understanding" (the human phenomenon, said); "The Canvas
  Vision Transformer" (the canvas named by its job); "Resetting the memory" ("does not have to start anew");
  "Toward action-aware neuro-AI research on vision" (what is kept across views, now comparable with trans-saccadic
  memory measures).
- **Keyword:** on the slide, as "integration across glimpses, as in trans-saccadic integration" (a glimpse is not a
  saccade). The fusion-versus-abstraction history in the notes.
- **Questions to expect:** "Human trans-saccadic memory holds three or four objects; the canvas holds a dense map.
  Which is CanViT a model of?" The canvas's capacity is a design choice; how much of it the model uses is measurable.
  "Najemnik & Geisler found that perfect integration across fixations barely helps search (below). Why a big
  memory?" Their task is finding one target; segmentation needs what is everywhere, and the reset measurement shows
  what the memory buys there.

## Coarse-to-fine, and why order matters

- **What it is:** scene recognition proceeds from coarse (low spatial frequency, global layout) to fine (high spatial
  frequency, detail); viewing behaviour shows a similar drift, from long saccades and short fixations to short saccades
  and long fixations.
- **Sources:**
  - Same images, two orders: Musel B., Chauvin A., Guyader N., Chokron S., Peyrin C. 2012. Is coarse-to-fine strategy
    sensitive to normal aging? *PLoS ONE* 7(6):e38493. doi:10.1371/journal.pone.0038493. Musel B., Kauffmann L.,
    Ramanoël S., Giavarini C., Guyader N., Chauvin A., Peyrin C. 2014. Coarse-to-fine categorization of visual scenes
    in scene-selective cortex. *J. Cogn. Neurosci.* 26(10):2287–2297. doi:10.1162/jocn_a_00643. Kauffmann L., Chauvin
    A., Guyader N., Peyrin C. 2015. Rapid scene categorization: role of spatial frequency order, accumulation mode and
    luminance contrast. *Vision Research* 107:49–57. doi:10.1016/j.visres.2014.11.013. Peyrin C., Michel C.M., Schwartz
    S., Thut G., Seghier M., Landis T., Marendaz C., Vuilleumier P. 2010. The neural substrates and timing of top–down
    processes during coarse-to-fine categorization of visual scenes: a combined fMRI and ERP study. *J. Cogn.
    Neurosci.* 22(12):2768–2780. doi:10.1162/jocn.2010.21424. Read: abstracts.
  - The framework: Schyns P.G., Oliva A. 1994. From blobs to boundary edges: evidence for time- and spatial-scale-
    dependent scene recognition. *Psychological Science* 5(4):195–200. doi:10.1111/j.1467-9280.1994.tb00500.x. Read:
    abstract (Crossref). Oliva A., Schyns P.G. 1997. Coarse blobs or fine edges? Evidence that information
    diagnosticity changes the perception of complex visual stimuli. *Cognitive Psychology* 34(1):72–107.
    doi:10.1006/cogp.1997.0667. Read: abstract. Bar M. 2003. A cortical mechanism for triggering top-down facilitation
    in visual object recognition. *J. Cogn. Neurosci.* 15(4):600–609. doi:10.1162/089892903321662976. Bar M. et al.
    2006. Top-down facilitation of visual recognition. *PNAS* 103(2):449–454. doi:10.1073/pnas.0507062103. Hochstein
    S., Ahissar M. 2002. View from the top: hierarchies and reverse hierarchies in the visual system. *Neuron*
    36(5):791–804. doi:10.1016/S0896-6273(02)01091-7 (title from Europe PMC; Crossref gives "View from the Top"). Hegdé
    J. 2008. Time course of visual perception: coarse-to-fine processing and beyond. *Progress in Neurobiology*
    84(4):405–439. doi:10.1016/j.pneurobio.2007.09.001. Read: abstracts. Navon 1977 is in `citations.md`; its abstract
    was not available this session.
  - Eye movements: Over E.A.B., Hooge I.T.C., Vlaskamp B.N.S., Erkelens C.J. 2007. Coarse-to-fine eye movement
    strategy in visual search. *Vision Research* 47(17):2272–2280. doi:10.1016/j.visres.2007.05.002. Read: abstract.
    Unema P.J.A., Pannasch S., Joos M., Velichkovsky B.M. 2005. Time course of information processing during scene
    perception: the relationship between saccade amplitude and fixation duration. *Visual Cognition* 12(3):473–494.
    doi:10.1080/13506280444000409. Read: abstract and introduction (Zotero).
  - The preview effect: Huber-Huber C., Buonocore A., Melcher D. 2021. The extrafoveal preview paradigm as a measure
    of predictive, active sampling in visual perception. *J. Vis.* 21(7):12. doi:10.1167/jov.21.7.12. Read: part
    (introduction, "An active-vision interpretation of preview effects", conclusions). Edwards G., VanRullen R.,
    Cavanagh P. 2018. Decoding trans-saccadic memory. *J. Neurosci.* 38(5):1114–1123 (online 2017).
    doi:10.1523/JNEUROSCI.0854-17.2017. Read: abstract.
- **Quotes:** "participants categorized coarse-to-fine sequences more rapidly than fine-to-coarse sequences" (Kauffmann
  et al. 2015, abstract; the sequences are six filtered versions of the same scene). "By attending first to the coarse
  scale, the visual system can get a quick and rough estimate of the input to activate scene schemas in memory,
  attending to fine information allows refinement, or refutation, of the raw estimate" (Schyns & Oliva 1994). A
  blurred image sent ahead to prefrontal cortex activates expectations "back-projected as an 'initial guess' to the
  temporal cortex" (Bar 2003). "vision at a glance ... For later vision with scrutiny, reverse hierarchy routines
  focus attention to specific, active, low-level units" (Hochstein & Ahissar 2002). "We hypothesize an intrinsic
  coarse-to-fine strategy for visual search that is even used when such a strategy is not optimal" (Over et al. 2007).
- **CanViT, analogous:** C2F starts with the whole scene at low resolution (a low-pass view of everything), then
  quadrants, then sixteenths; F2C visits the same tiles in reverse, and C2F reaches the higher peak accuracy (the
  paper: "highlighting the importance of contextually informed processing"). This is the Musel/Kauffmann contrast
  with viewpoints in place of filtered images. In C2F the first glimpse previews every region the later glimpses zoom
  into, as extrafoveal vision previews a saccade target.
- **CanViT, differs:** the human sequences keep coverage fixed and change spatial frequency; a CanViT zoom changes
  coverage and resolution together. Human preview is simultaneous (periphery during fixation); CanViT's is sequential.
  The mechanism behind C2F > F2C is not isolated. Hypotheses (none measured): reads condition the zoomed glimpses on
  the coarse guess (Bar's top-down facilitation); in F2C, late coarse writes overwrite fine detail written earlier;
  long monotone sequences of either kind differ from pretraining's random viewpoints in different ways.
- **Slide:** "Viewing policies" (F2C isolates order); "The Canvas Vision Transformer" (reads as top-down feedback).
- **Keyword:** on the slide, "coarse to fine, as in human scene recognition" with the CtF/FtC citation; Bar's
  "initial guess" and the preview effect in the notes.
- **Questions to expect:** "Is it top-down facilitation, or just that fine details get overwritten in F2C?" Not yet
  measured (hypotheses above); comparing canvas content after the same final glimpse under both orders would separate
  them. "People use the scale that is diagnostic for the task (Oliva & Schyns 1997); would a task-dependent order beat
  C2F?" EG-C2F is a first step: it keeps C2F's levels but reorders tiles by the readout's uncertainty.

## Looking where you are uncertain: local uncertainty and active sensing

- **What it is:** choosing the next fixation to gain the most information, either globally (expected information gain
  about the whole task) or locally (fixating the currently most uncertain location).
- **Sources:**
  - Renninger L.W., Verghese P., Coughlan J. 2007. Where to look next? Eye movements reduce local uncertainty. *J.
    Vis.* 7(3):6. doi:10.1167/7.3.6. Read: part (models, local-uncertainty results, general discussion; Zotero).
  - Najemnik J., Geisler W.S. 2005. Optimal eye movement strategies in visual search. *Nature* 434(7031):387–391.
    doi:10.1038/nature03390. Read: abstract.
  - Yang S.C.-H., Lengyel M., Wolpert D.M. 2016. Active sensing in the categorization of visual patterns. *eLife*
    5:e12215. doi:10.7554/eLife.12215. Yang S.C.-H., Wolpert D.M., Lengyel M. 2016. Theoretical perspectives on active
    sensing. *Curr. Opin. Behav. Sci.* 11:100–108. doi:10.1016/j.cobeha.2016.06.009. Read: abstracts.
  - Butko N.J., Movellan J.R. 2010. Infomax control of eye movements. *IEEE Trans. Auton. Ment. Dev.* 2(2):91–107.
    doi:10.1109/TAMD.2010.2051029. Read: abstract (OpenAlex).
  - Neurons: Foley N.C., Kelly S.P., Mhatre H., Lopes M., Gottlieb J. 2017. Parietal neurons encode expected gains in
    instrumental information. *PNAS* 114(16):E3315–E3323. doi:10.1073/pnas.1613844114. Gottlieb J., Oudeyer P.-Y.
    2018. Towards a neuroscience of active sampling and curiosity. *Nat. Rev. Neurosci.* 19(12):758–770.
    doi:10.1038/s41583-018-0078-0. Read: abstracts.
  - Machine learning: Houlsby N., Huszár F., Ghahramani Z., Lengyel M. 2011. Bayesian active learning for
    classification and preference learning. arXiv:1112.5745. Kendall A., Gal Y. 2017. What uncertainties do we need in
    Bayesian deep learning for computer vision? NeurIPS 2017 (arXiv comment), arXiv:1703.04977. Read: abstracts.
- **Quotes:** observers "may instead be using a local rule: fixate only the most informative locations, that is,
  reduce local uncertainty" (Renninger et al. 2007, abstract). "This early uncertainty map would then need to be
  combined with a stimulus-centered representation that incorporates knowledge gained from previous fixations"
  (Renninger et al. 2007, p. 13). Parietal neurons "encode, before an information sampling saccade, the reduction in
  uncertainty that the saccade is expected to bring for a subsequent action" (Foley et al. 2017).
- **CanViT, analogous:** EG-C2F is Renninger's local rule on a scene-coordinate map: the entropy of the class
  distribution read out from the canvas at each cell, averaged per tile, the most uncertain unvisited tile visited
  next within each quadtree level. The canvas is the "stimulus-centered representation that incorporates knowledge
  gained from previous fixations" that Renninger et al. say a local uncertainty map needs. It is "information
  sampling" in Gottlieb & Oudeyer's sense: reducing uncertainty for a known task.
- **CanViT, differs:** EG-C2F ranks by current entropy, not by expected information gain (Najemnik & Geisler's ideal
  searcher, Butko & Movellan's infomax, Houlsby et al.'s BALD): it does not predict what a glimpse would reveal.
  Predictive entropy mixes ambiguity that more looking cannot remove (aleatoric: class boundaries) with ignorance that
  it can (epistemic: unseen regions) (Kendall & Gal 2017); EG-C2F's per-tile averaging and once-per-level visits limit,
  but do not remove, glimpses spent on the first kind (a hypothesis, not measured). The quadtree schedule is fixed;
  only the order within a level adapts. Najemnik & Geisler found humans near-optimal "even though humans integrate
  information poorly across fixations".
- **Slide:** "The model's own uncertainty about what's where"; "Viewing policies" (EG-C2F's cell).
- **Keyword:** "looking where it is most uncertain" on the slide; "local uncertainty (Renninger et al. 2007)" as its
  citation; information gain, infomax and BALD in the notes. AME's attention-map entropy as the precedent is already
  prepared (`AGENTS.md`, "Questions to anticipate").
- **Question to expect:** "Entropy or expected information gain?" Entropy; an information-gain policy on the same
  observer is untested.

## Metacognition: monitoring and control, and where the term stops

- **What it is:** mechanisms that form beliefs about one's own mental operations (monitoring) and use them to
  regulate behaviour (control). Its neighbour, certainty: degree of belief about states of the world.
- **Sources:**
  - Fleming S.M. 2024. Metacognition and confidence: a review and synthesis. *Annu. Rev. Psychol.* 75:241–268.
    doi:10.1146/annurev-psych-022423-032425. Read: abstract (Europe PMC) and the author's preprint
    (doi:10.31234/osf.io/ge7tz, OSF), sections on scope and definitions and on confidence formation.
  - Pouget A., Drugowitsch J., Kepecs A. 2016. Confidence and certainty: distinct probabilistic quantities for
    different goals. *Nat. Neurosci.* 19(3):366–374. doi:10.1038/nn.4240. Read: abstract.
  - Fleming S.M., Daw N.D. 2017. Self-evaluation of decision-making: a general Bayesian framework for metacognitive
    computation. *Psychological Review* 124(1):91–114. doi:10.1037/rev0000045. Read: abstract.
  - Nelson T.O., Narens L. 1990. Metamemory: a theoretical framework and new findings. *Psychology of Learning and
    Motivation*, pp. 125–173. doi:10.1016/S0079-7421(08)60053-5. Read: metadata only (Crossref lists Nelson alone);
    the monitoring/control distinction is taken from Fleming's preprint, which cites it.
  - Confidence driving information seeking: Desender K., Boldt A., Yeung N. 2018. Subjective confidence predicts
    information seeking in decision making. *Psychological Science* 29(5):761–778. doi:10.1177/0956797617744771.
    Schulz L., Fleming S.M., Dayan P. 2023. Metacognitive computations for information search: confidence in control.
    *Psychological Review* 130(3):604–639. doi:10.1037/rev0000401. Read: abstracts.
  - Neurons: Kiani R., Shadlen M.N. 2009. Representation of confidence associated with a decision by neurons in the
    parietal cortex. *Science* 324(5928):759–764. doi:10.1126/science.1169405. Read: abstract.
  - Measuring it: Fleming S.M., Lau H.C. 2014. How to measure metacognition. *Front. Hum. Neurosci.* 8:443.
    doi:10.3389/fnhum.2014.00443. Rahnev D. 2021. Visual metacognition: measures, models, and neural correlates. *Am.
    Psychol.* 76(9):1445–1453. doi:10.1037/amp0000937. Guo C., Pleiss G., Sun Y., Weinberger K.Q. 2017. On
    calibration of modern neural networks. ICML 2017 (arXiv comment), arXiv:1706.04599. Read: abstracts.
- **Quotes:** "Sensitivity to uncertainty is a central aspect of (first-order) Bayesian computation, but alone is not
  evidence for metacognition" (Fleming, preprint, section 5). Fleming separates uncertainty "in a 'world-centred'
  reference frame" from "confidence in our own propositions or actions ... in a 'self-centred' reference frame"
  (preprint, section 2). "the term certainty should be reserved to refer to the encoding of all other probability
  distributions over sensory and cognitive variables" (Pouget et al. 2016). "subjective confidence predicts
  information seeking" (Desender et al. 2018). "it is crucial to treat metacognitive monitoring and control as closely
  linked processes" (Schulz et al. 2023).
- **CanViT, analogous:** a policy reads the observer's own uncertainty about what is where and uses it to decide what
  to sample next: monitoring feeding control, the loop Desender et al. and Schulz et al. study in people, with the
  monitored quantity computed from the model's own state rather than from the world.
- **CanViT, differs:** the entropy is computed from the same readout as the segmentation, so it is a first-order
  quantity in Fleming & Daw's terms and certainty about the world in Pouget et al.'s, not propositional confidence in
  a decision. Its calibration (Guo et al.) and its metacognitive sensitivity (does entropy predict where the
  segmentation is wrong; Fleming & Lau's type-2 measures) have not been measured. The policy, not the canvas, does
  the monitoring.
- **Slide:** "The model's own uncertainty about what's where".
- **Keyword:** the title is safe as it stands. If "metacognition" is said, say which sense, for example "metacognition
  in its simplest form: using its own uncertainty to decide what to look at"; a metacognition researcher in the room
  will otherwise object with Fleming's sentence above.
- **Questions to expect:** "Is that metacognition or just uncertainty?" Fleming's distinction, granted; the
  measurable step toward the stricter sense is metacognitive sensitivity of the canvas readout. "Does it know which
  parts it inferred rather than saw?" Not measured; humans do not reliably (Ehinger et al. 2017, in the filling-in
  entry).

## Spatiotopic memory, which the brain may not have

- **What it is:** spatiotopic means coded in world (here, scene) coordinates; retinotopic, in eye coordinates.
  Remapping updates retinotopic representations around each saccade.
- **Sources:**
  - Cavanagh P., Melcher D. 2026. Steerable autoencoders underlying remapping, spatiotopy, and visual stability.
    *Curr. Opin. Neurobiol.* 99:103221. doi:10.1016/j.conb.2026.103221. Read: full (Zotero).
  - Gardner J.L., Merriam E.P., Movshon J.A., Heeger D.J. 2008. Maps of visual space in human occipital cortex are
    retinotopic, not spatiotopic. *J. Neurosci.* 28(15):3988–3999. doi:10.1523/JNEUROSCI.5476-07.2008. Golomb J.D.,
    Kanwisher N. 2012. Higher level visual cortex represents retinotopic, not spatiotopic, object location. *Cerebral
    Cortex* 22(12):2794–2810 (online 2011). doi:10.1093/cercor/bhr357. d'Avossa G. et al. 2007. Spatiotopic
    selectivity of BOLD responses to visual motion in human area MT. *Nat. Neurosci.* 10(2):249–255.
    doi:10.1038/nn1824. Burr D.C., Morrone M.C. 2011. Spatiotopic coding and remapping in humans. *Phil. Trans. R.
    Soc. B* 366(1564):504–515. doi:10.1098/rstb.2010.0244. Golomb J.D., Mazer J.A. 2021. Visual remapping. *Annu.
    Rev. Vis. Sci.* 7:257–277. doi:10.1146/annurev-vision-032321-100012. Read: abstracts.
  - Allocentric maps of viewed space: Rolls E.T., Robertson R.G., Georges-François P. 1997. Spatial view cells in the
    primate hippocampus. *Eur. J. Neurosci.* 9(8):1789–1794. doi:10.1111/j.1460-9568.1997.tb01538.x. Rolls E.T. 1999.
    Spatial view cells and the representation of place in the primate hippocampus. *Hippocampus* 9(4):467–480.
    doi:10.1002/(SICI)1098-1063(1999)9:4<467::AID-HIPO13>3.0.CO;2-F. Killian N.J., Jutras M.J., Buffalo E.A. 2012. A
    map of visual space in the primate entorhinal cortex. *Nature* 491(7426):761–764. doi:10.1038/nature11587. Nau M.,
    Navarro Schröder T., Bellmund J.L.S., Doeller C.F. 2018. Hexadirectional coding of visual space in human entorhinal
    cortex. *Nat. Neurosci.* 21(2):188–190. doi:10.1038/s41593-017-0050-8. Julian J.B., Keinath A.T., Frazzetta G.,
    Epstein R.A. 2018. Human entorhinal cortex represents visual space using a boundary-anchored grid. *Nat. Neurosci.*
    21(2):191–194. doi:10.1038/s41593-017-0049-1. Read: abstracts.
- **Quotes:** "Spatiotopic maps offer two critical benefits. The first is the sense of stability ... The second
  benefit is trans-saccadic integration. An object's representation would remain at the same location on a spatiotopic
  map so that the processing of its identity and properties could continue across saccades without having to start
  again from zero" and "the search for that spatiotopic map has not met with much success" (Cavanagh & Melcher 2026,
  p. 1). "many brain regions are tuned in spatiotopic coordinates, but only for items that are actively attended" (Burr
  & Morrone 2011). Spatial view cells give "a spatial representation that is allocentric, i.e. in world coordinates"
  suited to "memories of where in an environment an object was seen" (Rolls et al. 1997). Entorhinal neurons
  "encode space during visual exploration, even without locomotion" (Killian et al. 2012).
- **CanViT, analogous:** the paper's "retinotopic Vision Transformer backbone" and "spatiotopic scene-wide latent
  workspace", bound by scene-relative RoPE. The canvas has both properties Cavanagh & Melcher attribute to a
  spatiotopic map, at every position: nothing in it moves when the viewpoint moves, and processing continues across
  glimpses.
  Spatial view cells and the entorhinal map of visual space are where primate brains represent viewed space in world
  coordinates, a neural anchor for the paper's "cognitive map" (Tolman 1948).
- **CanViT, differs:** the brain is argued to approximate spatiotopy by remapping a few attended targets on
  retinotopic maps (Cavanagh & Melcher 2026; Golomb & Mazer 2021); CanViT builds the dense map the brain may lack,
  and receives the coordinates the brain must compute. "Cognitive map" stays a functional analogy (`AGENTS.md`,
  "Claims to state with their scope").
- **Slide:** "The Canvas Vision Transformer"; the closing slide.
- **Keyword:** "retinotopic glimpse, spatiotopic canvas" on the slide; the remapping contrast and spatial view cells in
  the notes.
- **Question to expect:** "The brain does not seem to have a spatiotopic map. Isn't the canvas unbiological?" CanViT
  measures what a dense scene-coordinate memory buys (the reset result); how the brain gets similar benefits from
  remapping is the comparison it makes possible. The viewpoint question is already prepared (`AGENTS.md`).

## Scene memory as objects bound to locations; change blindness

- **What it is:** change blindness is the failure to notice large changes across a disruption (a saccade, a blank, a
  cut). Visual memory accounts hold that abstract object representations nonetheless accumulate during viewing, bound
  to locations in a representation of the scene's layout.
- **Sources:**
  - Change blindness and sparse accounts: Rensink R.A., O'Regan J.K., Clark J.J. 1997. To see or not to see: the need
    for attention to perceive changes in scenes. *Psychological Science* 8(5):368–373.
    doi:10.1111/j.1467-9280.1997.tb00427.x. Read: abstract (Crossref). O'Regan J.K. 1992. Solving the "real" mysteries
    of visual perception: the world as an outside memory. *Can. J. Psychol.* 46(3):461–488. doi:10.1037/h0084327.
    Simons D.J., Levin D.T. 1997. Change blindness. *Trends Cogn. Sci.* 1(7):261–267.
    doi:10.1016/S1364-6613(97)01080-2. Simons D.J., Rensink R.A. 2005. Change blindness: past, present, and future.
    *Trends Cogn. Sci.* 9(1):16–20. doi:10.1016/j.tics.2004.11.006. Rensink R.A. 2000. The dynamic representation of
    scenes. *Visual Cognition* 7(1–3):17–42. doi:10.1080/135062800394667 (abstract from OpenAlex). O'Regan J.K., Noë
    A. 2001. A sensorimotor account of vision and visual consciousness. *Behav. Brain Sci.* 24(5):939–973.
    doi:10.1017/S0140525X01000115. Ballard D.H., Hayhoe M.M., Pelz J.B. 1995. Memory representations in natural
    tasks. *J. Cogn. Neurosci.* 7(1):66–80. doi:10.1162/jocn.1995.7.1.66. Horowitz T.S., Wolfe J.M. 1998. Visual
    search has no memory. *Nature* 394(6693):575–577. doi:10.1038/29068. Read: abstracts.
  - Accumulating accounts: Hollingworth A., Henderson J.M. 2002. Accurate visual memory for previously attended
    objects in natural scenes. *J. Exp. Psychol. Hum. Percept. Perform.* 28(1):113–136.
    doi:10.1037/0096-1523.28.1.113 (abstract from OpenAlex). Hollingworth A. 2004. Constructing visual representations
    of natural scenes: the roles of short- and long-term visual memory. *JEP:HPP* 30(3):519–537.
    doi:10.1037/0096-1523.30.3.519. Hollingworth A. 2006. Scene and position specificity in visual memory for objects.
    *J. Exp. Psychol. Learn. Mem. Cogn.* 32(1):58–69. doi:10.1037/0278-7393.32.1.58. Hollingworth A. 2007.
    Object-position binding in visual memory for natural scenes and object arrays. *JEP:HPP* 33(1):31–47.
    doi:10.1037/0096-1523.33.1.31. Read: abstracts. Hollingworth A. 2008. Visual memory for natural scenes. In Luck
    S.J., Hollingworth A. (eds), *Visual Memory*, Oxford University Press, pp. 123–162. Read: part (sections 5.1, 5.2
    to the evaluation of theories, 5.3.2.3 and 5.4; Zotero; metadata from the book's front matter). Mirpour K., Arcizet
    F., Ong W.S., Bisley J.W. 2009. Been there, seen that: a neural mechanism for performing efficient visual search.
    *J. Neurophysiol.* 102(6):3481–3491. doi:10.1152/jn.00688.2009. Read: abstract.
  - Static versus dynamic settings: Tatler B.W., Land M.F. 2011. Vision and the representation of the surroundings in
    spatial memory. *Phil. Trans. R. Soc. B* 366(1564):596–610. doi:10.1098/rstb.2010.0188. Read: abstract.
- **Quotes:** change blindness supports "the idea that observers never form a complete, detailed representation of
  their surroundings" (Rensink et al. 1997). "the outside world is considered as a kind of external memory store which
  can be accessed instantaneously by casting one's eyes (or one's attention) to some location" (O'Regan 1992).
  Against: "relatively detailed visual information is retained in memory from previously attended objects in natural
  scenes" (Hollingworth & Henderson 2002); "episodic scene representations are formed through the binding of objects to
  scene locations" (Hollingworth 2007); "change blindness is not necessarily symptomatic of impoverished scene
  representation" (Hollingworth 2008, p. 130). "While static scene viewing paradigms favour extensive, but perhaps
  abstracted, memory representations, dynamic settings suggest sparser and task-selective representation" (Tatler &
  Land 2011).
- **CanViT, analogous:** the canvas is an abstract, location-indexed scene representation that accumulates over
  glimpses, the kind Hollingworth's account describes, on static scenes, where Tatler & Land place extensive memory.
  Melcher (2001) called the human version "medium-term" or "disposable": kept while the scene is in use, which is what
  a canvas is within a rollout.
- **CanViT, differs:** human scene memory is object-based and gated by attention; the canvas is written densely at
  every glimpse. CanViT can only update where its writes reach, so a change to an unvisited region would go unnoticed
  until a glimpse covers it: a form of change blindness by construction (an inference from the architecture, not
  tested; the model has only seen static scenes).
- **Slide:** none on screen; the notes of "Resetting the memory" and the prepared answer.
- **Draft answer to the anticipated question** (`AGENTS.md`, "Questions to anticipate"; for the authors to adopt or
  reject): change blindness rules out a detailed, picture-like memory of the scene; it does not settle whether abstract
  representations accumulate, and the visual memory work shows that they do, bound to scene locations (Hollingworth
  2008 lays out the five competing accounts, from O'Regan's outside memory to Hollingworth & Henderson's). The canvas
  holds a best guess in features, not a picture; how much of it survives later glimpses is measurable, and comparing
  that with human scene memory is the closing slide's first question.

## Perceptual filling-in and completion

- **What it is:** the visual system represents content where its input is absent: interpolation across the blind
  spot, scotomas or occluders (filling-in), the parts of an object that give no stimulation (amodal completion, in the
  broad sense), or the periphery reconstructed from the fovea.
- **Sources:**
  - Weil R.S., Rees G. 2011. A new taxonomy for perceptual filling-in. *Brain Research Reviews* 67(1–2):40–55.
    doi:10.1016/j.brainresrev.2010.10.004. Komatsu H. 2006. The neural mechanisms of perceptual filling-in. *Nat. Rev.
    Neurosci.* 7(3):220–231. doi:10.1038/nrn1869. Nanay B. 2018. The importance of amodal completion in everyday
    perception. *i-Perception* 9(4):2041669518788887. doi:10.1177/2041669518788887. Read: abstracts.
  - From the fovea outward: Toscani M., Gegenfurtner K.R., Valsecchi M. 2017. Foveal to peripheral extrapolation of
    brightness within objects. *J. Vis.* 17(9):14. doi:10.1167/17.9.14. Otten M., Pinto Y., Paffen C.L.E., Seth A.K.,
    Kanai R. 2017. The uniformity illusion. *Psychological Science* 28(1):56–68. doi:10.1177/0956797616672270. Read:
    abstracts (Europe PMC); both reviewed in Stewart et al. 2020 (read).
  - Completion over time and from little: Tang H., Schrimpf M., Lotter W., Moerman C., Paredes A., Ortega Caro J.,
    Hardesty W., Cox D., Kreiman G. 2018. Recurrent computations for visual pattern completion. *PNAS*
    115(35):8835–8840. doi:10.1073/pnas.1719397115. Orlov T., Zohary E. 2018. Object representations in human visual
    cortex formed through temporal integration of dynamic partial shape views. *J. Neurosci.* 38(3):659–678 (online
    2017). doi:10.1523/JNEUROSCI.1318-17.2017. Craddock M., Martinovic J., Lawson R. 2011. An advantage for active
    versus passive aperture-viewing in visual object recognition. *Perception* 40(10):1154–1163. doi:10.1068/p6974.
    Read: abstracts.
  - Confidence in what was filled in: Ehinger B.V., Häusser K., Ossandón J.P., König P. 2017. Humans treat unreliable
    filled-in percepts as more real than veridical ones. *eLife* 6:e21761. doi:10.7554/eLife.21761. Read: abstract.
- **Quotes:** "Perceptual filling-in occurs when structures of the visual system interpolate information across
  regions of visual space where that information is physically absent" (Weil & Rees 2011). "Amodal completion is the
  representation of those parts of the perceived object that we get no sensory stimulation from" (Nanay 2018). Foveal
  brightness is extrapolated to the periphery, but "this mechanism is selectively applied within an object's boundary"
  (Toscani et al. 2017). "Making inferences from partial information constitutes a critical aspect of cognition"
  (Tang et al. 2018). "a percept that is partially inferred is paradoxically considered more reliable than a percept
  based on external input" (Ehinger et al. 2017).
- **CanViT, analogous:** the canvas labels the never-sampled middle of the table as table (interpolation between
  glimpses), and after a glimpse of one end alone already extends the table toward the other (extrapolation); the
  paper's appendix shows the last write of a glimpse extending beyond it. Aperture viewing (Orlov & Zohary 2018;
  Craddock et al. 2011) is the human paradigm closest to CanViT's input: an object known only through successive
  partial views.
- **CanViT, differs:** human filling-in is mostly within a single view (blind spot, scotoma, occluder), often of
  surface attributes; the table's middle lies outside every glimpse. Nanay's broad definition admits it as amodal
  completion; the classic, occlusion-based definition does not (`AGENTS.md` keeps "not amodal completion"). Human
  extrapolation respects object boundaries (Toscani et al.); over large ADE20K objects, the canvas also labels as the
  object a large share of the other pixels in the band between the glimpses (`AGENTS.md`, table corners, for the
  numbers). The canvas marks no difference between seen and inferred regions.
- **Slide:** "Extrapolation to unobserved regions"; "Integrating multiple viewpoints into a coherent understanding".
- **Keyword:** "filling-in" or "completion" on the slide (the model completes; it does not see); Toscani's boundary
  result in the notes beside the spill number; aperture viewing in the notes.
- **Question to expect:** "Does it know which parts it filled in?" Not measured: compare the readout's entropy over
  seen and never-seen cells. Humans tend to trust filled-in content too much (Ehinger et al. 2017).

## Trans-saccadic feature prediction

- **What it is:** the visual system learns associations between how an object looks in the periphery before a
  saccade and in the fovea after it, and uses them to predict one from the other.
- **Sources:** Herwig A., Schneider W.X. 2014. Predicting object features across saccades: evidence from object
  recognition and visual search. *J. Exp. Psychol. Gen.* 143(5):1903–1922. doi:10.1037/a0036781. Goktepe N., Schütz
  A.C. 2023. Familiar objects benefit more from transsaccadic feature predictions. *Atten. Percept. Psychophys.*
  85(6):1949–1961. doi:10.3758/s13414-022-02651-8. Read: abstracts; the re-calibration literature as summarized in
  Stewart et al. 2020 (read).
- **Quote:** "the visual system uses past experience to predict how peripheral objects will look in the fovea, and what
  foveal search templates should look like in the periphery" (Herwig & Schneider 2014).
- **CanViT, analogous:** pretraining teaches CanViT to predict, from low-resolution partial views, the features a
  high-resolution view of the whole scene yields. Familiar objects benefit more in people (Goktepe & Schütz 2023), as
  world knowledge from the teacher helps CanViT.
- **CanViT, differs:** the human teacher is the person's own later foveal view; CanViT's is a frozen passive model
  that sees the whole scene. The paper's proposed future work, active-to-active self-distillation, is closer to the
  human case (an inference from the two descriptions, for the authors to judge).
- **Slide:** "Passive-to-active dense latent distillation" (notes); the closing slide.
- **Keyword:** notes.

## Vision at a glance, then recurrent refinement

- **What it is:** the gist of a scene (its category, layout and much of its content) is available from a single brief
  glance; recurrent processing after the first feedforward sweep refines it.
- **Sources:** Oliva A., Torralba A. 2006. Building the gist of a scene: the role of global image features in
  recognition. *Progress in Brain Research* 155:23–36. doi:10.1016/S0079-6123(06)55002-2. Greene M.R., Oliva A. 2009.
  Recognition of natural scenes from global properties: seeing the forest without representing the trees. *Cognitive
  Psychology* 58(2):137–176. doi:10.1016/j.cogpsych.2008.06.001. Fei-Fei L., Iyer A., Koch C., Perona P. 2007. What
  do we perceive in a glance of a real-world scene? *J. Vis.* 7(1):10. doi:10.1167/7.1.10. Lamme V.A.F., Roelfsema
  P.R. 2000. The distinct modes of vision offered by feedforward and recurrent processing. *Trends Neurosci.*
  23(11):571–579. doi:10.1016/S0166-2236(00)01657-X. Read: abstracts. Potter 1976 and Oliva 2005 are in
  `cognition.md`; Kar et al. 2019 is in the paper.
- **Quotes:** "Humans can recognize the gist of a novel image in a single glance, independent of its complexity" (Oliva
  & Torralba 2006). "within a single glance, much object- and scene-level information is perceived" (Fei-Fei et al.
  2007).
- **CanViT, analogous:** one low-resolution glimpse of the full scene already gives 38.5% mIoU; RFS shows refinement
  from recurrence alone on a fixed input, which then declines (the paper's Results).
- **CanViT, differs:** the gist literature limits time (brief, masked presentations); CanViT's first glimpse is
  limited in resolution, not in time.
- **Slide:** "A new active-vision state of the art in accuracy and efficiency" (the single glimpse); "Viewing
  policies" (RFS).
- **Keyword:** "gist" in the notes of both; on a slide only if the single-glimpse number needs a human anchor.

## Priority map, inhibition of return, Bayesian surprise, active inference

Names for parts of EG-C2F and for policies not yet tried; notes only.

- **Priority map with inhibition of return.** Fecteau & Munoz 2006 and Bisley & Goldberg 2010 are in `citations.md`
  (abstracts read this session; Bisley & Goldberg: the lateral intraparietal area "acts as a priority map in which
  objects are represented by activity proportional to their behavioral priority"). Mirpour et al. 2009 (above): LIP
  responses "marked off targets that had been fixated by a reduction in activity. This reduction acted like inhibition
  of return". EG-C2F's per-tile entropy, with visited tiles excluded within a level, is a priority map with inhibition
  of return computed by the policy from the canvas readout. Kiani & Shadlen 2009 (above) found that the parietal
  neurons representing a decision also encode certainty about it (in LIP, per Fleming's preprint), which ties this
  entry to the metacognition one.
- **Bayesian surprise.** Itti L., Baldi P. 2009. Bayesian surprise attracts human attention. *Vision Research*
  49(10):1295–1306. doi:10.1016/j.visres.2008.09.007. Read: abstract. Surprise "measures how data affects an observer,
  in terms of differences between posterior and prior beliefs". The paper's figure 1 shows canvas updates as the cosine
  dissimilarity between consecutive canvases, a feature-space counterpart of belief change; no policy uses it. A
  surprise-guided policy is untried.
- **Active inference, saccades as experiments.** Friston K., Adams R.A., Perrinet L., Breakspear M. 2012. Perceptions
  as hypotheses: saccades as experiments. *Front. Psychol.* 3:151. doi:10.3389/fpsyg.2012.00151. Read: part (sections
  on Bayes-optimal control and on priors and saliency; Zotero). Salience is "the negative entropy of the counterfactual
  density", computed on a 32 × 32 grid that "extends beyond the field of view", with inhibition of return. EG-C2F
  shares the uncertainty map over unseen locations and the inhibition of return; it differs in using current entropy,
  with no generative model of what a glimpse would show.

## Words the paper uses that this audience hears differently

| Paper's word | What this audience hears | What to say |
|---|---|---|
| "cognitive map" (§4, citing Tolman 1948) | hippocampal and entorhinal maps | a functional analogy; primates do map viewed space in world coordinates (spatial view cells; Killian et al. 2012), which the canvas resembles in role only |
| "latent workspace" (abstract) | Global Workspace Theory: a limited-capacity stage broadcasting among specialist modules (VanRullen R., Kanai R. 2021. Deep learning and the Global Workspace Theory. *Trends Neurosci.* 44(9):692–704. doi:10.1016/j.tins.2021.04.005; Goyal A. et al. 2022. Coordination among neural modules through a shared global workspace. ICLR 2022, arXiv:2103.01197, with Yoshua Bengio among the authors. Read: abstracts) | the canvas is a large spatial memory read by one backbone, with no competition for access and no claim about consciousness; prefer "memory" on slides |
| "mental image" (§5) | visual mental imagery, which "can function much like a weak version of afferent perception" (Pearson J. 2019. The human imagination: the cognitive neuroscience of visual mental imagery. *Nat. Rev. Neurosci.* 20(10):624–634. doi:10.1038/s41583-019-0202-9. Read: abstract) | the canvas is spatially organized and holds a best guess, but it is filled from glimpses, not generated without input; avoid the phrase on slides |
| "high-capacity working memory" (abstract) | visual working memory of about four objects (Luck & Vogel 1997, `citations.md`; abstract read) and object files (Kahneman D., Treisman A., Gibbs B.J. 1992. The reviewing of object files: object-specific integration of information. *Cognitive Psychology* 24(2):175–219. doi:10.1016/0010-0285(92)90007-O. Read: abstract) | the canvas is indexed by location, not by object, and its capacity is closer to visual long-term memory (Brady T.F., Konkle T., Alvarez G.A., Oliva A. 2008. Visual long-term memory has a massive storage capacity for object details. *PNAS* 105(38):14325–14329. doi:10.1073/pnas.0803390105. Read: abstract) while lasting only one rollout; Melcher's "medium-term" fits |

## Leads not kept

- Baddeley's "visuospatial sketchpad" (and Logie's passive "visual cache" against the active "inner scribe", which
  would echo "the canvas holds, the backbone acts"): no abstract read this session names them (Baddeley 2000, 2012 and
  Logie 2011 abstracts checked). UNVERIFIED; worth one read of Baddeley 2003 (*Nat. Rev. Neurosci.* 4:829–839,
  doi:10.1038/nrn1201) if the authors want the word.
- POMDP belief state (Kaelbling, Littman & Cassandra 1998, *Artificial Intelligence* 101:99–134,
  doi:10.1016/S0004-3702(98)00023-X): abstract not available; `AGENTS.md` already scopes "belief state".
- Navon 1977 (global precedence): abstract not available; Hochstein & Ahissar's abstract carries "forest before trees".
- Sensorimotor contingencies (O'Regan & Noë 2001, read in abstract): CanViT is given its viewpoints and learns no
  sensorimotor laws; kept only as the radical end of the change-blindness accounts.
- Girgus, Gellman & Hochberg 1980 (spatial order in piecemeal shape recognition through an aperture, *Percept.
  Psychophys.* 28(2):133–138, doi:10.3758/BF03204338): metadata only; the publisher's free PDF did not download. Read
  it if the aperture-viewing link is used: its title promises an order effect on piecemeal views.
