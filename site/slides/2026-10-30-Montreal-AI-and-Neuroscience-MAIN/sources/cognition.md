# Inference from knowledge of the world: sources for the talk

[Compiled 2026-10-01 by a Claude Code research subagent; read by Claude Code, not by the authors.] For the slide
"Integrating multiple viewpoints into a coherent understanding" (`#table`). Each entry names what was read: the full
PDF, the abstract (PubMed, Crossref or the publisher page), or only the metadata. Citation metadata (authors, year,
venue, volume, pages, DOI) come from Crossref or PubMed unless stated. Quotes are verbatim. UNVERIFIED marks what was
not checked against the source itself. "Zotero" is a PDF in the user's library. The CanViT paper cites none of these
works except Rao & Ballard 1999.

## Recommendation

Four citations, one per spoken claim: three primary experiments and one review.

| Spoken sentence | Citation on the slide | Figure |
|---|---|---|
| "Yet you know what is between them: more table, because you know tables. Perception goes beyond the information given." | Intraub & Richardson 1989, *J. Exp. Psychol. Learn. Mem. Cogn.* | `intraub-richardson-1989-boundary-extension.png` |
| "We know what scenes contain and where things go," | Biederman, Mezzanotte & Rabinowitz 1982, *Cogn. Psychol.* | `biederman-1982-hydrant-on-mailbox.png` |
| "and that knowledge guides where we look, and where we do not need to." | Torralba, Oliva, Castelhano & Henderson 2006, *Psychol. Rev.* | `torralba-2006-contextual-guidance.png` |
| "it is inference from prior knowledge of the world." | Peelen, Berlot & de Lange 2024, *Nat. Rev. Psychol.* (review) | none |

- **Intraub & Richardson 1989** is the closest scene-level experiment reviewed here: people shown a close-up
  remember, and draw, the scene continuing past the picture's edges. Two limits to keep in the wording: it is
  extrapolation beyond the edges of one view (the slide shows interpolation between two views), and it is measured in
  memory, though as early as 42 ms after the view is interrupted (Intraub & Dickinson 2008). A safe sentence: "people
  remember seeing what lies beyond the edges of a view; what they add is what such a place should contain."
- **Biederman et al. 1982** shows that a 150 ms glance already gives access to what belongs in a scene and where:
  objects that violate those relations are detected less accurately. Say "where things go", not "context makes
  objects easier to see": the facilitation claim is contested (Hollingworth & Henderson 1998).
- **Torralba et al. 2006** shows search fixations landing, from the first ones, in the regions where scene context
  says the target can be, and a model built on that prior predicting them better than saliency does. It supports
  "where we look" directly; "where we need not look" is its complement: the 20% of each image that the model selects
  held 73% of the first five fixations, so the other 80% held about a quarter.
- **Peelen et al. 2024** is the modern statement that scene and object perception are joint probabilistic inference
  with priors, from behaviour and neural data. Only its abstract was read this session. If a review read in full is
  preferred, Kersten, Mamassian & Yuille 2004 (*Annu. Rev. Psychol.*) is the alternative, though it concerns object
  shape and material more than scenes. Helmholtz 1867 can be named in speech ("what Helmholtz called unconscious
  inference") without a slide citation.
- For a neuroscience-heavy audience, Park, Intraub, Yi, Widders & Chun 2007 (*Neuron*) can replace the review: scene
  areas (PPA, RSC) respond as if the view extended past its edges, object area LOC does not.

A risk to anticipate: the middle of a table can be filled by continuing its edges and surface, which amodal
completion does from geometry alone (Kanizsa's view); "because you know tables" claims more. Intraub's own reading of
the trash-can drawings grants both: edges constrain the completion and "World knowledge would support these
constraints as well". A demonstration in which geometry is ambiguous and only knowledge resolves the gap would be
stronger evidence for the knowledge claim than the table.

## 1. The candidates named in the outline

- **Helmholtz H. 1867. *Handbuch der physiologischen Optik*.** Leipzig: Leopold Voss (Allgemeine Encyklopädie der
  Physik, vol. IX). Read: the 1867 volume on archive.org (`handbuchderphysi00helm`, OCR text), §26, pp. 430 and
  447–449. The preface (dated December 1866) says the three parts appeared in 1856, 1860 and 1866, so §26 was first
  issued in 1866; the bound volume is dated 1867. The perceptual acts by which we judge that an object of a given
  kind is at a given place are "nicht bewusste Thätigkeiten, sondern unbewusste"; Helmholtz proposes to call them
  "unbewusste Schlüsse" (p. 430), inferences from sensation to its cause that match analogical inference from past
  experience, and on pp. 447–449 "unbewusst vollführte Inductionsschlüsse". **Support:** the origin of "perception
  is inference from prior knowledge"; a theoretical claim, not evidence. The English translation (Southall, ed.,
  1924–1925, from the third German edition) was not checked: UNVERIFIED.
- **Bruner J.S. 1957. Going beyond the information given.** In *Contemporary Approaches to Cognition: A Symposium
  Held at the University of Colorado*, Harvard University Press. Read: nothing of the text. The volume's existence and
  contributor list come from a bibliography page; pages 41–69 and the 1973 reprint (*Beyond the Information Given*,
  Norton, pp. 218–238) come from a secondary summary (jimdavies.org): UNVERIFIED. That summary describes an essay
  about coding systems, learning and concept formation, with perception as one example: UNVERIFIED. Bruner's
  perception-specific paper is **On perceptual readiness**, *Psychological Review* 64(2):123–152, 1957,
  doi:10.1037/h0043805 (metadata verified; content UNVERIFIED). **Support:** the source of the slide's phrase; do not
  cite it as evidence.
- **Kersten D., Mamassian P., Yuille A. 2004. Object perception as Bayesian inference.** *Annual Review of
  Psychology* 55:271–304. doi:10.1146/annurev.psych.55.090902.142005. Read: the authors' preprint in full
  (cs.jhu.edu), not the published version. Abstract: "Recent work in Bayesian theories of visual perception has shown
  how complexity may be managed and ambiguity resolved through the task-dependent, probabilistic integration of prior
  object knowledge with image features." It traces the framework to Helmholtz, who "proposed that the visual system
  resolves ambiguity through built-in knowledge of the scene and how retinal images are formed, and uses this
  knowledge to automatically and unconsciously infer the properties of objects." **Support:** strong for "perception
  is inference from prior knowledge" as a framework; its cases are shape, material, lighting and depth, not the
  contents of unseen scene regions.
- **Biederman I., Mezzanotte R.J., Rabinowitz J.C. 1982. Scene perception: detecting and judging objects undergoing
  relational violations.** *Cognitive Psychology* 14(2):143–177. doi:10.1016/0010-0285(82)90007-X. Read: full PDF
  (scan on a Princeton course page). Five relations organize a scene: Interposition, Support, Probability, Position,
  familiar Size. Line-drawn scenes (247 slides from 42 objects and 17 backgrounds) were flashed for 150 ms to 96
  subjects, who judged whether a cued object matched a named target. Objects violating relations were missed more
  often (45.0% versus 24.9% in the normal "Base" condition), and d′ fell with the number of violations (1.62, 1.14,
  .78, .54 for zero to three). Semantic relations (Probability, Position, Size) were accessed at least as fast as
  the physical ones: "the relations were accessed from the results of a single fixation". **Support:** strong for
  "we know what scenes contain and where things go", available within one glance. Caveat: Hollingworth & Henderson
  1998 replicated the effect but found no consistency advantage once response bias was controlled (below).
- **Bar M. 2004. Visual objects in context.** *Nature Reviews Neuroscience* 5(8):617–629. doi:10.1038/nrn1476. Read:
  abstract and figure titles on nature.com (paywalled). The review "proposes specific mechanisms for the contextual
  facilitation of object recognition". Figure titles include "Resolution of ambiguous objects by context" and "The
  proposed model for the contextual facilitation of object recognition". The model's details (a coarse,
  low-spatial-frequency image activating a "context frame" in parahippocampal cortex) were seen only in a search
  snippet: UNVERIFIED. **Support:** moderate, for context-based prediction in the brain; review only.
- **Oliva A., Torralba A. 2007. The role of context in object recognition.** *Trends in Cognitive Sciences*
  11(12):520–527. doi:10.1016/j.tics.2007.09.009. Read: full PDF (University of Giessen course page). A "statistical
  summary of the scene provides a complementary and effective source of information for contextual inference, which
  enables humans to quickly guide their attention and eyes to regions of interest in natural scenes." Figure 1:
  averages of photographs aligned on an object show that "an object extends its influence beyond its own boundaries"
  (a monitor and table around the average keyboard, a ground plane under the hydrant). Figure 2: a blob read as a
  pedestrian is "the same shape as the car, except for a 90° rotation". Conclusion: "In the absence of enough local
  evidence about an object's identity, the scene structure and prior knowledge of world regularities might provide
  the additional information needed for recognizing and localizing an object." Box 1 states the controversy
  (Hollingworth & Henderson's functional isolation hypothesis). **Support:** strong as a review of context in
  recognition and in guiding the eyes; Figure 1 visualizes the claim that what surrounds a glimpse is predictable from
  the world's statistics, which suits the ML part of the audience.
- **Torralba A., Oliva A., Castelhano M.S., Henderson J.M. 2006. Contextual guidance of eye movements and attention
  in real-world scenes: the role of global features in object search.** *Psychological Review* 113(4):766–786.
  doi:10.1037/0033-295X.113.4.766. Read: Zotero PDF, model and experiment sections. A Bayesian model combines local
  saliency with a prior on target location computed from global scene features. Three groups of 8 observers counted
  people (36 street scenes), paintings or mugs (36 indoor scenes). The model's top 20% of the image held 73% of the
  first five fixations, against 58% for saliency alone (target-present images, three tasks together); observers
  agreed with each other above 90% for people search. In their words, contextual information can be "integrated
  prior to the first saccade, thereby reducing the number of image locations that need to be considered by
  object-driven attentional mechanisms". **Support:** strong for "knowledge guides where we look and where we need
  not". The prior is learned from global image statistics; it encodes the scene's layout, not the identities of the
  objects in it.
- **Henderson J.M., Hayes T.R. 2017. Meaning-based guidance of attention in scenes as revealed by meaning maps.**
  *Nature Human Behaviour* 1(10):743–747. doi:10.1038/s41562-017-0208-0. Read: Zotero PDF. Crowd raters (N = 165)
  scored the meaningfulness of context-free patches; 65 viewers studied 40 scenes for 12 s each. Meaning explained 53%
  of the variance in fixation density, saliency 38%; uniquely, 19% versus 4%, from the first fixation on.
  **Support:** moderate for "knowledge of what matters guides where we look"; nothing about inferring unseen content.
  Contested: Pedziwiatr et al. 2021 (*Cognition* 206:104465) found that DeepGaze II, a network using high-level but
  not explicitly semantic features, "outperforms MMs" and that meaning maps miss object–scene manipulations that do
  change gaze. Keep off the slide.
- **Võ M.L.-H., Boettcher S.E.P., Draschkow D. 2019. Reading scenes: how scene grammar guides attention and aids
  perception in real-world environments.** *Current Opinion in Psychology* 29:205–210.
  doi:10.1016/j.copsyc.2019.03.009. Read: Zotero PDF. "This scene grammar provides strong priors regarding what
  objects tend to be where within certain scenes." Semantic violations (a fire hydrant in a kitchen) and syntactic
  ones (a toaster in the sink); "anchor objects" (shower, sink, stove) predict where smaller objects are. On where not
  to look: "When you look for a toothbrush you will, therefore, probably quickly exclude the shower and toilet phrases
  from your search and focus your attention on the sink." Figure 1 shows one observer's fixations when searching the
  same room image, which contains neither target, for a laptop versus a teddy. **Support:** strong as a short
  review of scene grammar guiding search; the toothbrush sentence is an illustration, not data.

## 2. Closer to the phenomenon: inferring what was not sampled

### Boundary extension (beyond the edges of a view)

- **Intraub H., Richardson M. 1989. Wide-angle memories of close-up scenes.** *J. Exp. Psychol. Learn. Mem. Cogn.*
  15(2):179–187. doi:10.1037/0278-7393.15.2.179. Read: PubMed abstract; the figure via Intraub 2010. After 20
  pictures of 15 s each, 37 undergraduates' drawings showed that "95% of their drawings included information that had
  not been physically present but that would have been likely to have existed just outside the camera's field of
  view"; in Experiment 2, 85 undergraduates rated targets as closer-up than before. **Support:** the most direct
  evidence reviewed here that people fill in unsampled parts of a scene with what such a scene contains. Extrapolation, in memory.
- **Intraub H. 2010. Rethinking scene perception: a multisource model.** *Psychology of Learning and Motivation*
  52:231–264. doi:10.1016/S0079-7421(10)52006-1. Read: author's copy, pp. 231–240. Scene representation is "fleshed
  out" from "visual sensory, amodal, conceptual, and contextual" sources. "Where the visual sensory input abruptly ends
  at the picture's boundaries, perception does not end." On the trash-can picture: "all of the participants' drawings
  of this proxy view in Intraub and Richardson (1989) depicted unbroken (whole) trash cans and the continuation of the
  fence at the top and side boundaries". **Support:** strong, as the author's synthesis.
- **Intraub H., Bender R.S., Mangels J.A. 1992. Looking at pictures but remembering scenes.** *J. Exp. Psychol.
  Learn. Mem. Cogn.* 18(1):180–191. doi:10.1037/0278-7393.18.1.180. Read: PubMed abstract. Boundary extension occurred
  when the picture contained no incomplete objects, which the authors take to rule out object completion; in
  immediate tests all one-directional distortions were extensions, while after 48 hours wide-angle pictures were
  remembered as closer up ("boundary restriction"). **Support:** moderate; rules out an alternative.
- **Intraub H., Dickinson C.A. 2008. False memory 1/20th of a second later: what the early onset of boundary
  extension reveals about perception.** *Psychological Science* 19(10):1007–1014.
  doi:10.1111/j.1467-9280.2008.02192.x. Read: PubMed abstract. "We report false memory beyond the boundaries of a
  view, boundary extension, after less than 1/20th of a second"; replicated when the interruption included a saccade.
  **Support:** strong for the claim that the extension happens across a gap as brief as a saccade.
- **Park S., Intraub H., Yi D.-J., Widders D., Chun M.M. 2007. Beyond the edges of a view: boundary extension in
  human scene-selective visual cortex.** *Neuron* 54(2):335–342. doi:10.1016/j.neuron.2007.04.006. Read: PubMed
  abstract. "cortical mechanisms extrapolate missing information with highly constrained predictions about the
  environment just beyond the edges of a view"; fMRI adaptation showed boundary extension in PPA and RSC, not in LOC.
  **Support:** strong neural counterpart for this audience.
- **Bainbridge W.A., Baker C.I. 2020. Boundaries extend and contract in scene memory depending on image properties.**
  *Current Biology* 30(3):537–543.e3. doi:10.1016/j.cub.2019.12.004. Read: PubMed abstract. With 1,000 images and
  2,000 participants, "boundary contraction" was "an equally robust phenomenon"; "object-oriented images cause more
  boundary extension, scene-oriented images cause more boundary contraction". Debated: Intraub 2020, "Searching for
  boundary extension", *Curr. Biol.* 30(24):R1463–R1464, doi:10.1016/j.cub.2020.10.031; reply, R1465–R1466,
  doi:10.1016/j.cub.2020.10.032 (titles only read). **Caveat:** boundary extension is not universal; do not say
  "always".

### Amodal completion (behind an occluder, between visible parts)

- **Michotte A., Thinès G., Crabbé G. 1964. *Les compléments amodaux des structures perceptives*.** Louvain: Institut
  de psychologie de l'Université de Louvain; English in Thinès, Costall & Butterworth (eds.) 1991, *Michotte's
  Experimental Phenomenology of Perception*, Erlbaum. Metadata from the reference list of Thielen et al. 2024; text not
  read: UNVERIFIED. The origin of "amodal completion": perceiving an occluded object as continuous without seeing it.
- **Kanizsa G. 1979. *Organization in Vision: Essays on Gestalt Perception*.** Praeger. Not read. Thielen et al.
  2024 summarize his position: amodal completion "operates as a purely stimulus-driven automatic bottom-up process
  and remains resistant to knowledge-driven cognitive top-down influences". **Against** the knowledge reading: the
  classic view is that completion follows geometry, not knowledge.
- **Kellman P.J., Shipley T.F. 1991. A theory of visual interpolation in object perception.** *Cognitive Psychology*
  23(2):141–221. doi:10.1016/0010-0285(91)90009-D. Metadata only; content UNVERIFIED. The geometric account of
  interpolation between visible edges.
- **Hazenberg S.J., van Lier R. 2016. Disentangling effects of structure and knowledge in perceiving partly occluded
  shapes: an ERP study.** *Vision Research* 126:109–119. doi:10.1016/j.visres.2015.10.004. Read: PubMed abstract.
  Familiar objects with the **middle** occluded; when the occluder was removed, an ERP at 300–400 ms reflected both
  structure-compatible and knowledge-compatible completions: "the interpretation of partly occluded shapes is not
  solely driven by stimulus structure, but that it can also be influenced by knowledge of objects." **Support:** the
  closest experiment reviewed here to "what is between the two ends", though with single objects, not scenes.
- **Thielen J., Bosch S.E., van Leeuwen T.M., van Gerven M.A.J., van Lier R. 2019. Neuroimaging findings on amodal
  completion: a review.** *i-Perception* 10(2):2041669519840047. doi:10.1177/2041669519840047. Read: PubMed abstract.
  V1/V2 involvement "is not yet clear"; LOC and FFA "seem invariant to occlusion". **Support:** adjacent; the modern
  review of the neural side.
- **Thielen J., van Leeuwen T.M., Hazenberg S.J., Wester A.Z.L., de Lange F.P., van Lier R. 2024. Amodal completion
  across the brain: the impact of structure and knowledge.** *Journal of Vision* 24(6):10. doi:10.1167/jov.24.6.10.
  Read: abstract and introduction (Europe PMC full text). In LOC, responses were suppressed for "structure and
  knowledge-compatible stimuli"; in early visual cortex, no difference. **Support:** moderate; knowledge shapes
  completion in higher visual cortex.

### Gist and context at a glance

- **Potter M.C. 1976. Short-term conceptual memory for pictures.** *J. Exp. Psychol. Hum. Learn. Mem.*
  2(5):509–522. doi:10.1037/0278-7393.2.5.509. Read: PubMed abstract. Sixteen photographs in rapid sequence at 113, 167
  or 333 ms each; detection of a target named only by a title ("a boat") far exceeded recognition memory; "a scene is
  understood and so becomes immune to ordinary visual masking within about 100 msec". **Support:** adjacent; a glance
  yields meaning, which is the input to inference, not inference about unseen parts.
- **Oliva A. 2005. Gist of the scene.** In Itti L., Rees G., Tsotsos J.K. (eds.), *Neurobiology of Attention*,
  Elsevier, pp. 251–256. doi:10.1016/B978-012375731-9/50045-8. Read: chapter PDF from the Oliva lab, pp. 251–253.
  "Concurrent with gist development is the automatic activation of a framework of semantic information, including …
  predictions of which objects are likely to be found in the environment." **Support:** moderate, as a definition.
- **Davenport J.L., Potter M.C. 2004. Scene consistency in object and background perception.** *Psychological
  Science* 15(8):559–564. doi:10.1111/j.0956-7976.2004.00719.x. Read: PubMed abstract. Photographs shown for 80 ms then
  masked: objects identified better in consistent settings and backgrounds better with consistent objects. "Semantic
  consistency information is available when a scene is glimpsed briefly". **Support:** moderate; photographs rather
  than line drawings.

### Transsaccadic integration

- **Higgins E., Rayner K. 2015. Transsaccadic processing: stability, integration, and the potential role of
  remapping.** *Attention, Perception, & Psychophysics* 77:3–27. doi:10.3758/s13414-014-0751-y. Read: Zotero PDF,
  abstract and introduction. Reviews visual stability and integration across fixations, and predictive remapping.
  **Support:** adjacent. It concerns combining what was sampled across saccades, not inferring what was not sampled.
  Irwin 1991, summarized in `citations.md` as undetailed, capacity-limited memory across saccades, falls in the same
  category (not read this session).

### Predictive processing and filling in, in visual cortex

- **Rao R.P.N., Ballard D.H. 1999.** *Nature Neuroscience* 2(1):79–87. doi:10.1038/4580. Predictive coding;
  already in `citations.md` and the CanViT paper. **Support:** adjacent; a mechanism for extra-classical
  receptive-field effects, not evidence about scene content.
- **Smith F.W., Muckli L. 2010. Nonstimulated early visual areas carry information about surrounding context.**
  *PNAS* 107(46):20099–20103. doi:10.1073/pnas.1000233107. Read: accepted manuscript (Glasgow eprints). Three
  photographs with the lower-right quadrant occluded; fMRI patterns in the non-stimulated V1/V2 region identified which
  scene surrounded it (65 ± 5.2% correct, chance 33%, 6 participants in Experiment 1). The authors leave the carried
  information open (feedback predictions, attention to expected features, or low-level surround statistics).
  **Support:** moderate; the unseen region's cortex is driven by the context, with three images and no claim about
  predicted content.
- **Morgan A.T., Petro L.S., Muckli L. 2019. Scene representations conveyed by cortical feedback to early visual
  cortex can be described by line drawings.** *J. Neurosci.* 39(47):9410–9423. doi:10.1523/JNEUROSCI.0852-19.2019.
  Read: PubMed abstract. "fMRI activity patterns corresponding to occluded visual information in early visual cortex
  fill in scene-specific features", and line drawings of the missing information correlate with them.
  **Support:** the closest neural evidence reviewed here that cortex represents the content of an unseen
  region.
- **Muckli L. et al. 2015. Contextual feedback to superficial layers of V1.** *Current Biology* 25(20):2690–2695.
  doi:10.1016/j.cub.2015.08.057. Read: PubMed abstract. Contextual information in occluded V1 peaks in superficial
  layers, consistent with feedback. **Support:** adjacent (laminar mechanism).
- **Peelen M.V., Berlot E., de Lange F.P. 2024. Predictive processing of scenes and objects.** *Nature Reviews
  Psychology* 3:13–26 (online 23 November 2023). doi:10.1038/s44159-023-00254-0. Read: abstract and figure titles on
  nature.com. Findings "can be understood by conceptualizing object and scene perception as the outcome of a joint
  probabilistic inference in which best guesses about objects act as priors for scene perception and vice versa."
  **Support:** strong as the current review of the slide's thesis; body not read.
- **Spaak E., Peelen M.V., de Lange F.P. 2022. Scene context impairs perception of semantically congruent objects.**
  *Psychological Science* 33(2):299–313. doi:10.1177/09567976211032676. Read: PubMed abstract. Congruent objects were
  perceived worse than incongruent ones (N = 300; change detection and a bias-free discrimination task), as reduced
  prediction error predicts; Peelen et al. 2024's reference note places this "when sensory input is unambiguous".
  **Caveat:** priors can also cost: with clear input, the expected object can be processed less.

### Knowledge and where the eyes go (beyond Torralba et al. 2006)

- **Loftus G.R., Mackworth N.H. 1978. Cognitive determinants of fixation location during picture viewing.** *J.
  Exp. Psychol. Hum. Percept. Perform.* 4(4):565–572. doi:10.1037/0096-1523.4.4.565. Read: PubMed abstract. Observers
  "fixate earlier, more often, and with longer durations on objects that have a low probability of appearing in a
  scene (e.g., an octopus in a farm scene)". The "earlier" part has not always replicated (next entry).
- **Võ M.L.-H., Henderson J.M. 2009. Does gravity matter? Effects of semantic and syntactic inconsistencies on the
  allocation of attention during scene perception.** *Journal of Vision* 9(3):24. doi:10.1167/9.3.24. Read: Zotero
  PDF, abstract and introduction. No evidence of extrafoveal detection of inconsistencies, but once fixated,
  inconsistent objects got longer gaze durations and more fixations. **Support:** expected objects need less looking;
  unexpected ones are not found from afar.
- **Pedziwiatr M.A., von dem Hagen E., Teufel C. 2023. Knowledge-driven perceptual organization reshapes information
  sampling via eye movements.** *J. Exp. Psychol. Hum. Percept. Perform.* 49(3):408–427. doi:10.1037/xhp0001080.
  Read: PubMed abstract (Zotero has the PDF). On images that look like meaningless patches until the depicted object
  is known, "fixations on identical images are more object-centered, less dispersed, and more consistent across
  observers once these images are organized into objects", from the first fixations on. **Support:** strong:
  same pixels, different knowledge, different sampling.

### Classic statements and the opposing view

- **Gregory R.L. 1980. Perceptions as hypotheses.** *Phil. Trans. R. Soc. Lond. B* 290(1038):181–197.
  doi:10.1098/rstb.1980.0090. Read: PubMed abstract. "Perceptions may be compared with hypotheses in science."
  **Support:** a classic restatement of Helmholtz.
- **Hollingworth A., Henderson J.M. 1998. Does consistent scene context facilitate object perception?** *J. Exp.
  Psychol. Gen.* 127(4):398–415. doi:10.1037/0096-3445.127.4.398. Read: PubMed abstract. Replicated Biederman's
  paradigm, but with response bias controlled or a forced choice, "no such advantage obtained"; "object perception is
  not facilitated by consistent scene context."
- **Pylyshyn Z. 1999. Is vision continuous with cognition? The case for cognitive impenetrability of visual
  perception.** *Behavioral and Brain Sciences* 22(3):341–365. doi:10.1017/S0140525X99002022. Read: Crossref
  abstract. "an important part of visual perception, corresponding to what some people have called early vision, is
  prohibited from accessing relevant expectations, knowledge, and utilities"; the abstract allows "top-down
  interactions that are internal to the early vision system".
- **Firestone C., Scholl B.J. 2016. Cognition does not affect perception: evaluating the evidence for "top-down"
  effects.** *Behavioral and Brain Sciences* 39:e229 (online 2015). doi:10.1017/S0140525X15000965. Read: Crossref
  abstract. "None of these hundreds of studies … provides compelling evidence for true top-down effects on
  perception", where the alleged influences are "beliefs, desires, emotions, motivations, intentions, and linguistic
  representations". The slide's "prior knowledge" means learned regularities of scenes; whether that counts as
  top-down penetration in this debate depends on wording. "Inference from knowledge of how the world is laid out"
  keeps the claim on the less contested side.
- **Brewer W.F., Treyens J.C. 1981. Role of schemata in memory for places.** *Cognitive Psychology* 13(2):207–230.
  doi:10.1016/0010-0285(81)90008-6. Metadata only. The often-told result (participants waiting in an office later
  recalled books that were not there) is UNVERIFIED; the publisher page was blocked.

## 3. Figures

Extracted to `/Users/yberreby/code/CanViT/site/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/assets/figures/`
(provenance rows in `PROVENANCE.md` there):

- `intraub-richardson-1989-boundary-extension.png` (442×885 px): close-up photograph, a participant's drawing from
  memory with the scene continued past the edges, the wider view. The recommended figure for the slide's main claim.
- `biederman-1982-hydrant-on-mailbox.png` (1364×1020 px): line-drawn street with a fire hydrant on a mailbox (a
  Position violation). Pairs with "where things go".
- `torralba-2006-contextual-guidance.png` (1553×1566 px), extra: the model's maps and fixations, including the same
  kitchen searched for paintings (walls) and for mugs (counter). The photographs are low-resolution in the source.
- `oliva-torralba-2007-object-averages.png` (1439×518 px), extra: average images around a face, a keyboard and a fire
  hydrant; the surroundings of an object are predictable from image statistics.

Not extracted, worth considering: Oliva & Torralba 2007 Figure 2 (the car and pedestrian blobs, identical but for a
90° rotation; embedded image 686×713 px, p. 521); Võ et al. 2019 Figure 1 (laptop versus teddy search in a room
image containing neither, one observer; 946×1428 px, p. 206); Biederman et al. 1982 Fig. 3 ("Goodyear Sofa", p. 154); the right column
of Intraub 2010 Figure 1 (Intraub et al. 1996, pictures shown for 250 ms).

## 4. Bibliography notes

- Helmholtz: cite 1867 for the volume; §26 appeared in the third part, 1866.
- Firestone & Scholl: online 2015, volume 39 dated 2016, article e229.
- Peelen, Berlot & de Lange: online 2023, issue 2024 (*Nat. Rev. Psychol.* 3:13–26).
- Intraub's WIREs review "Rethinking visual scene perception" (doi:10.1002/wcs.149) is online 2011, in print 2012,
  3(1):117–127; not read.
- Bruner 1957: pages, editors and content UNVERIFIED (above).
