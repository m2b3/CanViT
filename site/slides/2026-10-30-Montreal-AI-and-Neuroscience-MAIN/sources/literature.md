# Literature for the MAIN 2026 talk

Facts read in primary sources, kept so that nobody has to find them again. This file holds what papers say
(quotes, numbers, tables, figures, pages) and where it was read; it holds no plan for the talk (that is
`../TALK.md`). Merged on 2026-10-24 from research notes that Claude Code subagents wrote on 2026-10-01 (git history:
`sources/active-vision-timeline.md`, `active-vision-novelty.md`, `foundation-models.md`, `concepts.md`,
"Perception as inference" below, "Citation records"), with their slide recommendations removed; none of it was reviewed by the authors.

How to read it:
- Each entry says what was read: **full**, **part**, **abstract** or **metadata**. "Zotero" is a PDF in the
  authors' library. **UNVERIFIED** marks what was not checked in the source itself. Quotes are verbatim; numbers are
  the papers' own. "p." is the page index of the PDF read unless stated.
- Lines marked **[inferred]** and bullets marked **agent's mapping** are a subagent's reading, not a paper's claim
  and not the authors' position. Treat them as hypotheses.
- A paper has one entry. Where an entry here and `checked-facts.md` disagree, check the source and fix the wrong one.

## Prior active vision

### Deep active vision models, as read in their papers

#### How the entries were assembled

1. Read the CanViT paper's Introduction and Related work
   (`paper/latex/CanViT_Toward_AVFMs.tex`) and the `references.bib` entries it
   cites for active vision; read the notes this file was merged from.
2. Located PDFs in a copy of the Zotero database. RAM, DRAM, Saccader, GFNet, AdaGlimpse, AdaptiveNN, Bajcsy,
   Aloimonos et al., Rangrej & Clark 2021, GAE, TNet, DRAW and STN came from `~/Zotero/storage/`; STAM, AME,
   Ranzato, Denil et al. and Sermanet et al. from arXiv; Larochelle & Hinton from the author's site
   (`https://www.cs.toronto.edu/~hinton/absps/nips_eyebm.pdf`). Ballard 1991 is closed access and was not read.
3. Extracted text with `pdftotext`, read each paper's abstract, introduction, method and results, and located figure
   and quote pages with a script that splits the text on page breaks. "p." below is the page index in the PDF file
   named; for the Zotero PDFs that is the version stored there.
4. First-release dates come from the arXiv API (`published` of v1); venues from the PDF header, the arXiv comment or
   Crossref, as each entry says. Semantic Scholar rate-limited most queries and DBLP refused automated ones this
   session.

#### 2014: Recurrent Attention Model (RAM)

- **Paper.** Mnih V., Heess N., Graves A., Kavukcuoglu K. "Recurrent Models of Visual Attention." NIPS 2014
  (Advances in Neural Information Processing Systems 27). arXiv:1406.6247, v1 2014-06-24. Zotero PDF:
  `~/Zotero/storage/K2766XKM/` (arXiv v1).
- **What it did.** A recurrent network takes a retina-like glimpse (several concentric patches at decreasing
  resolution) at each step, updates its hidden state, and samples the next glimpse location from a Gaussian whose
  mean it outputs; the location network is trained with REINFORCE and a learned baseline, the classifier with
  cross-entropy (Sections 3.1–3.2, pp. 3–5).
- **Why it mattered.** "the amount of computation it performs can be controlled independently of the input image
  size. While the model is non-differentiable, it can be trained using reinforcement learning methods to learn
  task-specific policies" (Abstract, p. 1). The CanViT paper: deep active vision "traces back to the Recurrent
  Attention Model" (Related work).
- **Headline.** 60×60 Cluttered Translated MNIST: 5.23% error with 8 glimpses against 7.83% for a convolutional
  network with about the same number of parameters; 100×100: 10.83% against 16.51% (Table 2, p. 7).
- **Figure.** Figure 1 (A–C), p. 3: glimpse sensor, glimpse network, unrolled model. Already extracted as
  `../assets/figures/mnih-2014-ram.png`. Alternative: Figure 3, p. 7, glimpse paths on cluttered digits.
- **Note.** The paper also trains RAM to play a 24×24-pixel game of "Catch" from the reward alone (Section 4.2).

What it introduced, against the precursors (read 2026-10-01; [inferred] marks the reader's own reading):

**New.** One recurrent network does everything at once: it encodes each glimpse together with its location, keeps a
memory of past glimpses, chooses the next location, and gives the answer. The whole system is trained end to end from
the task alone: the classifier by ordinary gradient descent, the location choice by REINFORCE with a learned baseline,
rewarded only by whether the final answer is right (pp. 4–5). Against the precursors, what is new is the combination
of continuous locations, a non-greedy objective (the reward comes at the end of the sequence) and a single network
that also works as an agent in a dynamic game (Catch, p. 8). RAM's own statement of the difference: "Our formulation
which employs an RNN to integrate visual information over time and to decide how to act is, however, more general,
and our learning procedure allows for end-to-end optimization of the sequential decision process instead of relying
on greedy action selection" (p. 2).

**Evidence.** "the amount of computation it performs can be controlled independently of the input image size. While
the model is non-differentiable, it can be trained using reinforcement learning methods to learn task-specific
policies" (Abstract, p. 1). The glimpse is retina-like: "k square patches centered at location l, with the first
patch being gw × gw pixels in size, and each successive patch having twice the width of the previous. The k patches
are then all resized to gw × gw and concatenated" (p. 5). Headline (Table 2, p. 7): 60×60 Cluttered Translated MNIST,
5.23% error with 8 glimpses (12×12 retina, 3 scales) against 7.83% for a convolutional network of roughly the same
parameter count; 100×100, 10.83% (8 glimpses, 4 scales) against 16.51%. "the overall capacity and the amount of
computation of our model does not change from 60 × 60 images to 100 × 100" (p. 8). Detail worth knowing: "the first
glimpse is always random" (p. 6).

**Contestable.**
- Learning where to look from a reward predates RAM: Butko & Movellan 2008 (policy gradient, per RAM p. 2 and
  Saccader p. 2), Paletta et al. 2005 (Q-learning, cited by RAM), Denil et al. 2012 (bandits, read). RAM's novelty is
  the end-to-end deep recurrent version, not reinforcement learning for gaze as such.
- "Computation independent of image size" is also Ranzato 2014's stated point, on arXiv two months earlier and cited
  by RAM.
- The word "glimpse" and the multi-resolution retina come from Larochelle & Hinton 2010: "We will refer to this
  low-resolution representation as a glimpse [14]" (RAM p. 4).

#### 2014: Deep Recurrent Attention Model (DRAM)

- **Paper.** Ba J., Mnih V., Kavukcuoglu K. "Multiple Object Recognition with Visual Attention." ICLR 2015.
  arXiv:1412.7755, v1 2014-12-24. Zotero PDF: `~/Zotero/storage/NPYK4NGA/` (arXiv v2).
- **What it did.** A two-layer LSTM reads multi-resolution glimpses one per step and emits the next location; a
  "context network" sees "a down-sampled low-resolution version of the whole input image" and sets the initial
  state that places the first glimpse; it reads several objects in sequence and is trained by a variational bound
  that the authors show is equivalent to REINFORCE (Section 3, pp. 2–4).
- **Why it mattered.** "While RAM was shown to learn successful gaze strategies on cluttered digit classification
  tasks and on a toy visual control problem it was not shown to scale to real-world image tasks or multiple objects"
  (Related work, p. 2). Conclusion: it "outperformed the state-of-the-art ConvNets on a multi-digit house number
  recognition task while using both fewer parameters and less computation" (Section 6, p. 8). The CanViT paper
  groups it with work that "remained largely confined to simple tasks such as digit recognition" (Related work).
- **Headline.** SVHN whole-sequence error 3.9% (forward-backward DRAM with Monte Carlo averaging) against 3.96% for
  an 11-layer convolutional network and 4.11% for the authors' 10-layer one (Table 3, p. 7). Against that 10-layer
  network on 54×54 inputs: 0.7 against 2.1 GFLOPs and 28M against 51M parameters (Table 5, p. 8).
- **Figure.** Figure 1, p. 2 (model diagram with the coarse-image context network); Figure 2, p. 6 (learned glimpse
  sequences on digit pairs and digit addition).
- **Note.** `sota-history.md` lists DRAM at 67.5% on ImageNet-1k, a reimplementation by Saccader's authors.

What it introduced, against the precursors (read 2026-10-01; [inferred] marks the reader's own reading):

**New.**
- Several objects in one image, read one after another: the model emits a label sequence with an end-of-sequence
  symbol and learns to localize each object "despite being given only class labels during training" (Abstract, p. 1;
  Section 3.2, p. 5). RAM recognized one object.
- A context network that sees "a down-sampled low-resolution version of the whole input image" and sets the initial
  state that places the first glimpse (p. 3). It is wired to the upper LSTM layer only, so "the contextual information
  cannot be used directly by the classification network and only affects the sequence of glimpse locations" (p. 3).
- A deeper glimpse pathway (three convolutional layers, a two-layer LSTM separating classification from location),
  and a derivation of the location update as a variational bound that "is equivalent to the REINFORCE (Williams,
  1992) learning rule employed in Mnih et al. (2014)" (p. 4).
- A real-image benchmark at state-of-the-art accuracy: multi-digit house numbers (SVHN).

**Evidence.** "While RAM was shown to learn successful gaze strategies on cluttered digit classification tasks and on
a toy visual control problem it was not shown to scale to real-world image tasks or multiple objects" (p. 2). SVHN
images are "tightly cropped 64 x 64 images with multi-digits at the center", jittered to 54×54 and converted to
grayscale, 3 glimpses per digit, up to 18 glimpses (p. 7). Headline (Table 3, p. 7): whole-sequence error 3.9% for a
forward-backward DRAM pair with Monte Carlo averaging, against 3.96% (Goodfellow et al.'s 11-layer CNN) and 4.11% (the
authors' 10-layer CNN); a single DRAM gets 5.1% (4.4% with Monte Carlo averaging). Cost at 54×54 (Table 5, p. 8): 0.7
against 2.1 GFLOPs, 28M against 51M parameters. The context network's effect on MNIST digit pairs (Table 1, p. 6): DRAM
5%, DRAM without context 7%, RAM 9%.

**Contestable.**
- A coarse first look that decides where to look sharply is Ranzato 2014's key idea; DRAM does not cite Ranzato.
- The multiplicative "what × where" combination is credited by DRAM itself to Larochelle & Hinton 2010 (p. 3).
- "First on real images" would be wrong: Larochelle & Hinton 2010 used face photographs, Denil et al. 2012 real video.
  DRAM's claim is narrower and holds: matching the best convolutional networks on a real-image benchmark.
- "Outperforms the state of the art" (Conclusion, p. 8) rests on 3.9% against 3.96%, with an ensemble of two models
  and Monte Carlo averaging; a single model is worse than the CNNs. "Matches" is the defensible word.
- DRAM is not purely glimpse-based: its first glimpse is aimed from the whole image at low resolution.

#### 2019: Saccader

- **Paper.** Elsayed G.F., Kornblith S., Le Q.V. "Saccader: Improving Accuracy of Hard Attention Models for Vision."
  NeurIPS 2019. arXiv:1908.07644, v1 2019-08-20. Zotero PDF: `~/Zotero/storage/ZTLRTF8J/` (NeurIPS version).
- **What it did.** A BagNet-style network computes features with a 77×77-pixel receptive field at 361 locations of a
  224 px image in one pass; an attention network and a "Saccader cell" pick 6 locations in turn, masking visited
  ones. Training has three steps: the representation network, then a self-supervised pretraining of the location
  network to emit locations in order of their class logits, then REINFORCE on whether the final prediction is
  correct (Sections 3.1–3.2, pp. 3–5).
- **Why it mattered.** "training hard attention models with only class label supervision is challenging, and hard
  attention has proved difficult to scale to complex datasets. [...] Key to Saccader is a pretraining step that
  requires only class labels and provides initial attention locations for policy gradient optimization" (Abstract,
  p. 1). The CanViT paper: active vision "remained largely confined to simple tasks such as digit recognition until
  2019, when Saccader achieved 75% ImageNet-1k top-1 accuracy by introducing an intermediate pretraining step to
  stabilize learning" (Related work). That 75% point is the Saccader-NASNet variant with a separate NASNet classifier;
  the paper's Section 4.3 and Figure 6 describe that classifier explicitly.
- **Headline.** ImageNet-1k top-1 75.03 ± 0.08% and top-5 91.19%, with 6 glimpses, "processing only 29.47 ± 0.26% of
  the image" with a separate NASNet classifier on 331 px images (Section 4.3 text, p. 9; Figure 6, p. 8). Reused
  from `sota-history.md`, re-read here. The base model at 224 px scores 70.31% (`sota-history.md`).
- **Figure.** Figure 1, p. 2: glimpses chosen by Saccader against those of DRAM on ImageNet photographs. Alternative:
  Figure 4, p. 7 (Saccader, DRAM, ordered-logits and random policies side by side).
- **Caveat an audience may raise.** "the attention network has access to the entire image" (Conclusion, p. 9); only
  the classification path is restricted to the glimpses.

What it introduced, against the precursors (read 2026-10-01; [inferred] marks the reader's own reading):

**New.**
- A pretraining step for the location network before reinforcement learning: it is first trained, from class labels
  only, to emit locations in descending order of a patch classifier's logits, then fine-tuned with REINFORCE on
  whether the final answer is right (p. 5). This targets the sparse-reward problem: "Our pretraining procedure
  overcomes the sparse-reward problem that makes hard attention models difficult to optimize" (p. 1).
- An architecture in which glimpses are selections, not crops: a BagNet computes features with a 77×77-pixel
  receptive field at 361 locations of the whole 224 px image in one pass; an attention network with a wide receptive
  field and a "Saccader cell" with a memory of visited locations pick 6 of them (pp. 3–4).
- Hard attention brought to ImageNet at accuracy approaching standard CNNs.
- Its motivation is interpretability, not efficiency or partial observation: "One approach that offers some level of
  interpretability by design is hard attention, which uses only relevant portions of the image" (Abstract, p. 1).

**Evidence.** "Key to Saccader is a pretraining step that requires only class labels and provides initial attention
locations for policy gradient optimization" (Abstract, p. 1). "Our results also suggest that the pretraining procedure
is necessary to achieve this performance (see Figure Supp.3...)" (p. 7; the supplementary figure was not read). The
75%: "with 6 glimpses, the top-1 and top-5 accuracy were 75.03 ± 0.08% and 91.19 ± 0.22%, respectively, while
processing only 29.47 ± 0.26% of the image with the NASNet" (p. 9). That number is Saccader-NASNet: Saccader picks
locations on the image downsized to 224 px, and a separate NASNet, fine-tuned on patches, classifies the patches cut
from the 331 px image (p. 9). Saccader on its own at 224 px is listed at 70.31% (6 glimpses of 77² px) in AdaGlimpse's
Table 2 (AdaGlimpse p. 11); Saccader's own text gives that point only in Figure 3. Does the attention network see the
whole image? Yes: "the attention network has access to the entire image, and thus the patch selection process remains
difficult to interpret" (Conclusion, p. 9); only the classification path is restricted to 77 px patches (p. 7).

**Contestable.**
- The CanViT paper (Related work) and the notes both credit "Saccader" with 75%; the 75% needs a separate, larger
  classifier (NASNet) and 331 px inputs. Saccader alone: about 70% (70.31%, as tabulated by AdaGlimpse).
- Saccader does not save computation or work from partial views: its feature network processes the entire image
  before any choice is made. [inferred: it chooses which evidence to use rather than where to point its sensor.]
- Hard attention on real photographs before 2019: Sermanet, Frome & Real, "Attention for Fine-Grained
  Categorization", ICLR 2015 workshop, applied DRAM to Stanford Dogs; Saccader judges it "achieved only a modest
  accuracy improvement over a non-attentive baseline, and did not appear to derive significant benefit from glimpses
  beyond the first" (p. 2).

#### 2020: Glance and Focus Network (GFNet)

- **Paper.** Wang Y., Lv K., Huang R., Song S., Yang L., Huang G. "Glance and Focus: a Dynamic Approach to Reducing
  Spatial Redundancy in Image Classification." NeurIPS 2020. arXiv:2010.05300, v1 2020-10-11. Zotero PDF:
  `~/Zotero/storage/ME7L7SIJ/`. Journal version: Huang et al., "Glance and Focus Networks for Dynamic Visual
  Recognition", IEEE TPAMI, arXiv:2201.03014.
- **What it did.** A global encoder "glances" at the whole image down-sampled to the patch size (e.g., 96×96); if
  the prediction is not confident enough, a patch proposal network picks a full-resolution patch for a local encoder,
  and so on until a confidence threshold is passed or T steps elapse. Encoders are off-the-shelf CNNs; the patch
  policy is trained with proximal policy optimization (PPO) (Sections 1 and 3, pp. 2–6; PPO on p. 6).
- **Why it mattered.** "Experiments on ImageNet show that our method consistently improves the computational
  efficiency of a wide variety of deep models" (Abstract, p. 1); "With ResNets and DenseNets, GFNet reduces the
  number of required Multiply-Adds for the given test accuracy by approximately 2 − 3× times" (Section 4.1, p. 7).
  The CanViT paper: GFNet and AdaptiveNN "showed that active vision could deliver computational efficiency gains on
  real-world tasks, although both remained structurally limited to classification tasks and fixed zoom levels"
  (Related work).
- **Headline.** ImageNet-1k: 75.93% with ResNet-50, 96 px patches, T = 5 (Table 1, p. 8; reused from
  `sota-history.md`); MobileNet-V3-Large at 75.4% needs 22% less latency on a mobile phone, 12.7 ms against 16.3 ms
  (Section 4.1, p. 7).
- **Figure.** Figure 1, p. 2: an easy image (eagle) classified at the glance, harder ones (husky, stork) after one or
  two focus steps, with the fraction of computation used. Alternative: Figure 7, p. 8, patch sequences on test
  images.

What it introduced, against the precursors (read 2026-10-01; [inferred] marks the reader's own reading):

**New.**
- Active vision as a wrapper around standard, off-the-shelf CNNs (MobileNet-V3, RegNet, EfficientNet, ResNet,
  DenseNet), initialized from ImageNet-pretrained weights (pp. 1, 6): a global encoder "glances" at the whole image
  resized to the patch size (e.g., 96×96), a local encoder processes full-resolution 96×96 patches (pp. 2, 4).
- Per-image stopping: the sequence ends once the softmax confidence passes a threshold, with thresholds set from a
  computation budget, adjustable without retraining (pp. 2, 6).
- Measured wall-clock speed-ups on a phone, and a reward defined as the increase of the true-class probability,
  optimized with PPO after a stage of training on random patches (pp. 5–6).

**Evidence.** GFNet's own statement of the difference with RAM: "1) we adopt a flexible and general CNN-based
framework that is compatible with a wide variety of CNNs to achieve SOTA computational efficiency, instead of sticking
to a pure RNN model; and 2) our network focuses on performing adaptive inference for higher efficiency, and the
recurrent process can be terminated conditioned on each input" (p. 3). "With ResNets and DenseNets, GFNet reduces the
number of required Multiply-Adds for the given test accuracy by approximately 2 − 3× times" (p. 7). "our method
reduces the required latency to achieve 75.4% test accuracy (MobileNets-V3-Large) by 22% (12.7ms v.s. 16.3ms)" on an
iPhone XS Max (p. 7). Headline: ImageNet-1k 75.93% top-1, ResNet-50, 96 px patches, 5 steps (Table 1, p. 9).

**Contestable.**
- Every element of glance, focus and confident early stopping is in Ranzato 2014 (low-resolution first look, sharp
  patches, confidence-threshold cascade on jittered MNIST); GFNet does not cite Ranzato. DRAM (2014) also had the
  whole-image coarse look. RAM's Discussion reports preliminary experiments with a learned stop action (p. 8).
- "Active vision as a speed-up" was already RAM's, Ranzato's and DRAM's motivation (DRAM measured fewer FLOPs than a
  CNN, Table 5). GFNet's novelty is making it pay on ImageNet with modern efficient networks, in measured latency.
- An observation useful for the CanViT thesis: in GFNet's ablation (Table 1, p. 9), at 5 steps a random patch policy
  with the glance reaches 74.46% and a fixed centre-then-corners order with the glance 75.12%, against 75.93% for the
  learned policy. Most of the accuracy comes from the observer; the learned policy adds 0.8 to 1.5 points there.

#### 2022: Sequential Transformers Attention Model (STAM)

- **Paper.** Rangrej S.B., Srinidhi C.L., Clark J.J. "Consistency driven Sequential Transformers Attention Model for
  Partially Observable Scenes." CVPR 2022, pp. 2508–2517 (Crossref, doi:10.1109/CVPR52688.2022.00255).
  arXiv:2204.00656, v1 2022-04-01. Authors at McGill University and the University of Toronto (PDF header). Not in
  Zotero; downloaded from arXiv.
- **What it did.** The image is split into a 7×7 grid of 32×32 glimpses; starting from a random glimpse, a
  DeiT-distilled transformer encodes the glimpses seen so far, and an actor network picks the next unobserved
  location, trained with a one-step actor-critic. The reward and a consistency loss pull the agent's class
  distribution toward that of a teacher transformer which sees the complete image (Sections 3–4, pp. 2–5).
- **Why it mattered.** "Most hard attention models initially observe a complete scene to locate and sense
  informative glimpses [...] we develop a Sequential Transformers Attention Model (STAM) that only partially
  observes a complete image and predicts informative glimpse locations solely based on past glimpses" (Abstract,
  p. 1). "While RAM could not scale beyond MNIST dataset, our approach scales to large-scale real-world datasets"
  (Introduction, p. 2). Its Table 1 marks DRAM, GFNet, Saccader, TNet and PatchDrop as using the complete image for
  attention, and STAM as using it for neither attention nor classification.
- **Headline.** ImageNet-1k 80.78% with a DeiT-B core and 78.25% with DeiT-S, both at t = 26 (27.7K pixels sensed of
  50,176); 76.35% with DeiT-S at 20.5K pixels (Table 1, p. 8; results averaged over ten runs).
- **Figure.** Figure 5, p. 7: glimpses chosen on ImageNet and fMoW images from t = 0 to 15, with the full image shown
  for reference only. Alternative: Figure 1, p. 1 (schematic).
- **Glimpse count.** Time steps start at t = 0, so t = 26 means 27 glimpses (27 × 1,024 = 27,648 pixels, about
  55% of the image; the table prints 27.7K). `sota-history.md` says "26 glimpses of 32 px" for the same point; the
  pixel counts of the other rows (t = 13 gives 14.3K, t = 19 gives 20.5K) match t + 1 glimpses, and AdaGlimpse's
  Table 2 lists STAM's t = 13 result as "14 × 32²".

#### 2023: Attention-Map Entropy (AME)

- **Paper.** Pardyl A., Rypeść G., Kurzejamski G., Zieliński B., Trzciński T. "Active Visual Exploration Based on
  Attention-Map Entropy." IJCAI 2023, pp. 1303–1311 (Crossref, doi:10.24963/ijcai.2023/145). arXiv:2303.06457, v1
  2023-03-11. Zotero has only the publisher's web page; the arXiv v3 PDF was read.
- **What it did.** A masked autoencoder (MAE) with a ViT-L encoder takes the glimpses seen so far; its decoder
  predicts the whole image (reconstruction or segmentation), with a classification head on the encoder for
  classification. The next glimpse is the unobserved patch whose row in the decoder's self-attention map has the
  highest entropy; there is no policy network and no reinforcement learning (Section 3, pp. 2–3).
- **Why it mattered.** "It leverages the internal uncertainty of the transformer-based model to determine the most
  informative observations. In contrast to existing solutions, it does not require additional loss components, which
  simplifies the training" (Abstract, p. 1). The CanViT paper names AME and AdaGlimpse as "among the few exceptions"
  that achieve dense prediction, and "the state of the art on active ADE20K segmentation" (Related work).
- **Headline.** ADE20K segmentation: 27.6% IoU with a SETR-initialised encoder, 24.4% with MAE weights (Table 3,
  p. 6; reused from `sota-history.md`). SUN360 classification: 75.7% (Table 2, p. 6).
- **Figure.** Figure 4, p. 4: step-by-step selection of 8 glimpses on a 256×128 image, with the input, the
  reconstruction and the attention-entropy map at each step. Alternative: Figure 3, p. 3 (how the entropy map is
  computed from an attention map).
- **Caveat.** The SETR initialisation was already trained on ADE20K segmentation (`sota-history.md`, caveat 7).

What it introduced, against the precursors (read 2026-10-01; [inferred] marks the reader's own reading):

**New.** Glimpse selection without any component trained for it: no policy network, no auxiliary loss, no
reinforcement learning. A masked autoencoder (ViT-L encoder on the patches seen so far, decoder over the full grid)
is trained only for its task; the next glimpse is the unobserved patch whose row in the last decoder layer's
self-attention map has the highest entropy, summed over heads (pp. 2–3). The same decoder predicts the whole scene
(reconstruction or segmentation); a head on the encoder's CLS token classifies (p. 4). It also reports retina-like
glimpses with a transformer, "for the first time" by its own account (p. 2).

**Evidence.** "It leverages the internal uncertainty of the transformer-based model to determine the most informative
observations. In contrast to existing solutions, it does not require additional loss components, which simplifies the
training" (Abstract, p. 1). "we select the patch that has the highest entropy and confuses our model the most"
(p. 2). Headline (Table 3, p. 6): ADE20K 27.6% IoU with an encoder initialized from SETR (a model already trained for
ADE20K segmentation), 24.4% from MAE weights; the table does not state the glimpse regime, and AdaGlimpse's Table 3
lists AME's 24.4% as 8 glimpses of 48×48 px on 128×256 images, 56.25% of the pixels (AdaGlimpse p. 12). SUN360
classification 75.7% (Table 2, p. 6). The first glimpse comes from the same rule run on mask tokens only (p. 4), so it
lands in the same place on every image (Figure 9, p. 7).

**Contestable.**
- Choosing glimpses without reinforcement learning is old: Larochelle & Hinton 2010 (a controller fitted by
  regression), Ranzato 2014 (local search). AME's novelty is narrower: no extra module or loss at all.
- Predicting the whole scene densely from glimpses predates AME: Seifi & Tuytelaars 2019 (reconstruction of 360°
  images), "Attend and Segment" 2020 and "Glimpse-Attend-and-Explore" (Seifi, Jha & Tuytelaars, ICCV 2021), which AME
  compares against in its Tables 1 and 3 (pp. 5–6). Looking where uncertainty is highest is also Denil et al. 2012's
  reward (tracking uncertainty).
- The 27.6% starts from weights already trained on ADE20K segmentation from full images.
- The selection rule buys little over chance in AME's own ablation: SUN360 classification with 8 glimpses of 32² px,
  72.4% random against 73.4% attention-entropy (Table 5, p. 7).

#### 2024: AdaGlimpse

- **Paper.** Pardyl A., Wronka M., Wołczyk M., Adamczewski K., Trzciński T., Zieliński B. "AdaGlimpse: Active Visual
  Exploration with Arbitrary Glimpse Position and Scale." ECCV 2024, Lecture Notes in Computer Science, pp. 112–129;
  online 2024-10-26, print volume 2025 (Crossref, doi:10.1007/978-3-031-72664-4_7). arXiv:2404.03482, v1
  2024-04-04. Zotero PDF: `~/Zotero/storage/PRXKWUF4/` (arXiv v1).
- **What it did.** A ViT-B encoder with position encodings computed from each patch's coordinates (from "Beyond
  Grids") takes all glimpses so far; a Soft Actor-Critic agent outputs a continuous action (x, y, scale) for the next
  glimpse, rewarded by the decrease of the task loss; a MAE-like decoder over a full grid of mask tokens produces
  dense outputs (Section 3, pp. 4–8).
- **Why it mattered.** "While modern AVE methods have demonstrated impressive performance, they are constrained to
  fixed-scale glimpses from rigid grids. In contrast, existing mobile platforms equipped with optical zoom
  capabilities can capture glimpses of arbitrary positions and scales" (Abstract, p. 1); the approach "enables our
  model to rapidly establish a general awareness of the environment before zooming in for detailed analysis" (same).
- **Headline.** ImageNet-1k 77.54% with 14 glimpses of 32×32 px, 28.57% of the pixels (Table 2, p. 11); ADE20K
  25.7% mIoU with 8 glimpses of 48×48 px, 36.73% of the pixels (Table 3, p. 12). Both reused from
  `sota-history.md`.
- **Figure.** Fig. 1, p. 2: a wide low-resolution first glimpse, then zooms toward a dog, with the prediction and
  its probability at each step. Alternative: Fig. 4, p. 9 (seven ImageNet examples with glimpse boxes, visible pixels
  and predicted label).
- **Caveat.** Its Table 2 compares with STAM's DeiT-S result at the same 14 glimpses (76.13%), not with STAM's 80.78%
  at 27 glimpses.

What it introduced, against the precursors (read 2026-10-01; [inferred] marks the reader's own reading):

**New.** Each glimpse is a square of any position and any size, chosen from a continuous action (x, y, scale) by a
Soft Actor-Critic agent and resampled to a fixed sensor resolution, so a large glimpse is coarse and a small one sharp
(pp. 4–5, 7). A ViT encoder with position encodings computed from patch coordinates (from "Beyond Grids") takes all
patches of all glimpses so far (p. 5). The reward is the decrease of the task loss, r_t = L_{t−1} − L_t (p. 7). Dense
outputs come from a MAE-like decoder over a full grid of mask tokens (p. 6). Classification and segmentation are
trained to match a teacher that sees the entire scene, "as in STAM" (p. 6). Without being told to, it learns to take
the whole image as its first glimpse (Figure 8, p. 14).

**Evidence.** "While modern AVE methods have demonstrated impressive performance, they are constrained to fixed-scale
glimpses from rigid grids. In contrast, existing mobile platforms equipped with optical zoom capabilities can capture
glimpses of arbitrary positions and scales" (Abstract, p. 1). "a scale of 0 corresponds to the maximum camera zoom
level and a scale of 1 to the widest view possible" (p. 5). Headlines: ImageNet-1k 77.54% with 14 glimpses of 32×32
px on 224×224 images, 28.57% of the pixels, against STAM's 76.13% at the same budget (Table 2, p. 11); ADE20K 25.7%
mIoU with 8 glimpses of 48×48 px, 36.73% of the pixels (Table 3, p. 12). Its own limitation: "quadratic computational
cost relative to the number of sampled patches" (p. 15).

**Contestable.**
- Choosing the scale was proposed before, outside AdaGlimpse's comparison set: RAM's Discussion ("The network can
  also be allowed to control the scale at which the retina samples the image", p. 8), Ranzato 2014 ("we could also
  predict the width and the height of the patch", p. 3), and Denil et al. 2012, whose control pathway models "the
  location, orientation, scale and speed of the attended object" (p. 1). AdaGlimpse's novelty claim is scoped to
  active visual exploration methods with fixed-scale grid glimpses, and holds there.
- A coarse whole-scene first look is what DRAM, GFNet and Saccader impose; AdaGlimpse learns it.

#### 2025: AdaptiveNN

- **Paper.** Wang Y., Yue Y., Yue Y., Wang H., Jiang H., Han Y., Ni Z., Pu Y., Shi M., Lu R., Yang Q., Zhao A., Xia
  Z., Song S., Huang G. "Emulating human-like adaptive vision for efficient and flexible machine visual perception."
  Nature Machine Intelligence 7(11):1804–1822, online 2025-11-06 (Crossref, doi:10.1038/s42256-025-01130-7).
  arXiv:2509.15333, v1 2025-09-18. Zotero PDF: `~/Zotero/storage/GC6BMI9C/` (arXiv v1, 95 pages with supplement).
- **What it did.** A quick glance processes the scene down-sampled to the fixation size; then a vision agent (a
  policy network and a value network) picks P×P fixations, each processed by a "Perception Net" whose features update
  an internal representation, and stops when the predicted value of further looking falls below a threshold.
  Training combines representation learning with "self-rewarding" reinforcement learning, whose reward is the
  negative task loss (Theorem 1, p. 9), implemented with PPO and generalized advantage estimation (Methods, p. 29).
- **Why it mattered.** "AdaptiveNN reduces the inference cost of well-performing models by up to 28× without
  sacrificing accuracy [...] the perceptual behaviors of AdaptiveNN are indistinguishable from people in many cases"
  (Abstract, p. 1). The 28× is for traffic-sign recognition on road scenes recorded from moving vehicles; on seven
  recognition benchmarks the reduction is 4 to 8 times (Main, p. 5). The CanViT paper's assessment is the one quoted
  under GFNet.
- **Headline.** ImageNet-1k: AdaptiveNN-DeiT-S at 288² px with 112² fixations, 82.2 ± 0.11% at 4.85 GFLOPs (3.15
  fixations on average), against DeiT-S at 79.9% (224², 4.61 GFLOPs) and 81.6% (384², 15.52 GFLOPs) (Supplementary
  Data Tab. 2, PDF p. 56; mean of 5 trials, Tab. 3, p. 57). The 82.2 is reused from `sota-history.md`.
- **Figure.** Figure 2a, p. 7: the glance, the fixation loop, the internal representation and the stop decision. For
  the neuroscience audience, Figure 6, p. 19: behavioural comparisons with human vision (fixation locations and
  judged difficulty).

What it introduced, against the precursors (read 2026-10-01; [inferred] marks the reader's own reading):

**New.**
- A theorem that the gradient of the expected task loss splits into a representation-learning term and a policy
  gradient whose reward is the negative task loss ("self-rewarding reinforcement learning", Theorem 1, p. 9),
  implemented with PPO and generalized advantage estimation (p. 29).
- A value network that both stabilizes learning and decides when to stop looking: observation ends when the predicted
  gain of further fixations falls below a threshold that can be changed at run time (p. 8).
- Breadth: CNN and transformer backbones, "17 benchmarks organized into 9 different tasks" including fine-grained
  recognition, traffic signs in road scenes, visual search, chest X-rays and a robot-manipulation language model
  (p. 1).
- Behavioural comparisons with people: overlap of its fixations with human viewing data (SALICON), correlation of its
  value estimates with human difficulty ratings (n = 10, Pearson 0.54 to 0.80), and "visual Turing tests" in which
  judges (n = 39) pick out the machine at 50 to 51% (pp. 17, 19).
- Lineage: it is the direct successor of GFNet, from the same Tsinghua group (Yulin Wang, Shiji Song and Gao Huang are
  authors of both), with the same glance then fixed-size fixations then adaptive stopping, trained with PPO. It cites
  GFNet (refs [67, 68], p. 4) and compares with it (Supplementary Section B, PDF p. 49). A search of all 95 PDF pages finds
  no citation of AdaGlimpse or AME, and it has no zoom choice: fixations are fixed P×P patches (p. 7).

**Evidence.** "AdaptiveNN formulates visual perception as a coarse-to-fine sequential decision-making process,
progressively identifying and fixating on regions pertinent to a given task, incrementally combining information
across fixations, and actively concluding its observation when sufficient" (Abstract, p. 1). "AdaptiveNN reduces the
inference cost of well-performing models by up to 28× without sacrificing accuracy" (p. 1); the 28× (27.9×) is traffic
signs on high-resolution road scenes, matching ResNet-50 at 960² px (90.2%, about 76 GFLOPs) with about 2.7 GFLOPs
(p. 13); on seven recognition benchmarks the reduction is 4 to 8× (p. 5). ImageNet-1k (Supplementary Data Tab. 2,
PDF p. 56): AdaptiveNN-DeiT-S on 288² px images with 112² px fixations, 82.2 ± 0.11% at 4.85 GFLOPs (3.15 fixations on
average), against DeiT-S at 79.9% (224², 4.61 GFLOPs) and 81.6% (384², 15.52 GFLOPs). The human data: SALICON maps
from "∼60 subjects" on Amazon Mechanical Turk (p. 25); the SALICON paper (Jiang, Huang, Duan & Zhao, "SALICON:
Saliency in Context", CVPR 2015, p. 1) says its paradigm "allowed using a general-purpose mouse instead of an eye
tracker to record viewing behaviors".

**Contestable.**
- "For the first time, we develop theoretical analyses ... revealing the natural emergence of reinforcement learning
  rules" (Supplementary A.2, PDF p. 45): DRAM (2014) already derived the REINFORCE update from maximizing the task
  likelihood marginalized over glimpse locations ("such learning rule can also be motivated by simply approximately
  optimizing the free energy", DRAM p. 5), and RAM trained the classifier by cross-entropy and the locations by
  REINFORCE, the same split (RAM p. 5).
- Rewarding a fixation by how much it lowers the task loss is AdaGlimpse's reward (2024); GFNet's (2020) used the gain
  in true-class probability. AdaptiveNN's differenced reward (p. 10) has the same form [inferred].
- Combining "sample-wise and spatial-wise adaptive computation" (A.2, p. 45) is what GFNet already did.
- "AdaptiveNN never senses the visual environment in its entirety" (p. 7) sits on the same page as "the full
  sequential process initiates with a quick glance, where a network coarsely processes an unknown scene in a
  down-sampled scale". It sees the whole scene at low resolution first.
- "for the first time in the modern community of modeling visual hard attention, we comprehensively compare the
  perceptual behaviors of our vision models with those of humans" (A.3, p. 47): the CanViT bibliography includes Li,
  Watters, Sohn & Jazayeri, "Modeling Human Eye Movements with Neural Networks in a Maze-Solving Task" (Gaze Meets ML
  workshop, PMLR 2023), whose abstract reports deep models fitted to human eye-tracking data (abstract read in
  `references.bib`, paper not read). Whether that model counts as hard attention was not checked.
- The CanViT paper says GFNet and AdaptiveNN "remained structurally limited to classification tasks"; AdaptiveNN
  also does visual search (locating digits, p. 15) and robot manipulation (CALVIN, p. 15). "No dense, per-pixel
  outputs" would be exact; "classification" invites a reviewer's or questioner's correction.

### Precursors before RAM

#### Larochelle & Hinton 2010

- **Paper.** Larochelle H., Hinton G.E. "Learning to combine foveal glimpses with a third-order Boltzmann machine."
  NIPS 2010 (Advances in Neural Information Processing Systems 23). No arXiv version; PDF from the author's site.
- **What it did.** A retina with a high-resolution centre and hexagonal low-resolution periphery takes glimpses at
  positions on a grid; a restricted Boltzmann machine with third-order (glimpse × position × hidden) connections
  accumulates them; a linear controller, trained to predict the log-probability of the correct class after each
  candidate fixation, chooses the next one greedily (Sections 2–4, pp. 2–5).
- **Why it mattered.** "To our knowledge, this is the first implemented system for combining glimpses that jointly
  trains a recognition component (the RBM) with an attentional component (the fixation controller)" (Section 4.3,
  p. 6). RAM takes its term from it: "We will refer to this low-resolution representation as a glimpse [14]" (RAM,
  Section 3.1, p. 4); DRAM credits it for the multiplicative "what" and "where" interaction (DRAM, p. 3).
- **Headline.** Facial expression recognition (7 classes): 62.7% after 6 fixations against 58.2% for an RBF-kernel
  SVM on the full images (Figure 3 and Section 6.3, p. 8).
- **Figure.** Figure 1, p. 2: the retinal transformation, example glimpses, and the multi-fixation RBM.
- **Also.** Glimpses on a 7 × 7 grid; the controller is trained by regression (pp. 2, 5). No reinforcement learning.

#### Denil, Bazzani, Larochelle & de Freitas 2012

- **Paper.** Denil M., Bazzani L., Larochelle H., de Freitas N. "Learning Where to Attend with Deep Architectures for
  Image Tracking." Neural Computation 24(8):2151–2184, 2012 (Crossref, doi:10.1162/NECO_a_00312). arXiv:1109.3737, v1
  2011-09-16.
- **What it did.** An identity pathway (factored RBMs feeding a multi-fixation RBM) classifies foveated gaze
  observations; a control pathway tracks the target with a particle filter, and a gaze policy is learned online to
  reduce tracking uncertainty, with bandit algorithms (Hedge, EXP3) or Bayesian optimization over a continuous
  action space (Abstract, p. 1; Section 4).
- **Why it mattered.** "we introduce gaze selection strategies which operate in the presence of partial information
  and on a continuous action space" (Abstract, p. 1). RAM counts it among "the other attempts to implement
  attentional processing in a deep learning framework" (RAM, Section 2, p. 2).
- **Headline.** Tracking error in pixels per video sequence (Tables 1–3, pp. 21–23); no single summary number.
- **Figure.** Figure 1, p. 3 (the two pathways, gaze, belief state, reward and policy).
- **Also.** "the first successful attempt to combine dynamic state estimation from gazes with online policy
  learning for gaze adaptation" (p. 6).

#### Ranzato 2014

- **Paper.** Ranzato M. "On Learning Where To Look." arXiv:1405.5488, v1 2014-04-24; preprint, no venue found. Its
  arXiv date precedes RAM's by two months, and RAM cites it.
- **What it did.** "The key idea is to first aggressively down-sample the image to cheaply detect candidate
  locations where to look at higher resolution" (Introduction, p. 2); a network predicts each next location, the
  location is treated as a latent variable found by a local discrete search during training, and glimpse networks
  are trained one after another (Section 2, pp. 3–6).
- **Why it mattered.** It recognizes objects with "an amount of computation that scales with the complexity of the
  input rather than its number of pixels" (Abstract, p. 1); the low-resolution first look anticipates DRAM's
  context network and GFNet's glance.
- **Headline.** Jittered MNIST (48×48): 1.4% error with 2 glimpses after fine-tuning, equal to a convolutional
  network, at a computational cost "2 order of magnitude lower" (Table 1 and Section 4.1, p. 8).
- **Figure.** Figure 1, p. 4 (low-resolution network, location predictor, first glimpse network).
- **Also.** A confidence threshold lets easy inputs stop early: 78% of jittered-MNIST test digits are classified
  from the low-resolution view alone, a 12× saving in operations (pp. 8–9). No reinforcement learning.

#### Earlier neural precursors, verified only through later papers

- **Schmidhuber J., Huber R. 1991.** "Learning to generate artificial fovea trajectories for target detection."
  International Journal of Neural Systems 2(1–2):125–134 (Crossref, doi:10.1142/S012906579100011X). Saccader's related
  work: "Early work using backpropagation to train the controller computed the gradient with respect to its weights
  by backpropagating through a separate neural network model of the environmental dynamics" (Saccader, p. 2).
  [Primary text not read.]
- **Butko N.J., Movellan J.R. 2009.** "Optimal scanning for faster object detection." CVPR 2009, pp. 2751–2758
  (Crossref). Saccader: Butko and Movellan "proposed instead to use policy gradient, learning a convolutional
  logistic policy to maximize long-term information gain" (Saccader, p. 2, citing their 2008 paper); RAM calls their
  framework "related to ours as they also employ a policy gradient formulation" (RAM, p. 2). [Primary text not read.]
- **Butko N.J., Movellan J.R. 2008.** "I-POMDP: An infomax model of eye movement." ICDL 2008: the paper RAM and
  Saccader cite for the quotes above. [Primary text not read: the PDF link on the lab page returns HTML.] RAM also
  cites Paletta et al., "Q-learning of sequential attention for visual object recognition from informative local
  descriptors" (2005) [not read].

### Pre-deep-learning foundations

- **Bajcsy R. 1988.** "Active perception." Proceedings of the IEEE 76(8), pages 996–1005 by the PDF (see
  discrepancies). Zotero PDF: `~/Zotero/storage/P7748FSH/`. "perception is not passive, but active. Perceptual
  activity is exploratory, probing, searching; percepts do not simply fall onto sensors as rain falls onto ground. We
  do not just see, we look" (Section I, printed p. 996, PDF p. 1). Active sensing is "a problem of controlling
  strategies applied to the data acquisition process which will depend on the current state of the data
  interpretation and the goal or the task of the process" (Section II, same page).
- **Aloimonos J., Weiss I., Bandyopadhyay A. 1988.** "Active vision." International Journal of Computer Vision
  1(4):333–356 (Crossref; the PDF's header carries a 1987 copyright date). Zotero PDF: `~/Zotero/storage/MEB9XVTR/`.
  "Problems that are ill-posed and nonlinear for a passive observer become well-posed and linear for an active
  observer" (Abstract, printed p. 333, PDF p. 1). Their "active" means controlling the geometry of the sensor (moving camera, rotating
  eyes), applied to shape from shading, contour and texture, and structure from motion.
- **Ballard D.H. 1991.** "Animate vision." Artificial Intelligence 48(1):57–86, February 1991 (Crossref). [Primary
  text not read: closed access, not in Zotero.] Quoted by Bajcsy, Aloimonos & Tsotsos, "Revisiting Active
  Perception" (arXiv:1603.02729, p. 3): "An animate vision system with the ability to control its gaze can make the
  execution of behaviors involving vision much simpler", followed by six advantages, the last "Gaze control leads
  naturally to the use of object centered coordinate systems as the basis for spatial memory."

### Other works an informed audience may name

Each line was checked as stated.

**Bridging steps, 2014–2018**
- **Sermanet P., Frome A., Real E.** "Attention for Fine-Grained Categorization." ICLR 2015 workshop, arXiv:1412.7054
  (2014-12-22). DRAM with an ImageNet-pretrained Inception network (modified for 96×96 inputs) as glimpse encoder on
  Stanford Dogs: 76.8% mean accuracy with 3 glimpses against 75.5% for the standard Inception network on the whole
  224×224 image (p. 8). Saccader judged that it "achieved only a modest accuracy improvement over a non-attentive
  baseline, and did not appear to derive significant benefit from glimpses beyond the first" (Saccader, p. 2). An
  early case of an active model built on a pretrained passive network.
- **Xu K., Ba J., Kiros R., Cho K., Courville A., Salakhutdinov R. et al.** "Show, Attend and Tell: Neural Image
  Caption Generation with Visual Attention." arXiv:1502.03044 (2015-02-10); ICML 2015 [venue not verified]. Hard
  (sampled) and soft attention over convolutional features of the whole image for captioning, trained by
  backpropagation or by a variational lower bound (abstract). Attention over features of a fully seen image.
- **Gregor K., Danihelka I., Graves A., Rezende D.J., Wierstra D.** "DRAW: A Recurrent Neural Network For Image
  Generation." ICML 2015 (PDF header), arXiv:1502.04623. Differentiable attention that "mimics the foveation of the
  human eye" for reading and writing images; it builds its output on a "canvas matrix" in pixel space (pp. 1–2).
  The CanViT paper does not cite it; the shared word "canvas" may draw a question.
- **Jaderberg M., Simonyan K., Zisserman A., Kavukcuoglu K.** "Spatial Transformer Networks." arXiv:1506.02025
  (2015-06-05). A differentiable crop-and-warp module inserted in a network (abstract). In the CanViT bibliography.
- **Caicedo J.C., Lazebnik S.** "Active Object Localization with Deep Reinforcement Learning." ICCV 2015 (arXiv
  comment), arXiv:1511.06015. A deep RL agent deforms a box to localize an object "after analyzing only between 11
  and 25 regions" (abstract).
- **Mathe S., Pirinen A., Sminchisescu C.** "Reinforcement Learning for Visual Object Detection." CVPR 2016,
  pp. 2894–2902 (Crossref). Sequential search with a learned stopping rule, "almost two orders of magnitude speed-up
  over sliding window methods" on PASCAL VOC 2012 (abstract as recorded in `references.bib`). Cited by the CanViT
  paper.
- **Ren M., Zemel R.S.** "End-to-End Instance Segmentation with Recurrent Attention." CVPR 2017 (arXiv comment),
  arXiv:1605.09410. A recurrent network with attention produces one region of interest and one instance mask per
  step (abstract). Cited by the CanViT paper's Introduction.
- **Jayaraman D., Grauman K.** "Learning to Look Around: Intelligently Exploring Unseen Environments for Unknown
  Tasks." CVPR 2018, pp. 1238–1247 (Crossref), arXiv:1709.00507 (2017-09-01). An RL agent rewarded for reducing
  uncertainty about unobserved parts of panoramic scenes and 3D shapes; "the learned policies are not tied to any
  recognition task" (abstract). GAE, AME and AdaGlimpse also evaluate on 360° panoramas (SUN360); the dataset
  Jayaraman & Grauman used was not checked.

**Single-step selection from a low-resolution view** (not sequential, so outside the paper's definition of active models)
- **Katharopoulos A., Fleuret F.** "Processing Megapixel Images with Deep Attention-Sampling Models." ICML 2019,
  arXiv:1905.03711. Samples patches from an attention distribution computed on a low-resolution view, with an
  unbiased gradient estimator (abstract).
- **PatchDrop (Uzkent & Ermon 2020)** and **LookWhere (Fuller et al. 2025)**: see `sota-history.md`. LookWhere
  (83.0% ImageNet-1k, 44.6% ADE20K mIoU) is the likeliest audience question; the CanViT paper does not cite it.

**Sequential glimpse models, 2021, cited by the CanViT paper or in the same line**
- **Seifi S., Jha A., Tuytelaars T.** "Glimpse-Attend-and-Explore: Self-Attention for Active Visual Exploration."
  ICCV 2021, pp. 16117–16126 (Crossref). Self-attention guides exploration, for "both dense and sparse prediction
  tasks" (abstract); AME's main baseline. Cited by the CanViT paper.
- **Rangrej S.B., Clark J.J.** "A Probabilistic Hard Attention Model For Sequentially Observed Scenes." BMVC 2021
  (Semantic Scholar record), arXiv:2111.07534. Never sees the full image; chooses glimpses by Bayesian optimal
  experimental design, synthesizing features of unseen regions with a partial VAE (abstract). McGill.
- **Papadopoulos A., Korus P., Memon N.** "Hard-Attention for Scalable Image Classification" (TNet). NeurIPS 2021
  (CanViT `references.bib`), arXiv:2102.10212. Traverses image scale-space top-down, processing satellite images up to
  896×896 px (abstract). Cited by the CanViT paper.

**Recent, brain-facing or foundation-model-facing** (details under "Foundation models, world knowledge and brain alignment")
- **Thorat S., Doerig A., Kroner A., Amme C., Kietzmann T.C.** "Predicting upcoming visual features during eye
  movements yields scene representations aligned with human visual cortex." arXiv:2511.12715 (2025-11-16).
  Recurrent networks predict the next glimpse's embedding along human-like scanpaths and align with fMRI in mid- and
  high-level visual cortex (abstract).
- **Pourrahimi M., Bashivan P.** bioRxiv doi:10.1101/2025.06.06.658387 (2025). A network that searches natural
  scenes with saccades and whose units show fronto-parietal response properties; McGill and Mila (PDF header). Cited
  by the CanViT paper.
- **Kolner et al. 2025** (GAP, ICLR 2025), **Blauch et al. 2026** (FOVI, ICML 2026) and **Prisadnikov et al. 2026**
  (self-supervised pretraining for a foveal iterative transformer): see "Active vision and foundation models" below and
  `sota-history.md`.

**Montreal connections, for the local audience.** Rangrej & Clark 2021 and two of STAM's three authors are at McGill
(PDF headers); Pourrahimi & Bashivan at McGill and Mila.

#### Not checked in these readings

- Butko & Movellan 2008 and Paletta et al. 2005 (primary texts not read; described through RAM and Saccader).
- Saccader's supplementary Figure Supp.3 (with and without location pretraining), not in the NeurIPS PDF read.
- Whether Li et al. 2023 is a hard-attention model.

### Prior work's own ablations of where to look

Read on 2026-10-24 by a subagent in each paper's PDF (values marked "fig" were read off a plot, about ±1 point);
Claude Code re-checked GAE's ADE20K row and GFNet's traffic-sign row in the PDF text. "Baseline" is the paper's
random or trivial viewpoint choice, "chosen" its own (learned or rule-based), everything else as equal as the paper
makes it.

| Paper | Task, setting | Baseline → chosen | How chosen | Where |
|---|---|---|---|---|
| GFNet journal (Huang, Wang et al., TPAMI, arXiv:2201.03014) | ImageNet, ResNet-50, 96² patches, T = 5 | random + glance 72.32 → 73.71 (t = 2); 74.46 → 75.93 (t = 5); centre-then-corners + glance 75.12 (t = 5); random without glance 54.53 (t = 1), 70.70 (t = 5) | RL (PPO) | Table 5 |
| same | Swedish traffic signs, 960 × 1280, 192² patches, ResNet-18 | random patch 68.97 → 91.50 (both 2.8 GFLOPs) | RL | Table 4 |
| GFNet (Wang et al., NeurIPS 2020, arXiv:2010.05300) | ImageNet, ResNet-50, T = 5 | random + glance 74.46 → 75.93; centre-then-corners 75.12 | RL (PPO) | Table 1, p. 9 |
| AME (Pardyl et al., IJCAI 2023, arXiv:2303.06457) | SUN360, 8 glimpses of 32², all weights trained | random 72.4 → 73.4 | attention-map entropy (rule) | Table 5, p. 7 |
| AdaptiveNN (arXiv:2509.15333) | ImageNet; classifier sees fixation features only, glance excluded; 5 seeds | DeiT-S, 1/2/3/4 fixations: random 73.5/77.0/78.5/79.4 → 78.2/80.4/81.3/81.8; centre-corner 76.4/78.2/79.5/80.3. ResNet-50: random 66.1/70.8/73.3/74.8 → 73.6/77.1/78.4/79.1 | RL | Supp. Data Tab. 30–37 |
| Saccader (Elsayed et al., NeurIPS 2019, arXiv:1908.07644) | ImageNet 224, 77² receptive field, 6 glimpses | top-1 about 0.52 → 0.71 at 6 glimpses; random reaches about 0.69 near 60 glimpses (fig) | RL after pretraining; the policy sees full-image features | Fig. 3a |
| STAM (Rangrej et al., CVPR 2022, arXiv:2204.00656) | 7 × 7 grid of 32² glimpses | t = 8: ImageNet about 63 → 66.5, fMoW about 61 → 67; t = 20: about 74.5 → 76.5 and 69.5 → 71.5 (fig); text: +1.8 / +2.3 over the best baseline | RL actor-critic | Fig. 3 |
| Rangrej & Clark (BMVC 2021, arXiv:2111.07534) | 7 glimpses, 8² on 32² images | t = 6, random → expected information gain: SVHN about 89 → 95, CIFAR-10 77 → 81, CINIC 68.5 → 72, CIFAR-100 50.5 → 51, TinyImageNet 36.5 → 36.5; RAM (RL) below random: about 34 on CIFAR-100, 23 on TinyImageNet (fig) | Bayesian optimal experiment design | Fig. 2 |
| TORE (Olszewski et al., arXiv:2311.15335) | SUN360 (8 glimpses), CIFAR-100 (12) | evaluation random → entropy: about 73 → 77, 79 → 80 (fig); training on random glimpses: SUN360 70.3 → 77.1, CIFAR-100 73.8 → 80.2 | attention entropy (AME's rule) | Fig. 6; Table 3 |
| AdaFocus (Wang et al., ICCV 2021, arXiv:2105.03245) | video mAP (dataset not stated in the table) | patch 96/128/160/192: random 65.8/70.7/73.1/74.8 → 70.2/73.4/75.0/75.9 | RL | Table 5 |
| AdaFocusV2 (Yue et al., CVPR 2022, arXiv:2112.14238) | ActivityNet, 128² patches | 1/2/4/8/16 frames: random 37.7/46.0/56.8/67.6/73.9 → 44.8/52.5/61.7/71.0/76.4 | end to end | Table 7 |
| PatchDrop (Uzkent & Ermon, CVPR 2020, arXiv:2003.00425) | choose high-resolution patches of 16 | ImageNet fixed 68.6 / centre-biased 66.2 → 74.9; CIFAR-10 83.8 → 91.9; fMoW 64.9 → 68.3 | RL | Tables 1–2 |
| GAE (Seifi, Jha & Tuytelaars, ICCV 2021, arXiv:2108.11717) | 8 retina glimpses | reconstruction RMSE: SUN360 38.6–39.4 → 33.8, MS-COCO 47.8–48.5 → 40.3; **ADE20K mPA 51.5 and 52.4 (two random variants) → 52.4**; COCO-Stuff 46.0–47.7 → 47.8; SUN360 classification 49.6 → 56.4; removing the decoder +10.8 | self-attention argmax | Tables 1, 3 |
| SimGlim (Jha, Seifi & Tuytelaars, WACV 2023) | 37 glimpses of 16², 18.75% of pixels | RMSE random → learned: SUN360 28.5 → 26.2, ADE20K 31.1 → 27.2; an MAE on random glimpses beats CNN-based GAE (33.8) and Attend & Segment (37.6) on SUN360; learned loses to random at 13 glimpses on SUN360 | learned loss predictor | Tables 1–3, Fig. 6 |
| Where to Look Next (Seifi & Tuytelaars, arXiv:1909.10304) | SUN360 reconstruction, 8 glimpses | RMSE 34.9 → 28.5; the random version already beats Learning to Look Around (38.8) and Sidekick (39.0) | attention module | Table 1 |
| Attend & Segment (Seifi & Tuytelaars, ECCV 2020, arXiv:2007.11548) | Cityscapes, 10 glimpses | about 75.6 → 78 mPA (fig) | uncertainty prediction | Fig. 9 |
| Learning to Look Around (Jayaraman & Grauman, CVPR 2018, arXiv:1709.00507) | view-grid completion | MSE×1000 SUN360 31.88 → 23.16; ModelNet seen 3.46 → 3.25, unseen 6.22 → 5.65; categorization SUN360 about 58 → 66 at t = 6, ModelNet-10 91 → 92 (fig) | RL (REINFORCE) | Table 1, Fig. 5 |
| Sidekick (Ramakrishnan & Grauman, ECCV 2018, arXiv:1807.11010) | as above | average MSE SUN360 30.99 → 23.36; ModelNet hard 7.32 → 5.75 | RL actor-critic | Table 1 |
| FoveaTer (Jonnalagadda et al., ICLR 2023, arXiv:2105.14173) | ImageNet, hand-picked hardest 8% only | 5 fixations about 8.5 → 14 (the "+63%" is relative); removing peripheral features −13.4 | self-attention | Fig. 3, Table 1 |
| LookWhere (Fuller et al., NeurIPS 2025, arXiv:2505.18051; select once) | random high-res patches at the same budget | traffic signs 70.6 → 94.6; CUB 78.2 → 89.0; Billiard 27.8 → 97.3; ImageNet +2.1 (fig) | distilled selector | Tables 3–5, Fig. 7d |
| IPS (Bergner et al., ICLR 2023, arXiv:2210.13007; scores all patches) | traffic signs M = 10 / 2; megapixel MNIST M = 100 | 50.7 / 28.3 → 98.6 / 98.5; 33.8 → 99.7 | top-M scoring | Tables 1, 3 |
| ATS (Katharopoulos & Fleuret, ICML 2019, arXiv:1905.03711) | test error, uniform vs attention sampling | colon cancer 15.6% → 9.3%; speed limits 53.1% → 8.9%; megapixel MNIST uniform at chance | learned attention distribution | Tables 1–2 |
| Uzkent et al. (WACV 2020, arXiv:1912.03966) | xView detection, which patches to zoom | AP 24.1 → 38.1 at 31% high-res usage | RL | Table 1 |
| SUGARL (Shang & Ryoo, NeurIPS 2023, arXiv:2306.00975) | 26 Atari games, IQM normalized to full-view DQN | fovea 20/30/50: random 0.155/0.104/0.134 → 0.424/0.730/0.785 | RL | Table 2c |

Caveats read in the same papers: not every comparison holds everything else equal. Rangrej's random baseline has no
controller; the selectors of Saccader, LookWhere and IPS see the whole image; PatchDrop, Uzkent, GFNet and AdaFocus
have a low-resolution glance or global context; AdaptiveNN excludes the glance from the classifier. Gaps shrink as
coverage grows (more glimpses, bigger patches). Checked, with no equal-budget random ablation: RAM, DRAM, TNet
(arXiv:2102.10212), Variational Saccading (arXiv:1812.03170), Differentiable Patch Selection (arXiv:2104.03059),
GAP (arXiv:2409.20213); AdaGlimpse compares only against attention-entropy heuristics (reconstruction RMSE
23.2–25.95 → 14.5, Supp. Table 1). Not checked: Sermanet 2014, Mathe et al. 2016, Dynamic Zoom-in Net.

AdaGlimpse's own policy, read in its Fig. 8 (p. 14 of arXiv:2404.03482): averaged over test images, the
reconstruction policy takes the whole image first, then four quadrants in turn (top-left, bottom-right,
bottom-left, top-right), with sharp-edged average maps at t = 2–5; the classification policy takes the whole image,
then mostly the centre. AdaGlimpse's encoder re-encodes all patches gathered so far, each with its coordinates and
no time step (`architectures/mae.py` `forward_encoder`, `rl/shared_memory.py` `current_patches`, read 2026-10-24 in
the m2b3 fork at 9b3b72d), so its output after the last glimpse does not depend on their order.

## Foundation models, world knowledge and brain alignment

### 1. What a foundation model is, and the vision lineage

- **Bommasani R. et al. 2021. On the Opportunities and Risks of Foundation Models.** Stanford CRFM report,
  arXiv:2108.07258. https://arxiv.org/abs/2108.07258. Zotero:
  `Bommasani et al. - 2022 - On the Opportunities and Risks of Foundation Models.pdf`.
  Read: §1. Coins the term: "A foundation model is any model that is trained on broad data (generally using
  self-supervision at scale) that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks; current
  examples include BERT, GPT-3, and CLIP." Two consequences: emergence ("the behavior of a system is implicitly
  induced rather than explicitly constructed") and homogenization ("the defects of the foundation model are
  inherited by all the adapted models downstream"). The name stresses that "a foundation model is itself
  incomplete but serves as the common basis from which many task-specific models are built via adaptation."
  Figures: Fig. 2 (one model centralizes data from many modalities and is adapted to many tasks); Fig. 1
  (emergence and homogenization across 30 years of AI).
- **Radford A. et al. 2021. Learning Transferable Visual Models From Natural Language Supervision (CLIP).** ICML 2021,
  PMLR 139:8748–8763, arXiv:2103.00020. https://proceedings.mlr.press/v139/radford21a.html. Zotero:
  `Radford et al. - 2021 - Learning Transferable Visual Models From Natural Language Supervision.pdf`.
  Contrastive image–caption pretraining on 400 million pairs; language then names visual concepts for zero-shot
  transfer: "we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of
  the 1.28 million training examples it was trained on." Fig. 1 (the approach).
- **Caron M. et al. 2021. Emerging Properties in Self-Supervised Vision Transformers (DINO).** ICCV 2021,
  arXiv:2104.14294. https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.
  Zotero: `Caron et al. - 2021 - Emerging Properties in Self-Supervised Vision Transformers.pdf`.
  Self-distillation without labels: "self-supervised ViT features contain explicit information about the semantic
  segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets"; 80.1%
  ImageNet top-1 with a linear probe on ViT-B. Fig. 1: attention maps of the [CLS] token outline objects, with no
  label ever given.
- **He K. et al. 2022. Masked Autoencoders Are Scalable Vision Learners (MAE).** CVPR 2022, arXiv:2111.06377.
  https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.
  Zotero: `He et al. - 2022 - Masked Autoencoders Are Scalable Vision Learners.pdf`.
  Mask 75% of patches and reconstruct the pixels; the encoder sees only the visible patches. Relevant to active
  vision twice over: inferring unseen regions from a few seen ones is the glimpse problem, and the dense active
  models AME and AdaGlimpse use MAE-style decoders (CanViT paper, related work). Its features encode depth poorly
  (El Banani et al. 2024, below). Fig. 1 (architecture).
- **Kirillov A. et al. 2023. Segment Anything (SAM).** ICCV 2023, arXiv:2304.02643.
  https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html.
  Zotero: `Kirillov et al. - 2023 - Segment Anything.pdf`. "We aim to build a
  foundation model for segmentation": a promptable model, trained on SA-1B (over 1 billion masks on 11 million
  images), that "can transfer zero-shot to new image distributions and tasks". Fig. 1 (task, model, data engine).
- **Oquab M. et al. 2024. DINOv2: Learning Robust Visual Features without Supervision.** TMLR (PDF header 01/2024),
  arXiv:2304.07193. https://openreview.net/forum?id=a68SUt6zFt. Zotero:
  `Oquab et al. - 2023 - DINOv2 Learning Robust Visual Features without Supervision.pdf`.
  States the vision version of the paradigm: "general-purpose visual features, i.e., features that work across
  image distributions and tasks without finetuning", obtained by self-supervision "if trained on enough curated
  data from diverse sources"; a 1B-parameter ViT distilled into smaller models. Segmentation and monocular depth
  are read out with linear probes on frozen features (Table 11, Fig. 7). Figures: Fig. 1 (first three PCA
  components of patch features: "Same parts are matched between related images despite changes of pose, style or
  even objects"), Fig. 7 (linear-probe depth and segmentation), Fig. 10 (part matching across images and domains).
- **Siméoni O. et al. DINOv3.** arXiv:2508.10104 (2025); TMLR 2026 (the TMLR PDF header reads "04/2026").
  https://openreview.net/forum?id=2NlGyqNjns. Zotero: `Siméoni et al. - 2026 - DINOv3.pdf`.
  A 7B-parameter self-supervised teacher on a curated 1.7-billion-image set (LVD-1689M), distilled into a family of
  smaller models. Gram anchoring "addresses the known yet unsolved issue of dense feature maps degrading during
  long training schedules"; the result "outperforms the specialized state of the art across a broad range of
  settings, without fine-tuning". The frozen-feature ideal in their words: "achieve excellent performance while
  being kept frozen ... a single forward pass can deliver cutting-edge results across multiple tasks." Figures: Fig. 1 (ImageNet linear-probe accuracy of supervised, weakly supervised and self-supervised methods over
  the years, plus DINOv3's lead on dense tasks), Fig. 3 (patch similarity maps on a 4096² image), Fig. 5 (see §4).

### 2. World knowledge in vision foundation models

- **El Banani M. et al. 2024. Probing the 3D Awareness of Visual Foundation Models.** CVPR 2024,
  doi:10.1109/CVPR52733.2024.02059, arXiv:2404.08636. https://arxiv.org/abs/2404.08636 (full PDF read). Not in
  Zotero. Probes frozen features for depth, surface normals and correspondence across views. "Recent
  self-supervised models such as DINOv2 learn representations that encode depth and surface normals, with
  StableDiffusion being a close second", while CLIP shows "very poor performance despite its impressive semantic
  generalization capabilities"; "CLIP and MAE features do not encode depth" (Fig. 2 caption). The limit: "the
  models struggle with multiview consistency ... they perform very poorly at large viewpoint variations", and
  their cross-view matching "is semantic in nature". Single-view tasks correlate with each other above 0.82;
  multiview tasks much less. Figures: Fig. 1 (ratings per model, single-image 3D versus multiview
  consistency), Fig. 2 (depth maps from task-specific probes), Fig. 6 (correspondence accuracy falling with
  viewpoint change).
- **DINOv2 (above), Figs. 1, 7 and 10**: parts, depth and segmentation read linearly from frozen self-supervised
  features.
- **Amir S., Gandelsman Y., Bagon S., Dekel T. 2021. Deep ViT Features as Dense Visual Descriptors.**
  arXiv:2112.05814 (venue UNVERIFIED). https://arxiv.org/abs/2112.05814. DINO-ViT features "encode powerful,
  well-localized semantic information, at high spatial granularity, such as object parts", "shared across related,
  yet different object categories", enough for zero-shot co-segmentation and semantic correspondence.
- **Man Y. et al. 2024. Lexicon3D: Probing Visual Foundation Models for Complex 3D Scene Understanding.** NeurIPS
  2024 (arXiv comment), arXiv:2409.03757. https://arxiv.org/abs/2409.03757. Seven encoders probed on 3D scene
  tasks: "DINOv2 demonstrates superior performance".
- **Linsley D. et al. 2025. The 3D-PC: a benchmark for visual perspective taking in humans and machines.** ICLR 2025
  (arXiv comment), arXiv:2406.04138. https://arxiv.org/abs/2406.04138. Over 300 DNNs, linearly probed or prompted,
  "approached or exceeded human accuracy in analyzing object depth order", and this accuracy "correlated with their
  object recognition performance"; on basic visual perspective taking, "Humans were nearly perfect, whereas most
  DNNs were near chance", and fine-tuned DNNs "dropped back to chance" on a variant that defeats shortcut
  strategies. Tension: 3D properties are encoded; reasoning over them is not.
- **Garrido Q. et al. 2025. Intuitive physics understanding emerges from self-supervised pretraining on natural
  videos.** arXiv:2502.11831 (preprint; no peer-reviewed venue found). https://arxiv.org/abs/2502.11831 (full PDF
  read). Not in Zotero. V-JEPA, trained to predict masked video in a learned representation space, is "surprised"
  by physically impossible videos (violation-of-expectation, as in infant studies): pairwise accuracy 98% on
  IntPhys, 66% on GRASP, 62% on InfLevel-lab; untrained networks, pixel-space video predictors and multimodal LLMs
  are near chance. Above chance even with 128 hours of unique video; a 115M-parameter model exceeds 85% on IntPhys.
  Limits in the same paper: no significant gain on color constancy, solidity or collision in GRASP, and object
  interactions "are close to chance". The authors argue this "challenges the idea that core knowledge ... needs to
  be hardwired". Figures: Fig. 1 (accuracy by model family on three benchmarks), Fig. 2 (per-property
  results and comparison with humans), Fig. 3 (effect of data type, amount and model size).
- **Bordes F., Garrido Q., Kao J.T., Williams A., Rabbat M., Dupoux E. 2025. IntPhys 2: Benchmarking Intuitive
  Physics Understanding In Complex Synthetic Environments.** arXiv:2506.09849 (preprint).
  https://arxiv.org/abs/2506.09849. Zotero:
  `Bordes et al. - 2025 - IntPhys 2 Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments.pdf`.
  Same group, harder scenes: models face "significant challenges ... with most models performing at chance levels
  (50%), in stark contrast to human performance, which achieves near-perfect accuracy." Tension with Garrido et al.
- **Punzo S. et al. 2026. How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining
  Paradigms.** arXiv:2606.09646 (preprint). https://arxiv.org/abs/2606.09646. Frozen probes on IntPhys 2 and MVP:
  V-JEPA strongest; the information peaks at intermediate-to-late layers, and "its accessibility depends strongly
  on pretraining paradigm, representational depth, and readout mechanism". Relevant because a probe that fails is
  a statement about the readout as well as the representation.
- **Assran M. et al. 2025. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.**
  arXiv:2506.09985. https://arxiv.org/abs/2506.09985. Zotero:
  `Assran et al. - 2025 - V-JEPA 2 Self-Supervised Video Models Enable Understanding, Prediction and Planning.pdf`.
  Pretrained on over 1 million hours of video; an action-conditioned version post-trained on under 62 hours of
  robot video plans pick-and-place zero-shot in new labs. The "world model" end of the foundation-model story; in
  the CanViT bibliography.
- **Huh M., Cheung B., Wang T., Isola P. 2024. Position: The Platonic Representation Hypothesis.** ICML 2024, PMLR
  235, arXiv:2405.07987. https://proceedings.mlr.press/v235/huh24a.html. Zotero:
  `Huh et al. - 2024 - Position The Platonic Representation Hypothesis.pdf`.
  "Neural networks, trained with different objectives on different data and modalities, are converging to a shared
  statistical model of reality in their representation spaces." Evidence: vision models' mutual-nearest-neighbour
  alignment rises with the number of VTAB tasks they solve (Fig. 2); vision–language alignment rises with language
  model performance (Fig. 3); §2.4 cites brain alignment. Figure: Fig. 1 (images and text as projections of
  one reality Z).
- **Gröger F., Wen S., Brbić M. 2026. Revisiting the Platonic Representation Hypothesis: An Aristotelian View.**
  ICML 2026 (arXiv comment: camera-ready), arXiv:2602.14486. https://arxiv.org/abs/2602.14486. In Zotero without a
  PDF. Similarity metrics are "confounded by network scale"; after a permutation-null calibration, "the apparent
  convergence reported by global spectral measures largely disappears, while local neighborhood similarity, but
  not local distances, retains significant agreement across different modalities." Tension with Huh et al.

### 3. Performance and brain alignment

- **Yamins D.L.K. et al. 2014. Performance-optimized hierarchical models predict neural responses in higher visual
  cortex.** PNAS 111(23):8619–8624, doi:10.1073/pnas.1403112111. https://www.pnas.org/doi/10.1073/pnas.1403112111.
  Zotero: `Yamins et al. - 2014 - Performance-optimized hierarchical models predict .pdf`.
  "Within a class of biologically plausible hierarchical neural network models, there is a strong correlation
  between a model's categorization performance and its ability to predict individual IT neural unit response
  data"; the top layer of the performance-optimized model predicts IT, its middle layers V4. Figure: Fig. 1
  (categorization performance against IT explained variance; r = 0.55, 0.78 and 0.80 for random, performance-
  optimized and IT-optimized model selection in panel A; r = 0.87 ± 0.15 in panel B). The scope matters: one
  model class, 2014-era performance levels. Stimuli (read 2026-10-25 in PMC4060707): "5,760 images of
  photorealistic 3D objects drawn from eight natural categories" on randomly selected "cluttered natural scenes",
  with large variation in position, scale and pose. Controls: pixels, SIFT, V1-like, V2-like and HMAX, which all
  fall short of IT "by a large margin" on the high-variation images. The model: a four-layer CNN found by
  hierarchical modular optimization (HMO), not trained on ImageNet.
- **Schrimpf M. et al. 2018. Brain-Score: Which Artificial Neural Network for Object Recognition is most
  Brain-Like?** bioRxiv 407007, doi:10.1101/407007. https://www.biorxiv.org/content/10.1101/407007v1. Zotero:
  `Schrimpf et al. - 2018 - Brain-Score Which Artificial Neural Network for O.pdf`.
  "Gains in ANN ImageNet performance led to gains on Brain-Score. However, correlation weakened at ≥ 70% top-1
  ImageNet performance"; smaller ANNs were more brain-like "than many of the best-performing ImageNet models".
  Figure: Fig. 1 (ImageNet top-1 against Brain-Score; the relation flattens for the newest models).
- **Schrimpf M. et al. 2020. Integrative Benchmarking to Advance Neurally Mechanistic Models of Human
  Intelligence.** Neuron 108:413–423 (first page read; last page UNVERIFIED), doi:10.1016/j.neuron.2020.07.040.
  Zotero: `Schrimpf et al. - 2020 - Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence.pdf`.
  The programme: suites of neural and behavioural benchmarks that any model must answer. Fig. 3 draws the analogy:
  a shared evaluation on ImageNet "helped incentivize computer vision out of toy problems", and Brain-Score is
  meant to do the same for neuroscience.
- **Kubilius J. et al. 2019. Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs (CORnet-S).**
  NeurIPS 2019 (arXiv comment), arXiv:1909.06161. https://arxiv.org/abs/1909.06161. A shallow recurrent network
  tops Brain-Score; "recurrence is the main predictive factor of both Brain-Score and ImageNet top-1 performance".
- **Zhuang C. et al. 2021. Unsupervised neural network models of the ventral visual stream.** PNAS 118(3):e2014196118.
  https://www.pnas.org/doi/10.1073/pnas.2014196118. Zotero:
  `Zhuang et al. - 2021 - Unsupervised neural network models of the ventral .pdf`.
  Contrastive self-supervised models "achieve neural prediction accuracy in multiple ventral visual cortical areas
  that equals or exceeds that of models derived using today's best supervised methods", even when trained on
  child head-camera video. Labels are not what makes a model brain-like.
- **Konkle T., Alvarez G.A. 2022. A self-supervised domain-general learning framework for human ventral stream
  representation.** Nature Communications 13:491, doi:10.1038/s41467-022-28091-4.
  https://www.nature.com/articles/s41467-022-28091-4 (abstract via OpenAlex). Instance-level self-supervision;
  "category information implicitly emerges", and the hierarchy captures human ventral-stream responses "on par
  with category-supervised models".
- **Fel T., Felipe I., Linsley D., Serre T. 2022. Harmonizing the object recognition strategies of deep neural
  networks with humans.** NeurIPS 2022 (arXiv comment), arXiv:2211.04533. https://arxiv.org/abs/2211.04533. Across
  84 ImageNet DNNs, "a systematic trade-off between DNN categorization accuracy and alignment with human visual
  strategies"; "State-of-the-art DNNs are progressively becoming less aligned with humans as their accuracy
  improves." Fig. 3 (the trade-off).
- **Linsley D. et al. 2023. Performance-optimized deep neural networks are evolving into worse models of
  inferotemporal visual cortex.** NeurIPS 2023, arXiv:2306.03779. https://neurips.cc/virtual/2023/poster/71381
  (venue), https://arxiv.org/abs/2306.03779 (PDF read). "Across three independent experiments ... DNNs have become
  progressively worse models of IT as their accuracy has increased on ImageNet." Spatially resolved IT recordings
  show ImageNet DNNs "rely on different visual features than those encoded by IT and that this problem worsens as
  their accuracy increases"; aligning DNNs to human feature-importance maps (the "neural harmonizer") partly
  breaks the trade-off. Figures: Fig. 1 (104 Brain-Score models, ImageNet accuracy against IT predictivity,
  with the Pareto front), Fig. 3 (135 DNNs on new IT recordings).
- **Linsley D., Feng P., Serre T. 2025. Better artificial intelligence does not mean better models of biology.**
  Trends in Cognitive Sciences 30(7):599–610 (online 2025-12-23; issue data from OpenAlex),
  doi:10.1016/j.tics.2025.11.016, arXiv:2504.16940. https://arxiv.org/abs/2504.16940. "This alignment is now
  plateauing - and in some cases worsening - as DNNs scale to human or superhuman accuracy"; "vision science must
  chart its own course". The opinion-piece form of the 2023 result.
- **Gokce A., Schrimpf M. 2025. Scaling Laws for Task-Optimized Models of the Primate Visual Ventral Stream.** ICML
  2025 (spotlight), PMLR 267, arXiv:2411.05712. https://icml.cc/virtual/2025/poster/44987,
  https://arxiv.org/abs/2411.05712 (PDF read). Over 600 models trained under controlled conditions: "while
  behavioral alignment continues to scale with larger models, neural alignment saturates", across architectures
  and datasets. Scale helps higher areas most (Behavior > IT > V4 > V2 > V1, Fig. 5b); behavioural alignment
  tracks validation accuracy while neural alignment saturates (Fig. 6). Conclusion: scaling "will not yield
  improved models of the brain's visual ventral stream". Figure: Fig. 1b (fitted scaling curves: behaviour
  keeps rising, neural alignment flattens).
- **Conwell C., Prince J.S., Kay K.N., Alvarez G.A., Konkle T. 2024. A large-scale examination of inductive biases
  shaping high-level visual representation in brains and machines.** Nature Communications 15:9383,
  doi:10.1038/s41467-024-53147-y. https://www.nature.com/articles/s41467-024-53147-y. Zotero:
  `Conwell et al. - 2024 - A large-scale examination of inductive biases shaping high-level visual representation in brains and.pdf`.
  224 models against human occipitotemporal fMRI (Natural Scenes Dataset), over 1.8 billion regressions:
  architectures (CNN versus Transformer) and objectives (contrastive versus vision–language) "achieve near
  equivalent brain predictivity, when other factors are held constant"; "variation across visual training diets
  yields the largest, most consistent effect". Among trained models, ImageNet top-1 accuracy barely relates to
  predictivity (Spearman −0.0057 in classical RSA; 0.17, p = 0.088, in voxel-encoding RSA; Fig. 5C); 14 matched
  ImageNet-1K/21K pairs show no effect of the larger set. They warn that "standard methods used to link models to
  brains may be too flexible". Figure: Fig. 5C (top-1 accuracy against brain predictivity: a flat cloud).
- **Conwell C., Bonner M.F. 2025. Model manifold analysis suggests the human visual brain is less like an optimal
  classifier and more like a feature bank.** NeurIPS 2025 workshop (DBM). https://openreview.net/forum?id=7ESqeV4vsV.
  Zotero: `Conwell and Bonner - 2025 - Model manifold analysis suggests the human visual brain is less like an optimal classifier and more.pdf`.
  Across 117 models, manifold signal-to-noise ratio predicts occipitotemporal alignment better than top-k
  recognition accuracy; the ventral stream looks like "a basis set (or feature vocabulary) for object recognition
  rather than ... the actual locus of recognition per se." Fits the foundation-model framing: a general feature
  bank read out by many tasks.
- **Raugel J., Szafraniec M., Vo H.V., Couprie C., Labatut P., Bojanowski P., Wyart V., King J.-R. Disentangling the
  Factors of Convergence between Brains and DINOv3.** ICLR 2026; arXiv:2508.18226 (2025) under the title
  "Disentangling the Factors of Convergence between Brains and Computer Vision Models".
  https://openreview.net/forum?id=i99ccgfad8, https://arxiv.org/abs/2508.18226. Zotero (ICLR version, read in
  full): `Raugel et al. - 2025 - Disentangling the Factors of Convergence between Brains and DINOv3.pdf`;
  arXiv version: `Raugel et al. - 2025 - Disentangling the Factors of Convergence between Brains and Computer Vision Models.pdf`.
  What they did: trained DINOv3 models from scratch varying size (Small 21M to Giant 1.1B on 1.7B images, plus the
  7B model's training checkpoints) and image type (Large models on 10M human-centric, satellite or cell-microscopy
  images), and compared them by linear ridge encoding with 7T fMRI (Natural Scenes Dataset, 8 subjects) and MEG
  (THINGS-MEG, 4 subjects). Three scores: encoding (Pearson r), spatial (best layer against distance from V1) and
  temporal (best layer against MEG peak time). What they found:
  (1) DINOv3-7B predicts the visual pathway (peak R = .45 ± .039) and, beyond it, prefrontal areas (BA44, BA45,
  IFS); the layer hierarchy maps onto cortex (spatial r = 0.38) and onto MEG latency (temporal score R = 0.96 in
  the text; the Fig. 2D caption gives r = 0.84).
  (2) Size, training amount and image type each matter, and interact; "the largest DINOv3 models trained with the
  most human-centric images reach the highest brain-similarity". Size: voxel-averaged encoding R of .096 (Small),
  .101 (Base), .105 (Large), .107 (Giant), each step significant at p < 1e−3, the increase mostly in higher-level
  cortex (BA45, IFS).
  Human-centric images beat satellite and cell images even in V1. The findings "generalize across seven additional
  models".
  (3) A training chronology: the encoding score reaches half its final value at about 2% of training (about 800
  million images); early sensory areas and early MEG windows are matched first, prefrontal areas and late windows
  last.
  (3b) Checked 2026-10-25 in arXiv:2508.18226v1, §3.2: a region's half time correlates with its distance from V1
  (R = 0.91, p < 1e−5); across MEG time windows, half time correlates with the temporal peak (R = 0.84, p < 1e−5).
  Half times of the three scores: temporal about 0.7% of training, encoding about 2%, spatial 4%; before training,
  encoding is R = .03 and the spatial and temporal scores are negative (§4). The Discussion's "between 1% and 4%"
  disagrees with §3.2's 0.7%.
  (4) The regions acquired last are those with the greatest developmental expansion (r = 0.88), thickness (0.77),
  slowest intrinsic timescales (0.71) and least myelin (−0.85).
  Caveat (an agent's reading, 2026-10-25): these four properties, and distance from V1, all vary along the same
  sensory-to-association gradient, so the correlations may reflect a single gradient rather than four mechanisms;
  the paper does not test whether any property predicts half time once distance from V1 is controlled for.
  An idea for CanViT (an agent's proposal, not the authors'): compare the order in which brain-like representations
  emerge during training in a passive model and in an active one; they might reach similar final encoding scores
  along different trajectories.
  Their own limits: a single model family; passive viewing data; their argument is about which factors drive
  convergence, and it never relates brain similarity to benchmark accuracy. Figures: Fig. 3 (the three
  scores over training), Fig. 4 (half time per region against distance from V1), Fig. 7 (half time against
  cortical properties).
  (5) The ICLR version against arXiv v1 (both read in full, 2026-10-27; the ICLR PDF is the Zotero copy above,
  sha256 31a5aea5…, also saved by Yohaï as `~/Downloads/12743_Disentangling_the_Factor.pdf`). Figures renumbered:
  v1's Fig. 1 is the unnumbered Method Figure, v1's Fig. 5 is Fig. 4 (the deck's half-time panel, v1 Fig. 5E, is
  ICLR Fig. 4E). Region-wise half times are of the last layer's encoding score (§3.2: "lower half times of the last
  layer"; Fig. 4A's caption says "layer 1"). Added: seven other trained models (ResNet-50, ConvNeXt-Large, ViT-MAE,
  supervised ViT-L/16, CLIP, SigLIP2, OWL-ViT) with untrained controls: once trained they score "similar" on all
  three metrics, and trained models beat untrained ones (Figs. S1–S5); the spatial score along the Margulies
  sensory-to-transmodal gradient rises from −0.32 to 0.13 over training (S6); distillation (S8) and Gram anchoring
  (S9) leave the scores' behaviour unchanged. Discussion scoped: "the model–brain similarity increases consistently
  with larger DINOv3 architectures"; the developmental trajectories "have not been tested in other models", only
  conjectured to generalize.
  (6) Its review (OpenReview forum i99ccgfad8, read 2026-10-27 from the page as Yohaï pasted it and through the
  API). Accept (Poster), 26 Jan 2026. Ratings 8, 6, 4, 2. The meta-review: the main concerns were "only for a single
  model (DINOv3) with a new performance metric that may be tailed for this model and that similar results have
  already been shown"; the seven added models address the first, and disentangling the factors sets the work apart;
  the AC expected the 2 and the 4 to rise to 6. Reviewers' points worth knowing: the title was too general (hence
  "…and DINOv3"); the human-centric advantage may come from the brain datasets' images (NSD, THINGS) resembling the
  training images, which the authors cannot check and now state as a limitation; larger models aligning better
  "substantially replicates" Huh et al. 2024 (Platonic Representation Hypothesis); the temporal score may owe much to
  architecture. The authors' answers: representations are the average of patch tokens (the CLS token encoded worse;
  registers unused); "small models trained longer still perform worse than larger models trained for fewer steps";
  RSA was too costly at this scale (70k × 70k), decoding not comparable across models. No reviewer questioned the
  size of the effects.
- **Raugel J. et al. 2026. Misalignment Between Backpropagation and the Hierarchy of Brain Responses to Images.**
  arXiv:2605.28693 (preprint). https://arxiv.org/abs/2605.28693. Zotero:
  `Raugel et al. - 2026 - Misalignment Between Backpropagation and the Hierarchy of Brain Responses to Images.pdf`.
  Backpropagated gradients of DINOv3 (and eight other models) predict fMRI and MEG in higher visual cortex and at
  later latencies, but their order and spatial layout diverge from the brain's hierarchy: "although deep networks
  and the brain may share similar representational content, they likely rely on fundamentally different
  mechanisms to learn those representations."
- **Muttenthaler L., Dippel J., Linhardt L., Vandermeulen R.A., Kornblith S. 2023. Human alignment of neural
  network representations.** ICLR 2023 (arXiv comment), arXiv:2211.01201. https://arxiv.org/abs/2211.01201. On
  human similarity judgements, "model scale and architecture have essentially no effect on the alignment ...
  whereas the training dataset and objective function both have a much larger impact"; "scaling alone is unlikely
  to be sufficient".
- **Muttenthaler L. et al. 2025. Aligning machine and human visual representations across abstraction levels.**
  Nature 647:349–355, doi:10.1038/s41586-025-09631-6, arXiv:2409.06509. https://arxiv.org/abs/2409.06509 (abstract
  also read via OpenAlex). Vision foundation models miss the coarse levels of human conceptual hierarchy;
  fine-tuning them toward a teacher trained on human judgements improves both human alignment and "generalization
  and out-of-distribution robustness". Alignment and usefulness can move together when alignment is trained for.
- **Mehta Y., Bonner M.F. 2026. Extremely coarse learning objectives induce human-aligned representations in AI
  vision models.** arXiv:2605.05556 (preprint). https://arxiv.org/abs/2605.05556 (PDF read). Networks trained to
  separate as few as eight broad classes match or exceed 1,000-class networks on human fMRI and macaque
  electrophysiology alignment, and "surpass all tested models, including widely used large-scale models such as
  CLIP and DINOv3", on THINGS similarity judgements. Fig. 3c. A 2026 result that DINOv3 is not the most
  human-aligned model on behaviour.
- **Doerig A. et al. 2025. High-level visual representations in the human brain are aligned with large language
  models.** Nature Machine Intelligence 7:1220–1234, doi:10.1038/s42256-025-01072-0 (abstract via OpenAlex).
  Zotero (preprint version): `Doerig et al. - Visual representations in the human brain are alig.pdf`.
  Networks trained to map images to LLM caption embeddings are "better aligned with brain representations than a
  large number of state-of-the-art alternative models, despite being trained on orders-of-magnitude less data."
  In this comparison the training objective, with far less data, produced the better brain model.
- **Mahner F.P., Muttenthaler L., Güçlü U., Hebart M.N. 2025. Dimensions underlying the representational alignment
  of deep neural networks with humans.** Nature Machine Intelligence 7:848–859, doi:10.1038/s42256-025-01041-7.
  Zotero: `Mahner et al. - 2025 - Dimensions underlying the representational alignment of deep neural networks with humans.pdf`.
  On odd-one-out judgements, a VGG-16 shows "a clear dominance of visual over semantic properties", unlike humans.
  An older network; useful for method, weaker for claims about foundation models.
- **Bowers J.S. et al. 2023. Deep problems with neural network models of human vision.** Behavioral and Brain
  Sciences 46:e385 (online 2022-12-01), doi:10.1017/S0140525X22002813 (abstract via OpenAlex). Prediction
  benchmarks "do not test hypotheses regarding what features are contributing to good predictions", and "DNNs
  account for almost no results from psychological research." The target article that Rothkopf et al. (§5)
  answer.

### 4. Benchmarks versus understanding

- **Geirhos R. et al. 2019. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy
  and robustness.** ICLR 2019 (oral), arXiv:1811.12231. https://arxiv.org/abs/1811.12231. On cue-conflict images,
  "ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes, which is in stark
  contrast to human behavioural evidence"; training on Stylized-ImageNet gives a shape bias and robustness. Figure: Fig. 1 (a cat shape with elephant-skin texture, classified as an elephant).
- **Geirhos R. et al. 2020. Shortcut learning in deep neural networks.** Nature Machine Intelligence 2:665–673,
  doi:10.1038/s42256-020-00257-z, arXiv:2004.07780. https://arxiv.org/abs/2004.07780. "Shortcuts are decision rules
  that perform well on standard benchmarks but fail to transfer to more challenging testing conditions." Fig. 1
  (examples of shortcuts).
- **Geirhos R. et al. 2021. Partial success in closing the gap between human and machine vision.** NeurIPS 2021
  (oral), arXiv:2106.07411. https://arxiv.org/abs/2106.07411. 85,120 trials from 90 observers on 17
  out-of-distribution datasets: "The longstanding distortion robustness gap between humans and CNNs is closing";
  "There is still a substantial image-level consistency gap, meaning that humans make different errors than
  models"; "human-to-model consistency improves when training dataset size is increased by one to three orders of
  magnitude." Data, more than architecture or objective, moves models toward humans. Figures: Fig. 1 (core
  results), Fig. 3 (shape bias by model), Fig. 4 (error consistency with humans).
- **Wichmann F.A., Geirhos R. 2023. Are Deep Neural Networks Adequate Behavioural Models of Human Visual
  Perception?** Annual Review of Vision Science 9, arXiv:2305.17023. https://arxiv.org/abs/2305.17023. DNNs "should
  only be regarded as promising -- but not yet adequate -- computational models of human core object recognition
  behaviour"; distinguishes statistical tools from computational models.
- **Recht B., Roelofs R., Schmidt L., Shankar V. 2019. Do ImageNet Classifiers Generalize to ImageNet?** ICML 2019,
  arXiv:1902.10811. https://arxiv.org/abs/1902.10811. Zotero:
  `Recht et al. - 2019 - Do ImageNet Classifiers Generalize to ImageNet.pdf`.
  A re-collected test set costs 11–14 points of accuracy, yet "accuracy gains on the original test sets translate
  to larger gains on the new test sets" (slope 1.1 on ImageNet, Fig. 1). Evidence that benchmark progress was not
  overfitting to the test set. Figure: Fig. 1.
- **Beyer L. et al. 2020. Are we done with ImageNet?** arXiv:2006.07159. https://arxiv.org/abs/2006.07159. "Yes, and
  no": with re-collected labels, recent gains are "substantially smaller than those reported on the original
  labels". In the CanViT bibliography.
- **Taori R. et al. 2020. Measuring Robustness to Natural Distribution Shifts in Image Classification.** NeurIPS
  2020, arXiv:2007.00644. https://arxiv.org/abs/2007.00644. Across 204 models and 213 conditions, robustness to
  synthetic perturbations barely transfers to natural shifts; "the main exception is training on larger and more
  diverse datasets".
- **Miller J. et al. 2021. Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and
  In-Distribution Generalization.** arXiv:2107.04649 (ICML 2021 per my recollection; venue UNVERIFIED).
  https://arxiv.org/abs/2107.04649. Out-of-distribution accuracy is "strongly correlated with in-distribution
  performance for a wide range of models and distribution shifts", with documented exceptions.
- **Kornblith S., Shlens J., Le Q.V. 2019. Do Better ImageNet Models Transfer Better?** CVPR 2019 (oral),
  arXiv:1805.08974. https://arxiv.org/abs/1805.08974. Fixed-feature and fine-tuned transfer correlate with ImageNet
  accuracy (r = 0.99 and 0.96), but "many common forms of regularization slightly improve ImageNet accuracy but
  yield penultimate layer features that are much worse for transfer learning."
- **Fang A., Kornblith S., Schmidt L. 2023. Does progress on ImageNet transfer to real-world datasets?** NeurIPS
  2023, doi:10.52202/075280-1089, arXiv:2301.04644. https://arxiv.org/abs/2301.04644. On six real-world datasets
  (camera traps, satellites, ...), "models with higher ImageNet accuracy do not consistently yield performance
  improvements."
- **DINOv3 (§1), Fig. 5.** During long training of ViT-g and ViT-7B, ImageNet linear accuracy keeps rising while
  VOC segmentation falls: "As training progresses, these similarities increase and the performance on dense tasks
  decreases." Fig. 6 shows patch similarity maps becoming "less localized" and "noisier". Direct evidence that a
  global classification score can hide the loss of spatial scene structure, in the model family CanViT distills. Figure: Fig. 5 (two curves diverging), optionally with Fig. 6.
- **Chen X., Marks M., Cheng Z. 2025. Probing the Mid-level Vision Capabilities of Self-Supervised Learning.** CVPR
  2025, doi:10.1109/CVPR52734.2025.02801, arXiv:2411.17474. https://arxiv.org/abs/2411.17474. 22 self-supervised
  models on 8 mid-level tasks (object localization, 3D geometry): "a weak correlation between mid-level and
  high-level task performance."
- **El Banani et al. 2024 (§2)**: CLIP's semantic strength coexists with poor depth; **3D-PC (§2)**: depth order
  correlates with recognition accuracy, perspective taking does not.
- Search result, stated as such: I found no study that establishes dense probes (segmentation, depth) as a
  stronger test of scene understanding than classification in general. The sources above show that classification
  accuracy and dense or geometric quality dissociate. Calling dense probing "more demanding of spatial scene
  understanding" is an inference from that dissociation, not a cited finding.

### 5. Active vision and foundation models

- **Huang Y., Wang Z., Tang W., Lu C., Cai P. 2026. I-Perceive.** arXiv:2603.00600 (submitted 2026-02-28),
  https://arxiv.org/abs/2603.00600; NeurIPS 2026 poster. The arXiv abstract page titles it "A Foundation Model for
  Active Perception with Language Instructions"; the PDF and the NeurIPS listing, "for Vision-Language Active
  Perception". For mobile manipulators indoors: from context images and a language instruction, it predicts a 6D
  camera pose that fulfills the perception intent. Read in search results and the abstract on 2026-10-24, not in the
  paper. Not cited by CanViT: judged to belong to another tradition (3D viewpoint selection).
- **Rothkopf C.A., Bremmer F., Fiehler K., Dobs K., Triesch J. 2023. Models of vision need some action.** Behavioral
  and Brain Sciences 46:e405, doi:10.1017/S0140525X23001577 (commentary on Bowers et al.).
  https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/models-of-vision-need-some-action/5171353472045298D87828F4EE62C4F5.
  Zotero (PsyArXiv version): `Rothkopf et al. - 2023 - Models of vision need some action.pdf`.
  Bowers et al. "overlook a much more fundamental limitation of this literature: disregarding the importance of
  action and interaction for perception." Models should learn self-supervised from input structured by the
  observer's own actions; object recognition from shuffled photographs is "as if the whole goal of human vision
  was to learn to shout out an appropriate word while being presented a random pile of photographs" (quoted from
  the PsyArXiv version; the abstract matches the published record). In the CanViT bibliography.
- **Killick G., Henderson P., Siebert P., Aragon-Camarasa G. 2023. Foveation in the Era of Deep Learning.** BMVC
  2023, arXiv:2312.01450. https://arxiv.org/abs/2312.01450. Zotero:
  `Killick et al. - 2023 - Foveation in the Era of Deep Learning.pdf`.
  A method paper despite the title. Its introduction states the two obstacles: foveated sensors "are not naturally
  amenable to standard convolution operators", and reinforcement learning for the attention policy brings "the
  difficulty of training systems with this method from scratch".
- **Search result for "a review arguing active vision lagged for lack of large-scale pretraining"**: none found.
  The argument exists in the CanViT paper itself (Saccader "introducing an intermediate pretraining step to
  stabilize learning"; active models "have struggled to match the accuracy, efficiency, flexibility and
  representational richness of their passive counterparts"; CanViT pretrains on 13.2 million ImageNet-21k scenes,
  "an order of magnitude more than previous active models"). The published statements nearest to it are the
  training difficulties above and the trend below: recent active and foveated models increasingly start from a
  pretrained passive network.
- **Blauch N.M., Alvarez G.A., Konkle T. 2026. FOVI: A biologically-inspired foveated interface for deep vision
  models.** ICML 2026, arXiv:2602.03766. https://arxiv.org/abs/2602.03766. Zotero:
  `Blauch et al. - 2026 - FOVI A biologically-inspired foveated interface for deep vision models.pdf`.
  A retina- and V1-inspired sensor manifold; one use case is "a foveated adaptation of the DINOv3 ViT foundation
  model, leveraging low-rank adaptation", competitive "with a fraction of the pixels and computational cost".
  A foveated foundation model obtained by adapting DINOv3's own weights; CanViT instead distills DINOv3's features
  into a new recurrent architecture. Figure: Fig. 1.
- **Thorat S., Doerig A., Kroner A., Amme C., Kietzmann T.C. 2025. Predicting upcoming visual features during eye
  movements yields scene representations aligned with human visual cortex.** arXiv:2511.12715 (preprint).
  https://arxiv.org/abs/2511.12715. Zotero:
  `Thorat et al. - 2025 - Predicting upcoming visual features during eye movements yields scene representations aligned with h.pdf`.
  Glimpse Prediction Networks predict the embedding of the next glimpse, taken from a pretrained network
  (SimCLR ResNet-50; DINOv2 also tested), along human-like scanpaths over COCO. Recurrent variants integrate
  glimpses into a scene representation that aligns with 7T fMRI in mid and high-level visual cortex better than
  matched semantic controls and better than the best alternatives tested, including DINOv2-L. Close to CanViT's
  logic (predict a frozen pretrained network's features from glimpses), with a brain-alignment result. The
  Zotero PDF is an earlier version; the abstract at the URL (v2) is worded differently.
- **Prisadnikov N., Paudel D.P., Fu Y., Van Gool L. 2026. Self-supervised pretraining for an iterative image size
  agnostic vision transformer.** arXiv:2604.20392 (preprint). https://arxiv.org/abs/2604.20392. DINO-style
  self-distillation adapted to a "foveal-inspired transformer" that iterates over multi-zoom patches, "to unlock
  its potential as a foundational backbone"; constant compute regardless of input resolution, evaluated on
  classification.
- **Wang Y. et al. 2025. Emulating human-like adaptive vision for efficient and flexible machine visual perception
  (AdaptiveNN).** Nature Machine Intelligence 7:1804–1822, doi:10.1038/s42256-025-01130-7, arXiv:2509.15333.
  Zotero: `Wang et al. - 2025 - Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception.pdf`.
  Coarse-to-fine fixations trained with self-rewarding reinforcement learning; "up to 28 times inference cost
  reduction without sacrificing accuracy" across 17 benchmarks.
- **Kolner O., Ortner T., Woźniak S., Pantazi A. 2025. Mind the GAP: Glimpse-based Active Perception improves
  generalization and sample efficiency of visual reasoning.** ICLR 2025, arXiv:2409.20213. Zotero:
  `Kolner et al. - 2025 - Mind the GAP Glimpse-based Active Perception improves generalization and sample efficiency of visua.pdf`.
  Glimpse locations as a "where" signal improve relational reasoning and out-of-distribution generalization.
  Follow-up: **Kolner et al. 2026. Task-driven Processing with Coarse-to-Fine Glimpse-based Active Perception**,
  arXiv:2609.09025 (preprint), Zotero `Kolner et al. - 2026 - Task-driven Processing with Coarse-to-Fine Glimpse-based Active Perception.pdf`:
  a glimpse front-end for existing detectors, up to 20% AP gain on high-resolution instance detection.
- **Ebouky B. et al. 2026. GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning.**
  arXiv:2605.07817 (preprint). Zotero:
  `Ebouky et al. - 2026 - GazeVLM Active Vision via Internal Attention Control for Multimodal Reasoning.pdf`.
  A 4B vision–language model emits gaze tokens that bias its own attention toward a region; active vision inside a
  foundation model, trained with GRPO.
- **Caccavella C. et al. 2026. Efficient Semantic Understanding from Digital Foveation.** ECCV 2026 workshop
  (Human-inspired Computer Vision), arXiv:2609.04088. https://arxiv.org/abs/2609.04088. On ADE20K-Object, "a single
  foveated observation achieves 95.9% of the baseline Top-1 accuracy ... while requiring only 4.7% of the
  computational cost."
- **Foveated tokenization for segmentation, both CVPR 2025 and in the CanViT bibliography**: Schmidt T., Newcombe
  R. Segment This Thing (Zotero `Schmidt and Newcombe - 2025 - Segment This Thing Foveated Tokenization for Efficient Point-Prompted Segmentation.pdf`):
  patches grow with distance from a point prompt, cutting tokens without shrinking the model. Zeng H. et al.
  Foveated Instance Segmentation (Zotero `Zeng et al. - 2025 - Foveated Instance Segmentation.pdf`):
  segmentation restricted to the instance at the user's gaze, for AR/VR. Single-fixation and gaze-driven, not
  sequential models with memory.
- **Pourrahimi M., Bashivan P. 2025. Emergent brain-like representations in a goal-directed neural network model of
  visual search.** bioRxiv, doi:10.1101/2025.06.06.658387. Zotero:
  `Pourrahimi and Bashivan - 2025 - Emergent brain-like representations in a goal-directed neural network model of visual search.pdf`.
  A visual-search agent built on an ImageNet-pretrained CNN develops a retinocentric cue-similarity map and
  prospective fixation signals resembling fronto-parietal activity. In the CanViT bibliography (as "Neural
  signatures of associational cortex emerge in a goal-directed model of visual search").

### Tensions across these papers (a subagent's synthesis, 2026-10-01)

1. **Accuracy predicted the brain, then stopped.** Yamins 2014 and Brain-Score 2018 found the link; Brain-Score
   2018 already saw it weaken above 70% top-1; Fel 2022, Linsley 2023 and Linsley, Feng, Serre 2025 report it
   reversing for ImageNet models; Gokce and Schrimpf 2025 find neural alignment saturating while behavioural
   alignment keeps scaling; Conwell 2024 finds no relation among trained models. Within one controlled family,
   Raugel et al. find that larger, longer-trained DINOv3 models on human-centric images are more brain-like,
   consistently and significantly, a claim their ICLR version scopes to DINOv3. These studies use different data (macaque electrophysiology, human fMRI and MEG),
   metrics and model sets, so they do not contradict one another directly; together they say that benchmark
   accuracy is a weak proxy for brain similarity at today's performance levels, and that data and objective
   matter more than architecture (Conwell 2024, Muttenthaler 2023, Geirhos 2021).
2. **Representational convergence may be partly a measurement artefact.** The Platonic Representation Hypothesis
   (Huh 2024) against the calibrated reanalysis (Gröger 2026): global convergence largely disappears; local
   neighbourhood structure survives. Conwell 2024 makes the parallel point for brain mapping ("too flexible").
3. **World knowledge is real but bounded.** Depth, normals, parts and correspondence are linearly decodable from
   self-supervised features (DINOv2; El Banani 2024), yet multiview consistency is poor (El Banani 2024),
   perspective taking is at chance (3D-PC), and intuitive physics passes simple tests (Garrido 2025) while failing
   complex scenes (IntPhys 2).
4. **Benchmarks mislead and inform.** Shortcuts and texture bias (Geirhos 2019, 2020) coexist with progress that
   transfers to new test sets (Recht 2019), tracks out-of-distribution accuracy (Miller 2021) and, with more data,
   narrows the human–machine error gap (Geirhos 2021). Classification can hide degraded spatial features (DINOv3
   Fig. 5; Kornblith 2019; Chen 2025).
5. **The brain-like model is not always the best model.** An eight-class coarse network beats DINOv3 and CLIP on
   human similarity judgements (Mehta and Bonner 2026); human-aligned fine-tuning improves both alignment and
   robustness (Muttenthaler 2025). Alignment and capability are partly separate axes.

### Figures in these papers

| Paper | Figure | Shows |
|---|---|---|
| Bommasani 2021 | Fig. 2 | One model, many data sources, many adapted tasks |
| DINO (Caron 2021) | Fig. 1 | Unsupervised attention maps outlining objects |
| DINOv2 (Oquab 2024) | Fig. 1; Fig. 7; Fig. 10 | PCA part matching; linear depth and segmentation; cross-image matching |
| DINOv3 (Siméoni) | Fig. 1; Fig. 3; Fig. 5 | ImageNet linear probing over the years; 4096² similarity maps; ImageNet up while segmentation falls |
| SAM (Kirillov 2023) | Fig. 1 | Task, model, data engine |
| El Banani 2024 | Fig. 1; Fig. 2; Fig. 6 | 3D-awareness ratings; probed depth maps; correspondence against viewpoint change |
| Garrido 2025 | Fig. 1; Fig. 2 | Violation-of-expectation accuracy by model family; per-property and human comparison |
| Huh 2024 | Fig. 1 | The "cave": images and text as projections of one reality |
| Yamins 2014 | Fig. 1 | Categorization performance against IT predictivity |
| Brain-Score 2018 | Fig. 1 | ImageNet top-1 against Brain-Score, flattening above 70% |
| Linsley 2023 | Fig. 1 | IT predictivity falling as ImageNet accuracy rises (Pareto front) |
| Gokce and Schrimpf 2025 | Fig. 1b; Fig. 6 | Behavioural alignment scales, neural alignment saturates |
| Conwell 2024 | Fig. 5C | Top-1 accuracy against brain predictivity: no trend among trained models |
| Raugel (ICLR 2026) | Fig. 3; Fig. 4; Fig. 7 | Scores over training; region-wise half times; half time against cortical properties |
| Mehta and Bonner 2026 | Fig. 3c | Eight-class network above DINOv3 and CLIP on THINGS judgements |
| Geirhos 2019 | Fig. 1 | Cat shape with elephant texture |
| Geirhos 2021 | Fig. 1; Fig. 4 | Robustness gap closing; error-consistency gap remaining |
| Recht 2019 | Fig. 1 | Original against new test accuracy, on a line of slope above 1 |
| FOVI (Blauch 2026) | Fig. 1 | Retina-like sensor manifold and foveated ViT |

Check each figure's license before reuse; captions above were read from the PDFs, the images were not inspected.

## Cognitive science: concepts that name what CanViT does

Wording the authors use [Yohaï, 2026-10-01]: CanViT sees only its glimpses; it infers, guesses or completes what it did
not sample. The canvas holds; a policy chooses (EG-C2F chooses where the segmentation read out from the canvas is least
certain).

### Trans-saccadic integration and memory

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
    Rev.* 185:106644. doi:10.1016/j.neubiorev.2026.106644. Read: abstract. Higgins & Rayner 2015 is under "Transsaccadic integration" below.
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
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** the canvas carries what each glimpse brought into the processing of the next and accumulates
  it over the rollout. It stores features, the kind of
  abstract code Irwin found, not fused pixels. Reads condition the next glimpse's processing on what was seen, as a
  presaccadic preview conditions postsaccadic processing (next entry). The memory-reset measurement (`checked-facts.md`) is the model's version of "does not have to start anew".
- **CanViT, differs (agent's mapping, not confirmed by the authors):** a glimpse is a crop at a chosen zoom, with no periphery and no eye; each glimpse's position and
  scale are given, as if corollary discharge were exact and free. The canvas is dense and tied to absolute scene
  position (a grid of 1024-dimensional tokens, 32² in pretraining and up to 64² in the paper's evaluations), where
  human trans-saccadic memory holds three to four items and is "not strictly tied to absolute spatial position" (Irwin
  1991; Prime et al. 2011), though precise world-centred positions can be kept when the task requires it (Hayhoe et
  al. 1991). Human integration weights each view by its reliability; CanViT's write rule is learned and its weighting
  has not been measured. Scenes are static, so CanViT
  never has to decide whether the world changed between views (Stewart et al. 2020, "Integration's limits: causal
  inference"; Choi et al. 2025).

### Coarse-to-fine, and why order matters

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
    84(4):405–439. doi:10.1016/j.pneurobio.2007.09.001. Read: abstracts. Navon 1977 is under "Citation records"; its abstract
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
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** C2F starts with the whole scene at low resolution (a low-pass view of everything), then
  quadrants, then sixteenths; F2C visits the same tiles in reverse, and C2F reaches the higher peak accuracy (the
  paper: "highlighting the importance of contextually informed processing"). This is the Musel/Kauffmann contrast
  with viewpoints in place of filtered images. In C2F the first glimpse previews every region the later glimpses zoom
  into, as extrafoveal vision previews a saccade target.
- **CanViT, differs (agent's mapping, not confirmed by the authors):** the human sequences keep coverage fixed and change spatial frequency; a CanViT zoom changes
  coverage and resolution together. Human preview is simultaneous (periphery during fixation); CanViT's is sequential.
  The mechanism behind C2F > F2C is not isolated. Hypotheses (none measured): reads condition the zoomed glimpses on
  the coarse guess (Bar's top-down facilitation); in F2C, late coarse writes overwrite fine detail written earlier;
  long monotone sequences of either kind differ from pretraining's random viewpoints in different ways.

### Looking where you are uncertain: local uncertainty and active sensing

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
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** EG-C2F is Renninger's local rule on a scene-coordinate map: the entropy of the class
  distribution read out from the canvas at each cell, averaged per tile, the most uncertain unvisited tile visited
  next within each quadtree level. The canvas is the "stimulus-centered representation that incorporates knowledge
  gained from previous fixations" that Renninger et al. say a local uncertainty map needs. It is "information
  sampling" in Gottlieb & Oudeyer's sense: reducing uncertainty for a known task.
- **CanViT, differs (agent's mapping, not confirmed by the authors):** EG-C2F ranks by current entropy, not by expected information gain (Najemnik & Geisler's ideal
  searcher, Butko & Movellan's infomax, Houlsby et al.'s BALD): it does not predict what a glimpse would reveal.
  Predictive entropy mixes ambiguity that more looking cannot remove (aleatoric: class boundaries) with ignorance that
  it can (epistemic: unseen regions) (Kendall & Gal 2017); EG-C2F's per-tile averaging and once-per-level visits limit,
  but do not remove, glimpses spent on the first kind (a hypothesis, not measured). The quadtree schedule is fixed;
  only the order within a level adapts. Najemnik & Geisler found humans near-optimal "even though humans integrate
  information poorly across fixations".

### Metacognition: monitoring and control, and where the term stops

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
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** a policy reads the observer's own uncertainty about what is where and uses it to decide what
  to sample next: monitoring feeding control, the loop Desender et al. and Schulz et al. study in people, with the
  monitored quantity computed from the model's own state rather than from the world.
- **CanViT, differs (agent's mapping, not confirmed by the authors):** the entropy is computed from the same readout as the segmentation, so it is a first-order
  quantity in Fleming & Daw's terms and certainty about the world in Pouget et al.'s, not propositional confidence in
  a decision. Its calibration (Guo et al.) and its metacognitive sensitivity (does entropy predict where the
  segmentation is wrong; Fleming & Lau's type-2 measures) have not been measured. The policy, not the canvas, does
  the monitoring.

### Spatiotopic memory, which the brain may not have

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
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** the paper's "retinotopic Vision Transformer backbone" and "spatiotopic scene-wide latent
  workspace", bound by scene-relative RoPE. The canvas has both properties Cavanagh & Melcher attribute to a
  spatiotopic map, at every position: nothing in it moves when the viewpoint moves, and processing continues across
  glimpses.
  Spatial view cells and the entorhinal map of visual space are where primate brains represent viewed space in world
  coordinates, a neural anchor for the paper's "cognitive map" (Tolman 1948).
- **CanViT, differs (agent's mapping, not confirmed by the authors):** the brain is argued to approximate spatiotopy by remapping a few attended targets on
  retinotopic maps (Cavanagh & Melcher 2026; Golomb & Mazer 2021); CanViT builds the dense map the brain may lack,
  and receives the coordinates the brain must compute. "Cognitive map" stays a functional analogy (`checked-facts.md`).

### Scene memory as objects bound to locations; change blindness

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
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** the canvas is an abstract, location-indexed scene representation that accumulates over
  glimpses, the kind Hollingworth's account describes, on static scenes, where Tatler & Land place extensive memory.
  Melcher (2001) called the human version "medium-term" or "disposable": kept while the scene is in use, which is what
  a canvas is within a rollout.
- **CanViT, differs (agent's mapping, not confirmed by the authors):** human scene memory is object-based and gated by attention; the canvas is written densely at
  every glimpse. CanViT can only update where its writes reach, so a change to an unvisited region would go unnoticed
  until a glimpse covers it: a form of change blindness by construction (an inference from the architecture, not
  tested; the model has only seen static scenes).

### Perceptual filling-in and completion

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
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** the canvas labels the never-sampled middle of the table as table (interpolation between
  glimpses), and after a glimpse of one end alone already extends the table toward the other (extrapolation); the
  paper's appendix shows the last write of a glimpse extending beyond it. Aperture viewing (Orlov & Zohary 2018;
  Craddock et al. 2011) is the human paradigm closest to CanViT's input: an object known only through successive
  partial views.
- **CanViT, differs (agent's mapping, not confirmed by the authors):** human filling-in is mostly within a single view (blind spot, scotoma, occluder), often of
  surface attributes; the table's middle lies outside every glimpse. Nanay's broad definition admits it as amodal
  completion; the classic, occlusion-based definition does not (the authors keep "not amodal completion"). Human
  extrapolation respects object boundaries (Toscani et al.); over large ADE20K objects, the canvas also labels as the
  object a large share of the other pixels in the band between the glimpses (`checked-facts.md`, table corners, for the
  numbers). The canvas marks no difference between seen and inferred regions.

### Trans-saccadic feature prediction

- **What it is:** the visual system learns associations between how an object looks in the periphery before a
  saccade and in the fovea after it, and uses them to predict one from the other.
- **Sources:** Herwig A., Schneider W.X. 2014. Predicting object features across saccades: evidence from object
  recognition and visual search. *J. Exp. Psychol. Gen.* 143(5):1903–1922. doi:10.1037/a0036781. Goktepe N., Schütz
  A.C. 2023. Familiar objects benefit more from transsaccadic feature predictions. *Atten. Percept. Psychophys.*
  85(6):1949–1961. doi:10.3758/s13414-022-02651-8. Read: abstracts; the re-calibration literature as summarized in
  Stewart et al. 2020 (read).
- **Quote:** "the visual system uses past experience to predict how peripheral objects will look in the fovea, and what
  foveal search templates should look like in the periphery" (Herwig & Schneider 2014).
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** pretraining teaches CanViT to predict, from low-resolution partial views, the features a
  high-resolution view of the whole scene yields. Familiar objects benefit more in people (Goktepe & Schütz 2023), as
  world knowledge from the teacher helps CanViT.
- **CanViT, differs (agent's mapping, not confirmed by the authors):** the human teacher is the person's own later foveal view; CanViT's is a frozen passive model
  that sees the whole scene. The paper's proposed future work, active-to-active self-distillation, is closer to the
  human case (an inference from the two descriptions, for the authors to judge).

### Vision at a glance, then recurrent refinement

- **What it is:** the gist of a scene (its category, layout and much of its content) is available from a single brief
  glance; recurrent processing after the first feedforward sweep refines it.
- **Sources:** Oliva A., Torralba A. 2006. Building the gist of a scene: the role of global image features in
  recognition. *Progress in Brain Research* 155:23–36. doi:10.1016/S0079-6123(06)55002-2. Greene M.R., Oliva A. 2009.
  Recognition of natural scenes from global properties: seeing the forest without representing the trees. *Cognitive
  Psychology* 58(2):137–176. doi:10.1016/j.cogpsych.2008.06.001. Fei-Fei L., Iyer A., Koch C., Perona P. 2007. What
  do we perceive in a glance of a real-world scene? *J. Vis.* 7(1):10. doi:10.1167/7.1.10. Lamme V.A.F., Roelfsema
  P.R. 2000. The distinct modes of vision offered by feedforward and recurrent processing. *Trends Neurosci.*
  23(11):571–579. doi:10.1016/S0166-2236(00)01657-X. Read: abstracts. Potter 1976 and Oliva 2005 are in
  "Perception as inference" below; Kar et al. 2019 is in the paper.
- **Quotes:** "Humans can recognize the gist of a novel image in a single glance, independent of its complexity" (Oliva
  & Torralba 2006). "within a single glance, much object- and scene-level information is perceived" (Fei-Fei et al.
  2007).
- **CanViT, analogous (agent's mapping, not confirmed by the authors):** one low-resolution glimpse of the full scene already gives 38.5% mIoU; RFS shows refinement
  from recurrence alone on a fixed input, which then declines (the paper's Results).
- **CanViT, differs (agent's mapping, not confirmed by the authors):** the gist literature limits time (brief, masked presentations); CanViT's first glimpse is
  limited in resolution, not in time.

### Priority map, inhibition of return, Bayesian surprise, active inference

Names for parts of EG-C2F and for policies not yet tried; notes only.

- **Priority map with inhibition of return.** Fecteau & Munoz 2006 and Bisley & Goldberg 2010 are in "Citation records"
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

### Words the paper uses that this audience hears differently

| Paper's word | What this audience hears | What to say |
|---|---|---|
| "cognitive map" (§4, citing Tolman 1948) | hippocampal and entorhinal maps | a functional analogy; primates do map viewed space in world coordinates (spatial view cells; Killian et al. 2012), which the canvas resembles in role only |
| "latent workspace" (abstract) | Global Workspace Theory: a limited-capacity stage broadcasting among specialist modules (VanRullen R., Kanai R. 2021. Deep learning and the Global Workspace Theory. *Trends Neurosci.* 44(9):692–704. doi:10.1016/j.tins.2021.04.005; Goyal A. et al. 2022. Coordination among neural modules through a shared global workspace. ICLR 2022, arXiv:2103.01197, with Yoshua Bengio among the authors. Read: abstracts) | the canvas is a large spatial memory read by one backbone, with no competition for access and no claim about consciousness |
| "mental image" (§5) | visual mental imagery, which "can function much like a weak version of afferent perception" (Pearson J. 2019. The human imagination: the cognitive neuroscience of visual mental imagery. *Nat. Rev. Neurosci.* 20(10):624–634. doi:10.1038/s41583-019-0202-9. Read: abstract) | the canvas is spatially organized and holds a best guess, but it is filled from glimpses, not generated without input |
| "high-capacity working memory" (abstract) | visual working memory of about four objects (Luck & Vogel 1997, "Citation records"; abstract read) and object files (Kahneman D., Treisman A., Gibbs B.J. 1992. The reviewing of object files: object-specific integration of information. *Cognitive Psychology* 24(2):175–219. doi:10.1016/0010-0285(92)90007-O. Read: abstract) | the canvas is indexed by location, not by object, and its capacity is closer to visual long-term memory (Brady T.F., Konkle T., Alvarez G.A., Oliva A. 2008. Visual long-term memory has a massive storage capacity for object details. *PNAS* 105(38):14325–14329. doi:10.1073/pnas.0803390105. Read: abstract) while lasting only one rollout; Melcher's "medium-term" fits |

### Leads not kept

- Baddeley's "visuospatial sketchpad" (and Logie's passive "visual cache" against the active "inner scribe", which
  would echo "the canvas holds, the backbone acts"): no abstract read this session names them (Baddeley 2000, 2012 and
  Logie 2011 abstracts checked). UNVERIFIED; worth one read of Baddeley 2003 (*Nat. Rev. Neurosci.* 4:829–839,
  doi:10.1038/nrn1201) if the authors want the word.
- POMDP belief state (Kaelbling, Littman & Cassandra 1998, *Artificial Intelligence* 101:99–134,
  doi:10.1016/S0004-3702(98)00023-X): abstract not available.
- Navon 1977 (global precedence): abstract not available; Hochstein & Ahissar's abstract carries "forest before trees".
- Sensorimotor contingencies (O'Regan & Noë 2001, read in abstract): CanViT is given its viewpoints and learns no
  sensorimotor laws; kept only as the radical end of the change-blindness accounts.
- Girgus, Gellman & Hochberg 1980 (spatial order in piecemeal shape recognition through an aperture, *Percept.
  Psychophys.* 28(2):133–138, doi:10.3758/BF03204338): metadata only; the publisher's free PDF did not download. Read
  it if the aperture-viewing link is used: its title promises an order effect on piecemeal views.

### Perception as inference from knowledge of the world

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
  doi:10.1037/h0043805 (metadata verified; content UNVERIFIED). **Support:** the source of the phrase "beyond the information given"; do not
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
  change gaze.
- **Võ M.L.-H., Boettcher S.E.P., Draschkow D. 2019. Reading scenes: how scene grammar guides attention and aids
  perception in real-world environments.** *Current Opinion in Psychology* 29:205–210.
  doi:10.1016/j.copsyc.2019.03.009. Read: Zotero PDF. "This scene grammar provides strong priors regarding what
  objects tend to be where within certain scenes." Semantic violations (a fire hydrant in a kitchen) and syntactic
  ones (a toaster in the sink); "anchor objects" (shower, sink, stove) predict where smaller objects are. On where not
  to look: "When you look for a toothbrush you will, therefore, probably quickly exclude the shower and toilet phrases
  from your search and focus your attention on the sink." Figure 1 shows one observer's fixations when searching the
  same room image, which contains neither target, for a laptop versus a teddy. **Support:** strong as a short
  review of scene grammar guiding search; the toothbrush sentence is an illustration, not data.

### 2. Closer to the phenomenon: inferring what was not sampled

#### Boundary extension (beyond the edges of a view)

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

#### Amodal completion (behind an occluder, between visible parts)

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

#### Gist and context at a glance

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

#### Transsaccadic integration

- **Higgins E., Rayner K. 2015. Transsaccadic processing: stability, integration, and the potential role of
  remapping.** *Attention, Perception, & Psychophysics* 77:3–27. doi:10.3758/s13414-014-0751-y. Read: Zotero PDF,
  abstract and introduction. Reviews visual stability and integration across fixations, and predictive remapping.
  **Support:** adjacent. It concerns combining what was sampled across saccades, not inferring what was not sampled.
  Irwin 1991, summarized in "Citation records" as undetailed, capacity-limited memory across saccades, falls in the same
  category (not read this session).

#### Predictive processing and filling in, in visual cortex

- **Rao R.P.N., Ballard D.H. 1999.** *Nature Neuroscience* 2(1):79–87. doi:10.1038/4580. Predictive coding;
  already in "Citation records" and the CanViT paper. **Support:** adjacent; a mechanism for extra-classical
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
  **Support:** strong as the current review of perception as inference with priors; body not read.
- **Spaak E., Peelen M.V., de Lange F.P. 2022. Scene context impairs perception of semantically congruent objects.**
  *Psychological Science* 33(2):299–313. doi:10.1177/09567976211032676. Read: PubMed abstract. Congruent objects were
  perceived worse than incongruent ones (N = 300; change detection and a bias-free discrimination task), as reduced
  prediction error predicts; Peelen et al. 2024's reference note places this "when sensory input is unambiguous".
  **Caveat:** priors can also cost: with clear input, the expected object can be processed less.

#### Knowledge and where the eyes go (beyond Torralba et al. 2006)

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

#### Classic statements and the opposing view

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
  representations". "Prior knowledge" in the sense of learned regularities of scenes; whether that counts as
  top-down penetration in this debate depends on wording. "Inference from knowledge of how the world is laid out"
  keeps the claim on the less contested side.
- **Brewer W.F., Treyens J.C. 1981. Role of schemata in memory for places.** *Cognitive Psychology* 13(2):207–230.
  doi:10.1016/0010-0285(81)90008-6. Metadata only. The often-told result (participants waiting in an office later
  recalled books that were not there) is UNVERIFIED; the publisher page was blocked.

### 3. Figures

Extracted to `site/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/assets/figures/`
(provenance rows in `PROVENANCE.md` there):

- `intraub-richardson-1989-boundary-extension.png` (442×885 px): close-up photograph, a participant's drawing from
  memory with the scene continued past the edges, the wider view.
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

## Citation records

[Checked by a research subagent on 2026-10-01 against publisher, Crossref, PubMed or arXiv records.] Where the
paper's `references.bib` has a work, its entry is authoritative.

### Active vision in machines

- Bajcsy R. 1988. Active perception. *Proc. IEEE* 76(8):966–1005. doi:10.1109/5.5968. Choosing sensing and control strategies to gain information.
- Aloimonos J., Weiss I., Bandyopadhyay A. 1988. Active vision. *IJCV* 1(4):333–356. doi:10.1007/BF00133571. Problems ill-posed for a passive observer become well-posed for an active one.
- Ballard D.H. 1991. Animate vision. *Artificial Intelligence* 48(1):57–86. doi:10.1016/0004-3702(91)90080-4. (Summary not checked: publisher page unavailable.)
- Larochelle H., Hinton G.E. 2010. Learning to combine foveal glimpses with a third-order Boltzmann machine. *NIPS 23*.
- Mnih V., Heess N., Graves A., Kavukcuoglu K. 2014. Recurrent Models of Visual Attention. *NIPS 27*. arXiv:1406.6247. Reinforcement learning chooses where to look; compute independent of image size.
- Ba J., Mnih V., Kavukcuoglu K. 2015. Multiple Object Recognition with Visual Attention. *ICLR 2015*. arXiv:1412.7755.
- Elsayed G.F., Kornblith S., Le Q.V. 2019. Saccader. *NeurIPS 32*. arXiv:1908.07644. 75% top-1 with the separate NASNet classifier.
- Wang Y., Lv K., Huang R., Song S., Yang L., Huang G. 2020. Glance and Focus (GFNet). *NeurIPS 33*. arXiv:2010.05300.
- Pardyl A., Rypeść G., Kurzejamski G., Zieliński B., Trzciński T. 2023. Active Visual Exploration Based on Attention-Map Entropy (AME). *IJCAI 2023*. arXiv:2303.06457.
- Pardyl A. et al. 2024. AdaGlimpse. *ECCV 2024*. arXiv:2404.03482. Soft Actor-Critic chooses glimpses of any position and scale.
- Wang Y. et al. 2025. Emulating human-like adaptive vision for efficient and flexible machine visual perception (AdaptiveNN). *Nature Machine Intelligence* 7:1804–1822. arXiv:2509.15333.

### Passive vision

- Krizhevsky A., Sutskever I., Hinton G.E. 2012. AlexNet. *NIPS 25*.
- Dosovitskiy A. et al. 2021. An Image is Worth 16x16 Words (ViT). *ICLR 2021*. arXiv:2010.11929.
- Caron M. et al. 2021. Emerging Properties in Self-Supervised Vision Transformers (DINO). *ICCV 2021*. arXiv:2104.14294.
- Oquab M. et al. 2024. DINOv2. *TMLR*. arXiv:2304.07193.
- Siméoni O. et al. DINOv3. arXiv:2508.10104 (2025); *TMLR* 2026.
- He K. et al. 2022. Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*. arXiv:2111.06377.

### Neuroscience and cognitive science

- Yarbus A.L. 1967. *Eye Movements and Vision*. Springer. doi:10.1007/978-1-4899-5379-7. Task-dependent scanpaths (chapter "Eye Movements During Perception of Complex Objects", pp. 171–211).
- Henderson J.M., Hollingworth A. 1999. High-level scene perception. *Annual Review of Psychology* 50:243–271. doi:10.1146/annurev.psych.50.1.243. Fixation shifts "an average of three times each second"; mean fixation about 330 ms.
- Rayner K. 1998. *Psychological Bulletin* 124(3):372–422. Scene perception: about 330 ms fixations, 4° saccades.
- Strasburger H., Rentschler I., Jüttner M. 2011. Peripheral vision and pattern recognition: a review. *Journal of Vision* 11(5):13. Minimum angle of resolution roughly linear in eccentricity.
- Curcio C.A., Sloan K.R., Kalina R.E., Hendrickson A.E. 1990. *J. Comparative Neurology* 292(4):497–523. Peak foveal cone density about 199,000 cones/mm², falling tenfold by 1 mm.
- Hoppe D., Rothkopf C.A. 2019. *Scientific Reports* 9:144. Gaze is better explained by planning several steps ahead than greedily.
- Baddeley A.D., Hitch G. 1974. Working Memory. *Psychology of Learning and Motivation*, pp. 47–89.
- Melcher D. 2001. (In the paper's bibliography: persistence of visual memory across fixations.)
- Irwin D.E. 1991. *Cognitive Psychology* 23(3):420–456. Memory across saccades: undetailed, capacity-limited, not tied to absolute position.
- Luck S.J., Vogel E.K. 1997. *Nature* 390:279–281. Visual working memory holds about four objects.
- Duhamel J.-R., Colby C.L., Goldberg M.E. 1992. *Science* 255:90–92. Parietal receptive fields shift before saccades (remapping).
- Rao R.P.N., Ballard D.H. 1999. *Nature Neuroscience* 2(1):79–87. Predictive coding: feedback carries predictions.
- Gilbert C.D., Li W. 2013. (In the paper's bibliography: top-down influences on visual processing.)
- Kar K., Kubilius J., Schmidt K., Issa E.B., DiCarlo J.J. 2019. *Nature Neuroscience* 22:974–983. Recurrence for challenging images; IT identity emerges later.
- Tolman E.C. 1948. Cognitive maps in rats and men. *Psychological Review* 55(4):189–208.
- Fecteau J.H., Munoz D.P. 2006. *Trends in Cognitive Sciences* 10(8):382–390. Priority maps: salience and relevance.
- Bisley J.W., Goldberg M.E. 2010. *Annual Review of Neuroscience* 33:1–21. LIP as a priority map.
- Posner M.I., Cohen Y. 1984. Components of visual orienting. *Attention and Performance X*, Erlbaum. Klein R.M. 2000. Inhibition of return. *Trends in Cognitive Sciences* 4(4):138–147.
- Navon D. 1977. *Cognitive Psychology* 9(3):353–383. Global precedence. Hegdé J. 2008. *Progress in Neurobiology* 84(4):405–439. Coarse-to-fine time course. Bar M. 2003. *J. Cognitive Neuroscience* 15(4):600–609.
- Rensink R.A., O'Regan J.K., Clark J.J. 1997. *Psychological Science* 8(5):368–373. Change blindness. O'Regan J.K. 1992. The world as an outside memory. *Canadian Journal of Psychology* 46(3):461–488.
- Deep networks as models of the visual system, from the paper's introduction: Yamins et al. 2014 (*PNAS*), Yamins & DiCarlo 2016 (*Nature Neuroscience*), Schrimpf et al. 2018 (Brain-Score), Zhuang et al. 2021, Bakhtiari et al. 2021, Mehrer et al. 2021, Raugel et al. 2025.

### Images

| Use | Source | License |
|---|---|---|
| Yarbus's scanpaths | https://commons.wikimedia.org/wiki/File:Yarbus_The_Visitor.jpg | Public domain (uploader's release); cite Yarbus 1967 |
| Acuity against eccentricity | https://commons.wikimedia.org/wiki/File:AcuityHumanEye.svg | CC BY-SA 3.0 (labeled an approximation) |
| Photoreceptor density | https://commons.wikimedia.org/wiki/File:Human_photoreceptor_distribution.svg | CC BY-SA 3.0 or GFDL |

## Bibliography issues found for the paper

As found on 2026-10-24. AME's diacritics, AdaGlimpse's year and Raugel et al.'s year are fixed on the local branch
`paper-camera-ready-fixes` (not yet merged); GAE's pages and DINOv3's month are open.

- AdaGlimpse: the paper's entry says 2025 (print); it is ECCV 2024, online 2024-10-26.
- AME: "Rypesc" should be "Rypeść".
- DINOv3: TMLR 2026 confirmed; `month=feb` unconfirmed; name spellings differ across records.
- Baddeley & Hitch 1974: volume 8 not confirmed by Crossref.
- Yarbus 1967: publisher city differs between Springer (New York) and Crossref (Springer US).

- **Bajcsy 1988 pages.** Crossref and "Citation records" give 966–1005. The Zotero PDF has 10 pages, its first page
  carries "0018-9219/88/0800-0996" and page number 996, and its last page is numbered 1005: the article spans
  996–1005.
- **STAM glimpse count in `sota-history.md`.** "26 glimpses" there; Table 1's t = 26 counts from 0, so 27 glimpses
  (see the STAM entry). The accuracy and pixel count are unaffected.
- **Pourrahimi & Bashivan title.** The Zotero PDF (posted 2025-06-10) is titled "Emergent brain-like representations
  in a goal-directed neural network model of visual search"; `references.bib` uses "Neural Signatures of Associational
  Cortex Emerge in a Goal-Directed Model of Visual Search", presumably a later bioRxiv version [not checked].
- **Not in the CanViT bibliography**, so citing them would be new: STAM, Larochelle & Hinton 2010, Denil et al.
  2012, Ranzato 2014, Bajcsy 1988, Aloimonos et al. 1988, Ballard 1991. Of these, "Citation records" already lists
  Larochelle & Hinton, Bajcsy, Aloimonos et al. and Ballard; STAM, Denil et al. and Ranzato are new.
- **GAE pages.** `references.bib` gives 16137–16146; Crossref (doi:10.1109/ICCV48922.2021.01583) gives 16117–16126.
  Not resolved here.

- `raugelDisentanglingFactorsConvergence2025`: booktitle is ICLR 2026 (the Fourteenth ICLR) but `year = 2025`
  (the OpenReview posting date); the ICLR title ("... Brains and DINOv3") differs from the arXiv title ("... Brains
  and Computer Vision Models", arXiv:2508.18226). The ICLR paper reports the temporal score as R = 0.96 in the text
  and r = 0.84 in the Fig. 2D caption.
- `simeoniDINOv32026`: the TMLR PDF in Zotero carries the header "Published in Transactions on Machine Learning
  Research (04/2026)"; the entry says `month = feb` (the OpenReview date is 2026-02-22).

- Helmholtz: cite 1867 for the volume; §26 appeared in the third part, 1866.
- Firestone & Scholl: online 2015, volume 39 dated 2016, article e229.
- Peelen, Berlot & de Lange: online 2023, issue 2024 (*Nat. Rev. Psychol.* 3:13–26).
- Intraub's WIREs review "Rethinking visual scene perception" (doi:10.1002/wcs.149) is online 2011, in print 2012,
  3(1):117–127; not read.
- Bruner 1957: pages, editors and content UNVERIFIED (above).
