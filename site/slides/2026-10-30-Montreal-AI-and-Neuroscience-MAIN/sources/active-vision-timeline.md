# Deep active vision timeline: sources for the MAIN 2026 slide

[Assembled by Claude Code on 2026-10-01 from the papers' own text; not yet reviewed by the authors.] Evidence for a
timeline slide of deep active computer vision, from the Recurrent Attention Model (RAM, 2014) to AdaptiveNN (2025),
with the precursors and pre-deep-learning foundations the speaker may mention in passing. Numbers already verified in
`sota-history.md` (same directory) are reused and say so; every other number, quote and figure location below was
read in the paper itself this session.

## Recommended slide entries

Year is the first public release (arXiv), the convention of `sota-history.json`. Only DRAM's venue year differs
(arXiv 2014-12-24, ICLR 2015); write 2015 if the slide uses venue years throughout.

| Year | Name | Title (at most six words) | Spoken "why it mattered" |
|---|---|---|---|
| 2014 | RAM | Reinforcement learning picks each glimpse | Mnih and colleagues turned recognition into a sequence of glimpses chosen by a recurrent network trained with reinforcement learning, so computation no longer grows with image size, and it beat a convolutional network of similar size on cluttered digits. |
| 2014 | DRAM | Glimpses read multi-digit house numbers | Ba and colleagues took the recipe from toy digits to photographs, matching the best convolutional network on multi-digit house numbers with fewer parameters and less computation, and gave the model a coarse look at the whole image to aim its first glimpse. |
| 2019 | Saccader | Pretrained policy reaches 75% on ImageNet | Elsayed and colleagues brought hard attention to ImageNet: a pretraining step gave the reinforcement-learned policy good starting locations, and the model reached 75% top-1 while its classifier saw under a third of each image. |
| 2020 | GFNet | Glance, then focus, stop when sure | Wang and colleagues made active vision a speed-up for standard networks on ImageNet: a low-resolution glance, then high-resolution patches only when needed, stopping once confident, for 2 to 3 times fewer operations with ResNets at equal accuracy. |
| 2022 | STAM | Transformer that never sees whole image | Rangrej, Srinidhi and Clark (McGill and Toronto) dropped the look at the whole image that earlier ImageNet models relied on: a transformer picks each glimpse from past glimpses alone, learns from a teacher that sees the full image, and reaches 80.8% on ImageNet. |
| 2023 | AME | Looks where its attention is uncertain | Pardyl and colleagues chose glimpses without reinforcement learning, where a masked-autoencoder transformer's attention is most uncertain, and used the same model to predict the whole scene densely, including ADE20K segmentation. |
| 2024 | AdaGlimpse | Chooses where to look and zoom | Pardyl and colleagues let the policy choose both position and zoom from a continuous range, so the model can take in the whole scene coarsely before zooming in on detail, as a camera with optical zoom can. |
| 2025 | AdaptiveNN | Coarse-to-fine fixations across many tasks | Wang and colleagues made the case at scale: coarse-to-fine fixations trained end to end, tested on 17 benchmarks across 9 tasks with up to 28 times lower inference cost at equal accuracy, and compared with human fixations. |

Notes on the selection:

- **Seven of the eight are the CanViT paper's own related-work narrative** (Related work, "Deep active vision" and
  "Dense prediction in active vision" paragraphs): RAM, DRAM, Saccader, GFNet, AdaptiveNN, AME, AdaGlimpse.
- **STAM is the one the CanViT paper does not cite** (`references.bib` has no Rangrej entry). It earns its place on
  three counts, each verified below: the highest ImageNet-1k accuracy of a sequential glimpse model in
  `sota-history.json` before AdaptiveNN (80.78%), no view of the whole image at any resolution, and a full-image
  teacher distilled into a glimpse-based student, an idea related to CanViT's passive-to-active distillation (STAM
  matches class distributions, CanViT dense features; AdaGlimpse uses a full-image teacher "as in STAM"). Two of its
  three authors are at McGill. Dropping STAM leaves seven entries that stay inside the paper's bibliography.
- **If two more must go**, DRAM and AME are the least needed for the CanViT story: DRAM's coarse context glance
  reappears in GFNet and AdaptiveNN, and AdaGlimpse also does dense prediction.
- **Precursors to mention in passing** (details below): Larochelle & Hinton 2010 (foveal glimpses combined by a
  Boltzmann machine; RAM takes the word "glimpse" from it), Ranzato 2014 (where to look, learned from a low-resolution
  view; on arXiv two months before RAM), Denil et al. 2012 (learned gaze for tracking). Foundations: Bajcsy 1988
  ("We do not just see, we look"), Aloimonos, Weiss & Bandyopadhyay 1988 (ill-posed problems become well-posed for an
  active observer), Ballard 1991 (gaze control makes vision-based behavior simpler).
- **How the CanViT paper places the recent entries**, for the transition to CanViT: GFNet and AdaptiveNN "remained
  structurally limited to classification tasks and fixed zoom levels"; AME and AdaGlimpse produce dense outputs with
  a decoder over "a full grid of learnable mask tokens", which "becomes intractable at high scene resolutions", and
  reach 27.6% and 25.7% ADE20K mIoU (Related work).

## How this was assembled

1. Read the CanViT paper's Introduction and Related work
   (`paper/latex/CanViT_Toward_AVFMs.tex`) and the `references.bib` entries it
   cites for active vision; read `sota-history.md`, `citations.md` and `foundation-models.md` in this directory.
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

## The eight entries

### 2014: Recurrent Attention Model (RAM)

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
  `assets/figures/mnih-2014-ram.png`. Alternative: Figure 3, p. 7, glimpse paths on cluttered digits.
- **Note.** The paper also trains RAM to play a 24×24-pixel game of "Catch" from the reward alone (Section 4.2).

### 2014: Deep Recurrent Attention Model (DRAM)

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

### 2019: Saccader

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
  stabilize learning" (Related work).
- **Headline.** ImageNet-1k top-1 75.03 ± 0.08% and top-5 91.19%, with 6 glimpses, "processing only 29.47 ± 0.26% of
  the image" with a separate NASNet classifier on 331 px images (Section 4.3 text, p. 9; Figure 6, p. 8). Reused
  from `sota-history.md`, re-read here. The base model at 224 px scores 70.31% (`sota-history.md`).
- **Figure.** Figure 1, p. 2: glimpses chosen by Saccader against those of DRAM on ImageNet photographs. Alternative:
  Figure 4, p. 7 (Saccader, DRAM, ordered-logits and random policies side by side).
- **Caveat an audience may raise.** "the attention network has access to the entire image" (Conclusion, p. 9); only
  the classification path is restricted to the glimpses.

### 2020: Glance and Focus Network (GFNet)

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

### 2022: Sequential Transformers Attention Model (STAM)

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

### 2023: Attention-Map Entropy (AME)

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

### 2024: AdaGlimpse

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

### 2025: AdaptiveNN

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

## Precursors before RAM

### Larochelle & Hinton 2010

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

### Denil, Bazzani, Larochelle & de Freitas 2012

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

### Ranzato 2014

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

### Earlier neural precursors, verified only through later papers

- **Schmidhuber J., Huber R. 1991.** "Learning to generate artificial fovea trajectories for target detection."
  International Journal of Neural Systems 2(1–2):125–134 (Crossref, doi:10.1142/S012906579100011X). Saccader's related
  work: "Early work using backpropagation to train the controller computed the gradient with respect to its weights
  by backpropagating through a separate neural network model of the environmental dynamics" (Saccader, p. 2).
  [Primary text not read.]
- **Butko N.J., Movellan J.R. 2009.** "Optimal scanning for faster object detection." CVPR 2009, pp. 2751–2758
  (Crossref). Saccader: Butko and Movellan "proposed instead to use policy gradient, learning a convolutional
  logistic policy to maximize long-term information gain" (Saccader, p. 2, citing their 2008 paper); RAM calls their
  framework "related to ours as they also employ a policy gradient formulation" (RAM, p. 2). [Primary text not read.]

## Pre-deep-learning foundations

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

## Other works an informed audience may name

None of these is recommended for the slide; each line was checked as stated.

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

**Single-step selection from a low-resolution view** (not sequential, so outside the slide's definition)
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

**Recent, brain-facing or foundation-model-facing** (details in `foundation-models.md`)
- **Thorat S., Doerig A., Kroner A., Amme C., Kietzmann T.C.** "Predicting upcoming visual features during eye
  movements yields scene representations aligned with human visual cortex." arXiv:2511.12715 (2025-11-16).
  Recurrent networks predict the next glimpse's embedding along human-like scanpaths and align with fMRI in mid- and
  high-level visual cortex (abstract).
- **Pourrahimi M., Bashivan P.** bioRxiv doi:10.1101/2025.06.06.658387 (2025). A network that searches natural
  scenes with saccades and whose units show fronto-parietal response properties; McGill and Mila (PDF header). Cited
  by the CanViT paper.
- **Kolner et al. 2025** (GAP, ICLR 2025), **Blauch et al. 2026** (FOVI, ICML 2026) and **Prisadnikov et al. 2026**
  (self-supervised pretraining for a foveal iterative transformer): see `foundation-models.md` and
  `sota-history.md`.

**Montreal connections, for the local audience.** Rangrej & Clark 2021 and two of STAM's three authors are at McGill
(PDF headers); Pourrahimi & Bashivan at McGill and Mila.

## Discrepancies and bibliography notes

- **Bajcsy 1988 pages.** Crossref and `citations.md` give 966–1005. The Zotero PDF has 10 pages, its first page
  carries "0018-9219/88/0800-0996" and page number 996, and its last page is numbered 1005: the article spans
  996–1005.
- **STAM glimpse count in `sota-history.md`.** "26 glimpses" there; Table 1's t = 26 counts from 0, so 27 glimpses
  (see the STAM entry). The accuracy and pixel count are unaffected.
- **Pourrahimi & Bashivan title.** The Zotero PDF (posted 2025-06-10) is titled "Emergent brain-like representations
  in a goal-directed neural network model of visual search"; `references.bib` uses "Neural Signatures of Associational
  Cortex Emerge in a Goal-Directed Model of Visual Search", presumably a later bioRxiv version [not checked].
- **Not in the CanViT bibliography**, so a slide citation would be new: STAM, Larochelle & Hinton 2010, Denil et al.
  2012, Ranzato 2014, Bajcsy 1988, Aloimonos et al. 1988, Ballard 1991. Of these, `citations.md` already lists
  Larochelle & Hinton, Bajcsy, Aloimonos et al. and Ballard; STAM, Denil et al. and Ranzato are new.
- **GAE pages.** `references.bib` gives 16137–16146; Crossref (doi:10.1109/ICCV48922.2021.01583) gives 16117–16126.
  Not resolved here.
- **AdaGlimpse year** (2025 in `references.bib`, ECCV 2024 online 2024-10-26) and **AME's "Rypesc"** are already
  listed in `citations.md`.
