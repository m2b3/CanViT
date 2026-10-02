[Written by a Claude Code subagent on 2026-10-01 from the seven papers' PDFs and the precursors named in active-vision-timeline.md; the speaker notes it checks are those of #timeline before that day's corrections. Not reviewed by the authors.]

# Timeline slide: what each model introduced, and a check of the speaker notes

[Claude Code, 2026-10-01. Every quote, number and page below was read this session in the paper named. "p." is the
page index of the PDF read (the files listed in `sources/active-vision-timeline.md`; AME from arXiv v3; Larochelle &
Hinton from the author's site; Ranzato and Denil et al. from arXiv v1). Statements marked [inferred] are my reading,
not a paper's claim. Venues of AME, AdaGlimpse and AdaptiveNN were checked on Crossref this session.]

Precursors used as the yardstick, each read this session:

- Larochelle H., Hinton G.E., "Learning to combine foveal glimpses with a third-order Boltzmann machine", NIPS 2010.
  A retina with a sharp centre and coarse periphery, glimpses on a 7×7 grid, accumulated by a Boltzmann machine; a
  linear controller trained by regression to predict the log-probability of the right class after each candidate
  fixation, used greedily (pp. 2, 5). "To our knowledge, this is the first implemented system for combining glimpses
  that jointly trains a recognition component (the RBM) with an attentional component (the fixation controller)"
  (p. 6). No reinforcement learning.
- Denil M., Bazzani L., Larochelle H., de Freitas N., "Learning where to Attend with Deep Architectures for Image
  Tracking", Neural Computation 2012 (arXiv:1109.3737). Foveated gaze on real video; a gaze policy learned online from
  a reward (lower tracking uncertainty) with bandit algorithms or Bayesian optimization over a continuous action space
  (pp. 1–2, 4); "the first successful attempt to combine dynamic state estimation from gazes with online policy
  learning for gaze adaptation" (p. 6).
- Ranzato M., "On Learning Where To Look", arXiv:1405.5488, April 2014 (two months before RAM). "The key idea is to
  first aggressively down-sample the image to cheaply detect candidate locations where to look at higher resolution"
  (p. 2); location treated as a latent variable found by local search, glimpse networks trained one after another
  (pp. 4–5); a confidence threshold lets easy inputs stop early: 78% of jittered-MNIST test digits are classified from
  the low-resolution view alone, a 12× saving in operations (pp. 8–9). Computation "scales with the complexity of the
  input rather than its number of pixels" (p. 1). No reinforcement learning.
- Butko N.J., Movellan J.R., "I-POMDP: An infomax model of eye movement", ICDL 2008. [Primary text not read: the PDF
  link on the lab page returns HTML.] RAM says their framework "is related to ours as they also employ a policy
  gradient formulation" (RAM p. 2); Saccader: they "proposed instead to use policy gradient, learning a convolutional
  logistic policy to maximize long-term information gain" (Saccader p. 2). RAM also cites Paletta et al., "Q-learning
  of sequential attention for visual object recognition from informative local descriptors" (2005) [not read].

## RAM (Recurrent Attention Model)

Mnih V., Heess N., Graves A., Kavukcuoglu K., "Recurrent Models of Visual Attention", NeurIPS 2014 (venue as cited in
Saccader's bibliography), arXiv:1406.6247.

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

**Say.** "Mnih and colleagues built a single network that looks at a small patch, sharp in the middle and blurry
around, remembers what it has seen, and learns by trial and error where to look next, rewarded only for getting the
answer right."

**Contestable.**
- Learning where to look from a reward predates RAM: Butko & Movellan 2008 (policy gradient, per RAM p. 2 and
  Saccader p. 2), Paletta et al. 2005 (Q-learning, cited by RAM), Denil et al. 2012 (bandits, read). RAM's novelty is
  the end-to-end deep recurrent version, not reinforcement learning for gaze as such.
- "Computation independent of image size" is also Ranzato 2014's stated point, on arXiv two months earlier and cited
  by RAM.
- The word "glimpse" and the multi-resolution retina come from Larochelle & Hinton 2010: "We will refer to this
  low-resolution representation as a glimpse [14]" (RAM p. 4).

## DRAM (Deep Recurrent Attention Model)

Ba J., Mnih V., Kavukcuoglu K., "Multiple Object Recognition with Visual Attention", ICLR 2015 (PDF header),
arXiv:1412.7755.

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

**Say.** "Ba and colleagues taught the network to read a house number digit by digit in street photographs, giving it
a blurry look at the whole image to decide where to start."

**Contestable.**
- A coarse first look that decides where to look sharply is Ranzato 2014's key idea; DRAM does not cite Ranzato.
- The multiplicative "what × where" combination is credited by DRAM itself to Larochelle & Hinton 2010 (p. 3).
- "First on real images" would be wrong: Larochelle & Hinton 2010 used face photographs, Denil et al. 2012 real video.
  DRAM's claim is narrower and holds: matching the best convolutional networks on a real-image benchmark.
- "Outperforms the state of the art" (Conclusion, p. 8) rests on 3.9% against 3.96%, with an ensemble of two models
  and Monte Carlo averaging; a single model is worse than the CNNs. "Matches" is the defensible word.
- DRAM is not purely glimpse-based: its first glimpse is aimed from the whole image at low resolution.

## Saccader

Elsayed G.F., Kornblith S., Le Q.V., "Saccader: Improving Accuracy of Hard Attention Models for Vision", NeurIPS 2019
(PDF header), arXiv:1908.07644.

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

**Say.** "Elsayed and colleagues got glimpse-based recognition to work on ImageNet by first teaching the model a
sensible guess about where the evidence is, and only then refining that guess by trial and error. But the part that
chooses where to look sees the whole image."

**Contestable.**
- The CanViT paper (Related work) and the notes both credit "Saccader" with 75%; the 75% needs a separate, larger
  classifier (NASNet) and 331 px inputs. Saccader alone: about 70% (70.31%, as tabulated by AdaGlimpse).
- Saccader does not save computation or work from partial views: its feature network processes the entire image
  before any choice is made. [inferred: on the slide's theme it is an outlier, a model that chooses which evidence
  to use rather than where to point its sensor.]
- Hard attention on real photographs before 2019: Sermanet, Frome & Real, "Attention for Fine-Grained
  Categorization", ICLR 2015 workshop, applied DRAM to Stanford Dogs; Saccader judges it "achieved only a modest
  accuracy improvement over a non-attentive baseline, and did not appear to derive significant benefit from glimpses
  beyond the first" (p. 2).

## GFNet (Glance and Focus Network)

Wang Y., Lv K., Huang R., Song S., Yang L., Huang G., "Glance and Focus: a Dynamic Approach to Reducing Spatial
Redundancy in Image Classification", NeurIPS 2020 (PDF header), arXiv:2010.05300.

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

**Say.** "Wang and colleagues used looking around to save computation: a quick low-resolution look at the whole
image, then sharp close-ups only for the images that need them, stopping as soon as the model is confident."

**Contestable.**
- Every element of glance, focus and confident early stopping is in Ranzato 2014 (low-resolution first look, sharp
  patches, confidence-threshold cascade on jittered MNIST); GFNet does not cite Ranzato. DRAM (2014) also had the
  whole-image coarse look. RAM's Discussion reports preliminary experiments with a learned stop action (p. 8).
- "Active vision as a speed-up" was already RAM's, Ranzato's and DRAM's motivation (DRAM measured fewer FLOPs than a
  CNN, Table 5). GFNet's novelty is making it pay on ImageNet with modern efficient networks, in measured latency.
- An observation useful for the CanViT thesis: in GFNet's ablation (Table 1, p. 9), at 5 steps a random patch policy
  with the glance reaches 74.46% and a fixed centre-then-corners order with the glance 75.12%, against 75.93% for the
  learned policy. Most of the accuracy comes from the observer; the learned policy adds 0.8 to 1.5 points there.

## AME (Attention-Map Entropy)

Pardyl A., Rypeść G., Kurzejamski G., Zieliński B., Trzciński T., "Active Visual Exploration Based on Attention-Map
Entropy", IJCAI 2023 (Crossref), arXiv:2303.06457.

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

**Say.** "Pardyl and colleagues dropped the separate 'where to look' network: the model looks next where its own
internal attention is most uncertain, and fills in a full map of the scene from what it has seen."

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

## AdaGlimpse

Pardyl A., Wronka M., Wołczyk M., Adamczewski K., Trzciński T., Zieliński B., "AdaGlimpse: Active Visual Exploration
with Arbitrary Glimpse Position and Scale", ECCV 2024 (Crossref: Lecture Notes in Computer Science, online
2024-10-26), arXiv:2404.03482.

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

**Say.** "Pardyl and colleagues let the model choose not only where to point but how far to zoom; it learns by itself
to start with a wide, blurry view of everything and then zoom in."

**Contestable.**
- Choosing the scale was proposed before, outside AdaGlimpse's comparison set: RAM's Discussion ("The network can
  also be allowed to control the scale at which the retina samples the image", p. 8), Ranzato 2014 ("we could also
  predict the width and the height of the patch", p. 3), and Denil et al. 2012, whose control pathway models "the
  location, orientation, scale and speed of the attended object" (p. 1). AdaGlimpse's novelty claim is scoped to
  active visual exploration methods with fixed-scale grid glimpses, and holds there.
- A coarse whole-scene first look is what DRAM, GFNet and Saccader impose; AdaGlimpse learns it.

## AdaptiveNN

Wang Y., Yue Y., Yue Y., Wang H., Jiang H., Han Y., Ni Z., Pu Y., Shi M., Lu R., Yang Q., Zhao A., Xia Z., Song S.,
Huang G., "Emulating human-like adaptive vision for efficient and flexible machine visual perception", Nature Machine
Intelligence 2025 (Crossref, online 2025-11-06), arXiv:2509.15333 (PDF read: arXiv v1 with supplement).

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

**Say.** "From the GFNet group, AdaptiveNN scaled the glance-then-fixate idea to many tasks, and its choices of where
to look match where people look about as well as one person matches the crowd, though that human data was recorded
with a mouse, not an eye tracker."

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

## The speaker notes, sentence by sentence

**"Machines that look around have a history too."** Supported (framing only).

**"In deep learning it starts in 2014 with the Recurrent Attention Model: take a glimpse, decide where to look next,
and the computation no longer grows with the size of the image."**
- "starts in 2014": contestable. The CanViT paper says the line "traces back to" RAM, and RAM is the usual starting
  point, but Larochelle & Hinton (NIPS 2010) already took foveal glimpses and learned where to look with a neural
  model, and Denil et al. 2012 call their gaze model a "deep architecture". Safer: "The deep-learning line usually
  starts in 2014 ...", or credit Larochelle & Hinton for the glimpse in one clause.
- "take a glimpse": supported. RAM's glimpse is retina-like, several concentric patches of doubling size resized to
  the same small resolution (p. 5); one experiment uses a single 8×8 patch (p. 6).
- "decide where to look next": supported, but it does not say what was new. The new part is that the decision is
  learned by trial and error from the final answer, in one network trained end to end (pp. 2, 5).
- "the computation no longer grows with the size of the image": supported (Abstract p. 1; p. 8). Ranzato 2014 made
  the same point two months earlier.

**"DRAM took it from toy digits to house numbers in photographs."** Supported (SVHN, "images of digits taken from
pictures of house fronts", p. 7), with two omissions: the inputs are 54×54 grayscale crops centred on the number, and
DRAM's own advance is reading several objects in sequence plus a coarse look at the whole image to aim the first
glimpse (pp. 1–3). The context network does see the whole image downsampled, so DRAM is not purely glimpse-based
(p. 3).

**"Then not much on real images until 2019: Saccader reached 75 percent on ImageNet, thanks to a pretraining step
that made the reinforcement learning work."**
- "not much on real images until 2019": defensible as a hedge and matches the CanViT paper's framing; Sermanet et al.
  2015 (Stanford Dogs) is the counterexample an expert may raise.
- "Saccader reached 75 percent on ImageNet": wrong as stated. 75.03% is Saccader choosing 6 locations plus a separate
  NASNet classifier on patches from 331 px images (p. 9); Saccader alone is about 70% (70.31%, AdaGlimpse Table 2).
  Correction: "Saccader brought it to ImageNet: 70 percent, 75 with a bigger classifier reading its patches."
- "thanks to a pretraining step that made the reinforcement learning work": supported ("Key to Saccader is a
  pretraining step ... provides initial attention locations for policy gradient optimization", p. 1; "necessary to
  achieve this performance", p. 7).
- Not in the notes but likely to be asked: Saccader's attention network sees the entire image (p. 9). Its goal was
  interpretability, not saving computation.

**"GFNet: glance at the whole image, then focus on patches only when needed; active vision as a speed-up."**
Supported (pp. 2, 7). Precision worth adding: it stops as soon as it is confident, and the speed-up is measured on a
phone. The glance-then-focus idea and confident early stopping were in Ranzato 2014; GFNet made it pay on ImageNet.

**"AME segments the whole scene from its glimpses, and picks the next one where its decoder's attention is most
spread out; no reinforcement learning there."** Supported: segmentation of ADE20K (Table 3, p. 6); "highest entropy"
of the decoder's attention rows, unobserved patches only (pp. 2–3); no policy network, no extra loss, no
reinforcement learning (p. 2). Caveats if asked: AME is mostly evaluated on reconstruction; the 27.6% starts from
ADE20K-trained weights; earlier models (Seifi & Tuytelaars 2020, Seifi et al. 2021) also segmented from glimpses.

**"AdaGlimpse chooses both where to look and how much to zoom."** Supported (Abstract, p. 1; action (x, y, scale),
p. 7). Worth adding: it learns by itself to take the whole scene as its first glimpse (p. 14).

**"And AdaGlimpse's successor in spirit, AdaptiveNN, last year: coarse-to-fine fixations across many tasks, compared
with where people look."**
- "AdaGlimpse's successor in spirit": wrong. AdaptiveNN is GFNet's successor: same group and authors, same glance then
  fixed-size fixations then adaptive stopping, PPO; it cites and compares with GFNet and never cites AdaGlimpse; it
  has no zoom choice, which is AdaGlimpse's defining feature. Correction: "And from the GFNet group, AdaptiveNN, last
  year: ...".
- "last year": supported (arXiv 2025-09-18; Nature Machine Intelligence, 2025-11-06).
- "coarse-to-fine fixations across many tasks": supported (Abstract p. 1; p. 7; 17 benchmarks, 9 tasks).
- "compared with where people look": supported with a caveat a neuroscience audience will care about: the human
  "gaze" maps are SALICON's, recorded with a mouse-contingent paradigm on Mechanical Turk, not an eye tracker
  (SALICON, CVPR 2015, p. 1). The comparison also covers difficulty judgments and visual Turing tests (pp. 17, 19).

**"Notice one thing: all but one learn where to look by reinforcement learning."** Supported for these seven, each
read in its method section: RAM REINFORCE (p. 5), DRAM a bound equivalent to REINFORCE (p. 4), Saccader REINFORCE
after pretraining (p. 5), GFNet PPO (p. 6), AdaGlimpse Soft Actor-Critic (p. 3), AdaptiveNN PPO (p. 29); AME is the
exception (p. 2). It is a property of this selection, not of the field: Larochelle & Hinton 2010, Ranzato 2014 and
Glimpse-Attend-and-Explore 2021 choose glimpses without reinforcement learning. Saying "of these seven" keeps it
exact.

**"Most of the effort went into choosing where to look."** Partly supported. The CanViT paper's wording is "prior ACV
work has often focused on action selection" (Introduction). Among the seven, the contribution centres on the choice
of where to look for RAM, Saccader, AME, AdaGlimpse and AdaptiveNN, less so for DRAM (several objects, coarse context)
and GFNet (efficiency wrapper around standard CNNs) [inferred]. "Most of the effort" is a quantity nobody measured;
"most of these papers put their novelty into choosing where to look" is defensible. Two ablations support the talk's
next point if wanted: GFNet's random policy reaches 74.46% against 75.93% learned at 5 steps (Table 1, p. 9), and
AME's random choice 72.4% against 73.4% (Table 5, p. 7).

**"And this year, at NeurIPS: CanViT. It learns no policy at all; it is the observer, and you bring the policy."**
Supported. NeurIPS 2026 acceptance is stated in the CanViT README ("2026-09-24: Accepted at NeurIPS 2026"). The paper
trains on random viewpoints (R-IID and F-IID rollouts) with no learned policy and calls the approach policy-agnostic
(Section "Policy agnosticism"; Introduction: "disentangles 'how to see in an active-vision setting' from 'where to
look'").

## Not checked

- Butko & Movellan 2008 and Paletta et al. 2005 (primary texts not read; described through RAM and Saccader).
- Saccader's supplementary Figure Supp.3 (with and without location pretraining), not in the NeurIPS PDF read.
- Whether Li et al. 2023 is a hard-attention model.
