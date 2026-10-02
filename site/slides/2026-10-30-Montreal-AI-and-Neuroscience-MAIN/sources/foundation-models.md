# Foundation models: sources for the talk

[Compiled 2026-10-01 by a Claude Code research subagent; read by Claude Code, not by the authors.] Each entry names
what was read: the local PDF (abstract plus the passages and captions quoted), the arXiv API record, or the
publisher record through OpenAlex. "Zotero" is the PDF in the user's library. UNVERIFIED marks what was not checked
against a primary source. Quotes are verbatim; numbers are the papers' own.

## 1. What a foundation model is, and the vision lineage

- **Bommasani R. et al. 2021. On the Opportunities and Risks of Foundation Models.** Stanford CRFM report,
  arXiv:2108.07258. https://arxiv.org/abs/2108.07258. Zotero:
  `Bommasani et al. - 2022 - On the Opportunities and Risks of Foundation Models.pdf`.
  Read: §1. Coins the term: "A foundation model is any model that is trained on broad data (generally using
  self-supervision at scale) that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks; current
  examples include BERT, GPT-3, and CLIP." Two consequences: emergence ("the behavior of a system is implicitly
  induced rather than explicitly constructed") and homogenization ("the defects of the foundation model are
  inherited by all the adapted models downstream"). The name stresses that "a foundation model is itself
  incomplete but serves as the common basis from which many task-specific models are built via adaptation."
  Slide figures: Fig. 2 (one model centralizes data from many modalities and is adapted to many tasks); Fig. 1
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
  label ever given. A strong first slide for "what emerges".
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
  are read out with linear probes on frozen features (Table 11, Fig. 7). Slide figures: Fig. 1 (first three PCA
  components of patch features: "Same parts are matched between related images despite changes of pose, style or
  even objects"), Fig. 7 (linear-probe depth and segmentation), Fig. 10 (part matching across images and domains).
- **Siméoni O. et al. DINOv3.** arXiv:2508.10104 (2025); TMLR 2026 (the TMLR PDF header reads "04/2026").
  https://openreview.net/forum?id=2NlGyqNjns. Zotero: `Siméoni et al. - 2026 - DINOv3.pdf`.
  A 7B-parameter self-supervised teacher on a curated 1.7-billion-image set (LVD-1689M), distilled into a family of
  smaller models. Gram anchoring "addresses the known yet unsolved issue of dense feature maps degrading during
  long training schedules"; the result "outperforms the specialized state of the art across a broad range of
  settings, without fine-tuning". The frozen-feature ideal in their words: "achieve excellent performance while
  being kept frozen ... a single forward pass can deliver cutting-edge results across multiple tasks." Slide
  figures: Fig. 1 (ImageNet linear-probe accuracy of supervised, weakly supervised and self-supervised methods over
  the years, plus DINOv3's lead on dense tasks), Fig. 3 (patch similarity maps on a 4096² image), Fig. 5 (see §4).

## 2. World knowledge in vision foundation models

- **El Banani M. et al. 2024. Probing the 3D Awareness of Visual Foundation Models.** CVPR 2024,
  doi:10.1109/CVPR52733.2024.02059, arXiv:2404.08636. https://arxiv.org/abs/2404.08636 (full PDF read). Not in
  Zotero. Probes frozen features for depth, surface normals and correspondence across views. "Recent
  self-supervised models such as DINOv2 learn representations that encode depth and surface normals, with
  StableDiffusion being a close second", while CLIP shows "very poor performance despite its impressive semantic
  generalization capabilities"; "CLIP and MAE features do not encode depth" (Fig. 2 caption). The limit: "the
  models struggle with multiview consistency ... they perform very poorly at large viewpoint variations", and
  their cross-view matching "is semantic in nature". Single-view tasks correlate with each other above 0.82;
  multiview tasks much less. Slide figures: Fig. 1 (ratings per model, single-image 3D versus multiview
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
  be hardwired". Slide figures: Fig. 1 (accuracy by model family on three benchmarks), Fig. 2 (per-property
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
  model performance (Fig. 3); §2.4 cites brain alignment. Slide figure: Fig. 1 (images and text as projections of
  one reality Z).
- **Gröger F., Wen S., Brbić M. 2026. Revisiting the Platonic Representation Hypothesis: An Aristotelian View.**
  ICML 2026 (arXiv comment: camera-ready), arXiv:2602.14486. https://arxiv.org/abs/2602.14486. In Zotero without a
  PDF. Similarity metrics are "confounded by network scale"; after a permutation-null calibration, "the apparent
  convergence reported by global spectral measures largely disappears, while local neighborhood similarity, but
  not local distances, retains significant agreement across different modalities." Tension with Huh et al.

## 3. Performance and brain alignment

- **Yamins D.L.K. et al. 2014. Performance-optimized hierarchical models predict neural responses in higher visual
  cortex.** PNAS 111(23):8619–8624, doi:10.1073/pnas.1403112111. https://www.pnas.org/doi/10.1073/pnas.1403112111.
  Zotero: `Yamins et al. - 2014 - Performance-optimized hierarchical models predict .pdf`.
  "Within a class of biologically plausible hierarchical neural network models, there is a strong correlation
  between a model's categorization performance and its ability to predict individual IT neural unit response
  data"; the top layer of the performance-optimized model predicts IT, its middle layers V4. Slide figure: Fig. 1
  (categorization performance against IT explained variance; r = 0.55, 0.78 and 0.80 for random, performance-
  optimized and IT-optimized model selection in panel A; r = 0.87 ± 0.15 in panel B). The scope matters: one
  model class, 2014-era performance levels.
- **Schrimpf M. et al. 2018. Brain-Score: Which Artificial Neural Network for Object Recognition is most
  Brain-Like?** bioRxiv 407007, doi:10.1101/407007. https://www.biorxiv.org/content/10.1101/407007v1. Zotero:
  `Schrimpf et al. - 2018 - Brain-Score Which Artificial Neural Network for O.pdf`.
  "Gains in ANN ImageNet performance led to gains on Brain-Score. However, correlation weakened at ≥ 70% top-1
  ImageNet performance"; smaller ANNs were more brain-like "than many of the best-performing ImageNet models".
  Slide figure: Fig. 1 (ImageNet top-1 against Brain-Score; the relation flattens for the newest models).
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
  breaks the trade-off. Slide figures: Fig. 1 (104 Brain-Score models, ImageNet accuracy against IT predictivity,
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
  improved models of the brain's visual ventral stream". Slide figure: Fig. 1b (fitted scaling curves: behaviour
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
  brains may be too flexible". Slide figure: Fig. 5C (top-1 accuracy against brain predictivity: a flat cloud).
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
  most human-centric images reach the highest brain-similarity". The size effect is small in absolute terms:
  voxel-averaged encoding R of .096 (Small), .101 (Base), .105 (Large), .107 (Giant), significant at p < 1e−3.
  Human-centric images beat satellite and cell images even in V1. The findings "generalize across seven additional
  models".
  (3) A training chronology: the encoding score reaches half its final value at about 2% of training (about 800
  million images); early sensory areas and early MEG windows are matched first, prefrontal areas and late windows
  last.
  (4) The regions acquired last are those with the greatest developmental expansion (r = 0.88), thickness (0.77),
  slowest intrinsic timescales (0.71) and least myelin (−0.85).
  Their own limits: a single model family; passive viewing data; their argument is about which factors drive
  convergence, and it never relates brain similarity to benchmark accuracy. Slide figures: Fig. 3 (the three
  scores over training), Fig. 4 (half time per region against distance from V1), Fig. 7 (half time against
  cortical properties).
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

## 4. Benchmarks versus understanding

- **Geirhos R. et al. 2019. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy
  and robustness.** ICLR 2019 (oral), arXiv:1811.12231. https://arxiv.org/abs/1811.12231. On cue-conflict images,
  "ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes, which is in stark
  contrast to human behavioural evidence"; training on Stylized-ImageNet gives a shape bias and robustness. Slide
  figure: Fig. 1 (a cat shape with elephant-skin texture, classified as an elephant).
- **Geirhos R. et al. 2020. Shortcut learning in deep neural networks.** Nature Machine Intelligence 2:665–673,
  doi:10.1038/s42256-020-00257-z, arXiv:2004.07780. https://arxiv.org/abs/2004.07780. "Shortcuts are decision rules
  that perform well on standard benchmarks but fail to transfer to more challenging testing conditions." Fig. 1
  (examples of shortcuts).
- **Geirhos R. et al. 2021. Partial success in closing the gap between human and machine vision.** NeurIPS 2021
  (oral), arXiv:2106.07411. https://arxiv.org/abs/2106.07411. 85,120 trials from 90 observers on 17
  out-of-distribution datasets: "The longstanding distortion robustness gap between humans and CNNs is closing";
  "There is still a substantial image-level consistency gap, meaning that humans make different errors than
  models"; "human-to-model consistency improves when training dataset size is increased by one to three orders of
  magnitude." Data, more than architecture or objective, moves models toward humans. Slide figures: Fig. 1 (core
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
  overfitting to the test set. Slide figure: Fig. 1.
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
  global classification score can hide the loss of spatial scene structure, in the model family CanViT distills. Slide figure: Fig. 5 (two curves diverging), optionally with Fig. 6.
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

## 5. Active vision and foundation models

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
  into a new recurrent architecture. Slide figure: Fig. 1.
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
  reduction without sacrificing accuracy" across 17 benchmarks. Already in the talk's citations.
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

## Tensions to keep visible

1. **Accuracy predicted the brain, then stopped.** Yamins 2014 and Brain-Score 2018 found the link; Brain-Score
   2018 already saw it weaken above 70% top-1; Fel 2022, Linsley 2023 and Linsley, Feng, Serre 2025 report it
   reversing for ImageNet models; Gokce and Schrimpf 2025 find neural alignment saturating while behavioural
   alignment keeps scaling; Conwell 2024 finds no relation among trained models. Within one controlled family,
   Raugel et al. find that larger, longer-trained DINOv3 models on human-centric images are more brain-like, with
   small absolute differences. These studies use different data (macaque electrophysiology, human fMRI and MEG),
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

## Points worth making to this audience

1. **A foundation model is a general feature bank, read out by many tasks.** Trained once on broad data without
   labels, then frozen and read out with linear probes (Bommasani 2021; DINOv2; DINOv3). Conwell and Bonner 2025
   describe the ventral stream in the same terms, a feature vocabulary rather than the locus of recognition, and
   self-supervised models match supervised ones as models of the ventral stream (Zhuang 2021; Konkle and Alvarez
   2022). This is why the teacher for an active model should be a self-supervised foundation model.
2. **What emerges is measurable world knowledge, with known limits.** Object parts, depth, surface orientation and
   cross-image correspondence emerge without labels (DINO Fig. 1; DINOv2 Figs. 1, 7, 10; El Banani 2024), and
   intuitive-physics surprise emerges from video prediction (Garrido 2025). The limits are as informative:
   viewpoint consistency, perspective taking and complex physics remain weak (El Banani 2024; 3D-PC; IntPhys 2).
3. **Higher scores are not automatically better brain models, as this audience's own literature shows.** The arc
   from Yamins 2014 through Brain-Score 2018 to Linsley 2023, Gokce and Schrimpf 2025 and Conwell 2024 is the
   frame for any "benchmark state of the art" claim. What the scores do buy is breadth: features that transfer
   across tasks, shifts and datasets (Kornblith 2019; Recht 2019; Geirhos 2021). Raugel et al. show that, inside
   one family, scale and ecological data raise brain similarity and reproduce a developmental ordering of cortex.
   CanViT's claim should be scoped accordingly: it brings foundation-model features into an active, sequential
   observer; whether that makes a better brain model is an open, testable question.
4. **Dense, frozen-feature evaluation tests scene understanding that classification can miss, and active vision
   lacked a foundation to meet that bar.** Classification can rise while spatial structure decays (DINOv3 Fig. 5; Chen 2025), so
   evaluating segmentation from frozen features tests more of the scene (an inference from these dissociations,
   see §4). Active models were trained from scratch per task, often with reinforcement learning (Killick 2023;
   the CanViT paper's account of Saccader). The field is now moving toward building active and foveated observers
   on top of foundation models (FOVI adapting DINOv3; Thorat 2025 predicting pretrained glimpse embeddings;
   Prisadnikov 2026), and Thorat 2025 finds that next-glimpse prediction yields brain-aligned scene
   representations, a direct link between action, self-supervision and the brain (Rothkopf 2023).

## Figures that would work on a slide

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

## Bibliography notes for the camera-ready

- `raugelDisentanglingFactorsConvergence2025`: booktitle is ICLR 2026 (the Fourteenth ICLR) but `year = 2025`
  (the OpenReview posting date); the ICLR title ("... Brains and DINOv3") differs from the arXiv title ("... Brains
  and Computer Vision Models", arXiv:2508.18226). The ICLR paper reports the temporal score as R = 0.96 in the text
  and r = 0.84 in the Fig. 2D caption.
- `simeoniDINOv32026`: the TMLR PDF in Zotero carries the header "Published in Transactions on Machine Learning
  Research (04/2026)"; the entry says `month = feb` (the OpenReview date is 2026-02-22).
