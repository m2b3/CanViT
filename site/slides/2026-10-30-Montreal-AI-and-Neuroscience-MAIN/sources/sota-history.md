# Published accuracy over time: active and passive vision

[Assembled by Claude Code on 2026-10-01 from the papers' own text; not yet reviewed by the authors.] Data for the
history chart (x = year, y = accuracy) on ImageNet-1k top-1 and ADE20K mIoU. The data file is
`sota-history.json`; its `_about` field defines every field and series. This note says how the points were
verified, lists them with their sources, and records the leads that could not be confirmed and the caveats a
careful audience may raise. Every number below is in the JSON; where the two disagree, the JSON is the reference.

## What the data supports

The CanViT paper's wording is that active vision models "have struggled to match" passive ones; the numbers
below scope that claim.

- **ImageNet-1k.** At each milestone below, the best sequential active model trails the passive frontier of its
  date:
  - Saccader (2019-08-20, 75.03) trails the extra-data frontier by 11.37 points (FixRes, 86.4, 2019-06-14)
    and the ImageNet-1k-only frontier by 9.27 (GPipe and EfficientNet-B7, 84.3).
  - STAM (2022-04-01, 80.78) trails model soups (90.94, 2022-03-10) by 10.16 and MAE ViT-H (87.8) by 7.02.
  - AdaptiveNN (2025-09-18, 82.2) trails Lion/BASIC-L (91.1, 2023-02-13) by 8.9, Perceptual MAE (88.1,
    ImageNet-1k only, 2022-12-30) by 5.9, and the frozen DINOv3 7B linear probe (88.4, 2025-08-13) by 6.2.
- **ADE20K.** Only two sequential active models report ADE20K mIoU:
  - AME (2023-03-11, 27.6 with a SETR-initialised encoder) is 35.3 points below InternImage-H (62.9,
    multi-scale, 2022-11-10).
  - AdaGlimpse (2024-04-04, 25.7) is 37.3 below ONE-PEACE (63.0).
  - A plain linear probe on a frozen DINOv2 ViT-g/14 (49.0, 2023-04-14) beats both by more than 20 points.
- **Select-once models.** LookWhere (NeurIPS 2025) reaches 83.0 on ImageNet-1k and 44.6 mIoU on ADE20K. It
  selects high-resolution patches once, from a low-resolution view of the whole image. Neither number is in the
  CanViT paper's comparison, which does not cite LookWhere. Under the series definitions here it is not a
  sequential glimpse model, but an audience may ask about it (see the caveats).
- **Plotting.** The ImageNet active points are ResNet-50- to ViT-B-sized models on 224–331 px images. The passive frontier
  after 2019 is set by models of 0.5–2.4B parameters tested at 475–800 px and pretrained on hundreds of millions
  to billions of extra images. A fair plot labels each series by what it is (see the caveats).

## How the data was assembled

1. **Starting list.** I read the CanViT paper (`CanViT_Toward_AVFMs.tex`): introduction, related work,
   experiments and the comparison table. I also read its `references.bib` entries for the active models it cites.
2. **Candidate sources.**
   - The paper's citations.
   - The user's Zotero library (PDFs).
   - The Papers with Code API at paperswithcode.co (ImageNet-1k: `dataset_id=72&task_id=1`; ADE20K:
     `dataset_id=15&task_id=3`), paged through by the subagents.
   - Web searches for active, foveated and hard-attention models, and for papers citing Saccader, GFNet,
     AdaGlimpse and AdaptiveNN.

   A leaderboard value counted only as a lead. The Papers with Code ADE20K list has 83 entries and lacks
   ONE-PEACE and ViT-P. Its ImageNet-1k list includes implausible entries (below).
3. **Verification.** Each value was read in the paper's text, extracted from its PDF with `pdftotext -layout`.
   The check covered the table row, its caption, and the setup text that states the data, backbone and
   protocol. Error rates were converted to accuracy (100 − error).
4. **Dates.** A date is the first public appearance of the number, taken from the arXiv submission history (or
   the venue, OpenReview or journal date when there is no earlier arXiv version). When
   a number changed between arXiv versions, the date is that of the first version that carries it (e.g., MPL
   90.2, CoAtNet 90.88, VGG 76.3).
5. **Venues** come from the PDF header, the arXiv comment field, or Semantic Scholar's record, all read on
   2026-10-01.
6. **Figure-only values.** GFNet's best accuracies appear only in figures. They were read from the PDF's vector
   drawing (pdfplumber) by interpolating between the axis ticks, with the legend colours checked. These two
   values are approximate (± 0.1).
7. **Who did what.** Three research subagents collected the evidence for passive ImageNet-1k, passive ADE20K and
   further active models. Each saved the verbatim table lines for each point. I wrote the core active points
   (Saccader, GFNet, AME, AdaGlimpse, AdaptiveNN) and the DINO-family points, and re-read the verbatim evidence
   for every point in the JSON. The evidence files stayed in the session scratchpad and are not in the
   repository; each point's `where` field names the table to reopen.
8. **What counts as active.** A model is *active* if it chooses where to look, one glimpse after another, each
   choice depending on what it has already seen. A model counts as *select once* if a policy picks
   high-resolution patches in a single step from a low-resolution view. *Borderline* covers foveated models whose
   fixations are random, or whose early layers see the full-resolution image.

## Verified points

### ImageNet-1k top-1 (validation set, %)

#### Active, sequential glimpses

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2019-08-20 | DRAM, reimplemented by Elsayed et al. (8 glimpses of 77 px) | 67.5 | Table 1 (#Locs 8), which credits Elsayed et al. 2019 | [Papadopoulos et al. 2021](https://arxiv.org/abs/2102.10212) |
| 2019-08-20 | Saccader (Saccader-NASNet, 331 px images) | 75.03 | Section 4.3 text (Figure 6) | [Elsayed et al. 2019](https://arxiv.org/abs/1908.07644) |
| 2020-10-11 | GFNet (ResNet-50, 96 px patches, T=5) | 75.93 | Table 1, last row, t=5 | [Wang et al. 2020](https://arxiv.org/abs/2010.05300) |
| 2020-10-11 | GFNet (EfficientNet backbones, T=4), read from figure | 79.8 | Figure 4(c), highest point of the GFNet curve, read from the PDF's vector data (± 0.1) | [Wang et al. 2020](https://arxiv.org/abs/2010.05300) |
| 2022-01-09 | GF-EfficientNet-B2 (TPAMI version, 128 px patches, T=4) | 77.95 | Table 2(c), contrastive reward, t=4 | [Huang et al. 2022](https://arxiv.org/abs/2201.03014) |
| 2022-01-09 | MS-GFNet (EfficientNet-B3, T=4), read from figure | 80.0 | Figure 10(b), highest point of the 'MS-GFNet (one model)' curve, read from the PDF's vector data (± 0.05) | [Huang et al. 2022](https://arxiv.org/abs/2201.03014) |
| 2022-04-01 | STAM (DeiT-B-distilled, 26 glimpses of 32 px) | 80.78 | Table 1 | [Rangrej et al. 2022](https://arxiv.org/abs/2204.00656) |
| 2023-09-23 | Liu et al. (learned saccades, 4x4 fovea action space) | 73.7 | Table 1 | [Liu et al. 2024](https://openreview.net/forum?id=lOwkOIUJtx) |
| 2024-03-29 | FALcon (evaluated by Mukherjee, Ibrayev & Roy) | 72.97 | Table 1, 'Clean' row | [Mukherjee et al. 2024](https://arxiv.org/abs/2404.00185) |
| 2024-04-04 | AdaGlimpse (ViT-B, 14 glimpses of 32 px) | 77.54 | Table 2 | [Pardyl et al. 2024](https://arxiv.org/abs/2404.03482) |
| 2025-08-22 | Prisadnikov et al. (ViT-B size, multi-zoom patches, GRPO policy) | 65.0 | Table 2 ('With Policy - Step 8', printed as 0.65) | [Prisadnikov et al. 2025](https://arxiv.org/abs/2508.16317) |
| 2025-09-18 | AdaptiveNN-DeiT-S | 82.2 | Supplementary Data Tab. 2 and 3 | [Wang et al. 2025](https://arxiv.org/abs/2509.15333) |
| 2026-04-22 | Prisadnikov et al. (ViT-S, self-supervised, frozen + linear probe, learned policy) | 75.0 | Tables 1 and 2 | [Prisadnikov et al. 2026](https://arxiv.org/abs/2604.20392) |

#### Active, one selection step from a low-resolution view

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2020-03-01 | PatchDrop (ResNet-50 classifier, ~7.9 of 16 patches) | 76.0 | Table 2 (Ft-2 column) | [Uzkent & Ermon 2020](https://arxiv.org/abs/2003.00425) |
| 2021-02-20 | TNet (BagNet-77, 5 attended locations) | 74.62 | Table 1 | [Papadopoulos et al. 2021](https://arxiv.org/abs/2102.10212) |
| 2025-05-23 | LookWhere ViT-B (128 of 256 patches) | 83.0 | Table 1 (present in arXiv v1) | [Fuller et al. 2025](https://arxiv.org/abs/2505.18051) |
| 2026-09-04 | LookThere (EVA init, 92% of patches dropped) | 83.0 | Figure 5(a) table | [Rammohan et al. 2026](https://arxiv.org/abs/2609.04698) |

#### Borderline (foveated, but not glimpse-limited sensing with a learned policy)

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2022-05-31 | FoveaTer (DeiT-S, Type-B, up to 3 fixations, dynamic stop) | 78.31 | Table 2 of arXiv v2 | [Jonnalagadda et al. 2021](https://arxiv.org/abs/2105.14173) |
| 2026-02-03 | FOVI-ViT-H+ (DINOv3, 3 random fixations) | 85.3 | Table 1 (printed as 0.853) | [Blauch et al. 2026](https://arxiv.org/abs/2602.03766) |

#### Passive, extra training data

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2012 | AlexNet, 7-CNN ensemble (2 pretrained on ImageNet Fall 2011) | 63.3 | Table 2 (top-1 val error 36.7%) | [Krizhevsky et al. 2012](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 2018-05-02 | ResNeXt-101 32x48d, IG-940M pretraining | 85.4 | Table 6 | [Mahajan et al. 2018](https://arxiv.org/abs/1805.00932) |
| 2019-06-14 | FixRes ResNeXt-101 32x48d (IG-940M) | 86.4 | Table 2 | [Touvron et al. 2019](https://arxiv.org/abs/1906.06423) |
| 2019-11-11 | Noisy Student (EfficientNet-L2), arXiv v1 | 87.4 | Table 2 of arXiv v1 | [Xie et al. 2019](https://arxiv.org/abs/1911.04252) |
| 2019-12-24 | BiT-L (ResNet152x4) | 87.76 | Table 2 of arXiv v1 | [Kolesnikov et al. 2019](https://arxiv.org/abs/1912.11370) |
| 2020-01-07 | Noisy Student (EfficientNet-L2), arXiv v2 | 88.4 | Table 2 of arXiv v2 | [Xie et al. 2019](https://arxiv.org/abs/1911.04252) |
| 2020-03-18 | FixEfficientNet-L2 | 88.5 | Table 1 | [Touvron et al. 2020](https://arxiv.org/abs/2003.08237) |
| 2020-10-22 | ViT-H/14 (JFT-300M) | 88.55 | Table 2 | [Dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929) |
| 2020-12-04 | EfficientNet-L2 (Noisy Student) + SAM | 88.61 | Table 3 of arXiv v2 (top-1 error 11.39%) | [Foret et al. 2020](https://arxiv.org/abs/2010.01412) |
| 2021-01-05 | Meta Pseudo Labels (EfficientNet-L2) | 90.2 | Table 4 of arXiv v3 | [Pham et al. 2020](https://arxiv.org/abs/2003.10580) |
| 2021-02-11 | NFNet-F4+ (JFT-300M) | 89.2 | Table 5 | [Brock et al. 2021](https://arxiv.org/abs/2102.06171) |
| 2021-06-08 | ViT-G/14 (JFT-3B) | 90.45 | Table 1 of arXiv v1 | [Zhai et al. 2021](https://arxiv.org/abs/2106.04560) |
| 2021-09-15 | CoAtNet-7 (JFT-3B) | 90.88 | Table 5 of arXiv v2 | [Dai et al. 2021](https://arxiv.org/abs/2106.04803) |
| 2022-03-10 | Model soups (greedy soup of ViT-G/14) | 90.94 | Table 4 | [Wortsman et al. 2022](https://arxiv.org/abs/2203.05482) |
| 2022-05-04 | CoCa (fine-tuned) | 91.0 | Table 2 | [Yu et al. 2022](https://arxiv.org/abs/2205.01917) |
| 2023-02-13 | BASIC-L image encoder (CoAtNet-7) trained with Lion, fine-tuned | 91.1 | Table 1 and Section 4.2 text | [Chen et al. 2023](https://arxiv.org/abs/2302.06675) |
| 2023-11-07 | OmniVec (fine-tuned) | 92.4 | Table 13 (also Table 14) | [Srivastava & Sharma 2023](https://arxiv.org/abs/2311.05709) |
| 2024 | OmniVec2 (fine-tuned) | 93.6 | Main text and supplementary Table 4 | [Srivastava & Sharma 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Srivastava_OmniVec2_-_A_Novel_Transformer_based_Network_for_Large_Scale_CVPR_2024_paper.html) |

#### Passive, ImageNet-1k only

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2012 | AlexNet, 5-CNN ensemble | 61.9 | Table 2 (top-1 val error 38.1%) | [Krizhevsky et al. 2012](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 2013-11-28 | ZFNet, 6-convnet ensemble | 64.0 | Table 2 (top-1 val error 36.0%), arXiv v3 | [Zeiler & Fergus 2013](https://arxiv.org/abs/1311.2901) |
| 2013-12-30 | OverFeat, 7 accurate models | 66.04 | Table 3 of arXiv v2 (Table 2 in v4); top-1 val error 33.96% | [Sermanet et al. 2013](https://arxiv.org/abs/1312.6229) |
| 2014-12-19 | VGG (2-net ensemble, D+E) | 76.3 | Table 7 (top-1 val error 23.7%), first in arXiv v4 | [Simonyan & Zisserman 2014](https://arxiv.org/abs/1409.1556) |
| 2015-02-06 | PReLU-net (model C, single model) | 78.41 | Table 6 (top-1 val error 21.59%) | [He et al. 2015](https://arxiv.org/abs/1502.01852) |
| 2015-02-11 | BN-Inception, 6-model ensemble | 79.9 | Figure 4 table (top-1 val error 20.1%) | [Ioffe & Szegedy 2015](https://arxiv.org/abs/1502.03167) |
| 2015-12-02 | Inception-v3, 4-model ensemble | 82.8 | Table 5 (top-1 error 17.2%) | [Szegedy et al. 2015](https://arxiv.org/abs/1512.00567) |
| 2015-12-10 | ResNet-152 (single model) | 80.62 | Table 4 (top-1 val error 19.38%) | [He et al. 2015](https://arxiv.org/abs/1512.03385) |
| 2016-02-23 | Inception-v4 + 3x Inception-ResNet-v2 ensemble | 83.5 | Table 5 (top-1 val error 16.5%) | [Szegedy et al. 2016](https://arxiv.org/abs/1602.07261) |
| 2017-09-05 | SENet (named SENet-154 in later versions) | 82.72 | Table 3 of arXiv v1 (top-1 val error 17.28%) | [Hu et al. 2017](https://arxiv.org/abs/1709.01507) |
| 2017-12-02 | PNASNet-5 (N=4, F=216) | 82.9 | Table 3 of arXiv v1 (Table 5 in v3) | [Liu et al. 2017](https://arxiv.org/abs/1712.00559) |
| 2018-11-16 | GPipe AmoebaNet-B (6, 512) | 84.3 | Table 2 of arXiv v1 | [Huang et al. 2018](https://arxiv.org/abs/1811.06965) |
| 2019-05-28 | EfficientNet-B7 | 84.3 | Table 2 | [Tan & Le 2019](https://arxiv.org/abs/1905.11946) |
| 2019-11-21 | AdvProp EfficientNet-B8 | 85.5 | Section 5 text, arXiv v1 | [Xie et al. 2019](https://arxiv.org/abs/1911.09665) |
| 2020-03-18 | FixEfficientNet-B8 | 85.7 | Table 2 | [Touvron et al. 2020](https://arxiv.org/abs/2003.08237) |
| 2021-02-11 | NFNet-F6 + SAM | 86.5 | Table 3 | [Brock et al. 2021](https://arxiv.org/abs/2102.06171) |
| 2021-03-31 | CaiT-M48 ↑448 Υ | 86.5 | Table 5 | [Touvron et al. 2021](https://arxiv.org/abs/2103.17239) |
| 2021-06-24 | VOLO-D5 ↑512 | 87.1 | Table 4 | [Yuan et al. 2021](https://arxiv.org/abs/2106.13112) |
| 2021-11-11 | MAE ViT-H/14 (448 px) | 87.8 | Table 3 and Section 5 text | [He et al. 2021](https://arxiv.org/abs/2111.06377) |
| 2022-09-08 | dBOT ViT-H/14 (448 px) | 88.0 | Table 2 | [Liu et al. 2022](https://arxiv.org/abs/2209.03917) |
| 2022-12-30 | Perceptual MAE (MSG-MAE, StyleGANv2-ADA-P loss) ViT-L | 88.1 | Table 2 | [Tukra et al. 2022](https://arxiv.org/abs/2212.14504) |
| 2023-06-01 | Hiera-H (224 px) | 86.9 | Table 8 | [Ryali et al. 2023](https://arxiv.org/abs/2306.00989) |
| 2026-02-08 | ViT-5-L (384 px) | 86.0 | Table 5 and Section 4.1 text | [Wang et al. 2026](https://arxiv.org/abs/2602.08071) |

#### Passive, frozen self-supervised backbone + linear probe

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2021-04-29 | DINO ViT-B/8 | 80.1 | Table 2 | [Caron et al. 2021](https://arxiv.org/abs/2104.14294) |
| 2023-04-14 | DINOv2 ViT-g/14 | 86.5 | Table 4 | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2025-08-13 | DINOv3 ViT-7B/16 | 88.4 | Table 7 of arXiv v1; Table 9 of the TMLR version | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |


### ADE20K mIoU (validation set, %)

#### Active, sequential glimpses

| Date | Model | mIoU | Protocol | Where | Source |
|---|---|---|---|---|---|
| 2023-03-11 | AME (MAE-initialised ViT-L) | 24.4 | unstated | Table 3 ('IoU') | [Pardyl et al. 2023](https://arxiv.org/abs/2303.06457) |
| 2023-03-11 | AME (SETR-initialised ViT-L) | 27.6 | unstated | Table 3 ('IoU') | [Pardyl et al. 2023](https://arxiv.org/abs/2303.06457) |
| 2024-04-04 | AdaGlimpse (ViT-B, 4 glimpses of 48 px) | 22.7 | unstated | Table 3 ('IoU') | [Pardyl et al. 2024](https://arxiv.org/abs/2404.03482) |
| 2024-04-04 | AdaGlimpse (ViT-B, 8 glimpses of 48 px) | 25.7 | unstated | Table 3 ('IoU') | [Pardyl et al. 2024](https://arxiv.org/abs/2404.03482) |

#### Active, one selection step from a low-resolution view

| Date | Model | mIoU | Protocol | Where | Source |
|---|---|---|---|---|---|
| 2025-05-23 | LookWhere (k=1024 of 1369 patches, 518 px) | 44.6 | unstated | Table 2 (present in arXiv v1) | [Fuller et al. 2025](https://arxiv.org/abs/2505.18051) |
| 2026-09-04 | LookThere (DINOv2 init, 84% of patches dropped) | 42.0 | unstated | Figure 5(b) table | [Rammohan et al. 2026](https://arxiv.org/abs/2609.04698) |

#### Passive, decoder trained on ADE20K (any pretraining data)

| Date | Model | mIoU | Protocol | Where | Source |
|---|---|---|---|---|---|
| 2016-08-18 | Cascade-DilatedNet (VGG) | 34.9 | unstated | Table 2 (printed as 0.3490) | [Zhou et al. 2017](https://arxiv.org/abs/1608.05442) |
| 2016-12-04 | PSPNet ResNet-269 | 43.81 | single-scale | Tables 3 and 4 | [Zhao et al. 2016](https://arxiv.org/abs/1612.01105) |
| 2016-12-04 | PSPNet ResNet-269 | 44.94 | multi-scale | Tables 3 and 4 | [Zhao et al. 2016](https://arxiv.org/abs/1612.01105) |
| 2018-03-23 | EncNet ResNet-101 | 44.65 | unstated | Table 4 | [Zhang et al. 2018](https://arxiv.org/abs/1803.08904) |
| 2018-11-28 | CCNet ResNet-101 (arXiv v1) | 45.22 | unstated | Table 6 of arXiv v1 | [Huang et al. 2018](https://arxiv.org/abs/1811.11721) |
| 2019-11-05 | ACNet ResNet-101 | 45.9 | multi-scale | Table 5 | [Fu et al. 2019](https://arxiv.org/abs/1911.01664) |
| 2020-04-19 | ResNeSt-101 | 46.91 | multi-scale | Table 7 of arXiv v1 | [Zhang et al. 2020](https://arxiv.org/abs/2004.08955) |
| 2020-12-31 | SETR-MLA (ViT-L) | 48.64 | single-scale | Table 4 | [Zheng et al. 2020](https://arxiv.org/abs/2012.15840) |
| 2020-12-31 | SETR-MLA (ViT-L) | 50.28 | multi-scale | Table 4 | [Zheng et al. 2020](https://arxiv.org/abs/2012.15840) |
| 2021-03-25 | Swin-L + UperNet | 53.5 | multi-scale | Table 3 | [Liu et al. 2021](https://arxiv.org/abs/2103.14030) |
| 2021-11-18 | SwinV2-G + UperNet (896 px test) | 59.3 | single-scale | Table 4 | [Liu et al. 2021](https://arxiv.org/abs/2111.09883) |
| 2021-11-18 | SwinV2-G + UperNet | 59.9 | multi-scale | Table 4 (asterisk = multi-scale) | [Liu et al. 2021](https://arxiv.org/abs/2111.09883) |
| 2022-05-17 | ViT-Adapter-L + Mask2Former (BEiT init) | 60.5 | multi-scale | arXiv v1 text; Table 9 of the ICLR version | [Chen et al. 2022](https://arxiv.org/abs/2205.08534) |
| 2022-08-22 | BEiT-3 + ViT-Adapter + Mask2Former | 62.0 | single-scale | Table 8 | [Wang et al. 2022](https://arxiv.org/abs/2208.10442) |
| 2022-08-22 | BEiT-3 + ViT-Adapter + Mask2Former | 62.8 | multi-scale | Table 8 | [Wang et al. 2022](https://arxiv.org/abs/2208.10442) |
| 2022-11-10 | InternImage-H + Mask2Former | 62.5 | single-scale | Table 5 | [Wang et al. 2022](https://arxiv.org/abs/2211.05778) |
| 2022-11-10 | InternImage-H + Mask2Former | 62.9 | multi-scale | Table 5 | [Wang et al. 2022](https://arxiv.org/abs/2211.05778) |
| 2023-05-18 | ONE-PEACE + ViT-Adapter + Mask2Former | 62.0 | single-scale | Table 3 | [Wang et al. 2023](https://arxiv.org/abs/2305.11172) |
| 2023-05-18 | ONE-PEACE + ViT-Adapter + Mask2Former | 63.0 | multi-scale | Table 3 | [Wang et al. 2023](https://arxiv.org/abs/2305.11172) |
| 2024-03-12 | ViT-CoMer-L + Mask2Former (BEiTv2 init) | 62.1 | multi-scale | Table 7 | [Xia et al. 2024](https://arxiv.org/abs/2403.07392) |
| 2025-05-26 | ViT-P on InternImage-H + Mask2Former | 63.1 | single-scale | Table 1 (arXiv v1 and v2) | [Shahabodini et al. 2025](https://arxiv.org/abs/2505.19795) |
| 2025-05-26 | ViT-P on InternImage-H + Mask2Former | 63.6 | multi-scale | Table 1 (arXiv v1 and v2) | [Shahabodini et al. 2025](https://arxiv.org/abs/2505.19795) |
| 2025-08-13 | DINOv3 7B (frozen) + ViT-Adapter + Mask2Former | 62.6 | single-scale | Table 11 ('Simple') | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |
| 2025-08-13 | DINOv3 7B (frozen) + ViT-Adapter + Mask2Former | 63.0 | multi-scale | Table 11 ('TTA') | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |

#### Passive, frozen self-supervised backbone + linear probe

| Date | Model | mIoU | Protocol | Where | Source |
|---|---|---|---|---|---|
| 2021-04-29 | DINO ViT-B/8 (measured by Oquab et al.) | 31.8 | linear probe | Table 10 ('lin.') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2021-11-11 | MAE ViT-H/14 (measured by Oquab et al.) | 33.3 | linear probe | Table 10 ('lin.') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2021-11-15 | iBOT ViT-L/16 (measured by Oquab et al.) | 44.6 | linear probe | Table 10 ('lin.') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2023-04-14 | DINOv2 ViT-g/14 | 49.0 | linear probe | Table 10 ('lin.') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2023-04-14 | DINOv2 ViT-L/14 | 53.1 | linear probe + ms | Table 10 ('+ms') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2025-08-13 | DINOv3 ViT-B/16 (distilled) | 51.8 | linear probe | Table 14 (distilled models) | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |
| 2025-08-13 | DINOv3 ViT-7B/16 | 55.9 | linear probe | Table 3 | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |
| 2026-03-23 | DINOv3 ViT-B/16, probed by Berreby et al. (512 px) | 47.19 | linear probe | Appendix table 'Passive-vision comparison: ADE20K mIoU at t = 0' (printed 47.2); unrounded value from the paper's export in the CanViT repo, site/assets/paper/ade20k_seg.json, probe_table row input_px 512 | [Berreby et al. 2026](https://arxiv.org/abs/2603.22570) |


## Verified, but left out of the JSON

**ImageNet-1k**
- PolyNet (Very Deep PolyNet, 2016-11-17): 82.64 (multi-crop, single model). Below the 2016 ensemble.
- Base Saccader (BagNet-77-lowD, 224 px, 6 glimpses): 70.31, as tabulated by TNet's Table 1. The Saccader
  appendix gives DRAM 66.6 and Saccader 70.1 on a 3,906-image validation subset.
- Frozen weakly-supervised backbones, linear probe (DINOv3 Table 9): PE-core G/14 89.3 and SigLIP 2 g/16 89.1.
  Both are above DINOv3 7B's 88.4. They are not self-supervised, so they are not in the `frozen_ssl` series.
- CF-ViT (AAAI 2023, 2022-03-08): 84.1. It tokenizes the whole image coarsely, then re-splits patches chosen by
  attention, once. It is not active.
- Jérémie et al. (arXiv 2402.15480): 74.3 with a single central log-polar fixation on a retrained ResNet-101;
  77.4 when the fixation sits at the ground-truth box centre. Not active.
- Vuyyuru, Reddy et al. 2020 (arXiv 2006.16427): 62.90 with fixed "retinal fixations" on central 320 px crops.
  The test set may be a 5k-image subset.
- Bio-FCG (Lukanov et al., Frontiers in Computational Neuroscience, 2021-11-22): 65.17 with one saccade. It was
  evaluated on ILSVRC-2010, not ILSVRC-2012.
- FoveaTer v3 (2022-10-02): 76.69 with radial-polar pooling and 76.30 with five square fixations. The JSON keeps
  v2's 78.31.

**ADE20K**
- Further passive results below the year's best:
  - OCNet v1 (2018-09-04): 45.08.
  - OCR HRNetV2-W48 (2019-09-24): 45.66.
  - ResNeSt-200 (arXiv v2, 2020-12-30): 48.36 multi-scale.
  - Mask2Former Swin-L-FaPN (2021-12-02): 56.4 single-scale / 57.7 multi-scale.
  - BEiT-L+ with In-House-70M intermediate fine-tuning (BEiT arXiv v2, 2022-09-03): 57.9 / 58.4.
  - ViT-Adapter-L with BEiTv2 (ICLR version): 61.2 / 61.5.
  - EVA-02-L (2023-03-20): 61.7 / 62.0.
- Frozen DINOv2 ViT-g/14 with a trained ViT-Adapter and Mask2Former head: 60.2. The DINOv2 paper does not say
  whether this is single-scale or multi-scale.
- Linear probes on other frozen backbones in DINOv3's Table 3 (input of 1024 patch tokens):
  - AM-RADIOv2.5 g/14: 53.0.
  - PE-spatial G/14: 49.3.
  - Franca g/14: 46.3.
  - SigLIP 2 g/16: 42.7.
  - Web-DINO 7B/14: 42.7.
  - PE-core G/14: 38.9.
  - DINOv2 g/14: 49.5 under DINOv3's protocol, against 49.0 in its own paper.
- DINOv3's distilled models (Table 14): S 47.0, S+ 48.8, L 54.9, H+ 54.8.
- InternViT-6B, frozen, linear probe at 504 px (InternVL, 2023-12-21): 47.2. In the same table, OpenCLIP-G
  frozen scores 39.3.
- DINOv2 Table 10, "+ms" column for older models:
  - DINO ViT-B/8: 35.2.
  - iBOT ViT-L/16: 47.5.
  - MAE ViT-H/14: 30.7.
  - OpenCLIP ViT-G/14: 46.0 (39.3 plain).

## Unverified leads

**ImageNet-1k**
- AmoebaNet-A 83.9 (Real et al. 2018): not pinned to the arXiv version that first carries it. GPipe's later
  versions cite it as the previous best.
- Not read in their own papers:
  - InternImage-H 89.6.
  - Florence 90.05.
  - NEPA-L 85.3 (a 2026 lead).
- MaskDistill 88.3 and ConvNeXt V2-H 88.9 were seen but not read. They use a CLIP teacher and ImageNet-22K
  labels, so they would belong to the extra-data series, below the frontier of their year.
- SpatialBoost 90.2 (2026) is a linear probe on DINOv3, not a fine-tuned result. Its protocol was not checked.
- 2024–2026, overall: no fine-tuned ImageNet-1k result above Lion/BASIC-L's 91.1 was found, other than OmniVec2
  (flagged in the JSON). The search was not exhaustive.
- Excluded as implausible, though they appear on the Papers with Code leaderboard:
  - SATA (arXiv 2409.19850): 94.9% from a ViT-B/16 "with no additional training or fine-tuning".
  - TAPe+ML (arXiv 2609.20869): 88.1% with under 100k parameters.
  - "Toward Errorless Training ImageNet-1k" reports training-set accuracy.
- Report top-5 only: GoogLeNet (2014) and the multi-model results of PReLU-net and ResNet.
- Luo et al. 2015 (foveation against adversarial examples) and Recasens et al. 2018 (saliency-based sampling)
  were not checked for ImageNet-1k numbers.
- Dallain et al. 2026 (saccade-inspired classification with ViT attention maps) shows its numbers in plots only,
  and its fixations come from a DINO attention pass over the full image.
- FoveaTer's venue: the v3 PDF header says ICLR 2023, but this was not confirmed.

**ADE20K**
- SAC (ICCV 2017, 44.30) and UperNet (ECCV 2018, 42.66) were seen only in later papers' tables. Both are below
  PSPNet.
- FD-SwinV2-G 61.4, Mask DINO 60.8 and M3I (InternImage-H, 62.9) were seen only in other papers' tables. None was
  a year's best.
- No 2024–2026 result above ViT-P's 63.6 multi-scale was found; the search was limited.
- No other active (sequential) model reports standard ADE20K mIoU:
  - GAE (Glimpse-Attend-and-Explore) reports pixel accuracy only.
  - SimGlim reports reconstruction only.
  - Digital Foveation uses a 98-class ADE20K subset.

## Checked: no ImageNet-1k top-1 or ADE20K mIoU

- **No ImageNet-1k number:**
  - TORE (WACV 2025): SUN360, CIFAR-100, Flowers102 and Food101.
  - Pourrahimi & Bashivan 2025 (bioRxiv): visual search on COCO-Search18. Its CNN is pretrained on
    retina-transformed ImageNet, with no accuracy reported.
  - AME: SUN360 classification only.
- **Small datasets only:**
  - Rangrej & Clark (BMVC 2021): up to TinyImageNet.
  - Gaussian RAM: cluttered MNIST and CIFAR.
  - Lu 2019: MNIST.
  - "Where to Look" (2111.07169): MNIST, CIFAR-10, SVHN.
  - Variational Saccading: MNIST, MIT-5k.
  - TDFN: MNIST.
  - Saccade Mechanisms (2206.05102): CIFAR-10, COCO, DAVSOD, MOT17.
- **ImageNet-100 or other subsets:**
  - Killick et al. 2023.
  - Foveated Dynamic Transformer.
  - MGNet.
  - EVA (2603.27340).
  - Two-stream foveation (TCDS 2024): CUB and ImageNet birds.
  - FocL: a 2,000-image validation subset with oracle or SAM crops.
- **Other tasks:**
  - FALcon's own paper reports localization only.
  - AttSeg: Cityscapes, CamVid, KITTI.
  - Grimes et al. (TMLR 2023): robot camera control.
  - Video models: AdaFocus, Uni-AdaFocus, policy-based foveated imaging.
  - Coarse-to-Fine GAP: instance detection.
  - Beyond Grids: plots only.

## Caveats a careful audience may raise

1. **Active models stand on passive ones.** Most active entries start from an ImageNet-pretrained passive network
   or learn from a passive teacher that sees the whole image:
   - Start from a pretrained passive network: GFNet, Saccader-NASNet, FALcon, and Liu et al., whose evaluator is a
     frozen ResNet-101.
   - Learn from a whole-image teacher: STAM and AdaGlimpse.
   - Both: LookWhere is initialised from DINOv2 and distilled from it.

   Their accuracy therefore tracks passive progress; it is not an independent line.
2. **Many active models also see the whole image at low resolution.**
   - GFNet and AdaptiveNN start with a downsampled glance of the full image.
   - Saccader computes restricted-receptive-field features over the whole 224 px image in one pass and picks
     locations from them.
   - PatchDrop, TNet and LookWhere select from a low-resolution view.

   STAM never sees the full image. AdaGlimpse's variable-scale glimpses can cover the whole image at low
   resolution.
3. **The two series differ in scale.**
   - ImageNet active entries are ResNet-50-, NASNet-, EfficientNet-B3-, DeiT-S/B- or ViT-B-sized models on
     224–331 px images.
   - The passive frontier after 2019 uses 0.5–2.4B-parameter models tested at 475–800 px, pretrained on hundreds
     of millions to billions of extra images (JFT-300M, JFT-3B, JFT-5B, Instagram, ALIGN).
   - A like-for-like comparison exists inside AdaptiveNN's own table: DeiT-S scores 79.9 at 224 px and 81.6 at
     384 px, against AdaptiveNN-DeiT-S's 82.2 at 288 px.
   - Showing the ImageNet-1k-only and frozen-SSL lines next to the extra-data line lets the audience compare like
     with like.
4. **Sensing budgets differ.** Active models report accuracy at a fraction of the pixels. Examples: AdaGlimpse
   uses 28.6% of the pixels, STAM about 55%, Saccader 29.5% of the 331 px image. Passive models see every pixel.
5. **Evaluation protocols changed over time.**
   - Passive results up to 2016 are ensembles with 10–144 crops; from 2017 they are mostly single-crop.
   - Test resolution grew from 224 px to 800 px.
   - Inception-v3 was scored on 48,238 non-blacklisted validation images (the paper says the full set is about
     0.2 points worse).
   - BiT, ViT, ViT-G, SAM, STAM and AdaptiveNN report means or medians over runs.

   All ImageNet values use the original ILSVRC-2012 validation labels, not ImageNet-ReaL. The 2017 best (82.9,
   single crop) is below the 2016 ensemble (83.5), so a running maximum mixes protocols.
6. **Numbers change between arXiv versions.** Each date is that of the version first carrying the number:
   - MPL: 86.9 in 2020, 90.2 from 2021-01-05.
   - CoAtNet: 89.77, then 90.88.
   - Noisy Student: 87.4, then 88.4.
   - BiT: 87.76, then 87.54 in the final version.
   - CCNet: 45.22, then 45.76.
   - FoveaTer: ImageNet-100 in v1, 78.31 in v2, 76.69 in v3.

   Venue years often fall a year after the arXiv date; DINOv3, for instance, is arXiv 2025 and TMLR 2026.
7. **ADE20K protocols differ.**
   - Multi-scale testing adds 0.4–1.6 points over single-scale across these entries.
   - From 2022 the leaders train and test at 896 px; SETR used 512 px crops and Swin 640 px.
   - From 2022 the leaders add intermediate segmentation training on COCO-Stuff (DINOv3 also on Hypersim) before
     ADE20K.
   - The active models are evaluated on much smaller scenes: AME on 256×128 px images, AdaGlimpse on 224×224.
   - AME's SETR initialisation was already trained on ADE20K segmentation.
   - Several papers do not state single- or multi-scale (protocol "unstated").
   - Swin's "62.8" is a test-set score, not validation mIoU.
8. **Frozen probes depend on the probe.**
   - The same DINOv2 ViT-g/14 scores 49.0 in its own paper and 49.5 under DINOv3's protocol.
   - DINOv3's paper gives its distilled ViT-B/16 51.8, while the CanViT paper's own probe of that model at 512 px
     gives 47.19, the protocol it also uses to probe CanViT-B.

   I did not investigate the 4.6-point difference. The probe numbers of older models (DINO, MAE, iBOT) were
   measured by the DINOv2 authors in 2023, so the JSON dates those points by model release.
9. **Flagged entries.**
   - OmniVec (92.4) and OmniVec2 (93.6), by the same two authors, give no resolution or crop protocol, and no
     independent reproduction was found. A conservative extra-data line stops at 91.1 (2023).
   - Perceptual MAE's text misstates ViT-L's parameter count.
   - ViT-P (63.6) is a two-model system with no venue. Without it, the 2025 ADE20K best is DINOv3's 63.0, which
     ties ONE-PEACE.
   - GFNet's two figure-read values are approximate.
   - DRAM's 67.5 is a number later papers tabulate from a Saccader figure.
10. **The definition of "active" decides several points.**
    - LookWhere (2025): 83.0 on ImageNet-1k and 44.6 mIoU on ADE20K. DINOv2 with all patches gets 84.2 and 46.8.
    - LookThere (2026): 83.0 and 42.0.

    Both select patches once from a low-resolution view. They exceed every sequential active model on both
    benchmarks, and LookWhere is close to CanViT-B's 84.5 (fine-tuned) and 45.9 (frozen, linear probe). The
    CanViT paper defines active models as processing scenes "through sequential, localized glimpses"
    (Introduction) and cites neither. A question about LookWhere is likely from an informed audience; whether
    the talk should address it is the authors' decision.
11. **"Year" can mean different things.** A point's year is that of its first public appearance (usually arXiv),
    not of the venue. The 2012 AlexNet and 2024 OmniVec2 points carry the year only.

## Notes for the paper's bibliography

- The DINOv3 TMLR PDF (Zotero copy, 2026) reads "Published in Transactions on Machine Learning Research
  (04/2026)"; `references.bib` has `month = feb`.
- The DINOv2 TMLR PDF reads "(01/2024)".
