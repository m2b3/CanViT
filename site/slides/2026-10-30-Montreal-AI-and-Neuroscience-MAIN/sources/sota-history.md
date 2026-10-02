# Published accuracy over time: active and passive vision

[Assembled by Claude Code on 2026-10-01 from the papers' own text; model sizes, test resolutions, pretraining data and
the base-size series added the same afternoon. That evening two primary-source audits by Codex
(`throwaway/history-audit/REPORT.md` and `REPORT2.md`, not author-validated) filled fields from official code, added
base-size points and resolved leads; before merging the second audit, Claude Code found each added point's value
again in the saved paper text, but did not re-check its dates or rerun its counts. Not yet reviewed by the
authors.] Data for the history chart (x = year, y = accuracy) on ImageNet-1k top-1 and ADE20K mIoU. The data file
is `sota-history.json`; its `_about` field defines every field and series. This note says how the points were
verified, lists them with their sources, and records the leads that could not be confirmed and the caveats a careful
audience may raise. The points in the tables below are those of the JSON; where the two disagree, the JSON is the
reference. The tables were generated from the JSON and the readers' results on 2026-10-01
(`throwaway/history-base/gen_md_tables.py` in the local checkout, not in git); the audits' fields and points were
then added to them by hand.

## What the data supports

The CanViT paper's wording is that active vision models "have struggled to match" passive ones; the numbers
below scope that claim. The author asked that the chart compare models of the same size [Yohaï, 2026-10-01:
"comparing to nonsensically large models is indeed stupid and counterproductive"], so the base-size comparison comes
first. Base size means at most about 100M parameters (`size_class`; up to 110M is counted as base, and entries above
100M say so).

- **ImageNet-1k, base size.** The sequential active models are base size except Saccader-NASNet (124.5M) and the
  two Prisadnikov et al. models of 2025 and 2026 (counts unknown). At each model's date:
  - Saccader-NASNet (2019-08-20, 75.03, 331 px) trails the base ImageNet-1k-only line by 9.37 points
    (EfficientNet-B7, 84.4 in arXiv v1, 66M, 600 px) and lies 13.73 points above the base frozen line of its date
    (BigBiGAN RevNet-50 ×4, 61.3, 256 px).
  - GFNet's figure-read 79.8 (2020-10-11, EfficientNet backbones; 37.4M if the point is the EfficientNet-B3 model with
    144 px patches, as its curve suggests) trails FixEfficientNet-B8 (85.7, 87.4M, 800 px) by 5.9 and lies 2.0
    points above the base frozen line of its date (BYOL ResNet-200, 77.8, 63M).
  - STAM (2022-04-01, 80.78, 100.5M at test, 224 px) trails VOLO-D3↑448 (86.3, 86M, 448 px, ImageNet-1k only) by
    5.52 and the base frozen line (EsViT Swin-B/W=14, 81.3, 87M) by 0.52.
  - AdaGlimpse (2024-04-04, 77.54, 86.9M, 224 px) trails MOAT-2 (86.5, 73.4M, 512 px, ImageNet-1k only) by 8.96 and
    the frozen DINOv2 ViT-B/14 linear probe (84.5, 86.6M, 224 px) by 6.96.
  - AdaptiveNN-DeiT-S (2025-09-18, 82.2, 89.4M at test, 288 px) trails MOAT-2 (86.5) by 4.3, the base frozen line
    (Proteus ViT-B/14, 84.9, distilled from DINOv2-L/14 on ImageNet-1k images) by 2.7, and the base extra-data line
    (ViT-B/16 distilled from ViT-22B, 88.6, 384 px) by 6.4. Its own Supplementary Data Tab. 2 gives the like-for-like
    baseline: DeiT-S scores 80.9 at 288 px and 81.6 at 384 px.
  - The base frozen line ends at 85.0: the CanViT paper's own linear probe of DINOv3 ViT-B/16 at 512 px (2026-03-23).
    DINOv3's TMLR supplement may raise it (an open lead under "Unverified leads").
- **ADE20K, base size.**
  - AdaGlimpse (2024-04-04, 25.7 with 8 glimpses; ViT-B encoder and policy 86.1M, 117.1M with its decoder; 224 px
    scenes) trails the base fine-tuned line by 30.0 single-scale (PlainSeg BEiT-B, 55.70, 640 px crops) and 30.1
    multi-scale (RevColV2-B + Mask2Former, 55.8), and the base frozen linear line (DINOv2 ViT-B/14, 47.3) by 21.6.
  - AME's encoder is a ViT-L (303.1M; 348.5M with its decoder), so AME is large and the base lines do not bound it.
    At its date (2023-03-11) the base fine-tuned line stood at 55.1 multi-scale (Mask2Former Swin-B) and the base
    frozen linear line at 38.3 (iBOT ViT-B/16).
  - The base frozen linear line ends at 51.8 (DINOv3 ViT-B/16 under DINOv3's protocol, 2025-08-13); the CanViT
    paper's probe of the same model gives 47.19 under its own protocol. The same DINOv3 supplement lead may raise this
    end too.
- **Resolution still differs at base size.** The base passive records from 2019 on test at 384 to 800 px; the active
  models at 224 to 331 px (AdaptiveNN 288). The base frozen probes, where the test size is known, use 224 or 256 px
  (BigBiGAN) up to DINOv2 and 512 px in the CanViT paper's DINOv3 probe.
- **All sizes, for reference.** These lines are set by models of 0.5 to 7 billion parameters and the author asked not
  to draw them.
  - ImageNet-1k: Saccader (75.03) trails the extra-data frontier of its date by 11.37 points (FixRes, 86.4,
    2019-07-19) and the ImageNet-1k-only frontier by 9.37 (EfficientNet-B7, 84.4). STAM (80.78) trails model soups
    (90.94, 2022-03-10) by 10.16 and MAE ViT-H (87.8) by 7.02. AdaptiveNN (82.2) trails Lion/BASIC-L (91.1,
    2023-02-13) by 8.9, Perceptual MAE (88.1, ImageNet-1k only, 2022-12-30) by 5.9, and the frozen DINOv3 7B linear
    probe (88.4, 2025-08-13) by 6.2.
  - ADE20K: AME (2023-03-11, 27.6 with a SETR-initialised encoder) is 35.3 points below InternImage-H (62.9,
    multi-scale, 2022-11-10). AdaGlimpse (2024-04-04, 25.7) is 37.3 below ONE-PEACE (63.0). A plain linear probe on a
    frozen DINOv2 ViT-g/14 (49.0, 2023-04-14) beats both by more than 20 points.
- **Select-once models.** LookWhere (NeurIPS 2025) reaches 83.0 on ImageNet-1k and 44.6 mIoU on ADE20K with a ViT-B
  extractor and a 3-block selector, 110.7M in all (large by the 110M cut, by 0.7M). It selects high-resolution patches
  once, from a low-resolution view of the whole image. Neither number is in the CanViT paper's comparison, which does
  not cite LookWhere. Under the series definitions here it is not a sequential glimpse model, but an audience may ask
  about it (see the caveats).
- **Plotting.** Filter each passive series by `size_class == "base"` and draw its running maximum (ADE20K: per
  protocol). The ImageNet extra-data base line is the running maximum over the base points of `overall` and
  `in1k_only`, as for all sizes. The base frozen ImageNet line starts in 2016 at 35.4 (Split-Brain AlexNet) and first
  reaches 60 with Local Aggregation (60.2, 2019-03-29).

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
   - For the base-size series: the comparison tables of later papers (EfficientNet, CaiT, VOLO, MOAT, MetaFormer,
     EVA-02, DINOv2, DINOv3 and others) as lead lists.

   A leaderboard value, another paper's table or a number recalled from memory counted only as a lead. The Papers
   with Code ADE20K list has 83 entries and lacks ONE-PEACE and ViT-P. Its ImageNet-1k list includes claims the JSON
   leaves out (SATA and TAPe+ML, under "Verified, but left out").
3. **Verification.** Each value was read in the paper's text, extracted from its PDF with `pdftotext -layout`.
   The check covered the table row, its caption, and the setup text that states the data, backbone and
   protocol. Error rates were converted to accuracy (100 − error).
4. **Dates.** A date is the first public appearance of the number, taken from the arXiv submission history (or
   the venue, OpenReview or journal date when there is no earlier arXiv version). When
   a number changed between arXiv versions, the date and value are those of the first version that carries the
   number (e.g., MPL 90.2, CoAtNet 90.88, VGG 76.3, EfficientNet-B7 84.4).
5. **Venues** come from the PDF header, the arXiv comment field, or Semantic Scholar's record, all read on
   2026-10-01.
6. **Figure-only values.** GFNet's best accuracies appear only in figures. They were read from the PDF's vector
   drawing (pdfplumber) by interpolating between the axis ticks, with the legend colours checked. These two
   values are approximate (± 0.1).
7. **Sizes, resolutions and pretraining.** For every point, `params_m`, `test_px` and `pretraining` were read in the
   paper. Where a paper states no parameter count, the count was taken from the official released code (instantiated
   without weights and counted with `uv run`), or computed from an unambiguous architecture description; the
   provenance tables below say which, entry by entry, and give null with the reason otherwise. For the active models,
   every network run at test time was counted separately (`params_detail`); teachers and critics used only in
   training are named but not summed.
8. **Who did what.** In the morning, three research subagents collected the evidence for passive ImageNet-1k,
   passive ADE20K and further active models; I wrote the core active points (Saccader, GFNet, AME, AdaGlimpse,
   AdaptiveNN) and the DINO-family points, and re-read the verbatim evidence for every point. In the afternoon,
   seven subagents read the sizes of the existing passive ImageNet-1k and ADE20K points and of the active models (two
   readers), and collected the base-size series (ImageNet-1k end to end, frozen probes on both benchmarks, ADE20K
   fine-tuned). I re-read the verbatim evidence for every correction to an existing point and for the endpoints of
   each base-size series, and the counts behind AdaptiveNN and GFNet (official-code count logs), Saccader (Table
   Supp.1) and AdaGlimpse's teacher (official README). Each reader saved the verbatim table lines per point. The
   evidence is kept outside git in the local checkout:
   `throwaway/history-base/evidence/` (afternoon), `throwaway/history-base/evidence_2026-10-01_morning/` (morning)
   and `throwaway/history-base/pdf_text/` (the PDF text extractions the evidence cites); each point's `where` field
   names the table to reopen. In the evening, Codex audited the file against primary sources: official code for
   missing fields, then the unverified leads and missing base-size records. Its reports, the papers it saved
   (`sources/`) and its parameter-count scripts are in `throwaway/history-audit/`.
9. **What counts as active.** A model is *active* if it chooses where to look, one glimpse after another, each
   choice depending on what it has already seen. A model counts as *select once* if a policy picks
   high-resolution patches in a single step from a low-resolution view. *Borderline* covers foveated models whose
   fixations are random, or whose early layers see the full-resolution image.

## Verified points

Model sizes, test resolutions and pretraining data are in the JSON and in the section "Where each size, resolution and
pretraining was read". The base-size points collected in the afternoon and by the evening audit are listed in the
section "Base-size series".

### ImageNet-1k top-1 (validation set, %)

#### Active, sequential glimpses

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2019-08-20 | DRAM, reimplemented by Elsayed et al. (8 glimpses of 77 px) | 67.5 | Table 1 (#Locs 8), which credits Elsayed et al. 2019 | [Papadopoulos et al. 2021](https://arxiv.org/abs/2102.10212) |
| 2019-08-20 | Saccader (Saccader-NASNet, 331 px images) | 75.03 | Section 4.3 text (Figure 6) | [Elsayed et al. 2019](https://arxiv.org/abs/1908.07644) |
| 2020-10-11 | GFNet (ResNet-50, 96 px patches, T=5) | 75.93 | Table 1, last row, t=5 | [Wang et al. 2020](https://arxiv.org/abs/2010.05300) |
| 2020-10-11 | GFNet (EfficientNet backbones, T=4), read from figure | 79.8 | Figure 4(c), highest point of the GFNet curve, read from the PDF's vector data (± 0.1) | [Wang et al. 2020](https://arxiv.org/abs/2010.05300) |
| 2022-01-09 | GF-EfficientNet-B2 (TPAMI version, 128 px patches, T=4) | 77.93 | Table 2(c) of arXiv v1, contrastive reward, t=4 (77.95 ± 0.03 in v2, 2022-08-04) | [Huang et al. 2022](https://arxiv.org/abs/2201.03014) |
| 2022-01-09 | MS-GFNet (EfficientNet-B3, T=4), read from figure | 80.0 | Figure 8(b) of arXiv v1 (Figure 10(b) in v2), highest point of the 'MS-GFNet (one model)' curve, read from the PDF's vector data (± 0.05) | [Huang et al. 2022](https://arxiv.org/abs/2201.03014) |
| 2022-04-01 | STAM (DeiT-B-distilled, 27 glimpses of 32 px) | 80.78 | Table 1 (t = 26, counted from t = 0) | [Rangrej et al. 2022](https://arxiv.org/abs/2204.00656) |
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
| 2026-05-29 | FOVI-ViT-H+ (DINOv3, 3 random fixations) | 85.3 | Table 1 of arXiv v2 (printed as 0.853; v1 of 2026-02-03 prints 0.850) | [Blauch et al. 2026](https://arxiv.org/abs/2602.03766) |

#### Passive, extra training data

| Date | Model | Top-1 | Where | Source |
|---|---|---|---|---|
| 2012 | AlexNet, 7-CNN ensemble (2 pretrained on ImageNet Fall 2011) | 63.3 | Table 2 (top-1 val error 36.7%) | [Krizhevsky et al. 2012](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 2018-05-02 | ResNeXt-101 32x48d, IG-940M pretraining | 85.4 | Table 6 | [Mahajan et al. 2018](https://arxiv.org/abs/1805.00932) |
| 2019-07-19 | FixRes ResNeXt-101 32x48d (IG-940M) | 86.4 | Table 2 of arXiv v2 (not in v1 of 2019-06-14) | [Touvron et al. 2019](https://arxiv.org/abs/1906.06423) |
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
| 2019-05-28 | EfficientNet-B7 | 84.4 | Table 2 and abstract of arXiv v1 (84.3 from v4, 2020-09-04) | [Tan & Le 2019](https://arxiv.org/abs/1905.11946) |
| 2019-11-21 | AdvProp EfficientNet-B8 | 85.5 | Section 5 text, arXiv v1 | [Xie et al. 2019](https://arxiv.org/abs/1911.09665) |
| 2020-03-18 | FixEfficientNet-B8 | 85.7 | Table 2 | [Touvron et al. 2020](https://arxiv.org/abs/2003.08237) |
| 2021-02-11 | NFNet-F6 + SAM | 86.5 | Table 3 | [Brock et al. 2021](https://arxiv.org/abs/2102.06171) |
| 2021-04-07 | CaiT-M48 ↑448 Υ | 86.5 | Table 5 of arXiv v2 (not in v1 of 2021-03-31) | [Touvron et al. 2021](https://arxiv.org/abs/2103.17239) |
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
| 2022-05-17 | ViT-Adapter-L + Mask2Former (BEiT init) | 60.5 | multi-scale | Table 10 of arXiv v1; Table 9 of the ICLR version | [Chen et al. 2022](https://arxiv.org/abs/2205.08534) |
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
| 2025-08-13 | DINOv3 ViT-7B/16 | 55.9 | linear probe | Table 3 | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |
| 2025-08-13 | DINOv3 ViT-B/16 (distilled) | 51.8 | linear probe | Table 14 (distilled models) | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |
| 2026-03-23 | DINOv3 ViT-B/16, probed by Berreby et al. (512 px) | 47.19 | linear probe | Table 5 of arXiv v1 (Table 9 of v2), 'Passive-vision comparison: ADE20K mIoU at t = 0' (printed 47.2); unrounded value from the paper's export in the CanViT repo, site/assets/paper/ade20k_seg.json, probe_table row input_px 512 | [Berreby et al. 2026](https://arxiv.org/abs/2603.22570) |


## Base-size series (at most about 100M parameters)

Each table lists, by date, the points of size class base in a series: the records among base-size models (a point
that beat every earlier base point of the series; ADE20K: per protocol), the existing points that are base size, and
reference points marked "not a record". A base point of the extra-data series counts only if it beats both base
lines (ImageNet-1k only and extra data) at its date. Params are those of the predicting network (ImageNet), the
frozen backbone (probes) or the backbone (ADE20K); "—" means the paper does not state it (the provenance tables say
what is known). Wider or Deeper Model A2's backbone, counted from the official code, is 105.07M: above 100M, base
under the 110M bound.

### ImageNet-1k, trained end to end on ImageNet-1k only (`in1k_only`)

| Date | Model | Params (M) | Test px | Pretraining | Top-1 | Where | Source |
|---|---|---|---|---|---|---|---|
| 2012 | AlexNet, single CNN | 60 | 224 | none | 59.3 | Table 2 ('1 CNN', top-1 val error 40.7%) | [Krizhevsky et al. 2012](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 2013-11-28 | ZFNet, single convnet (Fig. 3 architecture) | 62.4 | 224 | none | 61.6 | Table 2 of arXiv v3 ('1 convnet as per Fig. 3', top-1 val error 38.4%) | [Zeiler & Fergus 2013](https://arxiv.org/abs/1311.2901) |
| 2013-12-19 | Howard 2013, single convnet (Krizhevsky architecture, new train/test transforms) | 60 | 224 | none | 62.5 | Table 1 ('New Training, Test Transforms', top-1 val error 37.5%) | [Howard 2013](https://arxiv.org/abs/1312.5402) |
| 2014-06-18 | SPP-net (ZF-5 convolutional layers, 4-level pyramid, multi-size training) | 77 | 224 | none | 65.84 | Table 1 of arXiv v1, row (e5) (top-1 val error 34.16%) | [He et al. 2014](https://arxiv.org/abs/1406.4729) |
| 2015-02-11 | BN-Inception, 6-model ensemble | — | 224 | none | 79.9 | Figure 4 table (top-1 val error 20.1%) | [Ioffe & Szegedy 2015](https://arxiv.org/abs/1502.03167) |
| 2015-12-02 | Inception-v3, 4-model ensemble | — | 299 | none | 82.8 | Table 5 (top-1 error 17.2%) | [Szegedy et al. 2015](https://arxiv.org/abs/1512.00567) |
| 2015-12-10 | ResNet-152 (single model) (not a record) | 60.19 | — | none | 80.62 | Table 4 (top-1 val error 19.38%) | [He et al. 2015](https://arxiv.org/abs/1512.03385) |
| 2017-12-02 | PNASNet-5 (N=4, F=216) | 86.1 | 331 | none | 82.9 | Table 3 of arXiv v1 (Table 5 in v3) | [Liu et al. 2017](https://arxiv.org/abs/1712.00559) |
| 2019-04-03 | PNASNet-5-Large evaluated at 500 px with a fine-tuned GeM pooling exponent (MultiGrain appendix) | 86.1 | 500 | none | 83.6 | Table E.1 of arXiv v2 (s* = 500, p* = 1.7) | [Berman et al. 2019](https://arxiv.org/abs/1902.05509) |
| 2019-05-28 | EfficientNet-B7 | 66 | 600 | none | 84.4 | Table 2 and abstract of arXiv v1 (84.3 from v4, 2020-09-04) | [Tan & Le 2019](https://arxiv.org/abs/1905.11946) |
| 2019-09-30 | EfficientNet-B7 with RandAugment | 66 | 600 | none | 85.0 | Table 3 (and Table 1) of arXiv v1 | [Cubuk et al. 2019](https://arxiv.org/abs/1909.13719) |
| 2019-11-21 | AdvProp EfficientNet-B8 | 88 | 672 | none | 85.5 | Section 5 text, arXiv v1 | [Xie et al. 2019](https://arxiv.org/abs/1911.09665) |
| 2020-03-18 | FixEfficientNet-B8 | 87.4 | 800 | none | 85.7 | Table 2 | [Touvron et al. 2020](https://arxiv.org/abs/2003.08237) |
| 2021-06-07 | Refined-ViT-L↑448 | 81 | 448 | none | 85.9 | Table 7 | [Zhou et al. 2021](https://arxiv.org/abs/2106.03714) |
| 2021-06-24 | VOLO-D3↑448 | 86 | 448 | none | 86.3 | Table 4 (also Table 3) | [Yuan et al. 2021](https://arxiv.org/abs/2106.13112) |
| 2022-10-04 | MOAT-2 (512 px) | 73.4 | 512 | none | 86.5 | Table 2 ('1K only' column) | [Yang et al. 2022](https://arxiv.org/abs/2210.01820) |

### ImageNet-1k, trained end to end with extra data (`overall`)

| Date | Model | Params (M) | Test px | Pretraining | Top-1 | Where | Source |
|---|---|---|---|---|---|---|---|
| 2012 | AlexNet, single CNN pretrained on ImageNet Fall 2011 | — | 224 | ImageNet Fall 2011 (15M images, 22K classes) | 61.0 | Table 2 ('1 CNN*', top-1 val error 39.0%) | [Krizhevsky et al. 2012](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 2019-05-02 | ResNeXt-101 32x8d, semi-weakly supervised (IG-1B-Targeted teacher) | 88 | — | IG-1B-Targeted (Instagram, semi-weakly supervised) | 84.3 | Table 6 ('ours (semi-weakly sup.)', ResNeXt-101 32x8) | [Yalniz et al. 2019](https://arxiv.org/abs/1905.00546) |
| 2019-11-11 | Noisy Student, EfficientNet-B7 student with EfficientNet-L2 teacher, arXiv v1 | 66 | 600 | JFT-300M, unlabeled | 86.8 | Table 9 of arXiv v1 ('Noisy Student (B7, L2)') | [Xie et al. 2019](https://arxiv.org/abs/1911.04252) |
| 2020-01-07 | Noisy Student, EfficientNet-B7 student with EfficientNet-L2 teacher, arXiv v2 | 66 | 600 | JFT-300M, unlabeled | 86.9 | Table 9 of arXiv v2 ('Noisy Student (B7, L2)') | [Xie et al. 2019](https://arxiv.org/abs/1911.04252) |
| 2020-03-18 | FixEfficientNet-B7 (Noisy Student weights) | 66 | 632 | JFT-300M, unlabeled | 87.1 | Table 1 of arXiv v1 | [Touvron et al. 2020](https://arxiv.org/abs/2003.08237) |
| 2021-06-09 | CoAtNet-2 (ImageNet-21K pretraining, 512 px) | 75 | 512 | ImageNet-21k | 87.3 | Table 4 ('21K+1K' column) and Table 13 of arXiv v1 | [Dai et al. 2021](https://arxiv.org/abs/2106.04803) |
| 2022-10-03 | BEiT v2 ViT-B/16 (ImageNet-21k intermediate fine-tuning, 384 px) | 86 | 384 | ImageNet-21k (supervised) + CLIP-B/16 teacher for MIM targets | 87.5 | Table 7 of arXiv v2 (not in v1) | [Peng et al. 2022](https://arxiv.org/abs/2208.06366) |
| 2022-10-04 | MOAT-2 (ImageNet-22K pretraining, 512 px) | 73.4 | 512 | ImageNet-21k | 87.7 | Table 2 ('22K+1K' column) | [Yang et al. 2022](https://arxiv.org/abs/2210.01820) |
| 2022-10-24 | CAFormer-B36 (ImageNet-21K pretraining, 384 px) | 99 | 384 | ImageNet-21k | 88.1 | Table 5 (↑384 column) | [Yu et al. 2022](https://arxiv.org/abs/2210.13452) |
| 2023-02-10 | ViT-B/16 distilled from ViT-22B (384 px) | 86 | 384 | JFT + ViT-22B teacher (JFT-4B) | 88.6 | Table 8 | [Dehghani et al. 2023](https://arxiv.org/abs/2302.05442) |

### ImageNet-1k, frozen self-supervised backbone + linear probe (`frozen_ssl`)

| Date | Model | Params (M) | Test px | Pretraining | Top-1 | Where | Source |
|---|---|---|---|---|---|---|---|
| 2016-11-29 | Split-Brain Autoencoder (AlexNet, conv3 linear probe) | — | — | none | 35.4 | Table 2 of arXiv v1 (conv3 column); Section 4.1.1 | [Zhang et al. 2016](https://arxiv.org/abs/1611.09842v1) |
| 2018-03-21 | RotNet AlexNet (conv3 linear probe) | — | — | none | 38.7 | Table 5 of arXiv v1 (conv3 column); Section 3.2 | [Gidaris et al. 2018](https://arxiv.org/abs/1803.07728v1) |
| 2018-05-05 | Instance Discrimination ResNet-50 (linear SVM probe) | — | — | none | 54.0 | Table 2 of arXiv v1 ('Ours Resnet50', conv5 column); Section 4.2 | [Wu et al. 2018](https://arxiv.org/abs/1805.01978v1) |
| 2019-01-25 | Rotation RevNet-50 (width factor 16, linear probe) | 85.765632 | 224 | none | 55.4 | Table 2 of arXiv v1 ('Rotation' row, 'Ours' ImageNet column); Section 4.2; Appendix B | [Kolesnikov et al. 2019](https://arxiv.org/abs/1901.09005v1) |
| 2019-03-29 | Local Aggregation ResNet-50 (10-crop linear probe) | — | — | none | 60.2 | Table 1 of arXiv v1 (ResNet-50 rows); Section 4.2 | [Zhuang et al. 2019](https://arxiv.org/abs/1903.12355v1) |
| 2019-07-04 | BigBiGAN RevNet-50 x4 (BN+CReLU linear probe) | — | 256 | none | 61.3 | Table 2 of arXiv v1 (official validation set); Sections 4 and 4.1; Appendix A | [Donahue & Simonyan 2019](https://arxiv.org/abs/1907.02544v1) |
| 2019-10-21 | CMC ResNet-101 (two color-view encoders, linear probe) | 84.990912 | 224 | none | 65.0 | Table 2 of arXiv v3; official code models/resnet.py and LinearProbing.py | [Tian et al. 2019](https://arxiv.org/abs/1906.05849v3) |
| 2019-12-04 | PIRL ResNet-50 c2x (linear probe) | 98 | — | none | 67.4 | Section 3.2 text and Figure 2 of arXiv v1; Appendix A ('Base Architecture for PIRL') | [Misra & van der Maaten 2019](https://arxiv.org/abs/1912.01991v1) |
| 2020-02-13 | SimCLR ResNet-50 (2×) | 94 | 224 | none | 74.2 | Table 6 of arXiv v1 | [Chen et al. 2020](https://arxiv.org/abs/2002.05709) |
| 2020-06-13 | BYOL ResNet-200 (1×) | 63 | 224 | none | 77.8 | Table 9 (Appendix C.2) of arXiv v1 | [Grill et al. 2020](https://arxiv.org/abs/2006.07733) |
| 2021-04-05 | MoCo v3 ViT-BN-B/7 | 86 | 224 | none | 79.5 | Figure 8 (table under the plot) of arXiv v1 | [Chen et al. 2021](https://arxiv.org/abs/2104.02057) |
| 2021-04-29 | DINO ViT-B/8 | 85 | 224 | none | 80.1 | Table 2 | [Caron et al. 2021](https://arxiv.org/abs/2104.14294) |
| 2021-06-17 | EsViT Swin-B/W=14 | 87 | 224 | none | 81.3 | Table 1 of arXiv v1 | [Li et al. 2021](https://arxiv.org/abs/2106.09785) |
| 2023-04-14 | DINOv2 ViT-B/14 | 86.58 | 224 | LVD-142M (self-supervised; distilled from DINOv2 ViT-g/14) | 84.5 | Table 4 | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2024-07-15 | Proteus ViT-B/14 (DINOv2-L teacher, linear probe) | — | — | ImageNet-1k images + DINOv2-L/14 teacher (LVD-142M, self-supervised) | 84.9 | Table 4 of arXiv v1; Sections 3.2.1 and 3.2.2; Appendix A.1.2 | [Zhang et al. 2024](https://arxiv.org/abs/2407.10366v1) |
| 2026-03-23 | DINOv3 ViT-B/16, probed by Berreby et al. (512 px) | 86 | 512 | distilled from DINOv3 7B: LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) | 85.0 | Table 17 of arXiv v1 (Table 10 of v2), 'IN1k top-1 (ours)', printed 85.00% | [Berreby et al. 2026](https://arxiv.org/abs/2603.22570) |

### ADE20K, backbone fine-tuned with a segmentation decoder (`overall`)

| Date | Model | Params (M) | Test px | Pretraining | mIoU | Protocol | Where | Source |
|---|---|---|---|---|---|---|---|---|
| 2016-11-30 | Wider or Deeper Model A2 (2-convolution decoder) | 105.070912 | — | ImageNet-1k + Places365 | 43.73 | single-scale | Table 4 of arXiv v1 (ADE20K val, 'Model A2, 2 conv.'); Sections 5 and 6.2 | [Wu et al. 2016](https://arxiv.org/abs/1611.10080v1) |
| 2016-12-04 | PSPNet ResNet-152 (not a record) | — | — | ImageNet-1k | 42.62 | single-scale | Table 3 (PSPNet(152)), arXiv v1 | [Zhao et al. 2016](https://arxiv.org/abs/1612.01105) |
| 2016-12-04 | PSPNet ResNet-152 | — | — | ImageNet-1k | 43.51 | multi-scale | Table 3 (PSPNet(152)+MS), arXiv v1 | [Zhao et al. 2016](https://arxiv.org/abs/1612.01105) |
| 2017 | SAC ResNet-101 (SAC-multiple + MS) | — | — | ImageNet-1k | 44.3 | multi-scale | Table 3 ('SAC-multiple + MS') | [Zhang et al. 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Scale-Adaptive_Convolutions_for_ICCV_2017_paper.pdf) |
| 2018-03-23 | EncNet ResNet-101 | — | — | — | 44.65 | unstated | Table 4 | [Zhang et al. 2018](https://arxiv.org/abs/1803.08904) |
| 2018-09-04 | OCNet ResNet-101 (arXiv v1) | — | — | ImageNet-1k | 45.08 | unstated | Table 9 of arXiv v1 | [Yuan & Wang 2018](https://arxiv.org/abs/1809.00916v1) |
| 2018-11-28 | CCNet ResNet-101 (arXiv v1) | — | — | ImageNet (1k or 21k not stated) | 45.22 | unstated | Table 6 of arXiv v1 | [Huang et al. 2018](https://arxiv.org/abs/1811.11721) |
| 2019 | APCNet ResNet-101 | — | — | ImageNet-1k | 45.38 | unstated | Table 8 (ADE20K validation, 'Ours ResNet101'); Section 4.1 | [He et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf) |
| 2019-09-24 | OCR HRNetV2-W48 (arXiv v1) | — | 520 | ImageNet-1k | 45.66 | unstated | Table 9 and Section 4 text of arXiv v1 | [Yuan et al. 2019](https://arxiv.org/abs/1909.11065) |
| 2019-11-05 | ACNet ResNet-101 | — | — | — | 45.9 | multi-scale | Table 5 | [Fu et al. 2019](https://arxiv.org/abs/1911.01664) |
| 2020-04-03 | CPNet ResNet-101 | — | 480 | ImageNet-1k | 46.27 | multi-scale | Table 4 (CPNet101) | [Yu et al. 2020](https://arxiv.org/abs/2004.01547) |
| 2020-04-19 | ResNeSt-101 | 48.3 | — | ImageNet-1k | 46.91 | multi-scale | Table 7 of arXiv v1 | [Zhang et al. 2020](https://arxiv.org/abs/2004.08955) |
| 2020-12-30 | DeepLabV3 ResNeSt-200 (arXiv v2) | 70.2 | — | ImageNet-1k | 48.36 | multi-scale | Table 5 of arXiv v2 | [Zhang et al. 2020](https://arxiv.org/abs/2004.08955) |
| 2021-03-24 | DPT-Hybrid (ViT-Hybrid: ResNet-50 stem + ViT-B) | 98.181952 | 480 | ImageNet (1k or 21k not stated) | 49.02 | multi-scale | Table 4 | [Ranftl et al. 2021](https://arxiv.org/abs/2103.13413) |
| 2021-03-25 | Swin-B + UperNet (ImageNet-22k) | 88 | 640 | ImageNet-21k | 51.6 | multi-scale | Table 3 (UperNet Swin-B‡); protocol from Appendix A2.3 | [Liu et al. 2021](https://arxiv.org/abs/2103.14030) |
| 2021-05-31 | SegFormer-B5 (MiT-B5) | 81.4 | 640 | ImageNet-1k | 51.0 | single-scale | Table 1(a), arXiv v1 | [Xie et al. 2021](https://arxiv.org/abs/2105.15203) |
| 2021-05-31 | SegFormer-B5 (MiT-B5) | 81.4 | 640 | ImageNet-1k | 51.8 | multi-scale | Table 1(a), arXiv v1 (also Section 4.3 text) | [Xie et al. 2021](https://arxiv.org/abs/2105.15203) |
| 2021-07-01 | CSWin-B + UperNet (ImageNet-21k) | 78 | 640 | ImageNet-21k | 51.8 | single-scale | Table 6 (CSWin-B†), arXiv v1 | [Dong et al. 2021](https://arxiv.org/abs/2107.00652) |
| 2021-07-01 | CSWin-B + UperNet (ImageNet-21k) | 78 | 640 | ImageNet-21k | 52.6 | multi-scale | Table 6 (CSWin-B†), arXiv v1 | [Dong et al. 2021](https://arxiv.org/abs/2107.00652) |
| 2021-07-13 | MaskFormer Swin-B (ImageNet-22k) | 88 | 640 | ImageNet-21k | 52.7 | single-scale | Table 1, arXiv v1 | [Cheng et al. 2021](https://arxiv.org/abs/2107.06278) |
| 2021-07-13 | MaskFormer Swin-B (ImageNet-22k) | 88 | 640 | ImageNet-21k | 53.9 | multi-scale | Table 1, arXiv v1 | [Cheng et al. 2021](https://arxiv.org/abs/2107.06278) |
| 2021-12-02 | Mask2Former Swin-B (ImageNet-22k) | 88 | 640 | ImageNet-21k | 53.9 | single-scale | Table V (appendix), arXiv v1 | [Cheng et al. 2021](https://arxiv.org/abs/2112.01527) |
| 2021-12-02 | Mask2Former Swin-B (ImageNet-22k) | 88 | 640 | ImageNet-21k | 55.1 | multi-scale | Table V (appendix), arXiv v1 | [Cheng et al. 2021](https://arxiv.org/abs/2112.01527) |
| 2022-10-04 | MOAT-2 + DeepLabv3+ (ImageNet-22k, 641 px) | 73.4 | 641 | ImageNet-21k | 54.7 | single-scale | Table 4 (MOAT-2†, 641 px), arXiv v1 | [Yang et al. 2022](https://arxiv.org/abs/2210.01820) |
| 2023-03-20 | EVA-02-B + UperNet | 86 | 512 | ImageNet-21k images (MIM) + EVA-CLIP teacher (image-text) | 55.3 | single-scale | Table 15(a), arXiv v1 | [Fang et al. 2023](https://arxiv.org/abs/2303.11331) |
| 2023-09-02 | RevColV2-B + Mask2Former (not a record) | 88 | — | ImageNet-1k (MIM) + ImageNet-21k (supervised) | 54.9 | single-scale | Table 3 ('RevColV2-B+M2F') | [Han et al. 2023](https://arxiv.org/abs/2309.01005) |
| 2023-09-02 | RevColV2-B + Mask2Former | 88 | — | ImageNet-1k (MIM) + ImageNet-21k (supervised) | 55.8 | multi-scale | Table 3 ('RevColV2-B+M2F') | [Han et al. 2023](https://arxiv.org/abs/2309.01005) |
| 2023-10-19 | PlainSeg BEiT-B | — | 640 | BEiT weights (data not restated in the paper) | 55.7 | single-scale | Table 3 of arXiv v1 (mIoU(SS) column); Section 4.1; Appendix Table 8 | [Hong et al. 2023](https://arxiv.org/abs/2310.12755v1) |

### ADE20K, frozen self-supervised backbone + linear head (`frozen_ssl`)

| Date | Model | Params (M) | Test px | Pretraining | mIoU | Protocol | Where | Source |
|---|---|---|---|---|---|---|---|---|
| 2021-04-29 | DINO ViT-B/8 (measured by Oquab et al.) | 85 | 512 | ImageNet-1k (self-supervised) | 31.8 | linear probe | Table 10 ('lin.') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2021-04-29 | DINO ViT-B/16 (measured by Zhou et al., iBOT) | 85 | 512 | ImageNet-1k (self-supervised) | 34.5 | linear probe | Table 6 ('Seg.†', linear head) and Table 14 ('Seg. w/ Lin.') of the iBOT paper, arXiv v1 and v3 | [Zhou et al. 2021](https://arxiv.org/abs/2111.07832) |
| 2021-04-29 | DINO ViT-B/8 (measured by Oquab et al.) | 85 | 640 | ImageNet-1k (self-supervised) | 35.2 | linear probe + ms | Table 10 ('+ms') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2021-11-15 | iBOT ViT-B/16 | 85 | 512 | ImageNet-1k (self-supervised) | 38.3 | linear probe | Table 6 ('Seg.†', linear head) and Table 14 ('Seg. w/ Lin.'), arXiv v1 and v3 | [Zhou et al. 2021](https://arxiv.org/abs/2111.07832) |
| 2023-04-14 | DINOv2 ViT-B/14 | 86.58 | 512 | LVD-142M (self-supervised; distilled from DINOv2 ViT-g/14) | 47.3 | linear probe | Table 10 ('lin.') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2023-04-14 | DINOv2 ViT-B/14 | 86.58 | 640 | LVD-142M (self-supervised; distilled from DINOv2 ViT-g/14) | 51.3 | linear probe + ms | Table 10 ('+ms') | [Oquab et al. 2023](https://arxiv.org/abs/2304.07193) |
| 2025-08-13 | DINOv3 ViT-B/16 (distilled) | 86 | 512 | distilled from DINOv3 7B: LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) | 51.8 | linear probe | Table 14 (distilled models) | [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104) |
| 2026-03-23 | DINOv3 ViT-B/16, probed by Berreby et al. (512 px) (not a record) | 86 | 512 | distilled from DINOv3 7B: LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) | 47.19 | linear probe | Table 5 of arXiv v1 (Table 9 of v2), 'Passive-vision comparison: ADE20K mIoU at t = 0' (printed 47.2); unrounded value from the paper's export in the CanViT repo, site/assets/paper/ade20k_seg.json, probe_table row input_px 512 | [Berreby et al. 2026](https://arxiv.org/abs/2603.22570) |

## Active models: every network at test time

Parameters of every network an active, select-once or borderline model runs at test time, as `params_detail` records
them. For ADE20K entries, `params_m` excludes the segmentation decoder and `params_total_m` (in parentheses) includes
it, matching the backbone convention of the passive ADE20K points. Teachers and critics used only in training are
named but not summed. No active paper in the file states a parameter count for its full system except Saccader (Table
Supp.1), DRAM as reimplemented in Saccader (Table Supp.1) and TNet (Table 1); the other counts come from the official
code or the stated architecture, as the last column says.

| Benchmark | Model | Value | Params (M) | Networks at test time | Image px | Glimpse px × count | Inference compute | Where the count comes from |
|---|---|---|---|---|---|---|---|---|
| IN-1k | DRAM, reimplemented by Elsayed et al. (8 glimpses of 77 px) | 67.5 | 45.61 | glimpse network ResNet-v2-50 on channel-stacked multi-resolution crops + classification LSTM (1024 units) + location LSTM (1024 units) + FC 1024 + FC 2 (next location) = 45,610,219 in total; per-part counts not stated | 224 | 77 × 8 | 5.58 B FLOPs per image at 8 glimpses, counting multiplications only, measured by Papadopoulos et al. (TNet paper Table 1; Appendix B.3 Eq. 13) | Saccader paper (arXiv 1908.07644v3) Table Supp.1: DRAM 45,610,219; TNet paper Table 1: 45.61M; components from Saccader Appendix C, DRAM model |
| IN-1k | Saccader (Saccader-NASNet, 331 px images) | 75.03 | 124.54 | Saccader 35,583,913 (BagNet-77-lowD representation network 20,628,393 + attention network, what/where mixing conv and Saccader cell 14,955,520 by difference) + NASNet classifier 88,953,851 (by difference) = 124,537,764 | 331 | 113 × 6 | — | Saccader paper (arXiv 1908.07644v3) Table Supp.1: Saccader-NASNet 124,537,764, Saccader 35,583,913, BagNet-77-LowD 20,628,393; NASNet part and attention part computed by difference |
| IN-1k | GFNet (ResNet-50, 96 px patches, T=5) | 75.93 | 103.63 | global (glance) encoder ResNet-50 without fc 23.51M + local (focus) encoder ResNet-50 without fc 23.51M + classifier GRU(2048->1024) + FC 1000 10.47M + patch-proposal network 46.15M (FC 18432->2048 37.75M, FC 2048->1024 2.10M, GRU 1024 6.30M, FC 2) = 103.63M; critic FC (1,025 parameters) training only | 224 | 96 × 5 | — | counted with uv-run torch by instantiating the official classes (github.com/blackfeather-wang/GFNet-Pytorch @ 8a6775c: models/resnet.py resnet50, network.py ActorCritic and Full_layer with configs.py resnet50: policy_hidden_dim 1024, fc_hidden_dim 1024, state_dim 2048*ceil(96/32)^2); not stated in the paper, whose Appendix A.1 describes the GRUs (1024 units, conv layer removed in pi for ResNets) |
| IN-1k | GFNet (EfficientNet backbones, T=4), read from figure | 79.8 | 37.41 | global (glance) encoder EfficientNet-B3 with its classifier 12.23M (its classifier gives the glance prediction) + local (focus) encoder EfficientNet-B3 without classifier 10.70M + cascaded linear classifiers fc_2..fc_4 13.83M + patch-proposal network 0.65M (1x1 conv 1536->32, FC 800->256, GRU 256, FC 2) = 37.41M for 144 px patches (37.33M for 128 px patches) | 300 | 144 × — | about 0.75 G multiply-adds per image on average (x-position of the point in Figure 4(c), read from the PDF vector data) | counted with uv-run torch by instantiating the official classes (GFNet-Pytorch @ 8a6775c: models/gen_efficientnet efficientnet_b3, network.py with configs.py efficientnet_b3); not stated in the paper. That the top point belongs to the EfficientNet-B3 GFNet with 144 px patches is inferred: it lies on the highest-budget of the three GFNet curves of Figure 4(c), and appendix Table 2 lists B2-128, B3-128, B3-144 |
| IN-1k | GF-EfficientNet-B2 (TPAMI version, 128 px patches, T=4) | 77.93 | 30.06 | global (glance) encoder EfficientNet-B2 with its classifier 9.11M + local (focus) encoder EfficientNet-B2 without classifier 7.70M + cascaded linear classifiers fc_2..fc_4 12.68M + patch-proposal network 0.57M (1x1 conv 1408->32, FC 512->256, GRU 256, FC 2) = 30.06M | 260 | 128 × 4 | — | counted with uv-run torch by instantiating the official classes (GFNet-Pytorch @ 8a6775c: efficientnet_b2, network.py, configs.py efficientnet_b2); not stated in the paper |
| IN-1k | MS-GFNet (EfficientNet-B3, T=4), read from figure | 80.0 | — | two EfficientNet-B3 encoders + cascaded linear classifiers + patch-proposal network adapted to several patch sizes; MS-GFNet code is not released; the paper says its parameter count is approximately identical to GFNet's (Sec. 5.2.5); the GFNet EfficientNet-B3 count is 37.3-37.4M | — | — × — | about 0.75 G multiply-adds per image on average (x-position of the top point in Figure 10(b) of v2 / Figure 8(b) of v1, read from the PDF vector data) | not stated as a number; TPAMI Sec. 5.2.5 ("the number of parameters is approximately identical", said of the ResNet-50 MS-GFNet); approximation from the official GFNet EfficientNet-B3 count |
| IN-1k | STAM (DeiT-B-distilled, 27 glimpses of 32 px) | 80.78 | 100.5 | core DeiT-B distilled with classifiers G and D (class and distillation heads) 87.34M + actor MLP 13.13M (3 x {FC-BN-ReLU}, 2304->2048->2048->2048, then FC->1) + glimpse-location embeddings 0.04M (49 x 768) = 100.50M; critic MLP 1.32M and teacher DeiT-B distilled 87.34M used only in training | 224 | 32 × 27 | — | computed: core counted with uv-run timm 1.0.30 deit_base_distilled_patch16_224 (DeiT paper Table 5: 87M); actor and critic from the released STAM code (STAMVisionTransformer.set_mode, run_imagenet.sh: mlp_layers 4, mlp_hidden_dim 2048, actor input 3 x embed_dim) with embed_dim 768; the release only registers the DeiT-Small agent, the paper gives the MLP form (Sec. 5) and the Base core (A.3); no count is stated in the paper |
| IN-1k | Liu et al. (learned saccades, 4x4 fovea action space) | 73.7 | 55.75 | reconstruction model 0.02M (3 ConvLSTM layers 3->8->16->3 and a 3x3 conv) + actor ResNet18 with 16 outputs 11.18M + evaluator ResNet-101 44.55M = 55.75M; critic ResNet18 11.18M training only. With the released code's 4x4 actor (BasicBlock [3,4,6,3], ResNet-34 layout, 21.29M) the sum is 65.86M | 224 | 56 × 5 | — | computed: paper names ResNet18 actor/critic (Sec. 3.4) and ResNet101 evaluator (Sec. 4.1); counted with uv-run torchvision (16-way head for the 4x4 action space); ConvLSTM 3x3 kernels from the released code (github.com/jliu206/Foveal_Saccadic_Vision_System @ 90a5f9f), the paper gives only the channel widths (A.1) |
| IN-1k | FALcon (evaluated by Mukherjee, Ibrayev & Roy) | 72.97 | 40.57 | localizer VGG16 15.01M (conv layers 14.71M + fovea-control MLP 0.16M + saccade-control MLP 0.13M) + classifier ResNet-50 25.56M = 40.57M | 256 | 96 × — | — | architectures stated in FALcon paper App. B.2 and Mukherjee et al. Sec. 5.1 (VGG16 localizer, ResNet50 classifier); counted with uv-run torch from the FALcon code (github.com/TimurIbrayev/FALcon @ 89212d0, FALcon_models_vgg.VGG with FALcon_config_test_as_WSOL) and torchvision resnet50 |
| IN-1k | AdaGlimpse (ViT-B, 14 glimpses of 32 px) | 77.54 | 86.88 | glimpse encoder ViT-B/16 (12 blocks, 768-d, layer scale) 85.67M + linear classifier 0.77M + SAC actor (policy) 0.45M = 86.88M; training only: SAC critic 0.51M per Q-network copy, teacher DeiT III ViT-B/16 86.43M (README checkpoint deit_3_base_224_21k.pth). Fixed sin-cos position table 0.15M not counted. | 224 | 32 × 14 | — | not stated in the paper (Sec. 4 and appendix Table 2 give ViT-B: 12 blocks, 768-d; RL hidden 256); counted from the official code (apardyl/AdaGlimpse @ e5f37fd, architectures/mae.py and rl/actor_critic.py) |
| IN-1k | Prisadnikov et al. (ViT-B size, multi-zoom patches, GRPO policy) | 65.0 | — | main transformer 'compatible in size with the ViT-Base models' (12 layers, 768-d, 12 heads; MLP width not stated) + position-embedding MLP (3 -> 768; hidden size not stated) + task head MLP (size not stated) + policy: DETR-like transformer with K+1 queries cross-attending to the 16 state vectors (depth, width and K not stated); total not computable. | — | 16 × 8 | — | not stated: A.3.1-A.3.3 give the main transformer's depth and width only; no code released |
| IN-1k | AdaptiveNN-DeiT-S | 82.2 | 89.41 | glance net = DeiT-S patch embedding + blocks 1-8 on the 112 px glance 14.51M; glance classifier = DeiT-S blocks 9-12 7.1M; fixation net = full 12-block DeiT-S on 112 px fixations 21.61M; task heads after fixations 1-4 = 4 x (4 DeiT-S blocks) 28.39M + 5 LayerNorm/linear 384->1000 classifiers 1.93M; policy network 7.02M; value network (decides when to conclude) 7.02M; feature reuse/update modules 1.84M; sum 89.41M (82.39M without the value network). Not counted: unused glance-net and fixation-net classifier heads (0.77M; the fixation-net head serves a training regulariser). No teacher. | 288 | 112 × 3.15 | 4.85 GFLOPs per image at 3.15 average fixations; 5.25 GFLOPs at 3.40 (Supplementary Data Tab. 2) | not stated in the paper (Sec. 5.3.2 describes the components); counted from the official code (LeapLabTHU/AdaptiveNN @ 020bf8d, models/dynamic_deitS.py with the repo's modified timm, README evaluation arguments) |
| IN-1k | Prisadnikov et al. (ViT-S, self-supervised, frozen + linear probe, learned policy) | 75.0 | — | main transformer ViT-S (12 blocks, 384-d, 6 heads, 8 state tokens; DINOv2 implementation) + position-embedding MLP on (x, y, z) (size not stated) + linear head on the first state token (384 x 1000 + 1000 = 0.39M, computed) + gaze policy: 6-block transformer with N_mixture + 1 queries (width not stated); total not computable. Training only: DINO EMA teacher (copy of the student) and 3-layer DINO projection head to 65,536 prototypes. | — | 16 × 8 | — | not stated: Sec. 4 gives the ViT-S shape and a 6-block policy without width; no code released |
| IN-1k | PatchDrop (ResNet-50 classifier, ~7.9 of 16 patches) | 76.0 | — | policy "ResNet10" on the 56 px image (count not stated; 4.91M if one BasicBlock per stage in torchvision layout) + HR classifier ResNet-50 on the sampled 56 px patches of the 224 px image (25.56M if torchvision-standard) + LR classifier ResNet-50 initialised from the HR one (25.56M); about 56.0M under these readings | 224 | 56 × 7.9 | — | not stated in the paper (Sec. 5.1 names ResNet10 and ResNet50 only); the released code (github.com/ermongroup/PatchDrop @ 35b45e7) builds BasicBlock [3,4,6,3] classifiers and a BasicBlock [2,2,2,2] agent that fail on a 224 px input (checked), so it does not settle the count |
| IN-1k | TNet (BagNet-77, 5 attended locations) | 74.62 | 21.86 | feature extraction (BagNet-77 variant) ~20.04M + location module ~0.79M + positional-encoding FC ~0.52M + classification FC 0.51M = 21.86M (total stated; module split computed from Table 4 shapes and BagNet-77's 20.55M) | 224 | 77 × 5 | 10.84 B FLOPs per image at 5 locations, counting multiplications only (Table 1; Appendix B.3 Eq. 13) | TNet paper Table 1 (21.86M, both arXiv versions); module split computed from Table 4 |
| IN-1k | LookWhere ViT-B (128 of 256 patches) | 83.0 | 110.72 | selector = first 3 blocks of DINOv2 ViT-B/14 with 4 registers on a 154 px view 21.82M + selector head MLP (768 -> 3072 -> 16) 2.41M; extractor = 12-block ViT-B/14-reg4 at 224 px 85.72M (its own cls/register tokens, 3,840 parameters, unused); linear classifier 0.77M; sum 110.72M. Training only: DINOv2 ViT-B/14-reg4 teacher (distillation pretraining). | 224 | 14 × 128 | 14.8 GFLOPs per image (Table 1, ViT-B) | not stated in the paper; counted from the official code (antofuller/lookwhere @ f75d035, modeling.py: timm vit_base_patch14_reg4_dinov2, selector depth 3 at 154 px, SelectorHead) with timm 1.0.30 |
| IN-1k | LookThere (EVA init, 92% of patches dropped) | 83.0 | — | extractor: ViT-B scale per Sec. 4.1 ('At ViT-B scale'; count not stated; a DINOv2 ViT-B/14-reg4 at 518 px is 86.58M by timm) + selector = first 3 layers of the same initialisation on a 154 px (P = 14) or 160 px (P = 16) view (about 21.8M for DINOv2 ViT-B/14-reg4) + selector logit head and hierarchical-selection coefficients (architecture not stated) + linear task head; total not computable from the paper, lower bound about 108M if the extractor is a ViT-B/14. No teacher (selection learned with GRPO); initialisation: EVA [49] = Fang et al., EVA (CVPR 2023); the paper does not name the EVA checkpoint or its size. | 518 | — × — | — | not stated; no code link in the paper. Sec. 3.4 (selector depth L_L = 3, R_L = 154/160 px) and Sec. 4.1 ('At ViT-B scale') give the shape only |
| IN-1k | FoveaTer (DeiT-S, Type-B, up to 3 fixations, dynamic stop) | 78.31 | 22.05 | DeiT-Small (convolutional patch embedding, 12 blocks of 384-d; 6 blocks on full-resolution features, foveation pooling, 6 blocks on 22 pooled vectors plus retained foveal features; classifier) 22.05M; the foveation module (average pooling) and the attention-based fixation guidance add no learned parameters per Sec. 3.1-3.2; no policy network, no teacher. | 224 | — × — | 489 images/s throughput (Table 2 of arXiv v2; no FLOPs reported) | not stated in arXiv v2 (v3's '24M parameters' refers to its different Patchconvnet model); computed as DeiT-Small from timm deit_small_patch16_224 (22.05M), since v2 describes DeiT-Small plus parameter-free pooling |
| IN-1k | FOVI-ViT-H+ (DINOv3, 3 random fixations) | 85.3 | 843.97 | DINOv3 ViT-H+/16 backbone (32 blocks, 1280-d) 840.51M (timm vit_huge_plus_patch16_dinov3; FOVI states no count) with its patch embedding replaced by a kNN-convolution initialised from it + LoRA r = 8 on every weight matrix of blocks 1-16 3.44M and on the patch embedding 0.02M (computed from Appendix: r = 8, all weight matrices, first half of the network) = 843.97M; + classifier (not described; a linear 1280 -> 1000 head would add 1.28M). No policy network: fixations are random near the centre. | 256 | 64 × 3 | 175.30 GFLOPs per image for 3 fixations (Table 1 of arXiv v2; 175.26 in v1) | not stated in the FOVI paper; backbone counted from timm 1.0.30 vit_huge_plus_patch16_dinov3; LoRA computed from the Appendix (r = 8, all weight matrices) and Sec. 6.2 (patch embedding and first half of the network) |
| ADE20K | AME (MAE-initialised ViT-L) | 24.4 | 303.1 (348.54 with decoder) | encoder ViT-L/16 (24 blocks, 1024-d, 16 heads, MLP 4096) 303.1M (params_m) + decoder (8 blocks, 512-d, linear head to 150 x 16^2 outputs) 45.44M = 348.54M (params_total_m); no separate policy network: the next glimpse is the argmax of the entropy of the last decoder block's attention map; no teacher. | 128 | 48 × 8 | — | architecture stated in Sec. 4.1 and the supplementary (repo supplementary.pdf, Sec. 1.1); count not stated, counted from the official code (apardyl/AME @ ff77f27, mae_vit_large_patch16, image 128 x 256, 150 classes) |
| ADE20K | AME (SETR-initialised ViT-L) | 27.6 | 303.1 (348.54 with decoder) | encoder ViT-L/16 (24 blocks, 1024-d, 16 heads, MLP 4096) 303.1M (params_m) + decoder (8 blocks, 512-d, linear head to 150 x 16^2 outputs) 45.44M = 348.54M (params_total_m); no separate policy network: the next glimpse is the argmax of the entropy of the last decoder block's attention map; no teacher. | 128 | 48 × 8 | — | architecture stated in Sec. 4.1 and the supplementary (repo supplementary.pdf, Sec. 1.1); count not stated, counted from the official code (apardyl/AME @ ff77f27, mae_vit_large_patch16, image 128 x 256, 150 classes) |
| ADE20K | AdaGlimpse (ViT-B, 4 glimpses of 48 px) | 22.7 | 86.11 (117.08 with decoder) | glimpse encoder ViT-B/16 85.67M + SAC actor (policy) 0.45M = 86.11M (params_m); + MAE-style segmentation decoder (8 blocks, 512-d, plus 4-conv upscaling head to 150 classes) 30.97M = 117.08M (params_total_m); training only: SAC critic 0.51M per Q-network copy, teacher DeepLabV3 ResNet-101 58.66M (150 classes). | 224 | 48 × 4 | — | not stated in the paper; counted from the official code (architectures/mae.py decoder_type 'segment', 150 classes; rl/actor_critic.py) |
| ADE20K | AdaGlimpse (ViT-B, 8 glimpses of 48 px) | 25.7 | 86.11 (117.08 with decoder) | glimpse encoder ViT-B/16 85.67M + SAC actor (policy) 0.45M = 86.11M (params_m); + MAE-style segmentation decoder (8 blocks, 512-d, plus 4-conv upscaling head to 150 classes) 30.97M = 117.08M (params_total_m); training only: SAC critic 0.51M per Q-network copy, teacher DeepLabV3 ResNet-101 58.66M (150 classes). | 224 | 48 × 8 | — | not stated in the paper; counted from the official code (architectures/mae.py decoder_type 'segment', 150 classes; rl/actor_critic.py) |
| ADE20K | LookWhere (k=1024 of 1369 patches, 518 px) | 44.6 | 110.81 (110.92 with decoder) | selector (3 blocks of ViT-B/14-reg4 at 154 px) 21.82M + selector head 2.41M + extractor ViT-B/14-reg4 at 518 px 86.58M = 110.81M (params_m); + linear head 768 -> 150 on interpolated patch tokens 0.12M = 110.92M (params_total_m). Training only: DINOv2 teacher. | 518 | 14 × 1024 | 32.8 GFLOPs per image (Table 2) | not stated in the paper; counted from the official code (modeling.py) with timm 1.0.30 at 518 px |
| ADE20K | LookThere (DINOv2 init, 84% of patches dropped) | 42.0 | — | extractor: ViT-B scale per Sec. 4.1 ('At ViT-B scale'; count not stated; a DINOv2 ViT-B/14-reg4 at 518 px is 86.58M by timm) + selector = first 3 layers of the same initialisation on a 154 px (P = 14) or 160 px (P = 16) view (about 21.8M for DINOv2 ViT-B/14-reg4) + selector logit head and hierarchical-selection coefficients (architecture not stated) + linear task head; total not computable from the paper, lower bound about 108M if the extractor is a ViT-B/14. No teacher (selection learned with GRPO); initialisation: DINOv2 [26]; ViT-B scale per Sec. 4.1. | 518 | 14 × — | — | not stated; no code link in the paper. Sec. 3.4 (selector depth L_L = 3, R_L = 154/160 px) and Sec. 4.1 ('At ViT-B scale') give the shape only |

## Where each size, resolution and pretraining was read

### ImageNet-1k (passive)

| Model | Value | Params (M) | Where params | Test px | Where test px | Pretraining (where) | Size |
|---|---|---|---|---|---|---|---|
| AlexNet, 5-CNN ensemble | 61.9 | — | not stated for the ensemble: abstract and Section 4 give 60M for the single CNN; Section 6 calls the members 'five similar CNNs' without per-member counts | 224 | Section 4.1: test prediction averages ten 224x224 patches cut from 256x256 images | none (Table 2 caption and Section 6: only the asterisked models were pretrained on ImageNet Fall 2011) | large |
| AlexNet, single CNN | 59.3 | 60 | Abstract and Section 4: '60 million parameters' | 224 | Section 4.1: five 224 x 224 patches and their reflections | none (Table 2: rows without asterisk are trained on ILSVRC-2012 only) | base |
| ZFNet, 6-convnet ensemble | 64.0 | — | not stated in the paper | 224 | Fig. 3 caption: 'A 224 by 224 crop of an image ... is presented as the input'; Section 3: 10 sub-crops of 224x224 from the 256x256 centre | none (Table 2 caption: only the asterisked (Krizhevsky) rows used ImageNet 2011) | large |
| ZFNet, single convnet (Fig. 3 architecture) | 61.6 | 62.4 | computed: not stated in the paper; Fig. 3 layers with the dense connections stated in Section 3 give 62,357,608 (conv 3.73M + fc 58.63M) | 224 | Section 3: 10 different sub-crops of size 224x224 | none (Section 3: trained on the ImageNet 2012 training set) | base |
| Howard 2013, single convnet (Krizhevsky architecture, new train/test transforms) | 62.5 | 60 | not stated in this paper: Table 1 says 'using the architecture of Krizhevsky et al', which the AlexNet paper states has 60 million parameters | 224 | Sections 2-3: 224x224 crops; scales 256, 228 and 284 of the shorter side; three square views | none (not stated as such: the paper trains on the ILSVRC 2012 training images and mentions no other data) | base |
| OverFeat, 7 accurate models | 66.04 | 1008 | computed: 7 x 144M; v4 Table 4 gives the 'accurate' model 144M parameters (v2 calls the same model 'big'; v2 has no parameter table) | — | multi-scale dense evaluation over 4 of the 6 input sizes in v4 Table 5 / v2 Table 2 (245x245 up to 461x569); no single test size | none (Section 3: trained on the ImageNet 2012 training set) | large |
| SPP-net (ZF-5 convolutional layers, 4-level pyramid, multi-size training) | 65.84 | 77 | computed: not stated in the paper; ZF fast conv layers 96(7x7), 256(5x5), 384(3x3), 384(3x3), 256(3x3), 50-bin pyramid (12,800-d) -> fc 4096 -> 4096 -> 1000 gives 77,037,672 with dense convolutions; the fc layers alone are 73.3M | 224 | Section 3.1 and Table 2: 224x224 crops from images with smaller side 256, plus full-image views at min(w,h)=256 | none (Section 3.1: trained on the 1000-category ImageNet 2012 training set) | base |
| VGG (2-net ensemble, D+E) | 76.3 | 282 | computed: Table 2 parameter counts D 138M + E 144M | — | multi-crop and dense evaluation at test scales Q in {256, 384, 512} (Section 4.3); no single test size | none (Table 7 caption: 'Only the results obtained without outside training data are reported') | large |
| PReLU-net (model C, single model) | 78.41 | 330.6 | computed from Table 3 (model C: 7x7 conv 96; 6 conv 3x3 at 384, 768 and 896 channels; SPP 63 bins; fc 4096, 4096, 1000) = 330.62M with biases and channel-wise PReLU coefficients (count.py/count.log); the paper states no count | — | multi-scale dense ('multi-view testing on feature maps', Testing section); Table 4 lists scales 256, 384, 480 for model A; the scales behind Table 6's model-C result are not listed | none (Section 4: ImageNet 2012 training set) | large |
| BN-Inception, 6-model ensemble | 79.9 | — | not stated for the ensemble: Section 4.2 gives 13.6M for the Inception network; the six members are BN-x30 variants ('increased initial weights', dropout changes, per-activation BN) with no stated counts; 6 x 13.6M = 81.6M before BN's per-feature-map gamma and beta | 224 | Figure 4 table: Resolution 224, Crops 144, Models 6 | none (Section 4.2.2: 'all trained on the LSVRC2012 training data') | base |
| Inception-v3, 4-model ensemble | 82.8 | — | not stated for the ensemble: abstract says each network uses 'less than 25 million parameters', so four networks total under 100M; timm inception_v3 (port of the released TF weights) counts 23.83M, i.e. 95.3M for four (count2.log), not from the paper | 299 | Table 1: input 299x299x3; Table 5: 144 crops | none (evaluated on ILSVRC 2012; no extra data mentioned) | base |
| ResNet-152 (single model) | 80.62 | 60.19 | counted from torchvision 0.29.1 resnet152 (60,192,808; count.log); the paper states FLOPs (11.3B) but no parameter count; torchvision puts the stride in the 3x3 conv, which does not change the count | — | Section 3.4: fully-convolutional multi-scale testing, shorter side in {224, 256, 384, 480, 640}; no single test size | none (trained on the 1.28M ImageNet training images) | base |
| Inception-v4 + 3x Inception-ResNet-v2 ensemble | 83.5 | — | not stated in the paper; timm ports of the released TF-slim weights count Inception-v4 42.68M and Inception-ResNet-v2 55.84M (count2.log), i.e. about 210M for the ensemble | 299 | Figures 9 and 15 schemas: input 299x299x3; Table 5: 144 crops/dense evaluation | none (Table 5: ILSVRC 2012 validation, no extra data mentioned) | large |
| SENet (named SENet-154 in later versions) | 82.72 | 115.09 | counted from timm 1.0.30 legacy_senet154 (115,088,984; port of the official SENet-154 release); neither v1 nor v4 states a count | 320 | v1 Table 3 and text: 320x320 centre crop, single crop | none (v1: trained on ImageNet 2012; v4 Table 9 marks no extra data) | large |
| PNASNet-5 (N=4, F=216) | 82.9 | 86.1 | v1 Table 3 (Params 86.1M) | 331 | v1 Table 3: image size 331x331, single crop | none (ImageNet only) | base |
| GPipe AmoebaNet-B (6, 512) | 84.3 | 557 | v1 Table 2 (557M) | 480 | v1 Table 2: image size 480x480 | none (ImageNet 2012 only) | large |
| PNASNet-5-Large evaluated at 500 px with a fine-tuned GeM pooling exponent (MultiGrain appendix) | 83.6 | 86.1 | not stated in MultiGrain: PNASNet-5 (N=4, F=216) is 86.1M in Table 3 of the PNASNet paper (arXiv 1712.00559v1); GeM adds one scalar | 500 | 'Input size and cropping' paragraph and Table E.1 caption: for s* > 224 the largest side is resized to s*, no crop (500 px is the longer side) | none (Appendix E: off-the-shelf ImageNet networks from Cadene/pretrained-models.pytorch) | base |
| EfficientNet-B7 | 84.4 | 66 | Table 2 (66M), all versions | 600 | not stated in the paper; 600 from the official efficientnet_builder.py (github.com/tensorflow/tpu @ 7dc921f: 'efficientnet-b7': (2.0, 3.1, 600, 0.5)); also FixRes v2 Table 2 lists 600/600 | none (Section 5.2: ImageNet only) | base |
| EfficientNet-B7 with RandAugment | 85.0 | 66 | not stated in RandAugment: EfficientNet-B7 is 66M in Table 2 of the EfficientNet paper (arXiv 1905.11946) | 600 | Appendix: 'for EfficientNet B7 it was 600 by 600' | none (not stated as such: the ImageNet experiments use the ImageNet training set and mention no other data) | base |
| AdvProp EfficientNet-B8 | 85.5 | 88 | v1 Section 5 text ('829M vs. 88M') | 672 | not stated in the AdvProp paper (v1, v2); 672 from the official efficientnet_builder.py ('efficientnet-b8': (2.2, 3.6, 672, 0.5)); FixEfficientNet Table 2 also lists 672/672 for AdvProp B8 | none (v1 Section 1 and 5: 'without any extra data') | base |
| FixEfficientNet-B8 | 85.7 | 87.4 | Table 2 (#params 87.4M) | 800 | Table 2: FixEfficientNet test res 800 (train 672) | none (Table 2 caption: 'without external data'; starts from AdvProp B8) | base |
| NFNet-F6 + SAM | 86.5 | 438.4 | Table 3 (438.4M) | 576 | Table 1: F6 train 448px, test 576px | none (Section 4 text: 'without extra data') | large |
| CaiT-M48 ↑448 Υ | 86.5 | 356 | v2 Table 5 (356M) | 448 | v2 Table 5: train 224, test 448 | none (v2 Table 5 caption: 'models trained without external data'; hard-distillation teacher RegNetY-16GF (84M, 82.9%) trained by the authors, listed in the same table) | large |
| Refined-ViT-L↑448 | 85.9 | 81 | Table 7: 81M | 448 | Table 7: test size 448 | none (Table 7 caption: 'All models are trained without external data') | base |
| VOLO-D5 ↑512 | 87.1 | 296 | v1 Table 4 (296M) | 512 | v1 Table 4: train 224, fine-tune/test 512 | none (v1 Table 4 caption: 'All models are trained without external data'; token-labeling annotator per LV-ViT (arXiv 2104.10858 v3, Section 4): NFNet-F6 trained on ImageNet (86.3%), 576x576, 'without extra data') | large |
| VOLO-D3↑448 | 86.3 | 86 | Table 4: 86M (Table 3: 86.3M) | 448 | Table 4: test size 448 | none (Table 4 caption: 'All models are trained without external data'; LV-ViT (arXiv 2104.10858) 'Re-labeling' paragraph of v1 and training section of v3: annotator NFNet-F6 trained on ImageNet, no JFT-300M or ImageNet-22K) | base |
| MAE ViT-H/14 (448 px) | 87.8 | 632 | Table 11 (ViT-H 632M) | 448 | Table 3 caption and Section 5: fine-tuned with a 448 size | none (Table 3 caption: pre-training data is the ImageNet-1K training set) | large |
| dBOT ViT-H/14 (448 px) | 88.0 | 632 | not stated in dBOT; it uses 'ViT-H/14 [13]', and [13] (ViT paper, arXiv 2010.11929 v1, Table 1) gives ViT-Huge 632M | 448 | Table 2 column ViT-H448 and text 'with an image size of 448' | none (Table 2: ImageNet-1K; teachers bootstrapped from random initialisation) | large |
| MOAT-2 (512 px) | 86.5 | 73.4 | Table 2: 73.4M | 512 | Table 2: eval size 512 | none (Table 2 caption: '1K only: Using ImageNet-1K only') | base |
| Perceptual MAE (MSG-MAE, StyleGANv2-ADA-P loss) ViT-L | 88.1 | 304 | not stated for ViT-L; Section 4: 'We use the ViT-B and ViT-L architectures from the MAE paper', and MAE Table 11 gives ViT-L 304M | 224 | not stated directly; Section 4.1 contrasts with MAE ViT-H448 'with input image of size 448 rather than 224', implying 224 for their ViT-L | none (Table 2 'Pre-training Data: IN1K'; abstract: 'without use of additional pre-trained models or data') | large |
| Hiera-H (224 px) | 86.9 | 673 | Table 2 and Table 8 (673M); video tables list 672M | 224 | Table 2 caption: measured with 224x224 input; Table 8 at 224 | none (Table 8: MAE pretraining on ImageNet-1K) | large |
| ViT-5-L (384 px) | 86.0 | 305 | Table 5 (305M at 384; 304M at 224) | 384 | Table 5 and Section 4.1: 384x384 | none (Section 4.1: trained from scratch on ImageNet-1k (DeiT-III recipe)) | large |
| AlexNet, 7-CNN ensemble (2 pretrained on ImageNet Fall 2011) | 63.3 | — | not stated: 60M is the single CNN of Section 4; the two pretrained CNNs add a sixth convolutional layer whose size is not given | 224 | Section 4.1: ten 224x224 patches | ImageNet Fall 2011 (15M images, 22K classes), 2 of the 7 CNNs (Section 6 and Table 2 caption) | large |
| AlexNet, single CNN pretrained on ImageNet Fall 2011 | 61.0 | — | not stated: the 60M of the base network plus one extra convolutional layer whose size is not given; base size on any plausible layer size, but not computable | 224 | Section 4.1: 224 x 224 patches | ImageNet Fall 2011 (15M images, 22K classes) (Section 6 text and Table 2 caption) | base |
| ResNeXt-101 32x48d, IG-940M pretraining | 85.4 | 829 | Table 6 (829M) | 224 | Table 6: image size 224, single crop | IG-940M (Instagram hashtags, weakly supervised) (Table 6 row 'IG-940M-1.5k') | large |
| ResNeXt-101 32x8d, semi-weakly supervised (IG-1B-Targeted teacher) | 84.3 | 88 | Table 3: ResNext-101-32x8 88M | — | not stated in the paper (searched for crop, resolution, 224, pixel) | IG-1B-Targeted (Instagram, semi-weakly supervised) ('Datasets' paragraph and Table 6 caption) | base |
| FixRes ResNeXt-101 32x48d (IG-940M) | 86.4 | 829 | v2 Table 2 (829M) | 320 | v2 Table 2: train 224, test 320 | IG-940M (Instagram hashtags, weakly supervised) (v2 abstract: 'pre-trained in weakly-supervised fashion on 940 million public images') | large |
| Noisy Student (EfficientNet-L2), arXiv v1 | 87.4 | 480 | v1 Table 2 and Table 7 (480M) | 800 | v1 Table 7: EfficientNet-L2 train res 475, test res 800 | JFT-300M, unlabeled (v1 Table 2 '300M unlabeled images') | large |
| Noisy Student, EfficientNet-B7 student with EfficientNet-L2 teacher, arXiv v1 | 86.8 | 66 | Table 9: Noisy Student (B7) 66M; Table 7: EfficientNet-B7 66M | 600 | Table 7: B7 train 600, test 600 | JFT-300M, unlabeled (Table 2 (300M unlabeled images) and Table 9 caption) | base |
| BiT-L (ResNet152x4) | 87.76 | 930 | Section 3.3: 'ResNet-152x4 model, which has 0.93 billion trainable parameters' (NFNet and ViT-G tables list 928M) | — | not stated: Section 3.3 gives BiT-L training crops of 480x480 from 512x512 images; Section 3.1 says 'At test time, we only resize the image to a fixed size' without the size | JFT-300M (Section 3.3 / Table 2) | large |
| Noisy Student (EfficientNet-L2), arXiv v2 | 88.4 | 480 | v2 Table 2 and Table 8 (480M) | 800 | v2 Table 8: train res 475, test res 800 | JFT-300M, unlabeled (v2 Table 2 '300M unlabeled images') | large |
| Noisy Student, EfficientNet-B7 student with EfficientNet-L2 teacher, arXiv v2 | 86.9 | 66 | Table 9 of v2: Noisy Student (B7) 66M | 600 | Table 7: B7 train 600, test 600 | JFT-300M, unlabeled (Table 2 (300M unlabeled images from JFT) and Table 9 caption) | base |
| FixEfficientNet-L2 | 88.5 | 480 | Table 1 (480M) | 600 | Table 1: FixEfficientNet test res 600 (Noisy Student L2: train 475, test 800) | JFT-300M, unlabeled (Table 1 caption 'with extra training data'; text) | large |
| FixEfficientNet-B7 (Noisy Student weights) | 87.1 | 66 | Table 1: 66M | 632 | Table 1: FixEfficientNet test res 632 | JFT-300M, unlabeled (Table 1 caption: pre-trained with Noisy Student on 300M unlabeled images) | base |
| ViT-H/14 (JFT-300M) | 88.55 | 632 | Table 1 (ViT-Huge 632M) | 518 | Section 4.2: fine-tuned at 518 for ViT-H/14 with Polyak averaging | JFT-300M (Table 2 'Ours-JFT') | large |
| EfficientNet-L2 (Noisy Student) + SAM | 88.61 | 480 | not stated in the SAM paper; it fine-tunes the public NoisyStudent EfficientNet-L2 checkpoint, which Noisy Student's Table 8 lists at 480M | 475 | v2 Section 3.2: 'input resolution 475' | JFT-300M, unlabeled (v2 Section 3.2 and Table 3 caption) | large |
| Meta Pseudo Labels (EfficientNet-L2) | 90.2 | 480 | v3 Table 4 (480M) | — | not stated: v3 Section 4 gives the training resolution 512x512; no test resolution stated | JFT-300M, unlabeled (v3 Section 4 and Table 4) | large |
| NFNet-F4+ (JFT-300M) | 89.2 | 527 | Table 5 (527M) | — | not stated for NFNet-F4+: Appendix A.5 gives pretraining at 224 and fine-tuning hyperparameters but no fine-tuning/test resolution; Table 1 gives the F4 test size 512 (F4+ is a wider F4) | JFT-300M (Appendix A.5) | large |
| ViT-G/14 (JFT-3B) | 90.45 | 1843 | v1 Table 2 (G/14 1843M) | 518 | v1 appendix: '518 x 518 resolution for both ViT-g and ViT-G' fine-tuning | JFT-3B (v1 Section 2 and Table 1) | large |
| CoAtNet-2 (ImageNet-21K pretraining, 512 px) | 87.3 | 75 | Table 4: 75M | 512 | Table 4: eval size 512 | ImageNet-21k (Table 4 caption: '21K+1K denotes pre-training on ImageNet-21K and finetuning on ImageNet-1K') | base |
| CoAtNet-7 (JFT-3B) | 90.88 | 2440 | v2 Table 5 (2.44B) | 512 | v2 Table 5: eval size 512^2 | JFT-3B (v2 Table 5 caption: 'the last 3 rows use a larger dataset JFT-3B') | large |
| Model soups (greedy soup of ViT-G/14) | 90.94 | 1843 | not stated in the soups paper; the model is ViT-G/14 [102] (Zhai et al.), whose Table 2 lists 1843M | 518 | Appendix B.3.3: fine-tuned at 518x518; test images resized to 550x550 and centre-cropped to 518x518 | JFT-3B (Section 3.3.2 / Table 4 caption) | large |
| CoCa (fine-tuned) | 91.0 | 1000 | Table 1: image encoder 1B (the full model is 2.1B with the text decoder); fine-tuning uses the encoder plus an attentional pooler (Appendix A), whose size is not stated | — | not stated: Section 4 says pretraining continues 'for one epoch on a higher resolution of 576x576'; the fine-tuning table (Table 9) gives no resolution | JFT-3B + ALIGN image-text pairs (Section 4.1 Data) | large |
| BEiT v2 ViT-B/16 (ImageNet-21k intermediate fine-tuning, 384 px) | 87.5 | 86 | Table 7: 86M | 384 | Table 7: 384^2 column | ImageNet-21k (supervised) + CLIP-B/16 teacher for MIM targets ('Visual tokenizer training' paragraph (OpenAI CLIP-B/16 as VQ-KD teacher) and the text after Table 1 ('Only the intermediate fine-tuning phase uses the ImageNet-21k dataset')) | base |
| MOAT-2 (ImageNet-22K pretraining, 512 px) | 87.7 | 73.4 | Table 2: 73.4M | 512 | Table 2: eval size 512 | ImageNet-21k (Table 2 caption: '22K + 1K: ImageNet-22K pretraining and ImageNet-1K fine-tuning') | base |
| CAFormer-B36 (ImageNet-21K pretraining, 384 px) | 88.1 | 99 | Table 5: 99M | 384 | Table 5: ↑384 | ImageNet-21k (paragraph 'Pre-training on ImageNet-21K and fine-tuning on ImageNet-1K') | base |
| ViT-B/16 distilled from ViT-22B (384 px) | 88.6 | 86 | not stated in this paper: ViT-Base is 86M in Table 1 of the ViT paper (arXiv 2010.11929v1) | 384 | Table 8 caption: 'finetuned at 384 resolution' | JFT + ViT-22B teacher (JFT-4B) (Section 4.5.5 (distillation) and the 'Dataset' paragraph (JFT extended to around 4B images)) | base |
| BASIC-L image encoder (CoAtNet-7) trained with Lion, fine-tuned | 91.1 | 2400 | not stated in the Lion paper; BASIC (arXiv 2111.10050 v3) Table 5: BASIC-L image model CoAtNet-7, 2.4B | — | not stated in the Lion paper | JFT-5B + 6.6B image-text pairs (Section 4.2 text) | large |
| OmniVec (fine-tuned) | 92.4 | — | not stated anywhere in the paper (Table 14 only says the ViT baselines were enlarged to match OmniVec-4's parameter count) | — | not stated | multimodal: AudioSet, Something-Something v2, Wikipedia, SUN RGB-D, ModelNet40 (Section 4 'Masked pretraining') | — |
| OmniVec2 (fine-tuned) | 93.6 | — | not stated in the CVPR paper, its supplement or the arXiv copy (2507.13364v1) | — | not stated | multimodal: AudioSet, Something-Something v2, Wikipedia, SUN RGB-D, ModelNet40 and more (CVPR Section 4 'Masked pretraining') | — |
| Split-Brain Autoencoder (AlexNet, conv3 linear probe) | 35.4 | — | not stated in the paper; the probed features come from the convolutional layers of a split AlexNet (a full AlexNet is 60M), so base size | — | not stated: Section 4.1.1 gives frozen weights and spatially resized feature maps, not the image size | none (Section 4: 'We pre-train on the 1.3M ImageNet dataset [31] (without the use of labels)') | base |
| RotNet AlexNet (conv3 linear probe) | 38.7 | — | not stated in the paper; the probed features come from AlexNet convolutional layers (a full AlexNet is 60M), so base size | — | not stated in arXiv v1 | none (Table 5 caption: the models 'were pre-trained on ImageNet without labels', except the supervised and random rows) | base |
| Instance Discrimination ResNet-50 (linear SVM probe) | 54.0 | — | not stated in the paper; a ResNet-50 is 25.6M with its classifier by torchvision's definition | — | not stated | none (unsupervised; Section 4.2 learns the features on ImageNet) | base |
| Rotation RevNet-50 (width factor 16, linear probe) | 55.4 | 85.765632 | counted from the official code (google/revisiting-self-supervised, models/resnet.py revnet50, filters_factor=16, no classifier): 85,765,632 (throwaway/history-audit/count_revnet2.py, count_revnet2.json); not stated in the paper | 224 | not stated in the paper; the official config/evaluation/rotation_or_exemplar.sh sets --crop_size 224,224 | none (self-supervised on ImageNet; Table 2's 'Ours' column is the official validation set) | base |
| Local Aggregation ResNet-50 (10-crop linear probe) | 60.2 | — | not stated in the paper; a ResNet-50 is 25.6M with its classifier by torchvision's definition | — | not stated: Section 4.2 reports 10-crop validation accuracy without the crop size | none (self-supervised on ImageNet, Section 4) | base |
| BigBiGAN RevNet-50 x4 (BN+CReLU linear probe) | 61.3 | — | not stated in BigBiGAN; the paper ties its RevNet-50 ×4 to Kolesnikov et al.'s width factor 16, which is 85,765,632 in that study's official code (throwaway/history-audit/count_revnet2.json), so base size; the BigBiGAN encoder itself was not counted | 256 | Appendix A: encoder input 256 x 256, center crop | none ('We train a BigBiGAN on unlabeled ImageNet, freeze its learned representation') | base |
| CMC ResNet-101 (two color-view encoders, linear probe) | 65.0 | 84.990912 | counted from the official code (HobbitLong/CMC @ 7b227be, models/resnet.py, resnet101v2: the encoders of both views, unused contrastive fc heads excluded): 84,990,912 (throwaway/history-audit/count_cmc2.py, count_cmc2.log); not stated in the paper | 224 | not stated in the paper; the official LinearProbing.py validates with Resize(256), CenterCrop(224) | none (self-supervised on ImageNet; Table 2 caption: an encoder for each of the two views) | base |
| PIRL ResNet-50 c2x (linear probe) | 67.4 | 98 | Appendix A of arXiv v1: PIRL-c2x has 'a total of 98 million parameters'; the same paragraph gives 25.6 million for the standard ResNet-50, a count that includes its 1000-way fc layer, so the 98M may include a classifier too | — | not stated: Figure 2 reports single-crop top-1 without the crop size | none (Appendix A: 'the ImageNet training set (1.28 million images)'; self-supervised) | base |
| SimCLR ResNet-50 (2×) | 74.2 | 94 | Table 6 of arXiv v1 (Param. 94 for ResNet-50 (2x)) | 224 | not stated in v1 (Appendix B.6 states 224x224 center crops only for the transfer and semi-supervised protocols); the official google-research/simclr README's ImageNet linear-evaluation command uses --image_size=224 | none (Table 6 caption and Section 2.3: representations learned on ImageNet without labels) | base |
| BYOL ResNet-200 (1×) | 77.8 | 63 | Table 9 of arXiv v1 ('Weights' 63M for ResNet-200 1x) | 224 | Appendix C.1 of arXiv v1: test images resized to 256 px on the shorter side, then a 224x224 center crop | none (Section 4 / Appendix C.1: trained on the ImageNet ILSVRC-2012 training set without labels) | base |
| MoCo v3 ViT-BN-B/7 | 79.5 | 86 | Table 2 of arXiv v1 (ViT-Base 86 M); Section 6.3 text: the 7x7 patch 'keeps the model size unchanged' | 224 | not stated in the paper: linear probing reports 'single-crop top-1' without the crop size; Figure 8 caption says pretraining used two 224x224 crops. The official facebookresearch/moco-v3 main_lincls.py validates with Resize(256), CenterCrop(224) (the released protocol; the BN-B/7 checkpoint itself was not matched) | none (Section 6 (Experimental Results): pre-trained in the ImageNet-1k training set; Figure 8 caption) | base |
| DINO ViT-B/8 | 80.1 | 85 | Table 2 'Param.' column (85, in millions); frozen backbone, the linear head (768x1000 plus bias, about 0.77M) is not counted | 224 | paper: 'report accuracy on a central crop' (size not stated); official eval_linear.py (facebookresearch/dino @ 7c446df): Resize(256) + CenterCrop(224) | none (Section 4: pretrained on the ImageNet dataset without labels) | base |
| EsViT Swin-B/W=14 | 81.3 | 87 | Table 1 of arXiv v1 (#Parameters 87) | 224 | not stated in the paper (Section 3 mentions 224x224 only as a typical input when discussing sequence length); the official microsoft/esvit eval_linear.py validates with Resize(256), CenterCrop(224) | none (Section 4: unsupervised pre-training on ImageNet-1K without labels) | base |
| DINOv2 ViT-g/14 | 86.5 | 1100 | Section 4 text: 'our ViT-g backbone counts 1.1B parameters'; frozen backbone, linear head not counted | 224 | Table 4 caption: 'at resolution 224 x 224 unless stated otherwise' | LVD-142M (self-supervised) (Table 4 'Data' column) | large |
| DINOv2 ViT-B/14 | 84.5 | 86.58 | counted from the official code (github.com/facebookresearch/dinov2 @ 7764ea0f, dinov2.hub.backbones._make_dinov2_model('vit_base'), 12 blocks): 86,580,480; not stated in the paper, whose Table 17 lists 18 blocks for ViT-B/14 (contradicted by the paper's own layer list {3, 6, 9, 12} for ViT-S/B and by the code) | 224 | Table 4 caption: 'at resolution 224 x 224 unless stated otherwise' | LVD-142M (self-supervised; distilled from DINOv2 ViT-g/14) (Table 4 (Data: LVD-142M); Section 5 (distillation from ViT-g); Table 15 (LVD-142M composition)) | base |
| Proteus ViT-B/14 (DINOv2-L teacher, linear probe) | 84.9 | — | not stated in the paper; a ViT-B/14 (DINOv2's ViT-B/14 is 86.58M by its official code) | — | not stated: Appendix A.1.2 follows DINOv2's linear-probing protocol without the test size | ImageNet-1k images + DINOv2-L/14 teacher (LVD-142M, self-supervised) (Section 3.1: pre-training on the ImageNet-1K training set; Table 4 caption: Proteus-B is distilled from DINOv2-L) | base |
| DINOv3 ViT-7B/16 | 88.4 | 6716 | v1 Figure 16(a) table: ViT-7B 6716M (Table 2: 6.7B); frozen backbone, linear head not counted | 512 | v1 Table 7 caption: resolution adapted to 1024 patch tokens, 512x512 for patch size 16 | LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) (v1 Section 3.1) | large |
| DINOv3 ViT-B/16, probed by Berreby et al. (512 px) | 85.0 | 86 | DINOv3 paper, Figure 16(a) (ViT-B 86M); the CanViT paper does not restate it | 512 | Table 17 caption (v1) / Table 10 caption (v2): 512x512 input resolution | distilled from DINOv3 7B: LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) (DINOv3 arXiv v1 Section 3.1 (data) and Section 5.2 / 7 (distillation from ViT-7B)) | base |

### ADE20K (passive)

| Model | Value | Params (M) | Where params | Test px | Where test px | Pretraining (where) | Size |
|---|---|---|---|---|---|---|---|
| Cascade-DilatedNet (VGG) | 34.9 | — | Not stated in arXiv v1 or the CVPR 2017 version. DilatedNet is described as a dilated fully convolutional VGG-16; its count depends on how the converted fc6/fc7 layers are kept, which the paper does not say. | 512 | Section 4 footnote 2: benchmark images larger than this are rescaled to a shorter side of 512; smaller images keep their size. The network's test input size is not stated separately. | not stated (Not stated: neither version names pretraining data (searched both texts for ImageNet, pre-train, initialization).) | — |
| Wider or Deeper Model A2 (2-convolution decoder) | 43.73 | 105.070912 | counted from the official code (itijyou/ademxapp @ 0901990, util/symbol/resnet_v2.py rna_feat_a1, convolutions and batch-norm affine parameters; classifier and running statistics excluded): 105,070,912 (throwaway/history-audit/count_wider2.py, count_wider2.json); the official misc/places_model_a2.pdf graph has the same feature topology; not stated in the paper; above 100M, base under the 110M bound | — | not stated: the 500 px crop is a training setting; Section 5: no multi-scale testing 'except for the test set of ADE20K' | ImageNet-1k + Places365 (Section 6.2: Model A2 is initialised from Model A and tuned 'with the Places 365 data'; Model A is trained on ILSVRC 2012) | base |
| PSPNet ResNet-269 | 43.81 | — | Not stated: Table 3 lists pre-trained ResNet depths 50/101/152/269 without parameter counts; ResNet-269 is not one of He et al.'s configurations. | — | Not stated: Section 5.1 says only that 'an appropriately large cropsize' helps; no crop or test size is given. | not stated (Not stated: Section 3.3 says 'we use a pretrained ResNet [13]' without naming the dataset.) | — |
| PSPNet ResNet-269 | 44.94 | — | Not stated (as for the single-scale entry). | — | Not stated; multi-scale testing ('MS', Table 3/4) without listed scales or base size. | not stated (Not stated: 'pretrained ResNet' without naming the dataset.) | — |
| PSPNet ResNet-152 | 42.62 | — | not stated in the paper; ResNet sizes counted with torchvision 0.29 definitions (no weights), evidence/base-ade/resnet_param_counts.txt: ResNet-101 42.50M and ResNet-152 58.14M without the fc layer; PSPNet's own ResNet variant is not described, so no exact count | — | not stated (the paper only says an 'appropriately large cropsize') | ImageNet-1k (Section 3 text: 'we use a pretrained ResNet model [13]' ([13] = He et al. ImageNet ResNet)) | base |
| PSPNet ResNet-152 | 43.51 | — | not stated in the paper; ResNet sizes counted with torchvision 0.29 definitions (no weights), evidence/base-ade/resnet_param_counts.txt: ResNet-101 42.50M and ResNet-152 58.14M without the fc layer | — | not stated | ImageNet-1k (Section 3 text, as above) | base |
| SAC ResNet-101 (SAC-multiple + MS) | 44.3 | — | not stated; the backbone is a standard ResNet-101 with added scale-regression layers, so no exact count; ResNet sizes counted with torchvision 0.29 definitions (no weights), evidence/base-ade/resnet_param_counts.txt: ResNet-101 42.50M and ResNet-152 58.14M without the fc layer | — | not stated | ImageNet-1k (Section 4.1: 'ResNet101 model [11] pre-trained on ImageNet dataset [7]') | base |
| EncNet ResNet-101 | 44.65 | — | not stated in the paper; a ResNet-101 is 42.5M without its classifier by torchvision's definition, and OCR v1 states about 42.7M for its dilated ResNet-101 | — | Not stated: Section 3.1 says images are cropped to a fixed size with zero padding, without the size. | not stated (Not stated: 'EncNet augments a pre-trained Deep Residual Network (ResNet) [17]' (Introduction) without naming the dataset.) | base |
| OCNet ResNet-101 (arXiv v1) | 45.08 | — | not stated in OCNet v1; the same authors' OCR paper (v1, Section 4) states 'the backbone ResNet-101 is about 42.7M' | — | not stated (training resizes images to a random length in {300, 375, 450, 525, 600}) | ImageNet-1k (Section 4.2: 'We choose the ImageNet pretrained ResNet-101') | base |
| CCNet ResNet-101 (arXiv v1) | 45.22 | — | not stated in the paper; a ResNet-101 is 42.5M without its classifier by torchvision's definition, and OCR v1 states about 42.7M for its dilated ResNet-101 | — | Not stated: Section 4.2 gives training augmentation only (short side random in {300, 375, 450, 525, 600}); no test size. | ImageNet (1k or 21k not stated) (Section 4.2 'Network Structure': 'we choose the ImageNet pre-trained ResNet-101 [16] as our backbone' (arXiv v1).) | base |
| APCNet ResNet-101 | 45.38 | — | not stated in the paper; a ResNet-101 is 42.5M without its classifier by torchvision's definition | — | not stated: Section 4.1 sets the ADE20K training crop to 576; no test size | ImageNet-1k (Section 4.1: ResNet 'pre-trained on ImageNet [29]', where [29] is the ILSVRC paper of Russakovsky et al.) | base |
| OCR HRNetV2-W48 (arXiv v1) | 45.66 | — | HRNetV2-W48 count not stated in OCR v1 (it states about 42.7M for its ResNet-101 backbone). Swin v1 Table 3 lists 'OCRNet HRNet-w48' at 71M for the whole system (an mmsegmentation reproduction), so the backbone is under 71M | 520 | Section 4: 'ADE20K: ... crop size as 520 x 520' | ImageNet-1k (Section 3.2: 'ResNet-50/ResNet-101 pretrained over the ImageNet dataset'; HRNetV2-W48 pretraining not restated) | base |
| ACNet ResNet-101 | 45.9 | — | not stated in the paper; a ResNet-101 is 42.5M without its classifier by torchvision's definition, and OCR v1 states about 42.7M for its dilated ResNet-101 | — | Not stated for ADE20K: Section 4.4 says multi-scale input and multi-scale testing; the scale list in Section 4.3 is for Cityscapes and no base size is given. | not stated (Not stated: Section 4.2 'We employ a dilated pretrained ResNet architecture' without naming the dataset.) | base |
| CPNet ResNet-101 | 46.27 | — | not stated; ResNet sizes counted with torchvision 0.29 definitions (no weights), evidence/base-ade/resnet_param_counts.txt: ResNet-101 42.50M and ResNet-152 58.14M without the fc layer (the paper's 'off-the-shelf' ResNet-101 variant is not specified) | 480 | Section 4.1: random crops of 480 x 480 for ADE20K training; test-time base size not stated | ImageNet-1k (Section 4.1: 'We adopt the ResNet [13] as our pre-trained model') | base |
| ResNeSt-101 | 46.91 | 48.3 | Table 3 (arXiv v1): ResNeSt-101 48.3M (ImageNet classification network, including its classifier). | — | Not stated: Section 6.3 gives multi-scale evaluation with flipping, no crop or base size. | ImageNet-1k (Section 6.3 transfers the backbone trained in Section 5 on 'the ImageNet 2012 dataset' (1.28M images); Table 7 caption: 'trained without coarse labels or extra data'.) | base |
| DeepLabV3 ResNeSt-200 (arXiv v2) | 48.36 | 70.2 | ResNeSt v1 Figure 1 inset table: ResNeSt-200 70.2M (the ImageNet classification model, including its classifier; v2's table rounds to 70M) | — | not stated for ADE20K (GluonCV DeepLabV3 implementation) | ImageNet-1k (Section 6.3: transfer learning from the ImageNet-trained ResNeSt) | base |
| SETR-MLA (ViT-L) | 48.64 | 307 / 310.57 system | Backbone: SETR Table 1 'T-Large' is 24 layers, hidden 1024, 16 heads, which is ViT-Large, 307M in the ViT paper's Table 1 (arXiv 2010.11929v2); SETR does not state the backbone count. Whole model: SETR Table 4, SETR-MLA #Params 310.57M. | 512 | Section 4.1 'Implementation details': random 512 crops for ADE20K in training; test uses a sliding window ('e.g., 480 x 480 for Pascal Context', whose training crop is 480) after scaling to a uniform size that is not given. | ImageNet-21k (Table 4 'Pre' column 21K; transformer initialised from ViT weights pretrained on ImageNet-21K (Section 4.1 'Pre-training').) | large |
| SETR-MLA (ViT-L) | 50.28 | 307 / 310.57 system | As for the single-scale entry (ViT Table 1; SETR Table 4). | 512 | Multi-scale test with factors 0.5-1.75 (Section 4.1, mmsegmentation default) around the 512 crop setting. | ImageNet-21k (Table 4 'Pre' 21K.) | large |
| DPT-Hybrid (ViT-Hybrid: ResNet-50 stem + ViT-B) | 49.02 | 98.181952 | backbone count not stated in the paper (Table 9 gives 123M for the whole DPT-Hybrid monocular-depth model, DPT-Base 112M; the segmentation model has a different head). Counted from the official code: dpt/vit.py _make_pretrained_vitb_rn50_384 builds timm 0.4.5 vit_base_resnet50_384, 98,950,952 minus its unused 769,000-parameter ImageNet head = 98,181,952 (throwaway/history-audit/count_dpt.py, count_dpt.log) | 480 | Section 4.2: 'We train on square random crops of size 480' | ImageNet (1k or 21k not stated) (Section 4.2: 'The encoder is again initialized from ImageNet-pretrained weights') | base |
| Swin-L + UperNet | 53.5 | 197 / 234 system | Backbone: Table 1(b), Swin-L 197M. System: Table 3, UperNet Swin-L 234M. | 640 | Appendix A2.3 (ADE20K): Swin-B/L trained with 640x640 input; multi-scale test at [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]x the training resolution. | ImageNet-21k (Table 3 '‡ indicates that the model is pre-trained on ImageNet-22K'.) | large |
| Swin-B + UperNet (ImageNet-22k) | 51.6 | 88 / 121 system | Swin-B 88M: Table 1 (ImageNet classification); system 121M: Table 3 | 640 | Appendix A2.3: 'Swin-B and Swin-L with ‡ ... trained with the input of 640x640' | ImageNet-21k (Table 3 caption: '‡ indicates that the model is pre-trained on ImageNet-22K') | base |
| SegFormer-B5 (MiT-B5) | 51.0 | 81.4 / 84.7 system | Table 1(a): encoder 81.4M, decoder 3.3M (sum 84.7M); Table 4 states 84.7M for SegFormer-B5 | 640 | Section 4.1: 'we set crop size to 640 x 640 on ADE20K for our largest model B5'; evaluation rescales the short side to the crop size | ImageNet-1k (Section 4.1: 'We pre-train the encoder on the Imagenet-1K dataset') | base |
| SegFormer-B5 (MiT-B5) | 51.8 | 81.4 / 84.7 system | Table 1(a) | 640 | Section 4.1 | ImageNet-1k (Section 4.1) | base |
| CSWin-B + UperNet (ImageNet-21k) | 51.8 | 78 / 109.2 system | CSWin-B 78M: architecture/ImageNet table in Section 3-4; UperNet system 109.2M: Table 6 | 640 | Table 6 footnote: '† means the model is pretrained on ImageNet-21K and finetuned with 640x640 resolution' | ImageNet-21k (Table 6 footnote) | base |
| CSWin-B + UperNet (ImageNet-21k) | 52.6 | 78 / 109.2 system | as above | 640 | as above | ImageNet-21k (as above) | base |
| MaskFormer Swin-B (ImageNet-22k) | 52.7 | 88 / 102 system | backbone not restated here; Swin-B is 88M in the Swin paper's Table 1; MaskFormer system 102M: Table 1 | 640 | Table 1 crop size column | ImageNet-21k (Table 1 caption: 'Backbones pre-trained on ImageNet-22K are marked with †') | base |
| MaskFormer Swin-B (ImageNet-22k) | 53.9 | 88 / 102 system | as above | 640 | as above | ImageNet-21k (as above) | base |
| SwinV2-G + UperNet (896 px test) | 59.3 | 3000 | Table 2: SwinV2-G 3.0B; Section 3.6 'Model configurations': '3 billion parameters'. The UperNet system count is not stated (ViT-Adapter v1 Table 10 lists '>3.0B'). | 896 | Table 4: trained at 640 (window 40), tested at 896 (window 56). | ImageNet-22K-ext (70M private images) (Section 3.4, Section 4.2 and Appendix A2.2: two-step pretraining on the privately collected ImageNet-22K-ext (70M images): self-supervised, then image classification.) | large |
| SwinV2-G + UperNet | 59.9 | 3000 | As above (Table 2). | 896 | Table 4: 896 (56) test size, '*' = multi-scale. | ImageNet-22K-ext (70M private images) (As above.) | large |
| Mask2Former Swin-B (ImageNet-22k) | 53.9 | 88 / 107 system | backbone: Swin-B 88M (Swin paper Table 1); system 107M: Table V | 640 | Table V crop size column | ImageNet-21k (Table V caption: 'Backbones pre-trained on ImageNet-22K are marked with †') | base |
| Mask2Former Swin-B (ImageNet-22k) | 55.1 | 88 / 107 system | as above | 640 | as above | ImageNet-21k (as above) | base |
| ViT-Adapter-L + Mask2Former (BEiT init) | 60.5 | 303.3 / 571 system | Table 1 (arXiv v1): ViT-L 303.3M; with the adapter (23.7M) ViT-Adapter-L is 327.0M. System: Table 10 of arXiv v1, Mask2Former† ViT-Adapter-L 571M. | 896 | Table 10 of arXiv v1: crop size 896x896; Appendix A.2 'we fix the crop size to 896x896 pixels'. | ImageNet-21k (BEiT-L) + COCO-Stuff (Appendix A.2: 'we adopt the ImageNet-22K pre-trained BEiT-L [4]' and 'additionally use the COCO-Stuff-164K dataset for 80k iterations of pre-training'.) | large |
| BEiT-3 + ViT-Adapter + Mask2Former | 62.0 | 1009 | Computed from Table 2 (arXiv v2): vision experts 692M + shared self-attention 317M = 1009M, the parameters active for vision; Section 2.3 says 'about 1B'. Full model 1.9B. The segmentation system's count is not stated in BEiT-3. | 896 | Table 8: crop size 896^2; Table 15: input resolution 896^2. | ImageNet-21k + 21M image-text pairs + 160GB text (Table 3 and Section 2.3 'Pretraining Data'.) | large |
| BEiT-3 + ViT-Adapter + Mask2Former | 62.8 | 1009 | As for the single-scale entry. | 896 | Table 8: crop 896^2, '+MS' column. | ImageNet-21k + 21M image-text pairs + 160GB text (Table 3.) | large |
| MOAT-2 + DeepLabv3+ (ImageNet-22k, 641 px) | 54.7 | 73.4 / 80.5 system | MOAT-2 73.4M: ImageNet table (Table 2); system 80.5M: Table 4 | 641 | Appendix A.4: inputs resized and padded to 513 x 513 or 641 x 641; this row is 641 | ImageNet-21k (Table 4 caption: '†: use ImageNet-22K pretrained weights') | base |
| InternImage-H + Mask2Former | 62.5 | 1080 / 1310 system | Backbone: Table 1, InternImage-H 1.08B. System: Table 5, InternImage-H + Mask2Former 1.31B. | 896 | Table 5: crop size 896^2; Appendix: 'with a crop size of 896'. | 427M image-text pairs (LAION-400M, YFCC-15M, CC12M) + COCO-Stuff (Section 4.1 (427M joint dataset, M3I pretraining) and the Appendix ADE20K paragraph (backbone from the 427M-joint-dataset weights; pretraining and fine-tuning on COCO-Stuff and ADE20K, 80k iterations each).) | large |
| InternImage-H + Mask2Former | 62.9 | 1080 / 1310 system | As above. | 896 | Table 5 'MS' column, crop 896. | 427M image-text pairs (LAION-400M, YFCC-15M, CC12M) + COCO-Stuff (As above.) | large |
| EVA-02-B + UperNet | 55.3 | 86 | Table 4 and Table 15: 86M encoder; system count not given | 512 | Table 15 crop column (512) and Table 34 'crop size 512 / 512 / 640' | ImageNet-21k images (MIM) + EVA-CLIP teacher (image-text) (Table 4: 'EVA-02-B  EVA-CLIP  IN-21K (14M)'; Section 3.3.2: 'we primarily evaluate pure MIM pre-trained representations') | base |
| ONE-PEACE + ViT-Adapter + Mask2Former | 62.0 | 1520 | Table 3: 'Enc. #Params' 1.52B (consistent with Table 1: V-Adapter 3.4M + V-FFN 1.15B + shared attention 378M). The segmentation system's count is not stated in ONE-PEACE. | 896 | Table 3 crop size 896^2; Table 15 ADE20K image resolution 896^2. | LAION-2B image-text pairs + COCO-Stuff (Section 4 'Pretraining Details', 'Pretraining Datasets' (LAION-2B) and Appendix B.1 (coco-stuff intermediate fine-tuning, 80k steps).) | large |
| ONE-PEACE + ViT-Adapter + Mask2Former | 63.0 | 1520 | As above. | 896 | As above, multi-scale column. | LAION-2B image-text pairs + COCO-Stuff (As above.) | large |
| RevColV2-B + Mask2Former | 54.9 | 88 / 325 system | Table 2: RevColV2-B 88M (classification model). Table 1 lists 101M for the base model with its top-down columns, which the segmentation model also uses ('both bottom-up and top-down columns', Section 3.1.3). Table 3 states 325M for the Mask2Former system | — | not stated for the Mask2Former runs (the UperNet ImageNet-22k runs use 640 x 640, Section 3.1.3) | ImageNet-1k (MIM) + ImageNet-21k (supervised) (Section 3.1.1 and Table 2 (ImageNet-1K pre-train + 22K intermediate fine-tune)) | base |
| RevColV2-B + Mask2Former | 55.8 | 88 / 325 system | as above | — | not stated | ImageNet-1k (MIM) + ImageNet-21k (supervised) (as above) | base |
| PlainSeg BEiT-B | 55.7 | — / 105 system | Backbone alone not stated. Whole model: Table 3, 105M (22% randomly initialised), which bounds the backbone below 110M | 640 | Appendix Table 8: ADE20K crop 640; Section 4.1: sliding-window inference, single-scale results only | BEiT weights (data not restated in the paper) (Section 4.1 names BEiT among the pre-trained plain ViTs; the BEiT-B checkpoint is not identified) | base |
| ViT-CoMer-L + Mask2Former (BEiTv2 init) | 62.1 | — / 604 system | Backbone alone not stated: the paper gives system counts only (Table 7: Mask2Former + ViT-CoMer-L 604M; Table 5: UperNet + ViT-CoMer-L 383.4M). Its ViT branch is a ViT-L (307M in the ViT paper's Table 1). | — | Not stated: implementation details for Table 7 are deferred to a supplementary that arXiv v1-v3 do not contain. | BEiT v2 weights (data not restated in the paper) (Section 4.2 and Table 7 'Pre-train' column: BEiTv2 [31]. COCO-Stuff intermediate training is not mentioned.) | large |
| ViT-P on InternImage-H + Mask2Former | 63.1 | — / 1610 system | Two backbones, no separate count in the paper: Mask2Former† InternImage-H mask generator (InternImage-H is 1.08B in the InternImage paper's Table 1) and a DINOv2-L point classifier (Section 4.2; 304.4M counted with timm 1.0.30 vit_large_patch14_dinov2). System: Table 1, 1.61B. | 896 | Table 1: mask-generator crop 896x896; Section 4.2: ViT-P crop 518x518 on ADE20K. | InternImage-H and DINOv2-L weights + COCO (Table 1 caption: '† Models pre-trained on COCO' (both the Mask2Former and ViT-P marked †).) | large |
| ViT-P on InternImage-H + Mask2Former | 63.6 | — / 1610 system | As above. | 896 | As above, m.s. column. | InternImage-H and DINOv2-L weights + COCO (As above.) | large |
| DINOv3 7B (frozen) + ViT-Adapter + Mask2Former | 62.6 | 6716 / 7643 system | Backbone: Figure 16(a) table 'DINOv3 family of models', ViT-7B 6716M (Table 2: 6.7B; Table 11: 7B). System: computed 6716M + decoder 927M (Table 11) = 7643M; Table 11 gives encoder and decoder separately, trainable 927M. | 896 | Table 11 caption and Section 6.3.2: trained and evaluated at 896; App. D.10: single-scale sliding inference after resizing to the training resolution. | LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised); decoder: COCO-Stuff, Hypersim (Section 3.1 (data) and Section 6.3.2 (COCO-Stuff 80k, Hypersim 10k, ADE20K 20k iterations).) | large |
| DINOv3 7B (frozen) + ViT-Adapter + Mask2Former | 63.0 | 6716 / 7643 system | As above. | 896 | Multi-scale (TTA): ratios 0.9-1.1 of 896 (Section 6.3.2; App. D.10 lists [0.9, 0.95, 1.0, 1.05, 1.1] with flips). | LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised); decoder: COCO-Stuff, Hypersim (As above.) | large |
| DINO ViT-B/8 (measured by Oquab et al.) | 31.8 | 85 | DINO paper (arXiv 2104.14294v2) Table 1: ViT-B/8 #params 85M, without the projection head. timm 1.0.30 vit_base_patch8_224 counts 85,807,872. | 512 | DINOv2 Section 7.4 'Semantic segmentation', 'Linear' setup: logits upsampled to full resolution 512x512 (arXiv 2304.07193v1) | ImageNet-1k (self-supervised) (DINO trains on ImageNet without labels; DINOv2 Table 4 lists DINO's data as INet-1k.) | base |
| DINO ViT-B/16 (measured by Zhou et al., iBOT) | 34.5 | 85 | DINO paper Tables 1-2 (85M, feature extractor); iBOT Table 1 (85) | 512 | iBOT Appendix, ADE20K fine-tuning recipe: '160k iterations with 512 x 512 images. We do not use multi-scale training and testing' | ImageNet-1k (self-supervised) (DINO paper Section 3: ImageNet without labels) | base |
| DINO ViT-B/8 (measured by Oquab et al.) | 35.2 | 85 | DINO paper Tables 1-2 (85M) | 640 | DINOv2 Section 7: '+ms' uses image resolution 640 with multiscale test-time augmentation | ImageNet-1k (self-supervised) (DINO paper Section 3: ImageNet without labels) | base |
| MAE ViT-H/14 (measured by Oquab et al.) | 33.3 | 632 | MAE paper (arXiv 2111.06377v3) Table 12: MAE ViT-H 632M (also ViT paper Table 1: ViT-Huge 632M). | 512 | DINOv2 Section 7.4 'Semantic segmentation', 'Linear' setup: logits upsampled to full resolution 512x512 (arXiv 2304.07193v1) | ImageNet-1k (self-supervised) (DINOv2 Table 4: MAE ViT-H/14, data INet-1k.) | large |
| iBOT ViT-L/16 (measured by Oquab et al.) | 44.6 | 307 | iBOT paper (arXiv 2111.07832v3) Table 1 and Table 15: ViT-L/16 307 (M). | 512 | DINOv2 Section 7.4 'Semantic segmentation', 'Linear' setup: logits upsampled to full resolution 512x512 (arXiv 2304.07193v1) | ImageNet-21k (self-supervised) (DINOv2 Table 4: iBOT ViT-L/16, data INet-22k (Table 10 does not repeat the data; the iBOT paper has ViT-L/16 checkpoints for ImageNet-1K and ImageNet-22K, Table 15).) | large |
| iBOT ViT-B/16 | 38.3 | 85 | iBOT Table 1 (Par. 85 for ViT-B/16) | 512 | iBOT Appendix, ADE20K recipe: 512 x 512 images, no multi-scale training and testing | ImageNet-1k (self-supervised) (iBOT Section 4 and Table 15 (pre-train data ImageNet-1K for ViT-B/16)) | base |
| DINOv2 ViT-g/14 | 49.0 | 1100 | DINOv2 Section 4 (arXiv v1): 'our ViT-g backbone counts 1.1B parameters'. timm 1.0.30 vit_giant_patch14_dinov2 counts 1,136,479,232. | 512 | DINOv2 Section 7.4 'Semantic segmentation', 'Linear' setup: logits upsampled to full resolution 512x512 (arXiv 2304.07193v1) | LVD-142M (self-supervised) (DINOv2 Section 3 and Table 4 (data LVD-142M).) | large |
| DINOv2 ViT-L/14 | 53.1 | 304.4 | Not stated in DINOv2 (Table 17 gives the architecture only: 1024 dim, 16 heads, 24 blocks, MLP FFN for the distilled ViT-L/14). Counted with timm 1.0.30 vit_large_patch14_dinov2: 304,367,616. | 640 | DINOv2 Section 7.4 '+ms' setup: 'a larger image resolution of 640' with multiscale test-time augmentation. | LVD-142M (self-supervised) (DINOv2 Table 4 rows (LVD-142M).) | large |
| DINOv2 ViT-B/14 | 47.3 | 86.58 | counted from the official code (github.com/facebookresearch/dinov2 @ 7764ea0f): 86,580,480; not stated in the paper (Table 17 lists 18 blocks, contradicted by the code's 12 and the paper's layer list {3, 6, 9, 12}) | 512 | Section 7 'Semantic segmentation': logit map upsampled to full resolution (512x512) | LVD-142M (self-supervised; distilled from DINOv2 ViT-g/14) (Table 4 (Data: LVD-142M), Table 15 (composition)) | base |
| DINOv2 ViT-B/14 | 51.3 | 86.58 | counted from the official code (as for the 'lin.' point) | 640 | DINOv2 Section 7: '+ms' uses image resolution 640 | LVD-142M (self-supervised; distilled from DINOv2 ViT-g/14) (Table 4, Table 15) | base |
| DINOv3 ViT-7B/16 | 55.9 | 6716 | Figure 16(a) table: ViT-7B 6716M. | 512 | Table 3 caption: input adapted to 1024 patch tokens, 512x512 for patch size 16. | LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) (Section 3.1.) | large |
| DINOv3 ViT-B/16 (distilled) | 51.8 | 86 | Figure 16(a) table 'DINOv3 family of models' (distilled models): ViT-B 86M; Section 5.2: 'B (86M)'. | 512 | Section 7.1 (the comparison that Table 14 reports): 'for model with a patch size of 16 we input images of size 512 x 512 versus 448 x 448 when models are using patch size 14' | distilled from DINOv3 7B: LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) (Section 5.2 (distillation of the 7B teacher) and Section 3.1 (teacher data).) | base |
| DINOv3 ViT-B/16, probed by Berreby et al. (512 px) | 47.19 | 86 | DINOv3 paper Figure 16(a) table: ViT-B 86M (the CanViT paper states no count for it). | 512 | CanViT paper, Appendix 'Task: ADE20K segmentation', Evaluation: 'we resize all images and masks to a fixed size (here, 512 x 512)'; passive-comparison table row 512 px, 32x32 grid. | distilled from DINOv3 7B: LVD-1689M + retrieval-curated web images + ImageNet-1k/21k + Mapillary (DINOv3 data, self-supervised) (DINOv3 paper Section 5.2 / 3.1.) | base |

## Verified, but left out of the JSON

**ImageNet-1k**
- PolyNet (Very Deep PolyNet, 2016-11-17): 82.64 (multi-crop, single model, 92M). Below the 2016 ensemble, and at
  base size below the Inception-v3 4-model ensemble (82.8).
- Base Saccader (BagNet-77-lowD, 224 px, 6 glimpses): 70.31, as tabulated by TNet's Table 1. The Saccader
  appendix gives DRAM 66.6 and Saccader 70.1 on a 3,906-image validation subset.
- Frozen weakly-supervised backbones, linear probe (DINOv3 Table 9): PE-core G/14 89.3 and SigLIP 2 g/16 89.1.
  Both are above DINOv3 7B's 88.4. They are not self-supervised, so they are not in the `frozen_ssl` series.
- SpatialBoost (arXiv 2603.22057, 2026): 90.2 in Table 5's linear ImageNet column for DINOv3 + SpatialBoost, whose
  backbone is DINOv3 ViT-7B/16 (Section 4.1), so it is large.
- CF-ViT (AAAI 2023, 2022-03-08): 84.1. It tokenizes the whole image coarsely, then re-splits patches chosen by
  attention, once. It is not active.
- S3TA (Towards Robust Image Classification Using Sequential Attention Models, CVPR 2020, arXiv 1912.02184): S3TA-16
  72.54 nominal accuracy (Table 1). Its recurrent attention queries a feature map that a modified ResNet-152 computes
  on the full image (Section 3), so it is not glimpse-limited sensing; no single system count is stated.
- SparseFormer-B (SparseFormer: Sparse Visual Recognition via Limited Latent Tokens, ICLR 2024, arXiv 2304.03768):
  82.6, 81M (v1 Table 2). It samples image features over several steps, but its first layers (a 7×7 convolution and
  max pooling) see the full-resolution image: borderline under the series definitions, not added. Its bootstrapped
  successor (arXiv 2312.01987) keeps the feature-space sampling.
- Jérémie et al. (arXiv 2402.15480): 74.3 with a single central log-polar fixation on a retrained ResNet-101;
  77.4 when the fixation sits at the ground-truth box centre. Not active.
- Vuyyuru, Reddy et al. 2020 (arXiv 2006.16427): 62.90 with fixed "retinal fixations" on central 320 px crops.
  The test set may be a 5k-image subset.
- Bio-FCG (Lukanov et al., Frontiers in Computational Neuroscience, 2021-11-22): 65.17 with one saccade. It was
  evaluated on ILSVRC-2010, not ILSVRC-2012.
- FoveaTer v3 (2022-10-02): 76.69 with radial-polar pooling and 76.30 with five square fixations. The JSON keeps
  v2's 78.31.
- Base size, ImageNet-1k only, read but not records at their dates:
  - Single models in years when a base-size ensemble led: BN-Inception single network, multi-crop, 78.01 (13.6M,
    2015-02-11); Inception-v3 single model, 144 crops, 81.23 (2015-12-02); Inception-v4 single 82.3 and
    Inception-ResNet-v2 single 82.2 (2016-02-23; no count stated); NASNet-A (N=7) 82.3 (84.9M, arXiv v1 of
    2017-07-21; NASNet-A (6 @ 4032) 82.7 with 88.9M in v4).
  - AmoebaNet-A (6, 190), 86.7M: 82.7 in arXiv v1 (2018-02-05) and 82.8 in v2 (2018-02-06) of Regularized Evolution
    for Image Classifier Architecture Search (AAAI 2019, arXiv 1802.01548), ImageNet results table; v1's 82.8 row is
    AmoebaNet-A (6, 204), 99.6M. AutoAugment AmoebaNet-B (6,190) 82.75 (arXiv v1, 2018-05-24; 82.8 in v3); FixRes
    PNASNet-5 at 480 px 83.7 (2019-06-14); AdvProp EfficientNet-B7 85.2 (66M); CaiT-S36↑384Υ 85.4 (68M) and
    CaiT-S48↑384 85.1 (89M); HaloNet H7 84.9 (67M); EfficientNetV2-M 85.1 (55M; arXiv 2104.00298 v1); LV-ViT-M↑384
    85.4 (56M; the 89.5 in its row is ImageNet-ReaL; arXiv 2104.10858 v1 Table 4); CoAtNet-2 at 512 px 85.9 (75M,
    2021-06-09, ties Refined-ViT-L↑448 of 2021-06-07); XCiT-M24/8Υ↑384
    85.8 (84M); MaxViT-S at 512 px 86.19 (69M); CAFormer-B36↑384 86.4 (99M); TransNeXt-Base at 384 px 86.2 (89.7M);
    H-ViT3-B 85.5 (94M, 2025-12-01).
  - Excluded as above 100M: ZFNet v3's wider single model (62.5, 107.6M computed); Howard 2013's "Double FC" net
    (63.0, size not stated, wider fully connected layers); OverFeat and VGG single nets (133M to 145M); PReLU-net
    model C (330.6M computed); Deep Image (75.12, 212.7M); MaxViT-B (120M).
- Base size, extra data, read but not records: Yalniz et al. ResNeXt-101 32x4d 83.4 (43M) and ResNet-50 81.2; Noisy
  Student B7 with a B7 teacher 85.9; Meta Pseudo Labels EfficientNet-B7 86.87 (Table 11 of arXiv v3); HaloNet H4 at
  512 px 85.8 (85M, ImageNet-21k); EfficientNetV2-M (21k) 86.1 (55M); SwinV2-B 87.1 (88M, ImageNet-22K, 384 px; arXiv
  2111.09883 v1 Table 2); DeiT III ViT-B↑384 86.7 (ImageNet-21k; the 86.9 in its row is the parameter count; arXiv
  2204.07118 v1 Table 8); FocalNet-B 86.5, DaViT-Base 86.9 and FT-CLIP ViT-B 86.6 (384 px); ConvNeXt V2-B
  87.7 (ImageNet-22K, 384 px, 2023-01-02); EVA-02-B 88.6 at 448 px (2023-03-20; 88.57 in its Table 18, a tie with
  the ViT-22B-distilled ViT-B); UniRepLKNet-B 87.4 (98M); MobileNetV4 Hybrid-L 87.0 (35.9M, JFT data and an
  EfficientNet-L2 teacher); ScaleKD ViT-B/14 86.43; Sun et al. 2017 ResNet-101 from JFT-300M 79.2.
- Base size, frozen self-supervised + linear probe, read but not records: DINO ViT-B/16 78.2 (same paper as ViT-B/8);
  MoCo v3 ViT-B/16 76.7; iBOT ViT-B/16 79.5 (ImageNet-1k pretraining) and 79.0 (ImageNet-22K); MAE ViT-B 68.0; RELICv2
  ResNet200 (1x) 79.8 (63M); Mugs ViT-B/16 80.6; I-JEPA ViT-B/16 72.9; SimCLRv2 ResNet-152 (1x+SK) 77.2 (89M); SwAV
  ResNet-50 75.3; Franca ViT-B/14 82.0 (ImageNet-21K). MSN reports no full-label ViT-B linear result, data2vec no
  linear probe, and DINOv2 with registers only ViT-L. Also:
  - CMC ResNet-50 64.1 (two encoders, 47.0M counted from the official code), in the same table as the CMC point.
  - CPC v2 ResNet-50 63.8 (24M stated; Data-Efficient Image Recognition with Contrastive Predictive Coding, ICML
    2020, arXiv 1905.09272, Table 1 of v3): below the base frozen line from CMC's 65.0 (2019-10-21) on; its first
    public date is an open lead.
  - SwAV ResNet-50 (w2) 77.3 (94M, 400 epochs): a row of the official facebookresearch/swav README's "Larger
    architectures", absent from arXiv 2006.09882 v1. SwAV's v1 already follows BYOL's 77.8 (2020-06-13).
  - XCiT-M24/8 trained with DINO, 84M: 80.3 at 224 px and 80.9 at 384 px (XCiT, NeurIPS 2021, arXiv 2106.09681 v1
    Table 3, 2021-06-17), below EsViT's 81.3 of the same day.
- Frozen self-supervised, large: CMC ResNet-50 x2 68.4 (arXiv 1906.05849 v3, Table 2): two width-2 encoders,
  187.7M counted from the official code. v5's 70.6 uses other color views and RandAugment with the same encoders.
- Leaderboard claims kept out, though both papers do report them:
  - SATA (arXiv 2409.19850): SATA-B∗ 94.9 at 224 px in Table 1 (SATA-B, built on DeiT-B, 93.9). The starred ViT
    checkpoint's pretraining data is not identified, so no series can be assigned; Equation 7's outlier set, read
    literally with alpha = 1 (Section 5.6), is empty; the official code (github.com/nick-nikzad/SATA) could not be
    read.
  - TAPe+ML (arXiv 2609.20869): 88.1 on ImageNet-1k (Table 5, p. 23), trained from scratch on the standard split,
    with fewer than 100,000 parameters claimed in the abstract. Its limitations section (p. 36) says the full
    representation algorithm is proprietary and undisclosed, so the inference system can be neither specified nor
    counted. Accepted, it would top the base ImageNet-1k-only line (86.5).

**ADE20K**
- Further passive results below the year's best:
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
- DINOv3's distilled models (Table 14): S 47.0, S+ 48.8, L 54.9, H+ 54.8. In the same table, at base size: DINOv2
  ViT-B 48.4 (against 47.3 in its own paper), SigLIP 2 B 41.6, PE-core B 37.4; DINOv3 ConvNeXt-B (89M, Table 15) 46.3.
- InternViT-6B, frozen, linear probe at 504 px (InternVL, 2023-12-21): 47.2. In the same table, OpenCLIP-G
  frozen scores 39.3.
- DINOv2 Table 10, "+ms" column for older large models:
  - iBOT ViT-L/16: 47.5.
  - MAE ViT-H/14: 30.7.
  - OpenCLIP ViT-G/14: 46.0 (39.3 plain).
- Base size, backbone fine-tuned, read but not records: Focal-B + UperNet 49.0 / 50.5; BEiT-B+ (ImageNet-22k
  intermediate) + UperNet 53.6 / 54.2; ConvNeXt-B + UperNet 52.6 / 53.1; FD-CLIP ViT-B 52.8; HorNet-B 50.5 / 50.9;
  BEiT v2 ViT-B 53.1; MaskDistill ViT-B 54.3 (2022-10-19, after MOAT-2's 54.7); InternImage-B 50.8 / 51.3;
  ConvNeXt V2-B 52.1; SegViT BEiTv2-Base 54.0 / 54.9; DaViT-B with Florence-2 pretraining 54.9 / 55.5; UniRepLKNet-B
  53.5 / 53.9; TransNeXt-Base + Mask2Former 54.7 (ImageNet-1k only); ViT-CoMer-B 48.8 / 49.4; ViT-5-Base 49.1;
  SeMask-B FPN 49.35 / 50.98; iBOT ViT-B/16 + UperNet 50.0 (iBOT Table 6). Segmenter's comparison table uses ViT-L;
  EoMT's ADE20K semantic rows are ViT-L only; Kerssies et al. give figure values only (highest label 54.7). Also:
  - ANL ResNet-101 45.24 (Asymmetric Non-local Neural Networks for Semantic Segmentation, ICCV 2019, arXiv
    1908.07678, Table 4; multi-scale with flip; 63.17M for the whole model, Table 2; 2019-08-21), below APCNet's
    earlier 45.38.
  - DNL ResNet-101 45.97, not 46.0 (Disentangled Non-Local Neural Networks, ECCV 2020, arXiv 2006.06668, Table 4;
    multi-scale; 2020-06-11), below CPNet's 46.27 and ResNeSt-101's 46.91 of April 2020.
  - MVP ViT-B/16 + UperNet 52.4 (MVP: Multimodality-guided Visual Pre-training, arXiv 2203.05175, Table 2;
    fine-tuned), below Mask2Former Swin-B's 53.9 / 55.1.
  - Proteus ViT-B/14 + UperNet 54.4 (arXiv 2407.10366 v1, Table 5; single-scale at 518 px), below PlainSeg's 55.70.
- Base size, frozen linear, read but not records, all below DINOv2 ViT-B/14's 47.3 (or iBOT ViT-B/16's 38.3 before
  2023-04-14); the protocols differ from paper to paper:
  - Franca ViT-B/14 39.1 under its own protocol (DINOv2-B scores 42.6 under it; arXiv 2507.14137 v1 Table 2a).
  - VICRegL ConvNeXt-B 35.3 (85M; arXiv 2210.01571 v1 Table 2); its 43.2 belongs to ConvNeXt-XL (350M).
  - NeCo ViT-B/14 44.9 (arXiv 2408.11054 v1 Table 2).
  - FiT3D 45.93 (DINOv2 features after 3D-aware fine-tuning; arXiv 2407.20229 Table 3; its 58.71 is mAcc).
  - MoSiC ViT-B/14 43.6 and DINOv2R-B + MoSiC 44.4 (arXiv 2506.08694 v1 Tables 4 and 5).
  - DIP ViT-B/14 39.5 (arXiv 2506.18463 v1 Appendix Table 8; the 86.7 in its row is COCO).
- Frozen, distilled from a mix of self-supervised, supervised and language-supervised teachers: RADIO-AMP-B, 98M
  (RADIO Amplified: Improved Baselines for Agglomerative Vision Foundation Models, arXiv 2412.07679 v1 of
  2024-12-10; CVPR 2025 as RADIOv2.5), linear ADE20K 48.94 / 50.48 / 51.16 at 512 / 768 / 1024 px (Appendix Table
  A5). Its teachers are CLIP, DINOv2 and SAM-H (Section 2.1), so it is not self-supervised and stays out of
  `frozen_ssl`.

## Unverified leads

**ImageNet-1k**
- AmoebaNet-A 83.9 (Real et al. 2018), a larger model than the base-size AmoebaNet-A (6, 190): not pinned to the
  arXiv version that first carries it. GPipe's later versions cite it as the previous best.
- Not read in their own papers:
  - InternImage-H 89.6.
  - Florence 90.05.
  - NEPA-L 85.3 (a 2026 lead).
  - Mahajan et al. 2018 ResNeXt-101 32x8d (IG), 88M by its own Section 2.3: 82.2 per FixRes's Table 2, 82.7 per
    Yalniz et al.'s Table 6; Mahajan et al. show the accuracy only in figures. Below PNASNet-5 either way.
- MaskDistill 88.3 and ConvNeXt V2-H 88.9 were seen but not read. They use a CLIP teacher and ImageNet-22K
  labels, so they would belong to the extra-data series, below the frontier of their year.
- 2024–2026, overall: no fine-tuned ImageNet-1k result above Lion/BASIC-L's 91.1 was found, other than OmniVec2
  (flagged in the JSON). At base size, no result above MOAT-2's 86.5 (ImageNet-1k only), the ViT-22B-distilled
  ViT-B's 88.6 (extra data) or the CanViT paper's 85.0 (frozen) was found, other than the DINOv3 supplement below.
  The searches were not exhaustive: the comparison tables of TransNeXt, UniRepLKNet, ViT³, ScaleKD, MobileNetV4 and
  EVA-02, the paperswithcode.co mirror (132 ImageNet entries) and a few web searches.
- DINOv3's TMLR supplementary material (openreview.net/attachment?id=2NlGyqNjns&name=supplementary_material) holds
  Tables 36 and 37 on its distilled models. The PDF itself could not be opened (HTTP 403); the audit read its text as
  indexed by a web search (`throwaway/history-audit/dinov3_indexed_evidence2.md`). Table 36, linear classification,
  IN-1k val at 512 px: ViT-B 85.3, ConvNeXt-B 85.4. Table 37, linear segmentation, ADE20K at 1024 px: ViT-B 52.7 (51.8
  at 512 px, as in the JSON). Both models are base size (ViT-B 86M, ConvNeXt-Base 89M, arXiv v1 Figure 16(a)). The
  supplement's first public date is unknown and arXiv v1 (2025-08-13) lacks these tables, so they are not points;
  verified and dated, they would raise the base frozen ends to 85.4 and 52.7.
- First public dates still open: SwAV ResNet-50 (w2)'s 77.3 (a model-zoo row); CPC v2 ResNet-50's 63.8 (absent from
  arXiv v1 and v2, present in v3 of 2020-07-01, already tabulated by SimCLR's v1 of 2020-02-13), a base frozen record
  only if public between BigBiGAN (2019-07-04) and CMC (2019-10-21); RotNet's 38.7 in its ICLR 2018 OpenReview
  submission (OpenReview refused the request).
- SATA's implementation and checkpoint, and TAPe+ML's full inference system, are unresolved (both under "Verified,
  but left out").
- SAFER-AiD (SAFER-AiD: Saccade-Assisted Foveal-peripheral vision Enhanced Reconstruction for Adversarial Defense,
  WACV 2026, arXiv 2510.08761), a sequential active model: the accumulated reconstruction chooses each next saccade
  (Section 3.3); three glimpses with 56×56 foveae, 6% peripheral sampling and 224 px images (Section 4). Its tables
  give accuracy on attacked inputs, for two correctly classified images per class (Section 4.2); clean accuracy
  appears only in Figure 4, with no printed number or full-validation scope; the count of its inference networks
  (reconstruction network, actor, downstream classifier) is not stated. No point until a clean full-validation
  number and a count exist.
- APEX (Active Perception with EXploratory Sampling), in the Syracuse University dissertation Bio-Inspired Visual
  Intelligence: Improving Efficiency and Robustness through Foveal-Peripheral Sampling and Learned Saccades (2026,
  surface.syr.edu/etd/2372): the abstract names frozen CLIP features and adversarial and incremental evaluation but no
  ImageNet score or parameter count; the PDF download was refused (HTTP 403).
- Report top-5 only: GoogLeNet (2014) and the multi-model results of PReLU-net and ResNet.
- FoveaTer's venue: the v3 PDF header says ICLR 2023, but this was not confirmed.
- Howard 2013's Table 2 (37.1% error with 90 predictions) does not say which network it uses; Table 1 separates the
  "Double FC" network from the smaller one, and Table 3 calls 37.0% "One Base Net", matching the larger "Double FC"
  network. The JSON keeps the smaller network's 62.5.

**ADE20K**
- UperNet ResNet-101 42.66 (Unified Perceptual Parsing for Scene Understanding, ECCV 2018, arXiv 1807.10221) appears in
  later papers' tables, APCNet's Table 8 among them; the row was not located in UperNet's own text. It is below the
  2016–2017 base records either way.
- FD-SwinV2-G 61.4, Mask DINO 60.8 and M3I (InternImage-H, 62.9) were seen only in other papers' tables. None was
  a year's best.
- No 2024–2026 result above ViT-P's 63.6 multi-scale was found; the search was limited. At base size, no 2024–2026
  result above PlainSeg's 55.70 single-scale or RevColV2-B's 55.8 multi-scale was found (web searches and the Papers
  with Code mirror only).
- Scope of the frozen line: `frozen_ssl` admits self-supervised backbones only. RADIO-AMP-B (under "Verified, but left
  out"), distilled from CLIP, DINOv2 and SAM-H, would be a base frozen ADE20K record between DINOv2 (2023-04) and
  DINOv3 (2025-08) if the line admitted all frozen pretrained features. Whether to widen it is the authors' decision.
- The audit's search for base frozen linear ADE20K results between iBOT (2021-11) and DINOv3 (2025-08) found none
  above the line (candidates under "Verified, but left out" and "Checked"); it does not establish that none exist.

## Checked: no ImageNet-1k top-1 or ADE20K mIoU

- **No ImageNet-1k number:**
  - TORE (WACV 2025): SUN360, CIFAR-100, Flowers102 and Food101.
  - Pourrahimi & Bashivan 2025 (bioRxiv): visual search on COCO-Search18. Its CNN is pretrained on
    retina-transformed ImageNet, with no accuracy reported.
  - AME: SUN360 classification only.
  - Luo et al. 2015 (Foveation-based Mechanisms Alleviate Adversarial Examples, arXiv 1511.06292): top-5 accuracy
    only, with foveations placed from ground-truth object boxes (Section 2).
  - "Toward Errorless Training ImageNet-1k" (on the Papers with Code leaderboard): training-set accuracy.
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
  - FAVE (arXiv 2609.04392): General-2K (65.60 top-1) and object-size cohorts, on crops selected externally or from
    ground truth.
- **Other tasks:**
  - FALcon's own paper reports localization only.
  - AttSeg: Cityscapes, CamVid, KITTI.
  - Grimes et al. (TMLR 2023): robot camera control.
  - Video models: AdaFocus, Uni-AdaFocus, policy-based foveated imaging.
  - Coarse-to-Fine GAP: instance detection.
  - Beyond Grids: plots only.
  - Dallain et al. 2026 (A Saccade-inspired Approach to Image Classification using Vision Transformer Attention Maps,
    arXiv 2603.09613): classification shown in figures only; its fixations come from a single-pass static saliency
    map of a ViT over the full image.
  - Recasens et al. 2018 (Learning to Zoom, ECCV 2018, arXiv 1809.03355): iNaturalist and other non-ImageNet tasks;
    its saliency sampler runs in a single pass.
- **No standard ADE20K mIoU (active models):**
  - GAE (Glimpse-Attend-and-Explore) reports pixel accuracy only.
  - SimGlim reports reconstruction only.
  - Digital Foveation uses a 98-class ADE20K subset.
- **No frozen linear ADE20K mIoU (their ADE20K results fine-tune the backbone):**
  - LOCA (Location-Aware Self-Supervised Transformers, arXiv 2212.02400 v1; WACV 2024): Table 2 is end-to-end
    fine-tuning with a linear decoder.
  - SelfPatch (CVPR 2022, arXiv 2206.07990): Semantic FPN fine-tuning (Section 4).
  - Mugs (arXiv 2203.14415): Appendix B.3 fine-tunes the backbone.
  - Context Autoencoder (arXiv 2202.03026): UperNet fine-tuning (Section 4.4); its linear and attentive probes are on
    ImageNet.

## Caveats a careful audience may raise

1. **Active models stand on passive ones.** Most active entries start from an ImageNet-pretrained passive network
   or learn from a passive teacher that sees the whole image:
   - Start from a pretrained passive network: GFNet, Saccader-NASNet, FALcon, and Liu et al., whose evaluator is a
     frozen ResNet-101.
   - Learn from a whole-image teacher: STAM (a DeiT-B trained on ImageNet-1k) and AdaGlimpse. AdaGlimpse's ImageNet
     teacher is the DeiT III ViT-B checkpoint trained with ImageNet-21k (`deit_3_base_224_21k.pth`, named in the
     official README; the paper says only "a pre-trained ViT from [DeiT III]"), so its ImageNet-1k number used data
     beyond ImageNet-1k through distillation.
   - Both: LookWhere is initialised from DINOv2 and distilled from it.

   Their accuracy therefore tracks passive progress; it is not an independent line.
2. **Many active models also see the whole image at low resolution.**
   - GFNet and AdaptiveNN start with a downsampled glance of the full image.
   - Saccader computes restricted-receptive-field features over the whole 224 px image in one pass and picks
     locations from them.
   - PatchDrop, TNet and LookWhere select from a low-resolution view.

   STAM never sees the full image. AdaGlimpse's variable-scale glimpses can cover the whole image at low
   resolution.
3. **Size, resolution and data, at base size.** The base-size lines remove the gap in parameter count between the
   sequential and select-once models (22M to 125M on ImageNet-1k; AME on ADE20K is a 303M ViT-L) and the all-size
   frontier (0.5 to 7 billion parameters), but not every difference:
   - Test resolution: base passive records from 2019 on test at 384 to 800 px (MOAT-2 512, VOLO-D3 448,
     FixEfficientNet-B8 800); the active models use 224 to 331 px images.
   - Extra data: the base extra-data line uses JFT, ImageNet-21k and CLIP-family teachers; the base ImageNet-1k-only
     line does not, and is the closer match for the active models trained on ImageNet-1k (AdaGlimpse's teacher
     aside).
   - Ensembles and multi-crop testing: BN-Inception (6 networks, about 82M in total) and Inception-v3 (4 networks,
     each under 25M) are base size by total parameters but average 4 to 6 networks over 144 crops. Without them, the
     2015–2017 base line is lower (the single models listed under "Verified, but left out").
   - Compute at test time is not parameter count: STAM runs its DeiT-B core on up to 27 glimpses, AdaptiveNN reports
     4.85 GFLOPs at 3.15 fixations, LookWhere 14.8 GFLOPs.
   - A like-for-like comparison exists inside AdaptiveNN's own table: DeiT-S scores 79.9 at 224 px and 81.6 at
     384 px, against AdaptiveNN-DeiT-S's 82.2 at 288 px.
4. **Sensing budgets differ.** Active models report accuracy at a fraction of the pixels. Examples: AdaGlimpse
   uses 28.6% of the pixels, STAM about 55%, Saccader 29.5% of the 331 px image. Passive models see every pixel.
5. **Evaluation protocols changed over time.**
   - End-to-end results up to 2016 average ensembles or 10–144 crops; from 2017 they are mostly single-crop.
   - Test resolution grew from 224 px to 800 px.
   - Inception-v3 was scored on 48,238 non-blacklisted validation images (the paper says the full set is about
     0.2 points worse).
   - BiT, ViT, ViT-G, SAM, STAM and AdaptiveNN report means or medians over runs.

   All ImageNet values use the original ILSVRC-2012 validation labels, not ImageNet-ReaL. The 2017 best (82.9,
   single crop) is below the 2016 ensemble (83.5), so a running maximum mixes protocols.
6. **Numbers change between arXiv versions.** Each date is that of the version first carrying the number:
   - MPL: 86.9 in 2020, 90.2 from 2021-01-05.
   - CoAtNet: 89.77, then 90.88.
   - Noisy Student: 87.4, then 88.4; its B7 student 86.8, then 86.9.
   - BiT: 87.76, then 87.54 in the final version.
   - CCNet: 45.22, then 45.76.
   - CMC: ResNet-101 60.1 in v1 (2019-06-13) and v2, 65.0 from v3 (2019-10-21).
   - FoveaTer: ImageNet-100 in v1, 78.31 in v2, 76.69 in v3.
   - EfficientNet-B7: 84.4 in v1 to v3, 84.3 from v4 (2020-09-04).
   - FixRes: v1 (2019-06-14) has no ResNeXt-101 32x48d result; 86.4 first appears in v2 (2019-07-19).
   - CaiT: v1 (2021-03-31) tops out at CaiT-M36↑448Υ 86.3; the M48 row (86.5) first appears in v2 (2021-04-07).
   - GF-EfficientNet-B2: 77.93 in v1, 77.95 ± 0.03 in v2.
   - FOVI: 0.850 (3 fixations) and 0.840 (1 fixation) in v1, 0.853 and 0.844 in v2 (2026-05-29).
   - BEiT v2: the ViT-B row at 384 px with ImageNet-21k fine-tuning first appears in v2 (2022-10-03).
   - VOLO: v1's Table 1 puts VOLO-D5's 87.1 at 448 px; its Table 4 and v2 put it at 512 px.
   - Inception-v3: the abstract gives 17.3% top-1 error for the ensemble, Table 5 17.2%.

   Venue years often fall a year after the arXiv date; DINOv3, for instance, is arXiv 2025 and TMLR 2026.
7. **ADE20K protocols differ.**
   - Multi-scale testing adds 0.4–1.6 points over single-scale across these entries.
   - From 2022 the leaders train and test at 896 px; SETR used 512 px crops and Swin 640 px. The base-size records
     use 480 to 641 px crops.
   - From 2022 the large leaders add intermediate segmentation training on COCO-Stuff (DINOv3 also on Hypersim) before
     ADE20K.
   - The active models are evaluated on much smaller scenes: AME on 256×128 px images, AdaGlimpse on 224×224.
   - AME's SETR initialisation was already trained on ADE20K segmentation.
   - Several papers do not state single- or multi-scale (protocol "unstated"). EncNet's Section 3.1 says evaluation
     averages predictions over multiple scales, but its Table 4 does not mark the protocol. APCNet's Section 4.1
     describes multi-scale and flip evaluation, but neither its Table 8 nor its ADE20K section says whether the
     ADE20K result uses it.
   - Swin's "62.8" is a test-set score, not validation mIoU. Swin's own paper prints 51.6 for Swin-B + UperNet with
     multi-scale testing and no single-scale value; later papers (CSWin, ConvNeXt from Swin's GitHub, UniRepLKNet)
     cite 50.0 / 51.7.
8. **Frozen probes depend on the probe.**
   - The same DINOv2 ViT-g/14 scores 49.0 in its own paper and 49.5 under DINOv3's protocol. DINOv2 ViT-B/14 scores
     47.3 in its own paper, 48.4 under DINOv3's and 42.6 under Franca's.
   - DINO ViT-B: 31.8 (ViT-B/8, DINOv2's protocol) and 34.5 (ViT-B/16, iBOT's protocol). Both are dated by DINO's
     release; the base frozen ADE20K line mixes these protocols.
   - The base frozen ImageNet line mixes probes too: linear classifiers on intermediate AlexNet layers, best layer
     reported (Split-Brain, RotNet), a linear SVM on conv5 features (Instance Discrimination), 10-crop evaluation
     (Local Aggregation), a fixed BN+CReLU transform before the linear layer (BigBiGAN), and features of several
     layers concatenated (EsViT, DINOv2, Proteus).
   - DINOv3's paper gives its distilled ViT-B/16 51.8, while the CanViT paper's own probe of that model at 512 px
     gives 47.19, the protocol it also uses to probe CanViT-B. The CanViT paper resizes images and masks to 512×512
     for scoring, without keeping the aspect ratio.
   - The CanViT paper's ImageNet-1k probe of DINOv3 ViT-B/16 (85.0) reads the last-layer CLS token at 512 px, against
     224 px for DINOv2's 84.5, and its hyperparameters were chosen to maximise ImageNet-ReaL accuracy on the
     validation images, so the number is not from a held-out selection.

   I did not investigate the 4.6-point ADE20K difference. The probe numbers of older models (DINO, MAE, iBOT) were
   measured by the DINOv2 or iBOT authors, so the JSON dates those points by model release.
9. **Flagged entries.**
   - OmniVec (92.4) and OmniVec2 (93.6), by the same two authors, give no resolution, crop protocol or parameter
     count, and no independent reproduction was found. A conservative extra-data line stops at 91.1 (2023).
   - Perceptual MAE's v1 text calls its ViT-L "86M parameters"; that is ViT-B's count (ViT-L: 304M).
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
    not of the venue. The 2012 AlexNet, 2017 SAC, 2019 APCNet and 2024 OmniVec2 points carry the year only.
12. **Who states the parameter counts.** No active paper in the file states its full system's count except Saccader,
    DRAM (in Saccader's table) and TNet. The other active counts come from the official code or the stated
    architecture:
    - AdaptiveNN-DeiT-S is 89.4M at test time, four times a DeiT-S (22M): its code keeps a 12-block glance DeiT-S, a
      full fixation DeiT-S, a separate 4-block DeiT head for each of the four fixation steps, and 7.0M policy and 7.0M
      value networks (82.4M without the value network). The paper's Section 5.3.2 describes one head.
    - GFNet-ResNet-50 is 103.6M, of which the patch-proposal network is 46.2M (its first layer maps an 18,432-d
      feature map to 2,048 units).
    - STAM-B is 100.5M: the DeiT-B core (87.3M) and a 13.1M actor MLP.
    - Liu et al.: the paper names a ResNet18 actor (55.75M in all), the released code builds a ResNet-34 layout
      (65.86M). Its evaluator weights in the code are torchvision's IMAGENET1K_V2 (81.886% listed), while the paper
      reports 77.2% for its evaluator.
    - PatchDrop's count is unsettled: the released code builds networks that fail on a 224 px input.
    - LookThere and both Prisadnikov et al. papers give too little to count their systems (`params_m` null).
    - FoveaTer: arXiv v3's "24M parameters" belongs to a different model (Patchconvnet); v2's 78.31 uses a DeiT-S
      (22.05M).
13. **Passive counts not stated in their papers.** ResNet-152 (60.19M, torchvision), SENet-154 (115.09M, timm's port
    of the official release; NASNet, PNASNet, Mahajan et al. and GPipe list 145.8M to 146M), PReLU-net model C
    (330.6M, computed), ZFNet and SPP-net (computed), DINOv2 ViT-B/14 and ViT-L/14 (counted from code; DINOv2's
    Table 17 lists 18 blocks for ViT-B/14, contradicted by its own layer list and by the code's 12). The audits
    counted Wider or Deeper Model A2's backbone (105.07M), DPT-Hybrid's (98.18M), Rotation RevNet-50 (85.77M) and
    CMC's two ResNet-101 encoders (84.99M) from official code. The ResNet-101 backbones of EncNet, CCNet, ACNet, OCNet,
    SAC, APCNet and CPNet, the ResNet-152 of PSPNet, the AlexNet and ResNet-50 features of the early frozen probes,
    BigBiGAN's RevNet-50 ×4 and Proteus's ViT-B/14 have no stated count and are left null (base size under any
    variant). PSPNet's ResNet-269 and Cascade-DilatedNet are unpinned: if ResNet-269 is base size, PSPNet ResNet-269
    (43.81 single-scale, 44.94 multi-scale) becomes the 2016 base ADE20K record, above Wider or Deeper Model A2's
    43.73, and PSPNet ResNet-152 and SAC are not records. RevColV2-B is 88M in its classification table but 101M with
    the columns its segmentation model uses, and its Mask2Former system is listed at 325M, against 107M for Mask2Former
    Swin-B.
14. **Test-resolution conventions.** MultiGrain's 500 px is the longer image side; FOVI's 256 px is the maximum side;
    AME's 128 px is the shorter side of 256×128 scenes; EfficientNet-B7's 600 px and AdvProp B8's 672 px come from
    the official code, not the papers; so do DINO's, SimCLR's, MoCo v3's, EsViT's, Rotation RevNet-50's and CMC's
    224 px linear-probe crops and the two GFNets' 300 and 260 px. PlainSeg's 640 px is its sliding-window crop.
15. **The all-size frozen ImageNet line is incomplete.** The base frozen series is convnets until 2020, from the
    Split-Brain AlexNet (2016) to BYOL ResNet-200. The larger models of the same papers are not in the JSON: SimCLR
    ResNet-50 (4×), 375M, 76.5 (Table 6 of arXiv v1), BYOL ResNet-50 (4×), 375M, 78.6 and ResNet-200 (2×), 250M,
    79.6 (Table 1 of arXiv v1), and CMC's ResNet-50 x2 (under "Verified, but left out"). The larger MoCo v3 and
    EsViT models were not read. The running maximum of all
    `frozen_ssl` points is therefore not an all-size frontier, at least before DINOv2 (2023-04-14); the all-size
    frozen series was not searched for completeness.
16. **Pretraining data that touches the benchmark.** DINOv2's LVD-142M includes 1M images retrieved with ADE20K
    training images as seeds (DINOv2 Table 15); this applies to every DINOv2 ADE20K number and to models initialised
    from DINOv2 (LookWhere, LookThere). DINOv3's data includes ImageNet-1k and ImageNet-21k.

## Notes for the paper's bibliography

- The DINOv3 TMLR PDF (Zotero copy, 2026) reads "Published in Transactions on Machine Learning Research
  (04/2026)"; `references.bib` has `month = feb`.
- The DINOv2 TMLR PDF reads "(01/2024)".
