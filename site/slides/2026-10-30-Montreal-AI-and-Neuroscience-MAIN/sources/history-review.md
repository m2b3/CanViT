[A Claude Code subagent's review of the history charts, 2026-10-01, with five verification subagents, every number against its paper; their per-group reports and the papers' text were scratch files, not kept. Not reviewed by the authors.]

# Review of the slide "The wide gap between passive and active computer vision" (#history)

Reviewed 2026-10-01 on CanViT HEAD 888c045 (history.js and sources/sota-history.json unchanged since cb6aa60, clean
working tree for both). Slide: site/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/index.html, section #history.

Provenance tags used below:
- [me] I read it this session in the primary source (paper PDF text, arXiv abs page, or official README).
- [A], [B], [C], [E], [F] a verification agent read it this session in the primary source and saved quotes; its
  report was in a session scratchpad (not kept). Where I re-read the agent's number in the
  paper text myself, the tag is [me+X].
- All downloaded PDFs and their pdftotext extractions: a session scratchpad (not kept).

## Short answer

1. **The values are right.** Every number on both charts matches its paper to the printed digit: 4 ImageNet-1k
   active points and 2 ADE20K active points (checked by me), plus every passive point (checked by the five agents,
   with key numbers re-read by me). No value is wrong. The faults are in the comparisons the chart draws.
2. **On the rendered ImageNet-1k chart, two active models sit above the dashed passive line.**
   - **GFNet** (79.8, 2020-10-11) sits 2.0 points above it. At that date the dashed line is BYOL ResNet-200 at 77.8.
   - **Saccader** (75.03, 2019-08-20) sits above the dashed line's first level (74.2). The dashed line only starts
     six months later, on 2020-02-13.
   - No active model sits above the solid "trained end to end" line, on either chart.
   - These two crossings are what the author saw. Speaker-note claim for this step: "Below, every time." That claim
     is false for the dashed line.
3. **Cause: a protocol mismatch.** It is neither a data error nor a missing record.
   - The dashed line contains only self-supervised backbones read out by a linear probe. GFNet and Saccader are
     supervised and trained end to end from supervised ImageNet weights.
   - In 2019-2020, self-supervised linear probes trailed supervised training. In BYOL's own Table 9, ResNet-200 scores
     77.8 frozen against 79.3 trained with labels.
   - The slide labels the dashed line "frozen + linear decoding", which does not say "self-supervised". Under that
     label, frozen supervised features count. A frozen supervised ResNet-50 v2 with a linear probe scored 75.8 on
     2019-01-25 (Kolesnikov et al., Table 2) [me+C], above Saccader. With such points, the dashed line would track
     the solid one.
4. **The relevant passive competition is the solid line, or a like-for-like slice of it.** Like-for-like here means
   base size, ImageNet-1k data only, and the active model's resolution. Against it, every drawn active model trails
   at its date:

   | Active model | Gap to the like-for-like passive best |
   |---|---|
   | Saccader | 6.6 to 7.9 points |
   | GFNet | 2.2 points |
   | AdaGlimpse | 8.0 points |
   | AdaptiveNN | 1.6 to 3.3 points |
   | AME and AdaGlimpse on ADE20K | more than 20 points |

5. **Other mismatches, and which way each moves the drawn gap:**
   - The solid ImageNet-1k line mixes in extra data from 2019-11 on: JFT-300M, ImageNet-21k, CLIP teachers, and a
     ViT-22B teacher trained on JFT-4B. That widens the gap for every active model trained only on ImageNet-1k.
   - The solid line's records test at 384 to 800 px; the active models test at 224 to 331 px. That also widens the
     gap.
   - The active points are not filtered by size. Saccader-NASNet (124.5M) and AME (a ViT-L encoder, initialised from
     SETR weights already trained on ADE20K) are large, yet they are drawn against base-size lines. That narrows the
     gap in the active models' favour.
   - The ADE20K bracket ("24.2 points") subtracts a 2023 large active model from a 2025 base passive probe. That
     probe was measured under DINOv3's own protocol. Under the CanViT paper's protocol, the same model gives 47.19.

## 1. What the chart draws

Logic of history.js, replicated in a script in a session scratchpad (not kept) [me]:
- **Points:** the data comes from sources/sota-history.json.
- **Solid line:** the running maximum (a new step only when a point beats every earlier one) over
  `kind == "passive" && size_class == "base" && series != "frozen_ssl"`, minus the three FLAGGED entries. The
  ImageNet line therefore mixes the in1k_only and overall series. The ADE20K line mixes single-scale, multi-scale and
  unstated protocols.
- **Dashed line:** the running maximum over base-size frozen_ssl entries, excluding the boosted "linear probe + ms"
  protocol.
- **Active points:** the best entry of each paper in the ACTIVE list (Saccader, GFNet, AdaGlimpse, AdaptiveNN, AME).
  No size filter applies. STAM and the select-once models (LookWhere, LookThere) are not drawn.
- **Dates:** each point sits at its month. `yearOf` maps YYYY-MM-DD to year + (month - 0.5)/12, so two points in the
  same month share an x position.
- **Check on the live page:** I read the SVG of the live page [me], served by the session's existing server on port
  8765 (site/serve.mjs, the same checkout). Circle and path coordinates match the replication. Rendered positions on
  the ImageNet-1k chart (y = 500 - (v - 60) × 13.14):
  - Saccader circle at (481.1, 302.5), which is 75.03.
  - The dashed path starts at (509.0, 313.4), which is 74.2, 28 px to Saccader's right.
  - GFNet circle at (546.2, 239.8), which is 79.8. The dashed path there is at y = 266.1, which is 77.8.

**ImageNet-1k**

| Line | Steps drawn, as date value (model) |
|---|---|
| Solid, base, any data | 2012 59.3 (AlexNet), then 61.0 (AlexNet with Fall-2011 pretraining) · 2013-11 61.6 (ZFNet) · 2013-12 62.5 (Howard) · 2014-06 65.84 (SPP-net) · 2015-02 79.9 (BN-Inception, 6-model ensemble, 144 crops) · 2015-12 82.8 (Inception-v3, 4-model ensemble) · 2017-12 82.9 (PNASNet-5, 331 px) · 2019-04 83.6 (MultiGrain PNASNet, 500 px) · 2019-05 84.4 (EfficientNet-B7, 600 px) · 2019-09 85.0 (RandAugment B7) · 2019-11 85.5 (AdvProp B8), then 86.8 (Noisy Student B7, JFT) · 2020-01 86.9 · 2020-03 87.1 (FixEfficientNet-B7, Noisy Student weights, JFT, 632 px) · 2021-06 87.3 (CoAtNet-2, 21k) · 2022-10 87.5 (BEiT v2), then 87.7 (MOAT-2, 22k), then 88.1 (CAFormer-B36, 21k) · 2023-02 88.6 (ViT-B/16 distilled from ViT-22B, JFT-4B teacher, 384 px), flat to the end |
| Dashed, base, self-supervised | 2020-02 74.2 (SimCLR R50 2×) · 2020-06 77.8 (BYOL R200 1×) · 2021-04 79.5 (MoCo v3), then 80.1 (DINO B/8) · 2021-06 81.3 (EsViT Swin-B) · 2023-04 84.5 (DINOv2 B/14) · 2026-03 85.0 (the CanViT paper's probe of DINOv3 B/16 at 512 px) |

| Active point | Value | Solid line at its date | Dashed line at its date |
|---|---|---|---|
| Saccader-NASNet, 2019-08-20 | 75.03 | 84.4 (−9.4) | not yet drawn; starts at 74.2 on 2020-02-13 |
| GFNet, 2020-10-11 | 79.8 | 87.1 (−7.3) | **77.8 (+2.0, above)** |
| AdaGlimpse, 2024-04-04 | 77.54 | 88.6 (−11.1) | 84.5 (−7.0) |
| AdaptiveNN-DeiT-S, 2025-09-18 | 82.2 | 88.6 (−6.4) | 84.5 (−2.3) |

**ADE20K**

| Line | Steps drawn |
|---|---|
| Solid, base, mixed protocols | 2016-12 42.62, then 43.51 (PSPNet-152) · 2017 44.3 (SAC) · 2018-03 44.65 (EncNet) · 2018-09 45.08 (OCNet) · 2018-11 45.22 (CCNet) · 2019-09 45.66 (OCR) · 2019-11 45.9 (ACNet) · 2020-04 46.27, then 46.91 (ResNeSt-101) · 2020-12 48.36 (ResNeSt-200) · 2021-03 49.02 (DPT-Hybrid), then 51.6 (Swin-B, 22k) · 2021-05 51.8 (SegFormer-B5) · 2021-07 52.6 (CSWin-B), 52.7, 53.9 (MaskFormer Swin-B, 22k) · 2021-12 55.1 (Mask2Former Swin-B, 22k, multi-scale) · 2023-03 55.3 (EVA-02-B) · 2023-09 55.8 (RevColV2-B, multi-scale) |
| Dashed, base, linear probe | 2021-04 31.8, then 34.5 (DINO, measured in later papers) · 2021-11 38.3 (iBOT B/16) · 2023-04 47.3 (DINOv2 B/14) · 2025-08 51.8 (DINOv3 B/16, DINOv3's protocol) |

| Active point | Value | Solid line at its date | Dashed line at its date |
|---|---|---|---|
| AME (SETR-initialised ViT-L), 2023-03-11 | 27.6 | 55.1 on the date; 55.3 on the chart, because EVA-02-B (2023-03-20) shares the month | 38.3 |
| AdaGlimpse, 2024-04-04 | 25.7 | 55.8 | 47.3 |

The bracket on the "gap" step reads "24.2 points": 51.8 (dashed line end, 2025) minus 27.6 (AME, 2023). The CSS hides
the ImageNet-1k bracket (2.8 points) on #history.

## 2. Point-by-point verification

### Active points (all [me])

| Point | Value | Date | Params / size class | Resolution | Data, init, teacher | Verdict |
|---|---|---|---|---|---|---|
| Saccader-NASNet | 75.03 ± 0.08: arXiv 1908.07644v1, section before the Conclusion: "with 6 glimpses, the top-1 ... accuracy were 75.03 ± 0.08% ... processing only 29.47 ± 0.26% of the image with the NASNet" | v1 2019-08-20 | Table Supp.1 (v1): "Saccader-NASNet 124,537,764". **Large** (above 110M) | Locations chosen on the image downsized to 224 px; NASNet classifies 113 px patches cut from 331 px images (Table Supp.3 note) | NASNet trained by the authors (Tables Supp.2-3), fine-tuned on patches; no extra data mentioned (ImageNet-1k only is inferred) | verified; it is large but drawn against a base-size line |
| GFNet (EfficientNet backbones, T=4) | 79.8 is a figure reading. From the vector data of Fig. 4(c) of arXiv 2010.05300v1 (its extraction in a session scratchpad (not kept)), I get the top GFNet vertex at y = 87.6 pt against gridlines 80 at 85.76 and 79 at 95.56, i.e. **79.81**, at x = 0.753 G multiply-adds. In the same panel, the supervised "EfficientNets" baseline's highest marker is 79.80 at 1.0 G multiply-adds | v1 2020-10-11 (only version) | Not stated in the paper. 37.41M is the earlier agent's count from the official code, not re-run. Base is inferred: two EfficientNet-B2/B3 encoders, App. Table 2 | Not stated for the EfficientNet GFNets (JSON null, correct) | App. A.3: EfficientNets "train[ed] from scratch following all the details mentioned in their papers ... to match the reported performance": ImageNet-1k, supervised | value verified to about ±0.05; params unverifiable from the paper |
| AdaGlimpse (ImageNet-1k) | 77.54: arXiv 2404.03482v1 Tab. 2, "Ours 77.54 14 × 32² adaptive 28.57" | v1 2024-04-04 | Sec. 4: encoder "of the same size as standard ViT-B". 86.88M is a code count, not re-run. Base | 224 px | Teacher: "a pre-trained ViT from [43]" (DeiT III). The official README names `deit_3_base_224_21k.pth`, read in github.com/apardyl/AdaGlimpse README on master. That is DeiT III ViT-B trained on ImageNet-21k, which scores 85.7 on ImageNet-1k at 224 (DeiT III v1, arXiv 2204.07118, Table 8; checkpoint-to-row mapping inferred). Backbone pre-trained 600 epochs on random glimpses (Sec. 4) | verified; it used ImageNet-21k through the teacher |
| AdaGlimpse (ADE20K) | 25.7: v1 Tab. 3, "Ours 70.0 32.8 25.7 224 × 224 8 × 48² adaptive 36.73" (PA, mPA, IoU) | 2024-04-04 | as above (ViT-B encoder; decoder excluded) | 224 × 224 scenes | Teacher: DeepLabV3 ResNet-101 (Sec. 3.2). "In segmentation experiments we fine-tune a model trained for reconstruction" | verified |
| AdaptiveNN-DeiT-S | 82.2: arXiv 2509.15333v1, Supplementary Data Tab. 2, "3.15 4.85 82.2 ± 0.11" and "3.40 5.25 82.2 ± 0.12". The same table's own DeiT-S baselines, same recipe: 79.9 at 224, 80.9 at 288, 81.6 at 384 | v1 2025-09-18 | Not stated in the paper. 89.41M is a code count, not re-run. Base | 288 px image, 112 px glance and fixations (Supp. Tab. 1) | ImageNet-1K, 300 epochs, DeiT pipeline (Supp. Tab. 1); no teacher named | verified; params unverifiable from the paper |
| AME (SETR weights) | 27.6: arXiv 2303.06457v1 Table 3, "OURS (SETR-WEIGHTS) 35.6 69.5 27.6" (mPA, PA, IoU). With MAE weights: 24.4 | v1 2023-03-11 | "24 transformer blocks ... embedding size of 1024 (the same parameters as the ViT-L architecture)". **Large** | 256 × 128 images, 48 px glimpses | Encoder initialised from SETR weights "trained on ADE20k" | verified; it is large and initialised from an ADE20K-trained passive model |
| STAM (not drawn) | 80.78: arXiv 2204.00656v1 Table 1, "STAM (DeiTD-Base) ... 27.7K(t = 26) 80.78" | 2022-04-01 | DeiT-B distilled core, base | 224 px | Teacher: "publicly available weights" of DeiT-distilled. DeiT v1 (arXiv 2012.12877) names that model's teacher RegNetY-16GF, "that we trained with the same data augmentation as DeiT", i.e. ImageNet-1k | verified; not on the chart |

### Passive points

No value is wrong in any group. Details, quotes and locations are in the agent reports.

| Group | Report | Values | Field problems found |
|---|---|---|---|
| ImageNet-1k solid, before 2019 (9 entries) | verify-A-in1k-e2e-pre2019.md | all verified | (1) AlexNet note: the "39.0% without averaging" footnote belongs to the ILSVRC-2010 test result, not the 2012 "1 CNN". (2) ZFNet: 61.6 is not that paper's best base-size number: v1 (2013-11-12) has 61.7; v3 model (b) has 62.5 at 107.6M (computed). (3) Howard 62.5 then only ties ZFNet and is not a step. (4) AlexNet Fall-2011 "base" cannot be checked: the extra layer's size is unstated. (5) Inception-v3's 82.8 is on 48,238 images (82.7 on all 50,000). (6) The ensembles count as base only by summing members. |
| ImageNet-1k solid, 2019 on (17 entries) | verify-B-in1k-e2e-2019on.md | all verified | (1) EfficientNet-B7's 600 px and AdvProp B8's 672 px come from the official code and other papers, not their own papers. (2) Refiner's token labeling is inferred: its paper only cites the recipe. (3) VOLO's 86.3 is in Tables 4 and 7. (4) Yalniz's test size is not stated. (5) Yalniz 84.3 (2019-05-02) should be a step before EfficientNet-B7 (2019-05-28), but month rounding puts both at the same x (no visible effect). |
| ImageNet-1k dashed (9 entries) | verify-C-in1k-frozen.md | all verified | (1) EsViT note: 80.5 is the view-level-loss-only model. (2) DINOv2 B/14's 86.58M is not stated in the paper, whose Table 17 says 18 blocks; it is correct for the released 12-block model (computed). (3) "Table 9 of the DINOv3 TMLR version" is unverifiable (HTTP 403). (4) Test size not printed for SimCLR, MoCo v3 and EsViT. |
| ADE20K solid (25 entries) | verify-E-ade-e2e.md | all verified | (1) EncNet's protocol is multi-scale (Sec. 3.1), not "unstated". (2) The OCR, CPNet and DPT `test_px` values are training crops. (3) PSPNet's and CPNet's "ImageNet-1k" pretraining is inferred. (4) RevColV2-B is 88M or 101M depending on the paper's own table; base either way. (5) PSPNet ResNet-269's size is unknown; if base, the 2016 step is 44.94. |
| ADE20K dashed (13 entries) | verify-F-ade-frozen.md | all verified | (1) The DINO 31.8 and 34.5, MAE 33.3 and iBOT-L 44.6 are dated by model release. The JSON `_about` defines a date as the number's first public appearance: 2023-04-14 (DINOv2 paper) for 31.8, 33.3 and 44.6, and 2021-11-15 (iBOT paper) for 34.5. (2) The two DINO numbers come from different probes: four concatenated layers without BN over 160k iterations, against the last layer with BN over 40k. (3) DINOv2 B/14 params as in group C. |

Numbers I re-read myself in the paper text [me]:
- **Supervised and frozen baselines:**
  - Kolesnikov 2019 Table 2: supervised ResNet50 v2, linear probe, 75.8.
  - TResNet-XL at 224: 82.0.
  - CAFormer-B36, ImageNet-1k only: 85.5 at 224 and 86.4 at 384.
  - DeiT III v1 Tables 7 and 8: ViT-B 83.8 (1k, 224), ViT-B 85.7 (21k, 224); ConvNeXt-B 83.8 (224).
  - DeiT-B 81.8 at 224 and 83.1 at 384; distilled teacher RegNetY-16GF.
- **Other passive checks:**
  - FixRes PNASNet-5 at 480: 83.7.
  - Oct-ResNet-152+SE at 224: 81.6.
  - EVA-02-B: 55.3. RevColV2-B+M2F: 54.9 / 55.8.
  - DVT Table 2: DINOv2-reg 48.22; plain DINOv2 reproduced 47.29; arXiv 2401.02957 submitted 2024-01-05.

## 3. Is each line the relevant competition at each active model's date?

**The dashed ImageNet-1k line is not the relevant competition for any drawn active model.**
- All four drawn ImageNet-1k active models train their weights with ImageNet labels:
  - Saccader: NASNet trained and fine-tuned with labels.
  - GFNet: encoders from supervised EfficientNets, then trained with labels.
  - AdaGlimpse: KL to a supervised teacher.
  - AdaptiveNN: supervised DeiT recipe.
- The dashed line holds backbones that never saw a label, read by one linear layer.
- In 2020, no base-size self-supervised probe reached 77.8 or more before GFNet's date [C]. The closest:
  - SimCLRv2 R152 1×+SK (89M): 77.2.
  - BYOL R50 2×: 77.4.
  - SwAV R50-w2: 77.3.
- With no size cap, the self-supervised maximum at GFNet's date is 79.8 (SimCLRv2 R152 3×+SK, 795M) [C], which ties
  GFNet.
- So GFNet's position above the line is real for the data as defined. It says nothing about active against passive
  vision.
- The dashed line is the relevant comparison for a frozen, label-free readout. That covers CanViT-B's frozen probe on
  #results, and Prisadnikov et al. 2026, whose 75.0 is not drawn (inferred).

**The solid line is the relevant protocol, but it is mismatched in data and resolution.** Like-for-like references
at each active date, all base size:

| Active model (data, test px) | Chart's solid line | Base, ImageNet-1k only, any resolution | Base, ImageNet-1k only, near the active model's resolution | Same-family baseline in the active paper |
|---|---|---|---|---|
| Saccader 75.03 (1k, 331 px; 124.5M) | 84.4 | 84.4 (EfficientNet-B7, 600 px) [B] | 82.9 at 331 px (PNASNet-5, 86.1M, 2017-12) [A]; 81.6 at 224 px (Oct-ResNet-152+SE, 66.8M, 2019-04-10) [me+A] | not reported as a number |
| GFNet 79.8 (1k, px not stated) | 87.1 (JFT) | 85.7 (FixEfficientNet-B8, 800 px) in the file; 85.8 with KDforAA B8 (2020-03-25, missing) [B] | 82.0 at 224 px (TResNet-XL, 77.1M, 2020-03-30) [me+B] | supervised EfficientNet curve in the same Fig. 4(c): 79.80 at 1.0 G multiply-adds against GFNet's 79.81 at 0.75 G [me] |
| AdaGlimpse 77.54 (1k images + 21k teacher, 224 px) | 88.6 (JFT-4B teacher) | 86.5 (MOAT-2, 512 px) in the file; 86.6 with HIRI-ViT-L (768 px, 2024-03-18, missing) [B] | 85.5 at 224 px, 1k only (CAFormer-B36, 99M) [me+B]; its own teacher, DeiT III ViT-B 21k at 224: 85.7 [me] | STAM 76.13 at a matched glimpse budget (its Tab. 2) [me] |
| AdaptiveNN 82.2 (1k, 288 px) | 88.6 | 86.5 / 86.6 as above | 85.5 at 224 (CAFormer-B36); DeiT III ViT-B 83.8 at 224 [me] | DeiT-S 80.9 at 288 and 81.6 at 384 (Supp. Tab. 2) [me]. AdaptiveNN is above these, but they are a 22M model against AdaptiveNN's 89M (code count) |

- **ADE20K solid line, at AME's date:**
  - Mixed-protocol line: 55.1 [E].
  - Base size, ImageNet-1k pretraining only, single-scale: 52.4 (Mask2Former Swin-B 1k, 640 crop) [E].
- **At AdaGlimpse's date:** 55.8 mixed; 54.7, or 53.0 counting only explicitly labelled single-scale results, at
  ImageNet-1k-only single-scale [E].
- **Scene size:** the passive models test at a 512-640 px short side. AME uses 256 × 128 scenes, AdaGlimpse
  224 × 224 [E].
- **ADE20K dashed line at the two active dates:** 38.3 at AME and 47.3 at AdaGlimpse, or 48.22 with the missing DVT
  record [F, me].
- Every like-for-like line sits above both active points by more than 10 points.

### Missing records (read in their own papers by the agents; none moves an active model across a line)

- **ImageNet-1k solid:**
  - Pre-2015 steps: ZFNet v1 61.7 (2013-11-12), ZFNet v3 (b) 62.5, SPP-net v2 67.99 (2014-08-29; 66.06 if 100-110M
    is excluded), He & Sun 68.2 (2014-12-04) [A].
  - Yalniz 84.3 (2019-05-02), hidden by month rounding [B].
- **ImageNet-1k-only series (not drawn separately):** KDforAA B8 85.8 (2020-03-25), Dual-ViT-L 86.5 (2022-07-11),
  HIRI-ViT-L 86.6 (768 px, 2024-03-18) [B].
- **Extra-data series:** MaxViT-B 21k 88.38 (2022-04-04) is 119M, just over the cap [B].
- **ImageNet-1k dashed:**
  - InfoMin ResNeXt-101 74.5 (87M, 2020-05-20) [C].
  - Pre-2020 self-supervised records would start the line in 2019 [C]: Rotation 55.4, BigBiGAN 61.3, CMC 65.0, MoCo
    R50w2× 65.4, PIRL-c2x 67.4. The line would then stand at 61.3 at Saccader's date, so Saccader would sit even
    further above it.
- **ADE20K solid:** GFF 45.33 (2019-04-03), APCNet 45.38 (CVPR 2019), VOLO-D3 52.9 multi-scale (2021-06-24) [E]. No
  base-size result above 55.8 was found.
- **ADE20K dashed:** DINOv2-reg ViT-B/14 48.22 (DVT paper, 2024-01-05, 518 px) [F, me]. It raises the line at
  AdaGlimpse's date from 47.3 to 48.22.

## 4. Data mismatches between the passive lines and the active models

- **ImageNet-1k, the solid line includes extra data.** history.js keeps both the overall and in1k_only series. From
  2019-11 on, its steps are:
  - Noisy Student B7 (JFT-300M unlabeled, plus an EfficientNet-L2 teacher trained on JFT pseudo-labels).
  - FixEfficientNet-B7 (Noisy Student weights).
  - CoAtNet-2 and MOAT-2 (ImageNet-21k).
  - BEiT v2 (ImageNet-21k plus an OpenAI CLIP-B/16 tokenizer teacher).
  - CAFormer-B36 (21k).
  - ViT-B distilled from ViT-22B (a JFT-4B teacher) [B].
- **Which active models that mismatches:**
  - Saccader, GFNet and AdaptiveNN are ImageNet-1k only [me]. For GFNet and AdaptiveNN the chart compares them
    against JFT-trained models: 87.1 instead of 85.7 or 85.8, and 88.6 instead of 86.5 or 86.6.
  - AdaGlimpse is the exception: its teacher (`deit_3_base_224_21k.pth`) saw ImageNet-21k [me]. A 21k-matched line at
    its date is CAFormer-B36's 88.1 [B]; the JFT-4B-distilled 88.6 still exceeds that.
  - STAM's teacher chain is ImageNet-1k only [me], but STAM is not drawn.
- **The ImageNet-1k-only series itself allows ImageNet-1k teachers** [B]:
  - NFNet-F6 token labels (VOLO, Dual-ViT; inferred for Refiner).
  - A RegNetY-16GF teacher (DeiT distilled).
  - An EfficientNet-B7 teacher (KDforAA).
  - On the active side, STAM is matched in kind: its DeiT-B distilled teacher is likewise ImageNet-1k only.
- **ADE20K:**
  - The solid line is ImageNet-21k-pretrained from 2021-03 (Swin-B, CSWin-B, MaskFormer, Mask2Former); EVA-02-B adds
    a CLIP teacher; RevColV2 adds 21k fine-tuning [E].
  - AME's 27.6 starts from SETR, a ViT-L pretrained on ImageNet-21k and trained on ADE20K [me].
  - AdaGlimpse uses an ADE20K-trained DeepLabV3 teacher [me].
  - DINOv2's LVD-142M includes 1M images retrieved with ADE20K training images as seeds [F].
- **Resolution, ImageNet-1k:**
  - Solid-line records from 2019: 500 to 672 px through 2020 (MultiGrain 500, EfficientNet-B7 and Noisy Student B7
    600, AdvProp B8 672, FixEfficientNet-B7 632), then 384 to 512 px.
  - Active models: 224 to 331 px [B, me].
- **Resolution, ADE20K:**
  - Passive models: 512-640 px crops.
  - Active models: 256 × 128 (AME) and 224 × 224 (AdaGlimpse) scenes, of which they see 56% and 37% [E].
- **Size, which favours the active side:**
  - Saccader-NASNet is 124.5M and AME's encoder is a ViT-L. Both are drawn against base-size lines.
  - AGENTS.md ("Claims to state with their scope") says the passive lines are "restricted to models of the active
    models' size". That does not hold for these two.

## 5. Should any active model appear above the relevant passive line?

No.
- At its date, every drawn active model is below:
  - the base-size passive models trained end to end;
  - the ImageNet-1k-only slice;
  - the resolution-matched slice.
- **Where the chart shows otherwise:** GFNet above the dashed line, and Saccader above the dashed line's first level.
  - Not a data error: all values verified.
  - Not a missing record: no base-size self-supervised probe reached 77.8 before 2020-10-11, and adding the missing
    pre-2020 points lowers the line at Saccader's date.
  - The cause is a definitional choice. The dashed line is self-supervised only, its label omits that, and
    supervised, end-to-end-trained active models are drawn against it.
- **Smaller presentation defects found on the way:**
  1. Speaker note "Below, every time" is false for the dashed line.
  2. The ADE20K bracket mixes dates (2023 active, 2025 passive), size (ViT-L against ViT-B) and probe protocol. 51.8
     is DINOv3's protocol; 47.19 is the CanViT paper's, whose square-resize evaluation the paper says matches the
     active baselines [F]. At AME's own date the dashed gap is 10.7 points (38.3 − 27.6).
  3. Month rounding in `yearOf` hides real date order within a month (Yalniz before EfficientNet-B7; AME nine days
     before EVA-02-B).
  4. The ADE20K dashed line's early points are dated by model release, against the JSON's own date definition [F].

## 6. Options

These are for the authors; none was applied.
- **Option 1: draw the active models against the solid line only on #history.** Keep the dashed line for #results,
  where CanViT-B's frozen probe is like for like. Alternatively, relabel the dashed line "frozen self-supervised
  features + linear decoding" and change the notes.
- **Option 2: pick a data regime for the solid ImageNet-1k line.** Either the ImageNet-1k-only series (84.4, 85.8,
  86.6 at the active dates, adding the missing KDforAA and HIRI-ViT-L), which matches Saccader, GFNet and
  AdaptiveNN; or keep the mixed line and say it includes JFT and 21k.
- **Option 3: apply the base-size cap to active points too, or mark Saccader-NASNet and AME as large.** Saccader's base
  model number (70.3 per TNet's table) was not verified this session.
- **Option 4: make the ADE20K bracket date-matched and protocol-matched.**
