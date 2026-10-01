[Counted by a Claude Code subagent on 2026-10-01 from AdaptiveNN's and AdaGlimpse's code, papers and released checkpoints, under the rebuttal's conventions (training_flops.py); the script is throwaway/training_cost/baselines.py. Not reviewed by the authors.]

# Training compute of two prior active-vision models (2026-10-01)

Scope: own training compute behind AdaptiveNN-DeiT-S at 82.2% ImageNet-1k top-1, and behind AdaGlimpse at 25.7 mIoU on ADE20K (8 glimpses of 48 px). Counted by Claude from code, paper text and released checkpoint metadata on 2026-10-01. Nothing here was run on a GPU; per-forward costs are analytic, with two measured cross-checks.

- AdaptiveNN: "Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception" (Wang, Yue, Yue et al., Nature Machine Intelligence 2025, arXiv:2509.15333). It trains a glance network plus a fixation ("focus") network with a PPO-trained fixation policy.
- AdaGlimpse: "AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale" (Pardyl, Wronka, Wołczyk, Adamczewski, Trzciński, Zieliński, ECCV 2024, arXiv:2404.03482). It is a ViT-B encoder over accumulated glimpse patches, with an MAE-style decoder and a Soft Actor-Critic (SAC) agent that picks glimpse position and scale.

Reproduce: `uv run --no-sync --project ~/code/CanViT-paper-exporter python -u /Users/yberreby/.claude/jobs/2e07edf0/tmp/baseline_training_flops.py` (output saved in `/Users/yberreby/.claude/jobs/2e07edf0/tmp/baseline_training_flops.log`). Teacher measurement: `deeplab_teacher_flops.py` in the same directory.

## Results

| model, result | own training compute | ends of the range |
|---|---:|---|
| AdaptiveNN-DeiT-S, 82.2% IN-1k | **22.6 EF** per training run | 22.58 EF code-exact training; 22.93 EF if the per-epoch validation forwards are counted |
| AdaGlimpse, 25.7 mIoU ADE20K | **83.7 to 166.6 EF** | low: the code's 49-patch pretraining + 8-glimpse seg training; high: the paper's 196-patch pretraining + 12-glimpse seg training. Paper-stated configuration (196 patches, 8 glimpses, 10 frozen epochs): 165.7 EF |
| AdaGlimpse, if its 600-epoch pretraining is inherited | 32.4 to 33.5 EF | reconstruction stage + segmentation stage only (see "AdaGlimpse stage A lineage") |

EF = 10^18 FLOPs, 1 multiply-add = 2 FLOPs. The AdaGlimpse total is a chain of three stages. The segmentation stage alone is 2.25 to 3.28 EF; the rest is ImageNet-1k training the segmentation model inherits.

Training glimpse counts:
- AdaptiveNN: 4 fixations of 112 px plus 1 glance (whole image downsampled to 112 px) plus 1 regularization pass (whole image at 160 px) per image-step. That gives 1,533,542,400 fixations over 383,385,600 image-steps.
- AdaGlimpse: segmentation uses 8 glimpses of 48 px (9 patches each) per image-epoch, so 16,168,000 glimpses over 100 epochs (24,252,000 if trained at 12). Reconstruction uses 1,521,856,512 glimpses of 32 px. The pretraining uses random patches, not glimpses: 600 × 1,281,167 × {49, 56, 196} = 37.7B, 43.0B or 150.7B patches.

## Conventions

These follow `~/code/CanViT-Toward-AVFMs/rebuttal/training_flops.py`:
- 1 MAC = 2 FLOPs.
- Backward = 2 × forward for every part a loss backpropagates through; frozen parts are forward only.
- Pretrained weights and teachers a system starts from are excluded. A teacher run online during training has its forwards counted.
- Per-forward costs come from `canvit_paper_exporter.flops.primitives` (`vit_block`, `linear`, `patch_embed`, `conv2d`). These count matmul and conv FLOPs only; LayerNorm, softmax, GELU, interpolation and `grid_sample` are excluded.
- Validation forwards and hyperparameter search are excluded and reported separately where computable.

Code anchors: AdaptiveNN at `~/code/AdaptiveNN @ ba508e6`. For AdaGlimpse, the upstream state is `e5f37fd` (Adam Pardyl, 2024-07-30). `~/code/AdaGlimpse` is the m2b3 fork, whose later commits change only import paths, timing hooks and eval exploration mode; files were read from `git show e5f37fd:<path>`. Paper anchors: AdaptiveNN `~/code/AdaptiveNN/2509.15333v1.txt:<line>`. AdaGlimpse arXiv e-print extracted to `/Users/yberreby/.claude/jobs/2e07edf0/tmp/adaglimpse_src/` (`main.tex:<line>`, `tables_sup/hyperparams.tex:<line>`).

---

## 1. AdaptiveNN-DeiT-S on ImageNet-1k

### What one training iteration executes

The command is `GET_STARTED.md:33-40`: 8 processes × batch 512 = 4,096, `--input_size 288 --glance_input_size 112 --focus_patch_size 112 --focus_net_reg_size 160 --seq_l 4 --ppo_update_steps 5 --update_policy_freq 10`. Unspecified flags take the defaults in `main_ppo.py:186-237`. `main_ppo.py:11` puts `./models` first on `sys.path`, so `timm` resolves to the vendored, modified `models/timm`. Its `VisionTransformer` has no CLS token (`vision_transformer.py:263` is commented out; `pos_embed` has `num_patches` rows, `:265`). Token counts are therefore 49 at 112 px and 100 at 160 px. DeiT-S here is dim 384, MLP ratio 4 (`dynamic_deitS.py:515-518`).

Per image, `forward_backbone` (`dynamic_deitS.py:284-414`) runs:

| component | code | tokens | forward |
|---|---|---:|---:|
| regularization pass: focus net on the image resized to 160 px, 12 blocks + norm + pool + head | `:300`, `vision_transformer.py:341-367` | 100 | 4.491 GF |
| glance: image resized to 112 px, glance net blocks[:-4] (8) then blocks[-4:] (4, the step-0 task head) + head | `:302-314`, `vision_transformer.py:375-386` | 49 | 2.155 GF |
| ×4 feature reuse: depthwise 7×7 conv on the 7×7 state + MLP 384→1536→384 | `:363-371`, `:196-212`, `:417-427` | 49 | 0.470 GF |
| ×4 focus net on a 112 px crop of the 288 px image, 12 blocks (`remaining_blocks=0`, no norm/head) | `:361`, `:381` | 49 | 8.616 GF |
| ×4 recover MLP 384→1536→25 + (49×49)@(49×384) aggregation | `:387-402`, `:216-222` | 49 | 0.254 GF |
| ×4 per-step task head: 4 untied DeiT blocks + norm + pool + head | `:405-411`, `:135-147` | 49 | 2.837 GF |
| **sum of trained parts** | | | **18.822 GF** |
| ×4 policy net + value net on the detached 7×7×384 state (rollout, forward only) | `:326-337`, `:26-79` | | 0.152 GF |

Architecture cross-check against the paper. It says "the first eight blocks of DeiT for the initial processing of the down-sampled glance inputs" and "a task-specific head whose architecture adopts ... four DeiT blocks" (`2509.15333v1.txt:2325-2328`). The glance net is built at depth 12 (`main_ppo.py:199`) and split 8 + 4 by `remaining_blocks=4` (`main_ppo.py:196`).

All four fixations run every iteration: `for focus_step_index in range(seq_l)` with no break (`dynamic_deitS.py:319`). All four step outputs enter the loss (`engine_ppo.py:103-105`), as do the glance output (`:102`) and the regularization output (`:78-80`, weight 2.0).

### Which parts get gradients

`loss = loss_reg_focus_net + loss_focus + loss_glance + loss_KD` (`engine_ppo.py:106`), one backward per iteration (`:119-123`). The gradient reaches:
- the focus net, through the regularization pass and every fixation;
- the glance net, through its own head and through `updated_features` and the feature-reuse path, neither detached (`dynamic_deitS.py:363-371, 402`);
- all MLPs and per-step heads.

The policy and value nets read `updated_features.detach()` (`:328, :334`), and their outputs (sampled actions, `actions.detach()` at `:354`; log-probs; values) are not in this loss. They get no backward here. `skip_policy_net=True` only nulls `.grad` after `backward()` (`utils.py:388-401`); it does not change what backward computes. The self-distillation target is `x_focus[-1].detach()` (`engine_ppo.py:83`), with no extra forward.

Per image: 3 × 18.822 + 0.152 = **56.62 GF**.

### PPO

PPO runs every 10 iterations (`engine_ppo.py:134`) on the stored rollouts of those 10 iterations. States are the stored detached features (`dynamic_deitS.py:325`, `engine_ppo.py:170`), and rewards come from the stored logits (`engine_ppo.py:161-167`). **No backbone features are recomputed for PPO.** Only the policy and value nets run, via `evaluate_policy_net` (`dynamic_deitS.py:239-281`) over all 4 steps.

The loop runs `ppo_update_steps=5` epochs × `num_ppo_update_iters=1` minibatch, and that one minibatch is the whole buffer: `ppo_total_batch_size = batch_size × update_policy_freq` = 5,120 per GPU (`engine_ppo.py:196-204`, `main_ppo.py:237`). Each collected image therefore gets 5 × 4 × (policy + value) × (fwd + 2 × fwd). The policy and value nets are 19.05 MFLOPs each per state, so this is **2.286 GF per image**.

The collection buffer resets at `data_iter_step == 0` (`engine_ppo.py:127-129`). Only iterations 1 to 310 of the 312 per epoch reach PPO.

### Totals

Iterations per epoch: `DistributedSampler` pads 1,281,167 to ceil(N/8) = 160,146 per rank, and `drop_last=True` (`main_ppo.py:287-293`) gives 312 batches of 512. This matches `len(dataset_train) // 4096 = 312` (`main_ppo.py:388`). 300 epochs (`main_ppo.py:46`, Supp Data Tab. 1) gives 383,385,600 image-steps.

| term | EF |
|---|---:|
| trained parts, fwd + bwd | 21.648 |
| policy/value forward in rollout | 0.058 |
| PPO updates (380,928,000 images reach PPO) | 0.871 |
| **total, code-exact** | **22.577** |
| total with nominal 300 × 1,281,167 images | 22.640 |
| per-epoch validation forwards (excluded; `evaluate` runs `forward_backbone` incl. the regularization pass on 50,000 images, `main_ppo.py:475-476`) | 0.285 |

### Cross-check against the paper's own inference costs

Supp Data Tab. 2/3 (`2509.15333v1.txt:4355-4457`) lists 11 (average fixations, GFLOPs) rows, from (0.09, 1.20) to (3.40, 5.25). A least-squares fit gives 1.078 + 1.203 per fixation. The paper halts on the value net, "If Vπ(st) ≤ ηt" (`:925`), so inference runs one 4-block head at the stopping step. Under that reading my count, in MACs, gives:
- glance + one head = 1.077 G;
- per fixation (policy + reuse + focus + recover + value) = 1.187 G.

Both agree with the fit to within 1.3%. This checks the token counts, depths and the absence of a CLS token. It also indicates that the paper's "GFLOPs" are MACs, which this report converts at 2 FLOPs/MAC.

### Assumptions and what the code leaves open

1. Hyperparameters not on the README command take code defaults (`main_ppo.py:186-237`). The paper agrees on the architecture (8 + 4 glance blocks, 3×3 depthwise policy conv, C→128). Whether this exact command produced 82.2% cannot be observed; Supp Data Tab. 1 matches it.
2. Backward = 2 × forward throughout. Patch embeddings need no input gradient, which this convention overstates by about 0.03 GF/image.
3. **Five trials.** Fig. 3 error bars are "standard deviations of five independent trials with different random seeds" (`:1223-1224`), and Supp Tab. 3 averages "over 5 trials" (`:4457`). The per-trial columns differ (75.3 to 75.8 at 0.09 fixations). The text does not say whether the trials are five training runs: test-time fixations are deterministic, but the halting thresholds come from a genetic algorithm (`:2316`), which could also vary. If they are five training runs, the published 82.2 ± 0.12 sits on 5 × 22.6 = 113 EF. The slide quantity (one run) is 22.6 EF.
4. Excluded: the hyperparameter search on a 20% held-out split (`:2090`, compute undisclosed), validation forwards (0.285 EF), and non-matmul ops.
5. No external teacher: self-distillation uses the model's own final step. Training is from scratch (`pretrained=False`, `main_ppo.py:350`; `adaptivenn_training_facts.md` §1).

---

## 2. AdaGlimpse on ADE20K (25.7 mIoU, 8 × 48 px)

### What the 25.7 mIoU model trains from

The paper (`main.tex:326`) says: "In segmentation experiments we fine-tune a model trained for reconstruction". It also states "we pre-train the model for 600 epochs with 196 random glimpses per image" (also `tables_sup/hyperparams.tex:39`, "backbone pre-training epochs 600"). The chain is:

- **A.** 600-epoch backbone pretraining on ImageNet-1k.
- **B.** Reconstruction RL training on ImageNet-1k (the paper trains reconstruction on ImageNet-1k only, `main.tex:361`).
- **C.** Segmentation fine-tuning on ADE20K.

No segmentation checkpoint is released. The Hugging Face repo `apardyl/AdaGlimpse` holds two reconstruction checkpoints and one classification checkpoint. I read their pickled headers (training arguments and loop state) by HTTP range request, without downloading the weights; the script is `/Users/yberreby/.claude/jobs/2e07edf0/tmp/read_remote_ckpt_header.py` and the outputs are the `hdr_*.log` and `loops_*.log` files in that directory.

### Glimpse size and patches per glimpse

The paper's k = ceil(d_cam/16)² gives 9 patches for 48 px (`main.tex:260`). The code splits each glimpse into `glimpse_grid_size`² sub-patches resized to 16 px (`interactive_sampler.py:21-71`, `shared_memory.py:21`), so a glimpse's sampling resolution is 16 × grid. The default grid is 2 (`rl_glimpse.py:221-224`), but it is a command-line flag. The released checkpoints name glimpse size by exactly this rule:
- `Reconstruction_IMNET_12_glimpses_of_16_px.ckpt` has `glimpse_grid_size: 1`;
- `..._32_px.ckpt` (reconstruction and classification) have `glimpse_grid_size: 2`.

So the paper's 48 px segmentation glimpses are grid 3, 9 patches. This is consistent with the paper formula and with the table's pixel percentage: 8 × 48² / 224² = 36.73% (`tables/table_segmentation.tex:18`). This revises the reading in `rebuttal/adaglimpse_flops_empirical.md` and the `training_flops.py` header that the code "always" uses 4 patches; that holds only at the default flag. The grid-2 case is printed in the log (2.14 to 3.04 EF for stage C) but is excluded from the range ends.

### Per-step structure (all stages B and C)

- Each image batch becomes `num_glimpses + 1` training steps, t = 0..T (`glimpse_engine.py:62-70`; `rl_glimpse.py:444`).
- Each step re-encodes all patches gathered so far plus CLS (`rl_glimpse.py:509-511`, `shared_memory.py:47-49`, `mae.py:162-194`). It then runs the decoder over (1 + n) known tokens plus 196 mask tokens (`mae.py:196-232`, `mask=None`) and the task head (`rl_glimpse.py:513`, `:943`).
- Grad is enabled only for the final step's forward (`rl_glimpse.py:506-507`). The backbone backward runs once per episode at t = T, only in backbone epochs (`:749-750`). The paper agrees: "Optimization is performed only on the last exploration step" (`main.tex:270`).
- The segmentation head is `VisionTransformerUpHead` (`mae_utils.py:287-308`): 6 convs up to 224 × 224 at 150 classes, 85.0 GF per call, run at every step.
- Schedule `alternating` (`rl_glimpse.py:570-593`): RL-only for epochs < `freeze_backbone_epochs`, then RL on even epochs and backbone on odd epochs.
- SAC: torchrl 0.3.1 (pinned in `environment.yml`). No value network is passed (`rl_glimpse.py:104-110`), so this is SACLoss v2 with 2 Q nets (`torchrl/objectives/sac.py @ v0.3.1`, `_actor_loss`, `_compute_target_v2`, `_qvalue_v2_loss`).
  - Per sampled transition, forward is 2 actor passes + 6 Q passes; backward runs through actor + 2 Q (actor loss) and 2 Q (Q loss) (`rl_glimpse.py:600-627`).
  - One update per step for t = 1..T in RL epochs; step 0 returns early (`:709-712`).
  - The actor and critic see all T × grid² patch slots (per-patch conv, input MLPs, attention pooling; `actor_critic.py`). At 72 slots the actor forward is 59.9 MF and one sample-update is 1.08 GF.

Measured confirmation of this structure from the released reconstruction checkpoint. Its loop state has 3,220,074 training steps = 99 × 32,526, where 32,526 = 2502 batches × 13 steps per epoch, i.e. 4 GPUs × 128 images. Its optimizer-step count is 5,064,048 = 2502 × (55 RL epochs × 12 SAC updates × 3 optimizers + 44 backbone epochs × 1). That is exactly the alternating schedule with a 10-epoch freeze over 99 epochs. The checkpoint's `epoch` field says 76, which does not fit; the run was resumed from another run's `last.ckpt` (`hyper_parameters.resume`), which may explain it. The step counters and the OneCycle `total_steps = 3,252,600 = 100 × 32,526` agree with each other.

### Stage A: 600-epoch pretraining on ImageNet-1k

One encoder + reconstruction-decoder pass per image with a loss on it (`elastic_mae.py:65-74`), × 3 for fwd + bwd, × 600 × 1,281,167. Patches per image, in three cases as in `training_flops.py`:

| case | fwd per image | stage A |
|---|---:|---:|
| code `ElasticImageNet1k`, 49 random patches (`datasets/elastic.py:28` @ e5f37fd) | 22.21 GF | 51.2 EF |
| code RL `--pretraining`, 4 + 13 × 4 = 56 patches (`patch_sampler.py:83-103`) | 23.85 GF | 55.0 EF |
| paper, 196 patches (`main.tex:326`) | 57.75 GF | 133.2 EF |

The forward costs equal the FlopCounterMode measurements in `adaglimpse_flops_empirical.md` (b).

### Stage B: reconstruction RL training on ImageNet-1k

The configuration is from `Reconstruction_IMNET_12_glimpses_of_32_px.ckpt` hyper_parameters: `num_glimpses 12`, `glimpse_grid_size 2`, `epochs 100`, `freeze_backbone_epochs 10`, `backbone_training_type alternating`, `train_batch_size 128`, `rl_batch_size 128`, `rl_iters_per_step 1`, `pretrained_mae_path 'elastic_mae.ckpt'`. This matches the README reconstruction command and the reconstruction table's headline row "12 × 32²" (`tables/table_recon.tex:12`). Counted at 99 epochs, the steps executed by the released best checkpoint (55 RL + 44 backbone epochs), × 1,281,024 images per epoch (2502 × 128 × 4).

| term | EF |
|---|---:|
| encoder forward, t = 0..12 | 7.100 |
| decoder + linear head forward, t = 0..12 | 19.951 |
| backbone backward at t = 12 (44 epochs) | 2.477 |
| SAC updates (55 epochs, 12 per batch, rl/train batch 1) | 0.610 |
| actor forward in rollout + attention rollout | 0.062 |
| **stage B** | **30.200** |

The decoder runs over at least 197 tokens at every step and dominates. Forward per image-episode is 213.3 GF.

Sensitivity: if the segmentation model was fine-tuned from the unreleased 9 × 32² reconstruction model instead, stage B is 21.6 EF under the same schedule.

Wallclock cross-check: the README says "around 1 week on 4x A100 GPUs" for ImageNet-1k tasks, i.e. 672 A100-hours. That implies 12.5 TFLOP/s per GPU for stage B, about 8% of A100 TF32 dense peak. The run had `fp16: False` and `torch.set_float32_matmul_precision('high')` at `train.py:19`. This is plausible for an RL loop that samples glimpses on CPU workers; it is not a tight check.

### Stage C: segmentation on ADE20K

Paper-stated settings: 100 epochs (`main.tex:326`; `train.py` has no early-stopping callback, only best-checkpoint selection on `val/mPA`, `train.py:78-80`). Backbone batch 128 and RL batch 256 (`tables_sup/hyperparams.tex:50-51`). Teacher: DeepLabV3 with a ResNet-101 backbone on the full scene (`main.tex:270`), run online under `inference_mode` once per episode, every training epoch (`rl_glimpse.py:642-650, 688-689`). ADE20K train split: 20,210 images (`segmentation.py:21, 56` lists `images/training`; the count is the one `training_flops.py` uses).

Teacher forward: **92.37 GFLOPs** per 224 px image at 150 classes, measured with `FlopCounterMode` on torchvision 0.26 `deeplabv3_resnet101(num_classes=150)` (all `aten.convolution`; no aux head). The authors pinned torchvision 0.17.1; I assume the same graph (output stride 8, ASPP), which I have not checked against 0.17.1 source.

| case (48 px, 9 patches) | RL / backbone epochs | stage C |
|---|---|---:|
| T = 8, 10 frozen epochs (`tables_sup/hyperparams.tex:48`) | 55 / 45 | 2.290 EF |
| T = 8, 30 frozen epochs (main text, `main.tex:326`) | 65 / 35 | 2.249 EF |
| T = 12, 10 frozen epochs | 55 / 45 | 3.278 EF |
| T = 12, 30 frozen epochs | 65 / 35 | 3.237 EF |

Breakdown at T = 8 with 10 frozen epochs:

| term | EF |
|---|---:|
| decoder + up-head forward, t = 0..8 | 1.764 |
| backbone backward at t = 8 | 0.203 |
| teacher forward | 0.187 |
| encoder forward | 0.116 |
| SAC | 0.019 |

T = 12 is in range because the supplementary segmentation figure uses "12×48² adaptive glimpses" (`figures_sup/seg_ade/fig.tex:18`). The table's 4- and 8-glimpse rows may come from one model trained at a different T.

### AdaGlimpse stage A lineage (affects whether A is "own" compute)

- The released reconstruction run starts from `elastic_mae.ckpt` (unreleased).
- The repo's pretraining module `ElasticMae` initializes by default from `./elastic-224-30random70grid.pth` (`elastic_mae.py:38-41`). That checkpoint is published under `apardyl/BeyondGrids`, the authors' earlier "Beyond Grids: Exploring Elastic Input Sampling for Vision Transformers" (Pardyl, Kurzejamski, Olszewski, Trzciński, Zieliński, 2023, arXiv:2309.13353; venue not checked).
- Its saved args: supervised DeiT-III-style ViT-B (`deit_base_patch16_LS`, `bce_loss`, `ThreeAugment`), `epochs 800`, `random_patches 196` of 16 to 48 px, 70% grid, best checkpoint at epoch 780.
- The README's backbone for training AdaGlimpse, `BeyondGrids/base_224.pth`, is the same recipe with `random_patches 200`, best at epoch 768. The released classification checkpoint starts from `elastic_base.pth`.
- The paper's "600 epochs with 196 random glimpses ... sampled from a uniform distribution" resembles the Beyond Grids recipe (196 uniform random patches) but with a different epoch count.

I cannot tell from public material whether the 600-epoch pretraining is AdaGlimpse's own run, or a Beyond Grids model it inherits (which the convention would exclude). The headline counts it, as the paper says "we pre-train" and `training_flops.py` does; the "without stage A" line (32.4 to 33.5 EF) is the alternative. The same question bears on the ImageNet-1k pretraining term in `training_flops.py`.

### Assumptions and what could not be determined

1. The segmentation run's own arguments (T, freeze epochs, batch sizes, world size, whether the encoder was frozen) are not released. I used paper values; T and freeze define the stage C range. A frozen encoder (`--freeze-encoder`, default off) would remove part of the 0.2 EF backward.
2. Which reconstruction model seeded segmentation is not stated. I assumed the released 12 × 32² one (30.2 EF); the 9 × 32² alternative gives 21.6 EF.
3. Stage A uses the nominal 1,281,167 images per epoch; its world size and drop_last loss are unknown (under 0.1% at 4 GPUs).
4. Released code and published runs differ in version. The checkpoints' hyper_parameters contain keys absent from the released code (classification: `patch_mix_alpha`, `use_distilled_targets`) and lack keys it has (`teacher_type`, `pretrained_checkpoint`, `freeze_encoder`). I assume the FLOP-relevant structure is unchanged; this cannot be verified.
5. The DeepLabV3 teacher's own training is excluded as a teacher the system starts from. torchvision ships no 150-class ADE20K DeepLabV3, and the repo has a `--teacher-pretraining` mode that trains one on ADE20K (`rl_glimpse.py:970-983`), so it was plausibly the authors' own compute.
6. Excluded: validation forwards (stage B 1.06 EF, stage C 0.19 EF), hyperparameter search, and non-matmul ops.

---

## Related corrections to existing rebuttal notes (not edited)

- `rebuttal/adaglimpse_flops_empirical.md` ("Code-vs-paper contradictions" 1) and the `training_flops.py` header say the code always uses 4 patches per glimpse. The grid is a flag, and the authors' own 16 px and 32 px checkpoints use grid 1 and grid 2. The paper's 9-patch pricing of 48 px glimpses (835.01 GF) is the consistent one.
- `rebuttal/adaptivenn_training_facts.md` cites code anchors `@ 020bf8d`. The local clone is `ba508e6`; the line numbers I cite here are from `ba508e6`.
