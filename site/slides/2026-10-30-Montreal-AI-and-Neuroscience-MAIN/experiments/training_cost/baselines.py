"""Training FLOPs of AdaptiveNN-DeiT-S (ImageNet-1k) and AdaGlimpse (ADE20K segmentation), printed with their
derivation (sources/baseline-training-compute.md in the talk); export.py reads ann_total_code, lo and hi. Runs in the
paper exporter's environment, from the talk's directory:

    uv run --project ../../../paper/exporter python -u -m experiments.training_cost.baselines

Conventions (accounting.py, the rebuttal's): 1 MAC = 2 FLOPs; backward = 2x forward
for every part a loss backpropagates through; frozen parts forward only; teachers'
own training excluded, their online forwards counted. Matmul/conv FLOPs only
(LayerNorm, softmax, GELU, interpolation, grid_sample excluded, as in primitives).
Code anchors: AdaptiveNN @ ba508e6, AdaGlimpse upstream @ e5f37fd.
"""

from dataclasses import dataclass

import numpy as np
from canvit_paper_exporter.flops import primitives as P
from canvit_paper_exporter.flops.adaglimpse import ADAGLIMPSE, adaglimpse_flops

from experiments.training_cost.accounting import BWD_MULT, IN1K_TRAIN

GF = 1e9
EF = 1e18


def gf(x: float) -> str:
    return f"{x / GF:9.3f} GF"


def ef(x: float) -> str:
    return f"{x / EF:8.3f} EF"


# ============================================================ AdaptiveNN-DeiT-S
# DeiT-S as built by models/dynamic_deitS.py:515-518 on the vendored
# models/timm/models/vision_transformer.py (no CLS token: :263 commented out,
# pos_embed has num_patches rows :265).
DEIT_DIM = 384
DEIT_FFN = 4.0
N_CLASSES = 1000
SEQ_L = 4                      # --seq_l 4 (GET_STARTED.md:37)
GLANCE_TOKENS = (112 // 16) ** 2   # --glance_input_size 112 -> 7x7
FOCUS_TOKENS = (112 // 16) ** 2    # --focus_patch_size 112 -> 7x7
REG_TOKENS = (160 // 16) ** 2      # --focus_net_reg_size 160 -> 10x10
REMAINING_BLOCKS = 4           # main_ppo.py:196 default (not overridden)
GLANCE_DEPTH = 12              # main_ppo.py:199 default
FOCUS_DEPTH = 12               # main_ppo.py:205 default
POLICY_HIDDEN = 128            # main_ppo.py:191
POLICY_KERNEL = 3              # main_ppo.py:192
RECOVER_N = 5                  # main_ppo.py:195
PPO_EPOCHS = 5                 # --ppo_update_steps 5 (GET_STARTED.md:39)
UPDATE_POLICY_FREQ = 10        # --update_policy_freq 10
EPOCHS = 300                   # main_ppo.py:46 default, Supp Tab. 1
GLOBAL_BATCH = 4096            # 8 procs x 512 (GET_STARTED.md:33,36)
ITERS_PER_EPOCH = (IN1K_TRAIN + 7) // 8 // 512   # DistributedSampler pads to ceil(N/8); drop_last=True (main_ppo.py:292)
assert ITERS_PER_EPOCH == 312
assert ITERS_PER_EPOCH == IN1K_TRAIN // GLOBAL_BATCH  # main_ppo.py:388 agrees
# PPO consumes iterations 10, 20, ..., 310; the buffer resets at data_iter_step == 0
# (engine_ppo.py:127-129), so the last 2 iterations of each epoch never reach PPO.
PPO_ITERS_PER_EPOCH = (ITERS_PER_EPOCH // UPDATE_POLICY_FREQ) * UPDATE_POLICY_FREQ
assert PPO_ITERS_PER_EPOCH == 310


def deit_blocks(n_tokens: int, depth: int) -> int:
    return depth * P.vit_block(n_tokens, DEIT_DIM, DEIT_FFN)


def cls_head() -> int:  # LayerNorm + avgpool + Linear(384 -> 1000)
    return P.linear(1, DEIT_DIM, N_CLASSES)


def policy_net_patch(n_out: int) -> int:
    """dynamic_deitS.py:26-79 on the 7x7x384 state (feature_size = glance 112//16)."""
    side = 7
    depthwise = 2 * side * side * DEIT_DIM * POLICY_KERNEL ** 2
    pointwise = P.conv2d(side, side, DEIT_DIM, POLICY_HIDDEN, 1)
    mlp = (P.linear(1, POLICY_HIDDEN * side * side, 1024) + P.linear(1, 1024, 512)
           + P.linear(1, 512, n_out))
    return depthwise + pointwise + mlp


ANN = {}
ANN["reg_pass (focus net @160, 100 tok, 12 blk + head)"] = (
    P.patch_embed(REG_TOKENS, 16, DEIT_DIM) + deit_blocks(REG_TOKENS, FOCUS_DEPTH) + cls_head())
ANN["glance (glance net @112: 8 blk + 4-blk head, 49 tok)"] = (
    P.patch_embed(GLANCE_TOKENS, 16, DEIT_DIM)
    + deit_blocks(GLANCE_TOKENS, GLANCE_DEPTH - REMAINING_BLOCKS)
    + deit_blocks(GLANCE_TOKENS, REMAINING_BLOCKS) + cls_head())
per_fix = {
    "feature reuse (dw7x7 conv + MLP 384-1536-384)":
        2 * 49 * DEIT_DIM * 49 + P.linear(FOCUS_TOKENS, DEIT_DIM, 4 * DEIT_DIM) + P.linear(FOCUS_TOKENS, 4 * DEIT_DIM, DEIT_DIM),
    "focus net (12 blk, 49 tok)": P.patch_embed(FOCUS_TOKENS, 16, DEIT_DIM) + deit_blocks(FOCUS_TOKENS, FOCUS_DEPTH),
    "recover MLP 384-1536-25 + aggregation matmul":
        P.linear(FOCUS_TOKENS, DEIT_DIM, 4 * DEIT_DIM) + P.linear(FOCUS_TOKENS, 4 * DEIT_DIM, RECOVER_N ** 2)
        + 2 * GLANCE_TOKENS * FOCUS_TOKENS * DEIT_DIM,
    "per-step classifier (4 blk + head)": deit_blocks(GLANCE_TOKENS, REMAINING_BLOCKS) + cls_head(),
}
for k, v in per_fix.items():
    ANN[f"x{SEQ_L} {k}"] = SEQ_L * v
ANN_TRAINED_FWD = sum(ANN.values())
POLICY_FWD = policy_net_patch(2)
VALUE_FWD = policy_net_patch(1)
ANN_ROLLOUT_AGENT_FWD = SEQ_L * (POLICY_FWD + VALUE_FWD)   # dynamic_deitS.py:326-337, outputs not in the loss
ANN_PPO_PER_IMAGE = PPO_EPOCHS * SEQ_L * (POLICY_FWD + VALUE_FWD) * (1 + BWD_MULT)  # engine_ppo.py:200-225

ANN_MAIN_PER_IMAGE = ANN_TRAINED_FWD * (1 + BWD_MULT) + ANN_ROLLOUT_AGENT_FWD
images_code = EPOCHS * ITERS_PER_EPOCH * GLOBAL_BATCH
images_ppo = EPOCHS * PPO_ITERS_PER_EPOCH * GLOBAL_BATCH
images_nominal = EPOCHS * IN1K_TRAIN
ann_total_code = images_code * ANN_MAIN_PER_IMAGE + images_ppo * ANN_PPO_PER_IMAGE
ann_total_nominal = images_nominal * (ANN_MAIN_PER_IMAGE + ANN_PPO_PER_IMAGE)
ANN_EVAL = EPOCHS * 50_000 * (ANN_TRAINED_FWD + ANN_ROLLOUT_AGENT_FWD)  # evaluate() each epoch, forward_backbone incl. reg pass

print("================ AdaptiveNN-DeiT-S, ImageNet-1k, from scratch ================")
print("forward per training image:")
for k, v in ANN.items():
    print(f"  {k:<62} {gf(v)}")
print(f"  {'sum of trained parts (all receive gradients)':<62} {gf(ANN_TRAINED_FWD)}")
print(f"  {'policy+value nets in rollout, x4 (forward only)':<62} {gf(ANN_ROLLOUT_AGENT_FWD)}")
print(f"per image fwd+bwd (trained parts x3, agent fwd x1):            {gf(ANN_MAIN_PER_IMAGE)}")
print(f"PPO per image: {PPO_EPOCHS} epochs x {SEQ_L} states x (policy+value) x3:   {gf(ANN_PPO_PER_IMAGE)}"
      f"   [policy {POLICY_FWD / 1e6:.2f} MF, value {VALUE_FWD / 1e6:.2f} MF per state]")
print(f"image-steps: code-exact {images_code:,} (312 it x 4096 x 300); PPO-consumed {images_ppo:,}; nominal {images_nominal:,}")
print(f"TOTAL code-exact:    {ef(ann_total_code)}")
print(f"  main fwd+bwd:      {ef(images_code * ANN_TRAINED_FWD * (1 + BWD_MULT))}")
print(f"  rollout agent fwd: {ef(images_code * ANN_ROLLOUT_AGENT_FWD)}")
print(f"  PPO updates:       {ef(images_ppo * ANN_PPO_PER_IMAGE)}")
print(f"TOTAL nominal (300 x 1,281,167): {ef(ann_total_nominal)}")
print(f"per-epoch val forwards (excluded): {ef(ANN_EVAL)}")
print(f"fixations (112 px): {images_code * SEQ_L:,} code-exact; glances {images_code:,}; reg passes {images_code:,}")

# Cross-check against the paper's inference costs (Supp Data Tab. 2/3, (avg fixations, GFLOPs)).
# Halting uses the value net (V(s_t) <= eta_t, main text), so inference runs one task head at the
# stopping step: G = glance 8 blk + one 4-blk head; F = policy + reuse + focus + recover + value.
fix = np.array([0.09, 0.43, 0.77, 1.11, 1.46, 1.80, 2.14, 2.48, 2.82, 3.15, 3.40])
gfl = np.array([1.20, 1.61, 2.01, 2.42, 2.82, 3.23, 3.63, 4.04, 4.44, 4.85, 5.25])
slope, intercept = np.polyfit(fix, gfl, 1)
G_inf = ANN["glance (glance net @112: 8 blk + 4-blk head, 49 tok)"]
F_inf = sum(per_fix[k] for k in per_fix if not k.startswith("per-step")) + POLICY_FWD + VALUE_FWD
print(f"paper inference fit: {intercept:.3f} + {slope:.3f} x fixations  ('GFLOPs' as printed)")
print(f"ours, MACs (FLOPs/2): G = {G_inf / 2 / GF:.3f}, F = {F_inf / 2 / GF:.3f}   "
      f"(F incl. per-step 4-blk head: {(F_inf + per_fix['per-step classifier (4 blk + head)']) / 2 / GF:.3f})")
print()


# ============================================================ AdaGlimpse
ENC_DIM, DEC_DIM, FFN = 768, 512, 4.0
GRID_TOKENS = 196  # 224/16 squared: decoder mask tokens (mae.py:207-211)
SEG_HEAD = sum(P.conv2d(*layer) for layer in ADAGLIMPSE.cnn_layers)  # mae_utils.py:287-308
RECON_HEAD = P.linear(GRID_TOKENS, DEC_DIM, 16 * 16 * 3)              # mae.py:78 decoder_pred
TEACHER_DEEPLAB_FWD = 92.371e9  # measured: deeplab_teacher_flops.py (torchvision deeplabv3_resnet101, 150 cls, 224px)


def enc(n: int) -> int:  # mae.py:162-194: n patch tokens + CLS, 12 blocks
    return P.patch_embed(n, 16, ENC_DIM) + 12 * P.vit_block(n + 1, ENC_DIM, FFN)


def dec(n: int, head: int) -> int:  # mae.py:196-232 with mask=None: (1+n) known + 196 mask tokens
    return P.linear(n + 1, ENC_DIM, DEC_DIM) + 8 * P.vit_block(n + 1 + GRID_TOKENS, DEC_DIM, FFN) + head


def attention_rollout(n: int) -> int:  # mae.py:283-320: 12 matmuls of (n+1)^2 x (n+1); early return at n=0
    return 0 if n == 0 else 12 * 2 * (n + 1) ** 3


def agent_net(patch_num: int, critic: bool) -> int:
    """actor_critic.py: per-patch conv + 4 input MLPs + attention pooling + head (hidden 256)."""
    per_token = (2 * 12 * 12 * 3 * 8 * 25 + 2 * 2 * 2 * 8 * 32 * 25          # patch_net_conv on 16x16
                 + P.linear(1, 32 + 1 + 4 + ENC_DIM, 128)                     # input_layers, common_dim 2*256//4
                 + P.linear(1, 512, 256) + P.linear(1, 256, 256) + P.linear(1, 256, 8))  # pooling in + attention MLP
    per_sample = 2 * patch_num * 256 + P.linear(1, 256, 256)                  # att @ latent, pooling output
    head = (P.linear(1, 3, 256) + P.linear(1, 512, 256) + P.linear(1, 256, 1)) if critic else \
           (P.linear(1, 256, 256) + P.linear(1, 256, 6))
    return patch_num * per_token + per_sample + head


def sac_update_per_sample(patch_num: int) -> float:
    """torchrl 0.3.1 SACLoss v2, 2 Q nets: fwd = actor(s') + 2 Qtarget(s',a') + 2 Q(s,a) + actor(s) + 2 Q(s,a~);
    bwd (x2) through actor + 2 Q (actor loss) and 2 Q (q loss)."""
    a, q = agent_net(patch_num, critic=False), agent_net(patch_num, critic=True)
    fwd = 2 * a + 6 * q
    bwd = BWD_MULT * (a + 2 * q) + BWD_MULT * (2 * q)
    return fwd + bwd


@dataclass(frozen=True)
class RlStage:
    name: str
    num_glimpses: int
    grid: int
    head: int
    images_per_epoch: int
    rl_epochs: int
    backbone_epochs: int
    sac_samples_per_image_step: float  # rl_batch_size / train_batch_size
    teacher_fwd: float

    @property
    def patch_num(self) -> int:
        return self.num_glimpses * self.grid ** 2

    def per_image(self) -> dict[str, float]:
        steps = range(self.num_glimpses + 1)  # t = 0..T forwards (glimpse_engine.py:62-70)
        n = [self.grid ** 2 * t for t in steps]
        enc_cum = sum(enc(k) for k in n)
        dec_cum = sum(dec(k, self.head) for k in n)
        rollout_cum = sum(attention_rollout(k) for k in n)
        final = enc(n[-1]) + dec(n[-1], self.head)
        epochs = self.rl_epochs + self.backbone_epochs
        return {
            "encoder fwd, all steps": epochs * enc_cum,
            "decoder+head fwd, all steps": epochs * dec_cum,
            "attention rollout": epochs * rollout_cum,
            "actor fwd in rollout (steps 0..T-1)": epochs * self.num_glimpses * agent_net(self.patch_num, False),
            "teacher fwd (once per episode, frozen)": epochs * self.teacher_fwd,
            "backbone bwd at t=T (backbone epochs)": self.backbone_epochs * BWD_MULT * final,
            "SAC updates (RL epochs)": self.rl_epochs * self.num_glimpses * self.sac_samples_per_image_step
                                       * sac_update_per_sample(self.patch_num),
        }

    def total(self) -> dict[str, float]:
        return {k: v * self.images_per_epoch for k, v in self.per_image().items()}


def alternating(epochs: int, freeze: int) -> tuple[int, int]:
    """rl_glimpse.py:570-593: RL-only for epoch < freeze, then RL on even, backbone on odd epochs."""
    rl = sum(1 for e in range(epochs) if e < freeze or e % 2 == 0)
    return rl, epochs - rl


# Cross-check primitives against the exporter's AdaGlimpse model (paper's 835.01 GF, measured 835.27 GF).
seg_steps_1_8 = sum(enc(9 * t) + dec(9 * t, SEG_HEAD) for t in range(1, 9))
print("================ AdaGlimpse ================")
print(f"check: exporter adaglimpse_flops(8) = {gf(adaglimpse_flops(8))}; ours t=1..8 (adds decoder_embed) = {gf(seg_steps_1_8)}; measured 835.27 GF")

# Stage A: 600-epoch backbone pretraining on ImageNet-1k (paper main.tex Training; Supp hyperparams).
# One encoder+recon-decoder pass per image, loss on it: (1 + BWD) x fwd.
pre = {}
for label, n in {"code ElasticImageNet1k, 49 patches": 49, "code RL --pretraining, 56 patches": 56,
                 "paper, 196 patches": 196}.items():
    fwd = enc(n) + dec(n, RECON_HEAD)
    pre[label] = 600 * IN1K_TRAIN * (1 + BWD_MULT) * fwd
    print(f"stage A pretraining {label:<38} fwd/img {gf(fwd)}  total {ef(pre[label])}")

# Stage B: reconstruction RL training on ImageNet-1k; config from the released
# Reconstruction_IMNET_12_glimpses_of_32_px.ckpt (hyper_parameters + loop state):
# T=12, grid 2, batch 128 on 4 GPUs (2502 batches/GPU/epoch), rl batch 128, alternating, freeze 10;
# 3,220,074 steps = 99 epochs x 32,526; 5,064,048 optimizer steps = 2502 x (55 x 36 + 44 x 1).
rl_b, bb_b = alternating(99, 10)
assert (rl_b, bb_b) == (55, 44)
assert 2502 * (rl_b * 12 * 3 + bb_b) == 5_064_048
recon = RlStage("recon 12x32px", 12, 2, RECON_HEAD, 2502 * 128 * 4, rl_b, bb_b, 128 / 128, 0.0)
recon_total = recon.total()
print(f"\nstage B recon RL (IN1k, 12 x 32px glimpses, 99 epochs = 55 RL + 44 backbone, {recon.images_per_epoch:,} img/epoch):")
for k, v in recon_total.items():
    print(f"  {k:<44} {ef(v)}")
print(f"  {'TOTAL':<44} {ef(sum(recon_total.values()))}")
rp = recon.per_image()
print(f"  per image-episode fwd (enc+dec, t=0..12): {gf((rp['encoder fwd, all steps'] + rp['decoder+head fwd, all steps']) / 99)}")

# Stage C: ADE20K segmentation fine-tuning (paper: 100 epochs; teacher DeepLabV3-R101 on the full scene;
# backbone batch 128, rl batch 256 per Supp hyperparams; frozen-backbone epochs 10 (table) or 30 (text)).
ADE_TRAIN = 20_210
seg_cases = {}
for T in (8, 12):
    for grid, grid_label in ((3, "48px=9 patches (paper)"), (2, "32px=4 patches (code default)")):
        for freeze in (10, 30):
            rl_c, bb_c = alternating(100, freeze)
            stage = RlStage(f"seg T={T} {grid_label} freeze {freeze}", T, grid, SEG_HEAD, ADE_TRAIN,
                            rl_c, bb_c, 256 / 128, TEACHER_DEEPLAB_FWD)
            seg_cases[(T, grid, freeze)] = stage
print(f"\nstage C seg (ADE20K {ADE_TRAIN:,} img/epoch, 100 epochs):")
for key, stage in seg_cases.items():
    tot = stage.total()
    print(f"  {stage.name:<52} RL/bb epochs {stage.rl_epochs}/{stage.backbone_epochs}  {ef(sum(tot.values()))}")
ref = seg_cases[(8, 3, 10)]
print(f"  breakdown for {ref.name}:")
for k, v in ref.total().items():
    print(f"    {k:<44} {ef(v)}")
ep = ref.per_image()
print(f"    per image-episode fwd (enc+dec+head, t=0..8): {gf((ep['encoder fwd, all steps'] + ep['decoder+head fwd, all steps']) / 100)}")
print(f"    one SAC sample-update at P=72: {gf(sac_update_per_sample(72))}; actor fwd {agent_net(72, False) / 1e6:.1f} MF")

# Released checkpoints name glimpse size as 16 x glimpse_grid_size ("16_px" -> grid 1, "32_px" -> grid 2),
# so the paper's 48 px segmentation glimpses are grid 3. Range ends use grid 3; grid 2 is printed above only.
seg_totals = {k: sum(s.total().values()) for k, s in seg_cases.items() if k[1] == 3}
B = sum(recon_total.values())
recon_9 = RlStage("recon 9x32px", 9, 2, RECON_HEAD, 2502 * 128 * 4, rl_b, bb_b, 1.0, 0.0)
B_9 = sum(recon_9.total().values())
print(f"\nsensitivity: stage B if seg started from the (unreleased) 9 x 32px recon model, same schedule: {ef(B_9)}")
lo = pre["code ElasticImageNet1k, 49 patches"] + B + min(seg_totals.values())
hi = pre["paper, 196 patches"] + B + max(seg_totals.values())
paper_cfg = pre["paper, 196 patches"] + B + seg_totals[(8, 3, 10)]
print("\nAdaGlimpse ADE20K chain totals (stage C at 48 px = 9 patches):")
print(f"  low  (A=49 patches, C=T8 freeze30):             {ef(lo)}")
print(f"  paper-stated (A=196, B, C=T8 freeze10):          {ef(paper_cfg)}")
print(f"  high (A=196 patches, C=T12 freeze10):           {ef(hi)}")
print(f"  without stage A (if pretraining is inherited):  {ef(B + min(seg_totals.values()))} .. {ef(B + max(seg_totals.values()))}")
print(f"  stage C alone:                                  {ef(min(seg_totals.values()))} .. {ef(max(seg_totals.values()))}")
eval_recon = 99 * 50_000 * sum(enc(4 * t) + dec(4 * t, RECON_HEAD) for t in range(13))
eval_seg = 100 * 2_000 * sum(enc(9 * t) + dec(9 * t, SEG_HEAD) for t in range(9))
print(f"  per-epoch validation forwards (excluded): recon {ef(eval_recon)}, seg {ef(eval_seg)}")
print(f"glimpses: stage B {12 * recon.images_per_epoch * 99:,} (32 px); stage C {8 * ADE_TRAIN * 100:,} at T=8 (48 px), "
      f"{12 * ADE_TRAIN * 100:,} at T=12")
_a100 = 672 * 3600
print(f"README '1 week on 4x A100' = 672 A100-h; stage B implied FLOP/s per GPU: {B / _a100 / 1e12:.1f} TFLOP/s")
