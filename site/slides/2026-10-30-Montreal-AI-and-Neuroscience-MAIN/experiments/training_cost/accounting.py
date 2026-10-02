"""Each system's own training compute, as the NeurIPS rebuttal counted it (its training_flops.py, in the paper's
previous repository), on this repository's FLOP counts: canvit_pytorch.flops for CanViT-B and its DINOv3 teacher,
paper/exporter's formulas for AdaGlimpse and AME.

Conventions: 1 multiply-add = 2 FLOPs; a backward pass costs BWD_MULT forward passes (measured 1.996 on CanViT-B);
pretrained weights and teachers a system starts from are counted on no side (DINOv3 for CanViT-B, DeiT III for
AdaGlimpse, MAE or SETR for AME), but a teacher's forward passes during training are.
"""

from canvit_paper_exporter.flops import primitives as P
from canvit_paper_exporter.flops.adaglimpse import AdaGlimpseConfig, adaglimpse_flops
from canvit_paper_exporter.flops.ame import ame_flops
from canvit_paper_exporter.flops.canvit import CANVIT_B
from canvit_paper_exporter.flops.dinov3 import DINOV3_CONFIGS
from canvit_pytorch.flops import dinov3_flops, glimpse_flops, linear_flops

from experiments.glimpses import GLIMPSE_PX

BWD_MULT = 2.0
IN1K_TRAIN = 1_281_167
FFN_RATIO = 4.0

# CanViT-B's pretraining (the paper's pretraining hyperparameters): 2M steps of 64 scenes, an F-IID and an R-IID
# rollout per scene, 4 glimpses per rollout on average (K / p_stop = 2 / 0.5), a 32 x 32 canvas, DINOv3 ViT-B's 768-d
# features of the 512 px scene as targets.
STEPS, BATCH_SCENES, ROLLOUTS_PER_SCENE, GLIMPSES_PER_ROLLOUT = 2_000_000, 64, 2, 4
PRETRAINING_CANVAS_GRID, TEACHER_DIM, SCENE_PX = 32, 768, 512
NUM_SCENES = 13_200_000  # ImageNet-21k scenes whose teacher features were computed once
GLIMPSES = STEPS * BATCH_SCENES * ROLLOUTS_PER_SCENE * GLIMPSES_PER_ROLLOUT
# A pretraining glimpse: the forward pass, then the reconstruction heads (canvas patches and CLS to the teacher's dim).
PRETRAINING_GLIMPSE = (
    glimpse_flops(CANVIT_B, glimpse_size_px=GLIMPSE_PX, canvas_grid_size=PRETRAINING_CANVAS_GRID)
    + linear_flops(num_tokens=PRETRAINING_CANVAS_GRID**2, in_dim=CANVIT_B.canvas_dim, out_dim=TEACHER_DIM)
    + linear_flops(num_tokens=1, in_dim=CANVIT_B.backbone_spec.embed_dim, out_dim=TEACHER_DIM)
)
CANVIT_PRETRAINING = GLIMPSES * PRETRAINING_GLIMPSE * (1 + BWD_MULT)
TEACHER_FEATURES = NUM_SCENES * dinov3_flops(DINOV3_CONFIGS["DINOv3 ViT-B/16"], input_size_px=SCENE_PX)

# AdaGlimpse's ImageNet-1k pipeline: 600 epochs of reconstruction pretraining, then 100 of classification with 14
# glimpses of 32 px (encoder only, backward through the last step). Its paper and its code differ on the glimpses per
# pretraining image: 49 or 56 in the code, 196 in the paper.
ADAGLIMPSE_CLASSIFIER = AdaGlimpseConfig(glimpse_grid=2, dec_depth=0, cnn_layers=())
ADAGLIMPSE_CLASSIFICATION_GLIMPSES = 14
ADAGLIMPSE_PRETRAINING_GLIMPSES = {"code (49 patches)": 49, "code-RL (56)": 56, "paper (196)": 196}


def _adaglimpse_last_step(num_glimpses: int, config: AdaGlimpseConfig) -> int:
    return adaglimpse_flops(num_glimpses, config) - adaglimpse_flops(num_glimpses - 1, config)


def _adaglimpse_pretraining_forward(num_patches: int) -> int:
    """Encoder over the visible patches and CLS, then the decoder over them and the 196 mask tokens."""
    encoder = P.patch_embed(num_patches, 16, 768) + 12 * P.vit_block(num_patches + 1, 768, FFN_RATIO)
    decoder = 8 * P.vit_block(num_patches + 1 + 196, 512, FFN_RATIO)
    return encoder + decoder


ADAGLIMPSE_CLASSIFICATION = 100 * IN1K_TRAIN * (
    adaglimpse_flops(ADAGLIMPSE_CLASSIFICATION_GLIMPSES, ADAGLIMPSE_CLASSIFIER)
    + BWD_MULT * _adaglimpse_last_step(ADAGLIMPSE_CLASSIFICATION_GLIMPSES, ADAGLIMPSE_CLASSIFIER)
)
ADAGLIMPSE_PRETRAINING = {
    label: 600 * IN1K_TRAIN * (1 + BWD_MULT) * _adaglimpse_pretraining_forward(n)
    for label, n in ADAGLIMPSE_PRETRAINING_GLIMPSES.items()
}

# AME's ADE20K training (its paper): 75 epochs, 8 glimpses of 48 px, a loss at every step, so the backward pass runs
# through the whole rollout. ADE20K's 20,210 training images, or the 26K its paper states.
AME_SEGMENTATION = {n: 75 * n * (1 + BWD_MULT) * ame_flops(8) for n in (20_210, 26_000)}
