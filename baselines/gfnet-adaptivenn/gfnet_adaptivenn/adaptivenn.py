"""AdaptiveNN-DeiT-S on ImageNet-1k at a fixed number of fixations, measured with its released code.

    uv run python -m gfnet_adaptivenn.adaptivenn accuracy --root <sources> --imagenet-val <dir> --batch-size 32 \\
        --out adaptivenn_accuracy.json
    uv run python -m gfnet_adaptivenn.adaptivenn flops --root <sources> --out adaptivenn_flops.json
    uv run python -m gfnet_adaptivenn.adaptivenn parameters --root <sources> --out adaptivenn_parameters.json

`accuracy` runs the evaluation command of AdaptiveNN's GET_STARTED.md (a CUDA
GPU). Every image takes the glance and every fixation, with no early exit,
and its evaluator scores the prediction after each, so one pass gives top-1 at
every number of fixations. `flops` counts GFLOPs, at 2 FLOPs per
multiply-accumulate, with torch.utils.flop_counter over the same model's forward at each
number of fixations, on CPU and without the checkpoint, whose weights do not
change the count. Attention runs on PyTorch's math backend, whose matrix
products the counter sees. `parameters` counts the parameters of the modules
that forward runs up to each prediction (`active_parameters`); the
`active_params` command combines it with GFNet's count.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import torch
import tyro
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from gfnet_adaptivenn import imagenet, record
from gfnet_adaptivenn.active_params import cumulative_counts, millions, parameters_of
from gfnet_adaptivenn.sources import ADAPTIVENN, ADAPTIVENN_CHECKPOINT, Sources

EVALUATION_ARGUMENTS = (
    "--model", "dynamic_deitS", "--lr", "4e-3", "--seq_l", "4", "--clip_grad", "5.0",
    "--input_size", "288", "--glance_input_size", "112", "--focus_patch_size", "112",
    "--focus_net_reg_size", "160", "--loss_reg_focus_net_weight", "2.0",
    "--gamma", "0.5", "--ppo_lam", "0.84", "--ppo_update_steps", "5", "--update_policy_freq", "10",
)
"""The model and evaluation options of GET_STARTED.md's evaluation command (`main_ppo.py --eval true`)."""

TOP1_LINE = re.compile(r"^\* Acc@1 \S+ Acc@5 \S+ loss \S+\n((?:\S+\n)+)\* Glance Acc@1 (\S+) ", re.MULTILINE)
"""evaluate() prints top-1 after the last fixation, then after each fixation, then after the glance."""


def _options(sources: Sources) -> Any:
    """main_ppo.py's parsed options for the evaluation command, with its own defaults for the rest."""
    code = sources.repository(ADAPTIVENN)
    # main_ppo.py puts ./models first on the path, so that `timm` is the version vendored there.
    sys.path[:0] = [str(code / "models"), str(code)]
    import main_ppo  # pyright: ignore[reportMissingImports]

    return main_ppo.get_args_parser().parse_args([*EVALUATION_ARGUMENTS, "--eval", "true"])


def accuracy(sources: Sources, *, imagenet_val: Path, batch_size: int, log_dir: Path) -> dict[str, Any]:
    code, checkpoint = sources.repository(ADAPTIVENN), sources.checkpoint(ADAPTIVENN_CHECKPOINT)
    log = log_dir / f"{ADAPTIVENN_CHECKPOINT.file}.log"
    with imagenet.evaluation_root(imagenet_val) as root, log.open("w") as output:
        subprocess.run(
            (sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1", "main_ppo.py",
             "--data_path", str(root), "--output_dir", str(log_dir / "adaptivenn_output"),
             "--batch_size", str(batch_size), *EVALUATION_ARGUMENTS, "--eval", "true", "--resume", str(checkpoint)),
            cwd=code, stdout=output, stderr=subprocess.STDOUT, check=True,
        )
    [(after_each_fixation, after_glance)] = TOP1_LINE.findall(log.read_text())
    options = _options(sources)
    top1 = [float(after_glance), *map(float, after_each_fixation.split())]
    assert len(top1) == options.seq_l + 1, (log, top1)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    return record.provenance(ADAPTIVENN, device=torch.device("cuda")) | {
        "checkpoint": {"source": ADAPTIVENN_CHECKPOINT.url, "sha256": ADAPTIVENN_CHECKPOINT.sha256,
                       "params_total": sum(value.numel() for value in weights.values())},
        "accuracy": {
            "n_images": imagenet.num_images(imagenet_val), "early_exit": False, "input_size": options.input_size,
            "glance_input_size": options.glance_input_size, "focus_patch_size": options.focus_patch_size,
            "seq_l": options.seq_l, "batch_size": batch_size,
            "top1_by_k": {str(k): value for k, value in enumerate(top1)},
        },
    }


def build_model(options: Any) -> Any:
    """main_ppo.py's construction of the model, on the CPU, untrained."""
    from timm.models import create_model  # pyright: ignore[reportMissingImports]

    # main_ppo.py imports models.dynamic_deitS, which registers the model with timm.
    return create_model(
        options.model, seq_len=options.seq_l, feature_in_chans=options.feature_in_chans,
        policy_net_hidden_chans=options.policy_net_hidden_chans, policy_net_kernel_size=options.policy_net_kernel_size,
        recover_n=options.recover_n, remaining_blocks=options.remaining_blocks,
        glance_input_size=options.glance_input_size, glance_net_depth=options.glance_net_depth,
        glance_net_mlp_ratio=options.glance_net_mlp_ratio, glance_net_drop_path=options.glance_net_drop_path,
        focus_patch_size=options.focus_patch_size, focus_net_reg_size=options.focus_net_reg_size,
        focus_net_depth=options.focus_net_depth, focus_net_mlp_ratio=options.focus_net_mlp_ratio,
        focus_net_drop_path=options.focus_net_drop_path, multi_cls_drop_path=options.multi_cls_drop_path,
        pretrained=False, num_classes=options.nb_classes,
    ).eval()


def active_parameters(model: Any) -> list[int]:
    """The parameters of the modules `forward_backbone` runs for the prediction after the glance and each fixation.

    The glance and focus ViTs run their patch embedding, position embedding
    and blocks; their norm and head serve only the training regularizer
    (`infer_focus_net_only`). The glance adds `policy_net_norm[0]` and
    `policy_net_head[0]`. The first fixation adds the focus ViT, the patch
    policy's convolutions and linear layers, and the feature-reuse and
    recovery networks. Fixation i adds the layer norms indexed by i
    (`policy_net_patch_norm_list`, the policy's `ln_list`,
    `policy_net_recover_reuseNorm_list`, `policy_net_recover_reuseMLP_preLN_list`,
    `policy_net_recover_preLN_list`), its classifier `policy_net_classifier_list[i]`,
    `policy_net_norm[i + 1]` and `policy_net_head[i + 1]`. The value head
    (`policy_net_patch_stateV` and its norms) runs at every fixation but serves
    only training.
    """

    def vit(net: Any) -> set[torch.nn.Parameter]:
        return parameters_of(net) - parameters_of(net.norm, net.head)

    patch_policy = model.policy_net_patch
    first_fixation = vit(model.focus_net) | (parameters_of(patch_policy) - parameters_of(patch_policy.ln_list)) | (
        parameters_of(model.policy_net_recover_reuseConvHead, model.policy_net_recover_reuseMLP,
                      model.policy_net_recover)
    )
    added = [vit(model.glance_net) | parameters_of(model.policy_net_norm[0], model.policy_net_head[0])]
    for i, classifier in enumerate(model.policy_net_classifier_list):
        per_fixation = parameters_of(
            model.policy_net_patch_norm_list[i], patch_policy.ln_list[i], model.policy_net_recover_reuseNorm_list[i],
            model.policy_net_recover_reuseMLP_preLN_list[i], model.policy_net_recover_preLN_list[i],
            classifier, model.policy_net_norm[i + 1], model.policy_net_head[i + 1],
        )
        added.append(per_fixation | (first_fixation if i == 0 else set()))
    not_run = parameters_of(model.glance_net.norm, model.glance_net.head, model.focus_net.norm, model.focus_net.head,
                            model.policy_net_patch_stateV, model.policy_net_patch_norm_list_stateV)
    return cumulative_counts(added, not_run=not_run, every=parameters_of(model))


def parameters(sources: Sources) -> dict[str, Any]:
    weights = torch.load(sources.checkpoint(ADAPTIVENN_CHECKPOINT), map_location="cpu", weights_only=True)["model"]
    model = build_model(_options(sources))
    model.load_state_dict(weights)
    counts = active_parameters(model)
    return {
        "provenance": record.provenance(ADAPTIVENN, device=torch.device("cpu")),
        "model": {
            "whole_checkpoint_M": millions(sum(value.numel() for value in weights.values())),
            "per_k": {str(fixations): {"needed_M": millions(count), "parameters": count}
                      for fixations, count in enumerate(counts)},
        },
    }


def flops(sources: Sources) -> dict[str, Any]:
    options = _options(sources)
    model = build_model(options)
    image = torch.randn(1, 3, options.input_size, options.input_size)

    def counted(forward: Any) -> int:
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]), counter:
            forward()
        return counter.get_total_flops()

    # The forward also runs the focus network on a view of the whole image, a regularizer of the training loss
    # whose output evaluation discards; it is counted once and left out.
    regularizer = counted(lambda: model.infer_focus_net_only(image))
    per_k = {str(k): {"head_every_fixation_gflops": (counted(lambda: model(image, seq_l=k)) - regularizer) / 1e9}
             for k in range(options.seq_l + 1)}
    return record.provenance(ADAPTIVENN, device=torch.device("cpu")) | {
        "unit": "2 FLOPs per MAC", "image_px": options.input_size, "sdpa_math_forced": True,
        "regularizer_gflops_left_out": regularizer / 1e9, "per_k_cumulative": per_k,
    }


@dataclass(frozen=True)
class Accuracy:
    root: Path
    imagenet_val: Path
    """ImageNet-1k's validation images in ImageFolder layout: a directory per class."""
    batch_size: int
    """GET_STARTED.md uses 512 on a large GPU; evaluation takes 1.5 times this many images per batch."""
    out: Path
    """The record; AdaptiveNN's output goes beside it, as <checkpoint>.log."""


@dataclass(frozen=True)
class Flops:
    root: Path
    out: Path


@dataclass(frozen=True)
class Parameters:
    root: Path
    out: Path


def main() -> None:
    command = tyro.cli(
        Annotated[Accuracy, tyro.conf.subcommand("accuracy")]
        | Annotated[Flops, tyro.conf.subcommand("flops")]
        | Annotated[Parameters, tyro.conf.subcommand("parameters")]
    )
    sources = Sources(command.root)
    command.out.parent.mkdir(parents=True, exist_ok=True)
    match command:
        case Accuracy():
            result = accuracy(sources, imagenet_val=command.imagenet_val, batch_size=command.batch_size,
                              log_dir=command.out.parent)
        case Flops():
            result = flops(sources)
        case Parameters():
            result = parameters(sources)
    command.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
