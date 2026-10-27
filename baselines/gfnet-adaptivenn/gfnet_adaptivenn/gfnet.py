"""GFNet's released ImageNet-1k models at a fixed number of glimpses, measured with its released code.

    uv run python -m gfnet_adaptivenn.gfnet facts --root <sources> --out gfnet_checkpoint_facts.json
    uv run python -m gfnet_adaptivenn.gfnet accuracy --root <sources> --imagenet-val <dir> --out gfnet_accuracy.json
    uv run python -m gfnet_adaptivenn.gfnet flops --root <sources> --out gfnet_flops.json
    uv run python -m gfnet_adaptivenn.gfnet parameters --root <sources> --out gfnet_parameters.json

`facts` reads what each checkpoint stores: its configuration, the cumulative
multiply-accumulates (MACs) its authors computed, their top-1 at each number
of glimpses, and how many values each of its state dicts holds. `accuracy`
re-evaluates top-1 with GFNet's `inference.py --eval_mode 1`, in which every
image takes every glimpse ("anytime prediction"; a CUDA GPU). `flops` counts
GFLOPs, at 2 FLOPs per MAC, with torch.utils.flop_counter over that
evaluation's loop: the glance at the downsampled image and its classifier,
then per glimpse the policy, the local encoder and the cascaded classifier (a
CUDA GPU, which GFNet's policy requires). `parameters` counts the parameters
of the modules that loop runs up to each glimpse (`active_parameters`); the
`active_params` command combines it with AdaptiveNN's count.
"""

import itertools
import json
import math
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import torch
import tyro
from torch.utils.flop_counter import FlopCounterMode

from gfnet_adaptivenn import imagenet, record
from gfnet_adaptivenn.active_params import cumulative_counts, millions, parameters_of
from gfnet_adaptivenn.sources import GFNET, GFNET_CHECKPOINTS, Checkpoint, Sources

ANYTIME_EVAL_MODE = 1
"""`inference.py --eval_mode 1`: the checkpoint's own confidence thresholds, top-1 by number of glimpses on val."""
ANYTIME_LINE = re.compile(r"^anytime_classification : \[([^\]]*)\]$", re.MULTILINE)


def checkpoint_facts(path: Path) -> dict[str, Any]:
    stored = torch.load(path, map_location="cpu", weights_only=True)
    parameters = {key: sum(value.numel() for value in stored[key].values())
                  for key in ("model_prime_state_dict", "model_state_dict", "fc", "policy")}
    return {
        "file": path.name, "backbone": stored["model_name"], "patch": stored["patch_size"],
        "T": len(stored["flops"]), "model_flops": stored["model_flops"], "policy_flops": stored["policy_flops"],
        "fc_flops": stored["fc_flops"], "cum_macs": [int(x) for x in stored["flops"]],
        "anytime": [float(x) for x in stored["anytime_classification"]],
        "params": parameters, "params_total": sum(parameters.values()),
    }


def accuracy(sources: Sources, *, imagenet_val: Path, log_dir: Path) -> dict[str, Any]:
    code = sources.repository(GFNET)
    models = {}
    with imagenet.evaluation_root(imagenet_val) as root:
        for checkpoint in GFNET_CHECKPOINTS:
            path = sources.checkpoint(checkpoint)
            log = log_dir / f"{checkpoint.file}.log"
            with log.open("w") as output:
                subprocess.run(
                    (sys.executable, "inference.py", "--data_url", f"{root}/", "--checkpoint_path", str(path),
                     "--eval_mode", str(ANYTIME_EVAL_MODE)),
                    cwd=code, stdout=output, stderr=subprocess.STDOUT, check=True,
                )
            [printed] = ANYTIME_LINE.findall(log.read_text())
            facts = checkpoint_facts(path)
            top1 = [float(value) for value in printed.split(",")]
            assert len(top1) == facts["T"], (log, top1)
            models[checkpoint.file] = {
                "backbone": facts["backbone"], "patch": facts["patch"], "T": facts["T"],
                "top1_by_passes": top1, "stored_in_checkpoint": facts["anytime"],
            }
    return record.provenance(GFNET, device=torch.device("cuda")) | {
        "n_images": imagenet.num_images(imagenet_val), "eval_mode": ANYTIME_EVAL_MODE, "models": models,
    }


class GFNetModel(NamedTuple):
    """A released GFNet model as inference.py builds it, holding its checkpoint's weights, on the CPU."""

    glance_encoder: Any  # f_G, which also classifies the glance
    local_encoder: Any  # f_L
    policy: Any  # the patch proposal network
    classifier: Any  # the cascade of per-glimpse classifiers fc_2, fc_3, ...
    configuration: dict[str, Any]
    patch_px: int
    num_glimpses: int


@contextmanager
def _cuda_does_nothing() -> Iterator[None]:
    """GFNet's ActorCritic moves its action variance, a plain tensor, to CUDA in its constructor."""
    cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor  # pyright: ignore[reportAttributeAccessIssue]
    try:
        yield
    finally:
        torch.Tensor.cuda = cuda


def _import_gfnet(sources: Sources) -> None:
    """Put GFNet's checkout first on the import path, for its top-level modules."""
    code = str(sources.repository(GFNET))
    if sys.path[0] != code:
        sys.path.insert(0, code)


def load_model(sources: Sources, checkpoint: Checkpoint) -> GFNetModel:
    _import_gfnet(sources)
    from configs import model_configurations  # pyright: ignore[reportMissingImports]
    from models import create_model  # pyright: ignore[reportMissingImports]
    from network import ActorCritic, Full_layer  # pyright: ignore[reportMissingImports]

    stored = torch.load(sources.checkpoint(checkpoint), map_location="cpu", weights_only=True)
    architecture, patch = stored["model_name"], stored["patch_size"]
    configuration = model_configurations[architecture]
    # inference.py's construction of the two backbones it releases with a per-step classifier cascade.
    dropout = {"efficientnet": {"drop_rate": 0.3, "drop_connect_rate": 0.2},
               "mobilenetv3": {"drop_rate": 0.2, "drop_connect_rate": 0.2}}
    [family] = [name for name in dropout if name in architecture]

    def backbone(key: str) -> Any:
        model = create_model(architecture, pretrained=False, num_classes=imagenet.NUM_CLASSES, **dropout[family])
        # Saved from torch.nn.DataParallel, whose wrapper prefixes every key.
        model.load_state_dict({name.removeprefix("module."): value for name, value in stored[key].items()})
        return model.eval()

    with _cuda_does_nothing():
        policy = ActorCritic(configuration["feature_map_channels"],
                             configuration["feature_map_channels"] * math.ceil(patch / 32) ** 2,
                             configuration["policy_hidden_dim"], configuration["policy_conv"])
    policy.load_state_dict(stored["policy"])
    classifier = Full_layer(configuration["feature_num"], configuration["fc_hidden_dim"], configuration["fc_rnn"])
    classifier.load_state_dict(stored["fc"])
    return GFNetModel(
        glance_encoder=backbone("model_prime_state_dict"), local_encoder=backbone("model_state_dict"),
        policy=policy, classifier=classifier.eval(), configuration=configuration, patch_px=patch,
        num_glimpses=len(stored["flops"]),
    )


def active_parameters(model: GFNetModel) -> list[int]:
    """The parameters of the modules inference.py's loop runs up to each glimpse, from the glance.

    The glance runs the global encoder and its own classifier. The first later
    glimpse adds the local encoder, whose own classifier inference never calls,
    and the policy's state encoder, recurrent cell and actor; its critic serves
    training. Each later glimpse adds its entry of the classifier cascade.
    """
    cascade = {int(name.removeprefix("fc_")): entry for name, entry in model.classifier.named_children()}
    local_encoder = parameters_of(model.local_encoder) - parameters_of(model.local_encoder.classifier)
    policy = parameters_of(model.policy.state_encoder, model.policy.gru, model.policy.actor)
    added = [parameters_of(model.glance_encoder)] + [
        parameters_of(cascade[glimpse]) | (local_encoder | policy if glimpse == 2 else set())
        for glimpse in range(2, model.num_glimpses + 1)
    ]
    not_run = parameters_of(model.local_encoder.classifier, model.policy.critic,
                            *(entry for glimpse, entry in cascade.items() if glimpse > model.num_glimpses))
    every = parameters_of(model.glance_encoder, model.local_encoder, model.policy, model.classifier)
    return cumulative_counts(added, not_run=not_run, every=every)


def parameters(sources: Sources) -> dict[str, Any]:
    models = {}
    for checkpoint in GFNET_CHECKPOINTS:
        facts = checkpoint_facts(sources.checkpoint(checkpoint))
        counts = active_parameters(load_model(sources, checkpoint))
        models[checkpoint.file] = {
            "backbone": facts["backbone"], "patch": facts["patch"], "T": facts["T"],
            "whole_checkpoint_M": millions(facts["params_total"]), "parameters_by_glimpse": counts,
            "glance_active_M": millions(counts[0]), "cumulative_at_T_M": millions(counts[-1]),
        }
    return {"provenance": record.provenance(GFNET, device=torch.device("cpu")), "models": models}


def flops(sources: Sources) -> dict[str, Any]:
    assert torch.cuda.is_available(), "GFNet's policy network allocates on CUDA"
    _import_gfnet(sources)
    from network import Memory  # pyright: ignore[reportMissingImports]
    from utils import get_patch, get_prime  # pyright: ignore[reportMissingImports]

    models = {}
    for checkpoint in GFNET_CHECKPOINTS:
        facts, model = checkpoint_facts(sources.checkpoint(checkpoint)), load_model(sources, checkpoint)
        glance_encoder, local_encoder = model.glance_encoder.cuda(), model.local_encoder.cuda()
        policy, classifier = model.policy.cuda(), model.classifier.cuda()
        configuration, patch = model.configuration, model.patch_px
        memory = Memory()
        image = torch.randn(1, 3, configuration["image_size"], configuration["image_size"]).cuda()

        # inference.py's generate_logits, call for call, for these backbones: the glance, then each later glimpse.
        with torch.no_grad():
            with FlopCounterMode(display=False) as counter:
                features, state = glance_encoder(get_prime(image, patch, configuration["prime_interpolation"]))
                classifier(features, restart=True)
                glance_encoder.classifier(features)
            counts = [counter.get_total_flops()]
            for glimpse in range(1, model.num_glimpses):
                with FlopCounterMode(display=False) as counter:
                    action = policy.act(state, memory, restart_batch=glimpse == 1)
                    features, state = local_encoder(get_patch(image, action, patch))
                    classifier(features, restart=False)
                counts.append(counter.get_total_flops())
        cumulative = list(itertools.accumulate(counts))
        models[checkpoint.file] = {
            "backbone": facts["backbone"], "patch_px": patch, "T": model.num_glimpses,
            "cumulative_macs_ours": [count // 2 for count in cumulative],
            "cumulative_macs_theirs": facts["cum_macs"],
            "cumulative_gflops_our_unit": [count / 1e9 for count in cumulative],
        }
    return record.provenance(GFNET, device=torch.device("cuda")) | {
        "unit": "2 FLOPs per MAC; *_macs fields in MACs", "models": models,
    }


@dataclass(frozen=True)
class Facts:
    root: Path
    out: Path


@dataclass(frozen=True)
class Accuracy:
    root: Path
    imagenet_val: Path
    """ImageNet-1k's validation images in ImageFolder layout: a directory per class."""
    out: Path
    """The record; GFNet's output for each checkpoint goes beside it, as <checkpoint>.log."""


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
        Annotated[Facts, tyro.conf.subcommand("facts")]
        | Annotated[Accuracy, tyro.conf.subcommand("accuracy")]
        | Annotated[Flops, tyro.conf.subcommand("flops")]
        | Annotated[Parameters, tyro.conf.subcommand("parameters")]
    )
    sources = Sources(command.root)
    command.out.parent.mkdir(parents=True, exist_ok=True)
    match command:
        case Facts():
            result: Any = [checkpoint_facts(sources.checkpoint(checkpoint)) for checkpoint in GFNET_CHECKPOINTS]
        case Accuracy():
            result = accuracy(sources, imagenet_val=command.imagenet_val, log_dir=command.out.parent)
        case Flops():
            result = flops(sources)
        case Parameters():
            result = parameters(sources)
    command.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
