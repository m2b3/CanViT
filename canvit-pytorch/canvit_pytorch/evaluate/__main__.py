from typing import Protocol

import tyro

from canvit_pytorch.evaluate.batch import Batch
from canvit_pytorch.evaluate.latency.matrix import LatencyMatrix
from canvit_pytorch.evaluate.latency.measure import LatencyMeasurement
from canvit_pytorch.evaluate.latency.summary import LatencySummary
from canvit_pytorch.evaluate.processes import configure_logging
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import ADE20kSegmentationCanViT, ADE20kSegmentationDINOv3
from canvit_pytorch.evaluate.tasks.imagenet_classification import ImageNetClassification
from canvit_pytorch.evaluate.tasks.mask_iou.canvit import MaskIoUCanViT
from canvit_pytorch.evaluate.tasks.mask_iou.dinov3 import MaskIoUDINOv3
from canvit_pytorch.evaluate.tasks.reconstruction import Reconstruction


class Command(Protocol):
    def run(self) -> object: ...


COMMANDS: dict[str, type[Command]] = {
    "ade20k-segmentation-canvit": ADE20kSegmentationCanViT,
    "ade20k-segmentation-dinov3": ADE20kSegmentationDINOv3,
    "imagenet-classification": ImageNetClassification,
    "reconstruction": Reconstruction,
    "mask-iou-dinov3": MaskIoUDINOv3,
    "mask-iou-canvit": MaskIoUCanViT,
    "batch": Batch,
    "latency": LatencyMeasurement,
    "latency-matrix": LatencyMatrix,
    "latency-summary": LatencySummary,
}


def main() -> None:
    configure_logging()
    command: Command = tyro.extras.subcommand_cli_from_dict(
        COMMANDS, prog="python -m canvit_pytorch.evaluate", config=(tyro.conf.CascadeSubcommandArgs,),
    )
    command.run()


if __name__ == "__main__":
    main()
