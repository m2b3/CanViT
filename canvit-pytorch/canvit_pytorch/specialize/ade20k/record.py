"""What a probe-training run leaves in its directory: the selected probe and record.json beside it."""

import dataclasses
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvit_pytorch.probes import SegmentationProbe

RECORD_FILENAME = "record.json"


@dataclass(frozen=True)
class ProbeRecord:
    """record.json, rewritten after every validation by RunDirectory."""

    probe_dir: str
    """The selected probe: a directory next to this record, loadable with SegmentationProbe.from_pretrained."""
    selected_step: int
    """Optimizer steps the selected probe had taken."""
    validation_miou: list[float]
    """The selected probe's validation mIoU on each feature map (the canvas after each glimpse, or DINOv3's one map)."""
    steps_trained: int
    """Optimizer steps taken when the record was written: training.num_steps once training has finished."""
    config_type: str
    """The training command's configuration class, of which `config` holds the fields."""
    config: dict[str, Any]
    comet_experiment_key: str
    provenance: dict[str, Any]
    """canvit_pytorch.provenance.provenance() at the start of the run."""


class RunDirectory:
    """Keeps the probe of highest validation mIoU on the last feature map, and its record."""

    def __init__(
        self, path: Path, *, config_type: str, config: dict[str, Any], comet_experiment_key: str,
        provenance: dict[str, Any],
    ) -> None:
        path.mkdir(parents=True)
        self.path = path
        self.config_type = config_type
        self.config = config
        self.comet_experiment_key = comet_experiment_key
        self.provenance = provenance
        self.selected: ProbeRecord | None = None

    def record_validation(self, *, step: int, validation_miou: list[float], probe: SegmentationProbe) -> ProbeRecord:
        """Save the probe if it is the new selection, then rewrite record.json."""
        previous = self.selected
        if previous is None or validation_miou[-1] > previous.validation_miou[-1]:
            probe_dir = f"probe_step{step}"
            probe.save_pretrained(self.path / probe_dir)
            self.selected = ProbeRecord(
                probe_dir=probe_dir, selected_step=step, validation_miou=validation_miou, steps_trained=step,
                config_type=self.config_type, config=self.config,
                comet_experiment_key=self.comet_experiment_key, provenance=self.provenance,
            )
        else:
            self.selected = dataclasses.replace(previous, steps_trained=step)
        record_path = self.path / RECORD_FILENAME
        staged = record_path.with_suffix(".staged")
        staged.write_text(json.dumps(dataclasses.asdict(self.selected), indent=2, default=str) + "\n")
        os.replace(staged, record_path)
        if previous is not None and previous.probe_dir != self.selected.probe_dir:
            shutil.rmtree(self.path / previous.probe_dir)
        return self.selected
