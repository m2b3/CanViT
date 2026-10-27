"""Compute used after submission, by hardware: H100 hours on the SLURM cluster, Cloud TPU chip-hours and list price, and
RTX 4090 GPU-hours recorded by the runs themselves; with the April TPU fine-tuning in the same units. Input: a record
that derives each figure from cluster accounting, cloud billing and run records."""

import datetime as dt

from canvit_paper_exporter.core import Dataset, load_json
from canvit_paper_exporter.paths import Inputs

RECORD = Inputs("compute_reporting", ("paper_compute.json",))


def _day(timestamp: str) -> dt.date:
    return dt.date.fromisoformat(timestamp[:10])


def compute() -> dict:
    record = load_json(RECORD.file("paper_compute.json"))
    earlier, slurm, tpu, rtx = record["earlier_totals"], record["slurm_h100"], record["cloud_tpu"], record["rtx4090"]
    # The submission's totals end before the review period's records begin, so the two never count the same work.
    assert _day(slurm["first_job_start"]) > dt.date.fromisoformat(earlier["slurm_total_last_job"])
    assert _day(rtx["learned_policy_first_start"]) > dt.date.fromisoformat(earlier["workstation_estimate_written"])
    assert _day(tpu["review_period"]["first_day"]) > _day(tpu["april_finetuning"]["last_day"])
    # The workstation's sources do not overlap: their sum is their union.
    assert abs(rtx["learned_policy_gpu_hours"] + rtx["evaluation_gpu_hours"] - rtx["recorded_gpu_hours"]) < 0.01
    return record


dataset = Dataset(name="compute_reporting", compute=compute, inputs=(RECORD,))
