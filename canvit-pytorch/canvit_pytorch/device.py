"""The device that examples and demos run on unless told otherwise."""

import torch


def accelerator_or_cpu() -> torch.device:
    """The accelerator torch can use on this machine (CUDA, or MPS on Apple silicon), else the CPU."""
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
