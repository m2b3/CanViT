"""What a TPU fine-tuning run logs about its environment, its tensors' shardings and XLA compilations."""
# pyright: reportMissingImports=false
# (torch_xla is installed only in the tpu/ environment.)

import logging
import os

import torch
import torch_xla
import torch_xla.backends as xla_backends
import torch_xla.debug.metrics as xla_metrics
import torch_xla.runtime as xr
from torch import Tensor

log = logging.getLogger(__name__)

# Logged as environment diagnostics, except names containing a SECRET_SUBSTRINGS entry.
DIAGNOSTIC_ENV_PREFIXES = (
    "XLA_", "LIBTPU", "PJRT_", "PT_XLA", "TPU_", "OMP_", "MKL_", "TF_CPP", "SKYPILOT_", "PYTORCH_", "NEURON_", "GRPC_",
)
SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class CompileCounter:
    """Warns when XLA compiled a graph since the previous check: steady-state training compiles none."""

    def __init__(self) -> None:
        self.total = 0

    def check(self, step: int) -> None:
        data = xla_metrics.metric_data("CompileTime")
        if data is None:
            return
        new = data[0] - self.total
        self.total = data[0]
        if new:
            log.warning("xla_recompile step=%d compiles_since_last_log=%d total_compiles=%d", step, new, self.total)


_SHARDINGS_LOGGED: set[str] = set()


def log_sharding_once(tensor: Tensor, name: str) -> None:
    if name in _SHARDINGS_LOGGED:
        return
    _SHARDINGS_LOGGED.add(name)
    try:
        spec = torch_xla._XLAC._get_xla_sharding_spec(tensor)
    except Exception as error:  # noqa: BLE001  diagnostics only
        spec = f"<error: {error}>"
    log.info("sharding[%s] shape=%s dtype=%s spec=%s", name, tuple(tensor.shape), tensor.dtype, spec)


def log_environment() -> None:
    """Versions, precision, allowlisted environment variables with secrets redacted, device count."""
    import torchvision

    log.info("─── environment diagnostics ───")
    log.info("torch        = %s  (%s)", torch.__version__, torch.__file__)
    log.info("torchvision  = %s", torchvision.__version__)
    log.info("torch_xla    = %s  (%s)", torch_xla.__version__, torch_xla.__file__)
    try:
        import libtpu

        log.info("libtpu       = %s", getattr(libtpu, "__version__", "?"))
    except Exception as error:  # noqa: BLE001  diagnostics only
        log.info("libtpu       = (import failed: %s)", error)
    log.info("mat_mul_prec = %s (torch_xla)", xla_backends.get_mat_mul_precision())
    log.info("f32 matmul   = %s (torch core)", torch.get_float32_matmul_precision())
    log.info("CPU count    = %s", os.cpu_count())
    log.info("device count = %d (xr.global_runtime_device_count)", xr.global_runtime_device_count())
    log.info("─── env vars (allowlisted) ───")
    for name in sorted(os.environ):
        if name.startswith(DIAGNOSTIC_ENV_PREFIXES):
            redacted = any(secret in name.upper() for secret in SECRET_SUBSTRINGS)
            log.info("  %s = %s", name, "<redacted>" if redacted else os.environ[name])
    log.info("─── end diagnostics ───")
