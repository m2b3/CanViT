from pathlib import Path

import jax
import numpy as np
from flax import nnx
from jax.experimental.layout import Format, Layout
from jax.sharding import SingleDeviceSharding

from canvit_nnx.hub import load_state, save_state


def test_state_files_keep_the_values_of_a_column_major_parameter(tmp_path: Path) -> None:
    # TPUs store some parameters column-major (a v5p chose it for (768, 1000) and (384, 3) kernels); such an
    # array reaches the host as a Fortran-ordered copy.
    module = nnx.Linear(3, 5, rngs=nnx.Rngs(0))
    expected = np.asarray(module.kernel[...])
    column_major = Format(Layout(major_to_minor=(1, 0)), SingleDeviceSharding(jax.devices()[0]))
    module.kernel[...] = jax.device_put(module.kernel[...], column_major)
    assert not np.asarray(module.kernel[...]).flags.c_contiguous, "the fixture must reproduce a column-major host copy"

    save_state(module, tmp_path / "state.safetensors")
    restored = nnx.Linear(3, 5, rngs=nnx.Rngs(1))
    load_state(restored, tmp_path / "state.safetensors")

    np.testing.assert_array_equal(np.asarray(restored.kernel[...]), expected)
