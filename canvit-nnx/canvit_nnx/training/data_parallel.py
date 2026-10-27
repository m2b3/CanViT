"""Data parallelism over every device of one host or of a multi-host slice.

Every host runs the same program: it loads its own rows of each global batch,
and the model and optimizer state are replicated on every device. Each method
here that touches device arrays is a collective that every host must call,
never behind a leader check; leader checks guard file writes only.
"""

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh, NamedSharding, PartitionSpec

BATCH_AXIS = "batch"


@dataclass(frozen=True)
class DataParallel:
    mesh: Mesh

    @staticmethod
    def start(*, multi_host: bool) -> "DataParallel":
        """The data-parallel mesh over all devices; with multi_host, first join the slice's other hosts.

        jax.distributed.initialize must precede every other JAX call; it finds
        the other hosts of a Cloud TPU slice on its own.
        """
        if multi_host:
            jax.distributed.initialize()
        return DataParallel(
            jax.make_mesh((jax.device_count(),), (BATCH_AXIS,), axis_types=(jax.sharding.AxisType.Auto,))
        )

    @property
    def is_leader(self) -> bool:
        """Whether this host writes the run's files."""
        return jax.process_index() == 0

    @property
    def batched(self) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec(BATCH_AXIS))

    def host_rows(self, batch_size: int) -> slice:
        """The rows of each global batch that this host loads."""
        num_devices, num_hosts = jax.device_count(), jax.process_count()
        assert batch_size % num_devices == 0, f"batch size {batch_size} does not split over {num_devices} devices"
        rows_per_host = batch_size // num_hosts
        return slice(jax.process_index() * rows_per_host, (jax.process_index() + 1) * rows_per_host)

    def to_global(self, *host_rows: np.ndarray) -> tuple[jax.Array, ...]:
        """Each host's rows of global batches into those batches, split over the devices."""
        return tuple(jax.make_array_from_process_local_data(self.batched, rows) for rows in host_rows)

    def split_full_batch[T](self, full_batch: T, *, batch_size: int) -> T:
        """Arrays over a whole global batch (a pytree), identical on every host, as global arrays.

        Each host passes on its host_rows, so their rows pair with those of the
        data each host loads at the same host_rows.
        """
        rows = self.host_rows(batch_size)
        return jax.tree.map(
            lambda array: jax.make_array_from_process_local_data(self.batched, np.asarray(array)[rows]), full_batch
        )

    def to_host(self, *arrays: jax.Array) -> tuple[np.ndarray, ...]:
        """Global device arrays into NumPy arrays holding every row, on every host."""
        return tuple(np.asarray(array) for array in process_allgather(arrays, tiled=True))

    def replicate(self, *nodes: nnx.Pytree) -> None:
        replicated = NamedSharding(self.mesh, PartitionSpec())
        for node in nodes:
            nnx.update(node, jax.device_put(nnx.state(node), replicated))
