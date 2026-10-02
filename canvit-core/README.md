# canvit-core

`canvit-core` contains the backend-independent definitions shared by the
CanViT (Canvas Vision Transformer) packages. It provides `CanViTConfig`, the backbone specifications,
canvas-attention schedules, readout fusion, ImageNet preprocessing for native
arrays, and the native MLX/NNX checkpoint schema.

Install `canvit-mlx`, `canvit-nnx`, or `canvit-pytorch` for model execution.
The root
[backend documentation](https://github.com/m2b3/CanViT#backend-packages)
defines the shared model and tensor-shape contract.

## Install

```bash
uv add canvit-core
```

## Public API

The package exports `CanViTConfig` at the top level. Backend-specific modules
use the focused modules for preprocessing, checkpoint handling, and the other
shared definitions.

```python
from canvit_core import CanViTConfig

config = CanViTConfig()
```

## Develop from a checkout

```bash
cd canvit-core
uv sync
```

The repository's backend projects resolve this directory through their local
`uv` sources. An index or Git installation resolves the published
`canvit-core` distribution instead.
