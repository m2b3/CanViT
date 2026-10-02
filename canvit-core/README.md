# canvit-core: shared CanViT (Canvas Vision Transformer) definitions

`canvit-core` contains backend-independent CanViT definitions shared by the
PyTorch, MLX and JAX/Flax NNX packages. It owns `CanViTConfig`, backbone
specifications, the canvas-attention schedule, probe fusion and the strict
checkpoint schema. It also provides the shared NHWC ImageNet preprocessing
helpers used by the native packages.

The model and tensor-shape contract lives in the root
[Backend packages](https://github.com/m2b3/CanViT#backend-packages) section.
The package exports `CanViTConfig`; backend packages import the other
definitions from its focused modules.

## Develop from a checkout

```bash
cd canvit-core
uv sync
```

From the repository checkout, dependent packages resolve `canvit-core` through
a local `uv` source. Direct installation from an index or a Git subdirectory
requires a published `canvit-core` distribution.
