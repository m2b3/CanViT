# canvit-nnx: CanViT (Canvas Vision Transformer) for JAX and Flax NNX

The JAX and Flax NNX implementation of CanViT. Read the root
[Backend packages](https://github.com/m2b3/CanViT#backend-packages) section
for the shared API, tensor shapes and checkpoint contract.

## Develop from a checkout

```bash
cd canvit-nnx
uv sync
```

The checkout resolves `canvit-core` from its sibling directory. A Git
subdirectory install uses package-index dependencies, so it requires a
published `canvit-core` distribution:

```bash
uv add 'canvit-nnx @ git+https://github.com/m2b3/CanViT.git#subdirectory=canvit-nnx'
```

## Convert a checkpoint

From the repository root:

```bash
uv run --project canvit-nnx python -m tools.convert_checkpoints --help
```
