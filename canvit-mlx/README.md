# canvit-mlx: CanViT (Canvas Vision Transformer) for MLX

The MLX implementation of CanViT for Apple Silicon. Read the root
[Backend packages](https://github.com/m2b3/CanViT#backend-packages) section
for the shared API, tensor shapes and checkpoint contract.

## Develop from a checkout

```bash
cd canvit-mlx
uv sync
```

The checkout resolves `canvit-core` from its sibling directory. A Git
subdirectory install uses package-index dependencies, so it requires a
published `canvit-core` distribution:

```bash
uv add 'canvit-mlx @ git+https://github.com/m2b3/CanViT.git#subdirectory=canvit-mlx'
```

## Convert a checkpoint

From the repository root:

```bash
uv run --project canvit-mlx python -m tools.convert_checkpoints --help
```
