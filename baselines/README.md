# Baselines

The paper's measurements of other active-vision models. Each directory is its own uv project, apart from the
CanViT packages: nothing outside `baselines/` imports from here, and nothing here is published to PyPI.

| Directory | What it is | Paper |
|---|---|---|
| [`adaglimpse-nnx/`](adaglimpse-nnx/) | AdaGlimpse's architecture ported to JAX/Flax NNX, the baseline of the architecture control, with its cost model and tests against AdaGlimpse's own code | `supp:architecture-control`, `supp:coco-detection` |
| [`gfnet-adaptivenn/`](gfnet-adaptivenn/) | GFNet's and AdaptiveNN's released ImageNet-1k models measured with their own code: top-1, FLOPs and parameters at a fixed number of glimpses | `supp:in1k-efficiency` |

Third-party code and checkpoints are fetched from their authors' releases, pinned by commit and checksum, never
copied into this repository. `adaglimpse-nnx/` is not offered under the repository's MIT license; its README states
its terms.
