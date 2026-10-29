# Learned viewing policies (`canvit_pytorch.rl`)

A learned policy chooses where a frozen CanViT looks next. A
`ViewpointScorer` scores a fixed set of candidate viewpoints from the canvas
and from the segmentation that an ADE20K probe reads out of it; a
`LearnedPolicy`, a `canvit_pytorch.policies.Policy` like the paper's policies,
takes the full scene first and then, at each glimpse, the candidate of highest
score. The scorer is trained by reinforcement learning to choose the glimpses
that most reduce the probe's segmentation error, while CanViT and the probe
stay frozen. Each scorer records in its `config.json` the `Observer` it was
trained with: the CanViT and probe checkpoints, by Hub id (they load from a
local mirror when `CANVIT_HUB_ROOT` names one), the scene and glimpse sizes and
the canvas grid. The commands below run from `canvit-pytorch/`.

## Run a published policy

`LearnedPolicy.from_pretrained` loads a scorer with its observer (canvit-pytorch 0.3.0 or later).

```python
import torch
from PIL import Image

from canvit_pytorch.device import accelerator_or_cpu
from canvit_pytorch.hub.repos import Q_LEARNING_POLICIES
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.rl import LearnedPolicy

device = accelerator_or_cpu()
policy = LearnedPolicy.from_pretrained(Q_LEARNING_POLICIES[0], device)
scene_size_px = policy.scorer.observer.scene_size_px
scene = preprocess(scene_size_px)(Image.open("scene.jpg").convert("RGB")).unsqueeze(0).to(device)

with torch.inference_mode():
    steps = policy.rollout(scene, num_glimpses=5)
for step in steps:  # the full scene, then the policy's choices: (row, col) centers and scales
    print(step.t, step.viewpoint.centers.tolist(), step.viewpoint.scales.tolist())
labels = policy.model.logits(steps[-1].state.canvas).argmax(dim=1)  # ADE20K classes on the canvas grid
```

The policy keeps the state of its episode, so a policy object serves one
episode at a time; `rollout` starts each episode from the observer's initial
state, as in training. A scorer's Hub card states the length of its training
episodes.

## Evaluate

`ade20k-segmentation-learned-policy` measures ADE20K validation mIoU after
every glimpse of a learned policy, with the observer its scorer records, in the
format of the paper's `ade20k-segmentation-canvit` (see
[evaluate.md](evaluate.md); install with `uv sync --extra evaluate`). The
result's metadata identifies the scorer: its repo, the Hub commit evaluated
and the SHA-256 of its weights. To compare with EG-C2F, run
`ade20k-segmentation-canvit` with the probe and canvas grid of the scorer's
`observer`:

```bash
export ADE20K_ROOT=/path/to/ADEChallengeData2016
uv run python -m canvit_pytorch.evaluate ade20k-segmentation-learned-policy \
  --scorer-repo canvit/qpolicy-ade20k-c64-t5-qband-2026-07-04-s0 --num-glimpses 5 \
  --output results/ade20k_seg_qpolicy_s0.pt
uv run python -m canvit_pytorch.evaluate ade20k-segmentation-canvit \
  --probe-repo PROBE_REPO --episode.canvas-grid-size CANVAS_GRID_SIZE \
  --episode.policy entropy_coarse_to_fine --episode.num-glimpses 5 \
  --batch-size 8 --output results/ade20k_seg_egc2f.pt
```

`--scorer-repo` also accepts the `scorer/` directory of a training run, and
`--scorer-revision` pins a Hub commit, branch or tag.

## Train

```bash
uv sync --extra rl
export ADE20K_ROOT=/path/to/ADEChallengeData2016
uv run python -m canvit_pytorch.rl.train --out-dir runs/qpolicy-s0 --seed 0
uv run python -m canvit_pytorch.rl.train objective:reinforce --out-dir runs/pgpolicy-s0 --seed 0
```

The defaults of `rl.train.PolicyTraining` are the recipe of the published
policies; `--help` lists them, and `--observer.*` selects another CanViT and
probe. The run directory holds:

- `scorer/`: the selected scorer, and `last_step_scorer/`, that of the last
  training step, both loadable with `ViewpointScorer.from_pretrained`;
- `record.json`: the objective, the configuration, provenance, the selected
  step, and at every validation the mean training loss since the previous one
  and the validation cross-entropy after each glimpse.

`PolicyTraining.scenes` is the method to override to train on another
dataset. `python -m canvit_pytorch.hub.publish policy --run-dir RUN ...`
stages a run's selected scorer as a Hub repo, with its record and a card
generated from both.

## Method

**Candidates** (`rl.candidates`). At each scale, a grid of viewpoint centers
spans the positions where a glimpse of that scale fits inside the scene. The
spacing does not depend on the scale, so glimpses of a scale overlap.

**Features** (`rl.features`). The scorer reads, at each canvas cell: the
entropy of the probe's class distribution and the layer-normalized canvas, each
with its change since the previous decision; and the cosine dissimilarity of
the canvas to that of the previous decision and to that before the first glimpse.
The features are average-pooled to a fixed grid, whatever the canvas grid.
`--feature-groups` selects the groups a scorer reads (all by default), and the
scorer records them in its `config.json`.

**Scorer** (`rl.scorer`). Per-group projections and normalizations fuse the
features into a map, which a ConvNeXt-V2 U-Net pools down to a single cell and
expands back, so that every score depends on the whole scene. Each candidate's
score is read from the map at its center by a head shared across scales.

**Reward** (`rl.train.reward`). A glimpse's reward is the fraction of the
probe's remaining cross-entropy on the scene's labels that it removes. Rewards
are standardized by running statistics shared across scenes and kept
separately for each decision of an episode, since their scale changes from one
glimpse to the next.

**Objectives** (`rl.train.objectives`). Each training episode starts with the
full scene. Under Q-learning (the default, with discount 0), the scores are the
values of the candidates: a dueling head separates a scene's mean value from the
candidates' advantages, and the score of each viewpoint taken regresses on its
standardized reward. Episodes take a random candidate with probability
`epsilon`, otherwise the best-scoring one under the current scorer, so training
visits the states the deployed policy reaches. Under `objective:reinforce`, the
scores are the logits of a softmax policy that episodes sample, trained by
REINFORCE (the score-function estimator) with an entropy bonus whose weight
adapts to hold the policy's mean entropy at a floor.

**Selection** (`rl.train.validation`). Every `validation-interval` steps, the
deployed policy runs on the ADE20K validation set; the run keeps the scorer of
lowest mean cross-entropy after the glimpses it chose.

## Published policies

`hub.repos.Q_LEARNING_POLICIES` and `hub.repos.REINFORCE_POLICIES`
name the published scorers, one per training seed of each objective's recipe.
Each repo's card states the observer, the training run and the validation
cross-entropy of the selected step, and the repo holds the run's full record.
