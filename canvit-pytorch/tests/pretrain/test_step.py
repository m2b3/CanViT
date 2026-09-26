from contextlib import nullcontext

import torch

from canvit_pytorch.model.config import CanViTConfig
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import make_policy
from canvit_pytorch.pretrain.loss import DistillationTargets, distillation_losses
from canvit_pytorch.pretrain.step import sample_rollout_length, training_step
from canvit_pytorch.viewpoint import sample_at_viewpoint

POLICIES = ("full_then_random", "random")
GRID, GLIMPSE_PX, TEACHER_DIM, BATCH = 4, 32, 8, 2


def test_rollout_length_is_whole_chunks_with_mean_k_over_p_stop() -> None:
    lengths = torch.tensor([float(sample_rollout_length(tbptt_chunk_glimpses=2, stop_probability=0.5)) for _ in range(20_000)])
    assert torch.all(lengths % 2 == 0) and lengths.min() == 2
    assert abs(lengths.mean().item() - 4) < 0.1


def test_single_chunk_rollouts_get_the_gradient_of_the_mean_loss() -> None:
    """With p_stop = 1 each rollout is one chunk of K glimpses, so truncated BPTT is plain backpropagation of the
    loss averaged over glimpses and rollouts."""
    torch.manual_seed(0)
    model = CanViTForPretraining(canvit_config=CanViTConfig(backbone_name="vits16"), teacher_dim=TEACHER_DIM, teacher_patch_grid=GRID)
    images = torch.randn(BATCH, 3, GRID * 16, GRID * 16)
    targets = DistillationTargets(
        patches=torch.randn(BATCH, GRID**2, TEACHER_DIM), cls=torch.randn(BATCH, TEACHER_DIM),
        raw_patches=torch.randn(BATCH, GRID**2, TEACHER_DIM), raw_cls=torch.randn(BATCH, TEACHER_DIM),
    )
    losses_enabled = dict(enable_teacher_patch_loss=True, enable_teacher_cls_loss=True)

    torch.manual_seed(1)
    metrics = training_step(
        model=model, images=images, targets=targets, rollout_policies=POLICIES, tbptt_chunk_glimpses=2,
        stop_probability=1.0, glimpse_size_px=GLIMPSE_PX, canvas_grid_size=GRID, autocast=nullcontext(), **losses_enabled,
    )
    step_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    torch.manual_seed(1)
    losses = []
    for name in POLICIES:
        policy = make_policy(name, batch_size=BATCH, device=images.device, num_glimpses=2, canvas_grid_size=GRID)
        state = model.init_state(batch_size=BATCH, canvas_grid_size=GRID)
        for t in range(2):
            viewpoint = policy.step(t, state)
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=GLIMPSE_PX)
            state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state
            losses.append(sum(distillation_losses(model, state, targets, **losses_enabled).values()))
    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()

    assert metrics.num_glimpses == 2
    torch.testing.assert_close(metrics.loss, mean_loss.detach())
    assert step_grads.keys() == {name for name, p in model.named_parameters() if p.grad is not None}
    for name, p in model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(step_grads[name], p.grad, msg=name)
