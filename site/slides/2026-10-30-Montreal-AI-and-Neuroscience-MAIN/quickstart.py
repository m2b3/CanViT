import torch
from PIL import Image
from canvit_pytorch import CanViTForSemanticSegmentation, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess

torch.set_grad_enabled(False)
model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
    pretrained_repo="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02",
    probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k",
).eval()
scene = preprocess(512)(Image.open("street.jpg").convert("RGB")).unsqueeze(0)
state = model.init_state(batch_size=1, canvas_grid_size=64)

# A first glimpse: the whole scene, zoomed out
viewpoint = Viewpoint.full_scene(batch_size=1, device=scene.device)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)
logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)  # [1, 150, 64, 64]

# A second glimpse, zoomed in on the left of the street: centers (row, col), scales
viewpoint = Viewpoint(centers=torch.tensor([[0.0, -0.4]]), scales=torch.tensor([0.5]))
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)
logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
