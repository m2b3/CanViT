import torch
import torch.nn.functional as F
from PIL import Image
from canvit_pytorch import CanViTForPretraining, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess

model = CanViTForPretraining.from_pretrained(
    "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
).eval()
scene = preprocess(512)(Image.open("scene.jpg").convert("RGB")).unsqueeze(0)
state = model.init_state(batch_size=1, canvas_grid_size=64)

full_scene = Viewpoint.full_scene(batch_size=1, device=None)
top_left = Viewpoint(centers=torch.tensor([[-0.5, -0.5]]), scales=torch.tensor([0.5]))

with torch.inference_mode():
    for viewpoint in [full_scene, top_left]:
        glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)
        state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state

canvas = F.layer_norm(model.canvit.canvas_patch_grid(state.canvas), [1024])  # [1, 64, 64, 1024]
