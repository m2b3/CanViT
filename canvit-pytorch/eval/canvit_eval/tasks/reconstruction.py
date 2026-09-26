"""Reconstruction quality: cosine similarity between CanViT canvas and DINOv3 teacher."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch.teacher import load_teacher
from canvit_pytorch.preprocess import preprocess
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from canvit_specialize.training.utils import collect_metadata

from canvit_eval.config import TEACHER_REPO, EpisodeConfig, progress, require_existing_dir
from canvit_eval.runner import eval_batches, load_model
from canvit_eval.tasks.base import TaskConfig

log = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FlatImageDataset(Dataset):
    def __init__(self, root: Path, transform: Callable[..., Tensor]) -> None:
        self.paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        assert len(self.paths) > 0, f"No images in {root}"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        return (self.transform(Image.open(self.paths[idx]).convert("RGB")),)


@dataclass
class _Acc:
    scene_raw: float = 0.0
    cls_raw: float = 0.0
    scene_norm: float = 0.0
    cls_norm: float = 0.0
    n: int = 0


def _default_image_dir() -> Path:
    from canvit_eval.config import ade20k_root
    return ade20k_root() / "images" / "validation"


@dataclass(kw_only=True)
class Config(TaskConfig):
    model_repo: str
    output: Path = Path("results/recon.pt")
    batch_size: int = 16
    num_workers: int = 4
    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(policy="random", n_timesteps=10))
    image_dir: Path = field(default_factory=_default_image_dir)
    scene_size: int = 512
    teacher_cache: Path | None = None

    def run(self) -> Path:
        return evaluate(self)


@torch.inference_mode()
def evaluate(cfg: Config) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)
    require_existing_dir(cfg.image_dir, description="Reconstruction image directory")

    model = load_model(cfg.model_repo, device)
    canvas_grid = cfg.episode.canvas_grid or cfg.scene_size // model.backbone.patch_size_px
    cls_std, scene_std = model.standardizers(canvas_grid)
    assert scene_std.initialized

    dataset = FlatImageDataset(cfg.image_dir, preprocess(cfg.scene_size))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        pin_memory=True, shuffle=False)

    # Teacher features
    teacher = load_teacher(TEACHER_REPO, device)
    if cfg.teacher_cache is not None and cfg.teacher_cache.exists():
        cached = torch.load(cfg.teacher_cache, map_location="cpu", weights_only=True)
        t_patches, t_cls = cached["patches"], cached["cls"]
    else:
        plist, clist = [], []
        for batch in progress(loader, desc="Teacher"):
            feats = teacher.forward_norm_features(batch[0].to(device))
            plist.append(feats.patches.cpu())
            clist.append(feats.cls.cpu())
        t_patches, t_cls = torch.cat(plist), torch.cat(clist)
        if cfg.teacher_cache is not None:
            cfg.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"patches": t_patches, "cls": t_cls}, cfg.teacher_cache)
    del teacher
    torch.cuda.empty_cache()

    T = cfg.episode.n_timesteps
    accs = [_Acc() for _ in range(T)]
    idx = 0
    t0 = time.perf_counter()

    for br in eval_batches(model=model, loader=loader, episode_cfg=cfg.episode,
                           canvas_grid=canvas_grid, device=device, amp=cfg.amp):
        B = br.batch[0].shape[0]
        raw_p = t_patches[idx:idx + B].to(device).float()
        raw_c = t_cls[idx:idx + B].to(device).float()
        norm_p, norm_c = scene_std(raw_p), cls_std(raw_c.unsqueeze(1)).squeeze(1)
        idx += B

        for step in br.steps:
            ps = model.predict_teacher_scene(step.state.canvas)
            pc = model.predict_scene_teacher_cls(step.state.recurrent_cls)
            a = accs[step.t]
            a.scene_raw += F.cosine_similarity(ps, raw_p, dim=-1).mean().item() * B
            a.scene_norm += F.cosine_similarity(ps, norm_p, dim=-1).mean().item() * B
            a.cls_raw += F.cosine_similarity(pc, raw_c, dim=-1).mean().item() * B
            a.cls_norm += F.cosine_similarity(pc, norm_c, dim=-1).mean().item() * B
            a.n += B

    elapsed = time.perf_counter() - t0
    per_t = [{"t": t, "scene_cos_raw": a.scene_raw / a.n, "cls_cos_raw": a.cls_raw / a.n,
              "scene_cos_norm": a.scene_norm / a.n, "cls_cos_norm": a.cls_norm / a.n}
             for t, a in enumerate(accs)]

    for p in per_t:
        log.info("  t=%d  scene=%.4f  cls=%.4f", p["t"], p["scene_cos_norm"], p["cls_cos_norm"])

    results = {
        "per_timestep": per_t, "n_images": idx,
        "metadata": {**collect_metadata(cfg), "elapsed_s": round(elapsed, 1)},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info("Saved to %s", cfg.output)
    return cfg.output
