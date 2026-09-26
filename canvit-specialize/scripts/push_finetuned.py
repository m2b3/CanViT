"""Publish a finetuned CanViTForImageClassification checkpoint to HuggingFace Hub.

Invocation: `uv run python scripts/push_finetuned.py --help`
End-to-end workflow context: canvit_specialize/training/gcp_in1k_clf_ft/README.md.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from canvit_pytorch import CanViTForImageClassification, Viewpoint, sample_at_viewpoint

from scripts.upload_utils import augment_hf_config_with_comet

log = logging.getLogger("push_finetuned")

# Keys that exist in the training checkpoint but not in CanViTForImageClassification
# (pretraining heads carried through training, unused for classification).
_PRETRAINING_PREFIXES = (
    "scene_cls_head.",
    "scene_patches_head.",
    "cls_standardizers.",
    "scene_standardizers.",
)


@dataclass
class Config:
    checkpoint: Path
    """Local .pt file produced by gcp_in1k_clf_ft training."""
    pretrained_repo: str
    """HF repo of the pretrained CanViT checkpoint used at training time."""
    probe_repo: str
    """HF repo of the DINOv3 linear probe fused into the finetuning head."""
    canvas_grid: int
    """Canvas grid size the checkpoint was trained with (typically 32)."""
    repo_id: str
    """Destination HF repo (e.g. canvit/canvitb16-add-vpe-finetune-...)."""
    public: bool = False
    """Publish as public instead of private (default: private)."""


def _remap_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map training keys (`model.*`, `norm.*`, `head.*`) to CanViTForImageClassification
    keys (`canvit.*`, `norm.*`, `head.*`), dropping pretraining-only prefixes."""
    remapped: dict[str, torch.Tensor] = {}
    skipped = 0
    for k, v in sd.items():
        if k.startswith("model."):
            stripped = k.removeprefix("model.")
            if any(stripped.startswith(p) for p in _PRETRAINING_PREFIXES):
                skipped += 1
                continue
            remapped[f"canvit.{stripped}"] = v
        elif k.startswith(("norm.", "head.")):
            remapped[k] = v
        else:
            raise ValueError(f"Unexpected training-checkpoint key: {k!r}")
    log.info("Remapped %d keys, skipped %d pretraining-only prefixes", len(remapped), skipped)
    return remapped


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")

    log.info("Building CanViTForImageClassification: %s + %s", cfg.pretrained_repo, cfg.probe_repo)
    model = CanViTForImageClassification.from_pretrained_with_probe(
        pretrained_repo=cfg.pretrained_repo,
        probe_repo=cfg.probe_repo,
        canvas_grid=cfg.canvas_grid,
    )

    log.info("Loading checkpoint: %s", cfg.checkpoint)
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    log.info("  step=%s  best_val_acc=%s  comet_key=%s",
             ckpt.get("step"), ckpt.get("best_val_acc"), ckpt.get("comet_key"))

    remapped = _remap_state_dict(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    log.info("state_dict loaded (%d tensors)", len(remapped))

    log.info("Sanity forward pass...")
    model.eval()
    scene = torch.randn(1, 3, 512, 512)
    vp = Viewpoint.full_scene(batch_size=1, device=torch.device("cpu"))
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=vp, glimpse_size_px=128)
    with torch.inference_mode():
        state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)
        logits, _ = model(glimpse=glimpse, state=state, viewpoint=vp)
    assert logits.shape == (1, 1000), f"Unexpected logits shape: {logits.shape}"
    log.info("  OK logits %s", tuple(logits.shape))

    log.info("Pushing to %s (public=%s)...", cfg.repo_id, cfg.public)
    model.push_to_hub(
        cfg.repo_id,
        private=not cfg.public,
        commit_message=f"step={ckpt.get('step')} best_val_acc={ckpt.get('best_val_acc', 0):.4f}",
    )

    comet_key = ckpt.get("comet_key")
    if comet_key:
        augment_hf_config_with_comet(
            cfg.repo_id,
            comet_key,
            extra={"base_checkpoint": cfg.pretrained_repo, "probe_repo": cfg.probe_repo},
        )
    else:
        log.info("No comet_key in checkpoint — skipping config augmentation")

    log.info("Done: https://huggingface.co/%s", cfg.repo_id)


if __name__ == "__main__":
    main(tyro.cli(Config))
