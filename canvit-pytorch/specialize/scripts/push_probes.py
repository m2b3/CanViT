"""Push ADE20K segmentation probes to HuggingFace Hub.

Four modes:
  single    — push one probe with an explicit repo_id
  batch     — push every probe under a directory, auto-deriving repo_ids
  retrofit  — refresh the model card, collection note, and config.json
              sanitization for probes already on HF (no local .pt needed)
  reorder   — re-sort canvas + DINOv3 probe collections by canonical key

Every publish (single/batch):
  1. Uploads model.safetensors + config.json (probe weights + training metadata)
  2. Uploads README.md (minimal model card, HP table derived from metadata)
  3. Upserts the repo into the appropriate collection with a canonical note
  4. Reorders the collection so items appear smallest → largest

Usage:
    uv run python scripts/push_probes.py single \
        --probe PATH --repo-id canvit/probe-ade20k-... [--public] [--dry-run]

    uv run python scripts/push_probes.py batch \
        --probe-dir PATH [--match GLOB] [--owner canvit] [--public] [--dry-run]

    uv run python scripts/push_probes.py retrofit \
        --repo-ids canvit/probe-ade20k-... [... more ...] [--dry-run]

    uv run python scripts/push_probes.py reorder [--dry-run]
"""

import fnmatch
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import tyro
from huggingface_hub import HfApi, hf_hub_download

from canvit_pytorch import CANVIT_REPO_ROOT, resolve_canvit_repo
from canvit_pytorch.checkpoints import (
    ABLATION_MODEL_SHORTS,
    PRETRAIN_MODEL_SHORTS,
    ade20k_dinov3_probe_name,
    ade20k_probe_name,
)
from canvit_pytorch.probes import SegmentationProbe
from scripts.upload_utils import (
    json_sanitize,
    upload_model_card,
    upload_probe_to_hub,
    upsert_collection_item,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Stable HF collection slugs under the canvit-org root.
CANVAS_PROBE_COLLECTION = f"{CANVIT_REPO_ROOT}/canvit-ade20k-segmentation-probes-pytorch-69d550b66add770c509bb77a"
DINOV3_PROBE_COLLECTION = f"{CANVIT_REPO_ROOT}/dinov3-ade20k-segmentation-probes-pytorch-69d59b1eb69bbb3422f49b4f"

# Batch-mode repo naming: short id per base-model repo. Full name lives in config.json.
# Aliased CanViT-B repo names resolve to the same weights.
_MODEL_SHORT: dict[str, str] = {
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "dv3b",
    "facebook/dinov3-vits16-pretrain-lvd1689m": "dv3s",
    # Legacy alias for the IN21k flagship weights.
    resolve_canvit_repo("canvit-vitb16-pretrain-512px-in21k"): "in21k",
    **PRETRAIN_MODEL_SHORTS,  # flagship (in21k), in1k, sa1b
    **ABLATION_MODEL_SHORTS,
}



def _extract_state_dict_and_metadata(raw: dict) -> tuple[dict, dict]:
    """Supports canvas and DINOv3 probe checkpoint formats."""
    if "feat_type" in raw:
        sd = raw["probe_state_dict"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dict"}
        return sd, meta
    if "probe_state_dict" in raw:
        sd = raw["probe_state_dict"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dict"}
        meta["feat_type"] = "dinov3_spatial"
        return sd, meta
    raise AssertionError(f"Unknown probe checkpoint format. Keys: {sorted(raw.keys())}")


def _assert_not_finetune(raw: dict, name: str) -> None:
    """LP-FT checkpoints carry full CanViT weights; pushing via this script
    would silently drop them. Refuse until a paired-publish path exists."""
    if "model_state_dict" in raw or raw.get("config", {}).get("finetune") is True:
        raise NotImplementedError(
            f"{name} is an LP-FT checkpoint with full CanViT weights. "
            f"push_probes.py only handles standalone probe heads; pushing "
            f"would silently drop the CanViT state_dict."
        )


def load_probe(pt_path: Path) -> tuple[SegmentationProbe, dict]:
    """Return (probe, metadata). Probe weights strict-loaded and validated."""
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    _assert_not_finetune(raw, pt_path.name)
    sd, meta = _extract_state_dict_and_metadata(raw)

    embed_dim = sd["conv.weight"].shape[1]
    num_classes = sd["conv.weight"].shape[0]
    use_ln = "ln.weight" in sd
    dropout = meta.get("config", {}).get("dropout")
    assert dropout is not None, f"dropout missing in {pt_path}"

    probe = SegmentationProbe(
        embed_dim=embed_dim, num_classes=num_classes,
        dropout=dropout, use_ln=use_ln,
    )
    result = probe.load_state_dict(sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return probe, meta



def _resolve_feat_type(meta: dict) -> str:
    """Return 'canvas_hidden' or 'dinov3_spatial'. Falls back on schema sniffing."""
    ft = meta.get("feat_type")
    if ft is not None:
        return ft
    cfg = meta.get("config", {})
    if "resolution" in cfg and "model" in cfg:
        return "dinov3_spatial"
    return "canvas_hidden"



def _dinov3_variant(model_repo: str) -> str:
    """'facebook/dinov3-vitb16-...' -> 'ViT-B/16'."""
    m = re.search(r"dinov3-vit(b|s|l|h)(\d+)", model_repo)
    assert m is not None, f"Cannot parse DINOv3 variant from {model_repo!r}"
    return f"ViT-{m.group(1).upper()}/{m.group(2)}"



def derive_repo_id(owner: str, meta: dict) -> str:
    cfg = meta["config"]
    steps_k = cfg["max_steps"] // 1000

    if _resolve_feat_type(meta) == "dinov3_spatial":
        short = _MODEL_SHORT[cfg["model"]]
        name = ade20k_dinov3_probe_name(short, resolution=cfg["resolution"], steps_k=steps_k)
        return f"{owner}/{name}"

    short = _MODEL_SHORT.get(cfg["model_repo"])
    assert short is not None, (
        f"Unknown model_repo {cfg['model_repo']!r} — add to _MODEL_SHORT."
    )
    name = ade20k_probe_name(short, scene=cfg["scene_size"], grid=cfg["canvas_grid"], steps_k=steps_k)
    return f"{owner}/{name}"



def canvas_probe_note(cfg: dict) -> str:
    short = _MODEL_SHORT[cfg["model_repo"]]
    return f"CanViT {short}, {cfg['scene_size']}px scene, {cfg['canvas_grid']}×{cfg['canvas_grid']} canvas grid"


def dinov3_probe_note(cfg: dict) -> str:
    return f"DINOv3 {_dinov3_variant(cfg['model'])}, {cfg['resolution']}px input"



def _sci_katex(x: float) -> str:
    """Inline KaTeX math for a scientific-notation number.

    HF model cards render `\\\\(...\\\\)` as inline math (the markdown layer
    halves the double-backslash, then KaTeX sees `\\(...\\)`). We emit
    two backslashes literally so the source is markdown-safe.
    """
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10 ** exp)
    if abs(mantissa - 1) < 1e-9:
        return rf"\\( 10^{{{exp}}} \\)"
    if exp == 0:
        return f"{mantissa:g}"
    return rf"\\( {mantissa:g} \times 10^{{{exp}}} \\)"



def _common_hp_rows(cfg: dict) -> list[tuple[str, str]]:
    scale_lo, scale_hi = cfg["aug_scale_range"]
    return [
        ("Optimizer", "AdamW"),
        ("Peak LR", _sci_katex(cfg["peak_lr"])),
        ("Weight decay", _sci_katex(cfg["weight_decay"])),
        ("LR schedule", f"{cfg['warmup_steps']:,}-step warmup → cosine decay"),
        ("Batch size", str(cfg["batch_size"])),
        ("Max steps", f"{cfg['max_steps']:,}"),
        ("Dropout", str(cfg["dropout"])),
        ("Augmentation", f"RandomResizedCrop scale [{scale_lo:g}, {scale_hi:g}] + HFlip"),
        ("Precision", "bf16 (AMP)" if cfg.get("amp") else "fp32"),
    ]


def _canvas_hp_rows(cfg: dict) -> list[tuple[str, str]]:
    specific = [
        ("Scene size", f"{cfg['scene_size']} px"),
        ("Canvas grid", f"{cfg['canvas_grid']} × {cfg['canvas_grid']}"),
        ("Glimpse size", f"{cfg['glimpse_px']} px"),
        ("Timesteps (T)", str(cfg["n_timesteps"])),
        ("Training policy", "F-IID" if cfg.get("train_start_full") else "R-IID"),
    ]
    return specific + _common_hp_rows(cfg)


def _dinov3_hp_rows(cfg: dict) -> list[tuple[str, str]]:
    specific = [
        ("Input size", f"{cfg['resolution']} × {cfg['resolution']} px"),
    ]
    return specific + _common_hp_rows(cfg)


def _rows_to_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Hyperparameter | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)



_CARD_TEMPLATE = """\
---
license: mit
library_name: canvit-pytorch
pipeline_tag: image-segmentation
tags:
{tags}
datasets:
  - scene_parse_150
base_model: {base_model}
---

# {title}

{intro}

- **Paper**: [arXiv:2603.22570](https://arxiv.org/abs/2603.22570)
- **Training code**: [github.com/m2b3/CanViT-specialize](https://github.com/m2b3/CanViT-specialize)

## Usage

```bash
uv add "canvit-pytorch @ git+https://github.com/m2b3/CanViT-PyTorch.git"
```

```python
import torch
from canvit_pytorch.probes import SegmentationProbe

probe = SegmentationProbe.from_pretrained("{repo_id}").eval()

# {input_narrative}
features = torch.randn(1, {grid}, {grid}, {embed_dim})
with torch.inference_mode():
    logits = probe(features)    # [B, num_classes, H, W]
assert logits.shape == (1, {num_classes}, {grid}, {grid})
```

## Training

Architecture: `{arch}`.

{hp_table}
"""


def _arch_str(use_ln: bool) -> str:
    """Describe the probe architecture (LN is optional; DINOv3 probes omit it)."""
    return "LayerNorm → Dropout → BatchNorm → Conv1×1" if use_ln else "Dropout → BatchNorm → Conv1×1"


def _indent_tags(tags: list[str]) -> str:
    return "\n".join(f"  - {t}" for t in tags)


def build_canvas_card(
    repo_id: str, *, embed_dim: int, num_classes: int, use_ln: bool, cfg: dict,
) -> str:
    scene, grid = cfg["scene_size"], cfg["canvas_grid"]
    base_model = cfg["model_repo"]
    return _CARD_TEMPLATE.format(
        tags=_indent_tags(["canvit", "active-vision", "ade20k", "segmentation-probe"]),
        base_model=base_model,
        title=f"ADE20K Segmentation Probe — canvas {grid}×{grid} @ {scene}px scene",
        intro=(f"Linear segmentation probe on the canvas features of\n"
               f"[{base_model}](https://huggingface.co/{base_model})."),
        repo_id=repo_id,
        grid=grid,
        embed_dim=embed_dim,
        num_classes=num_classes,
        input_narrative="[B, H, W, D] canvas features from a CanViT forward pass",
        arch=_arch_str(use_ln),
        hp_table=_rows_to_table(_canvas_hp_rows(cfg)),
    )


def build_dinov3_card(
    repo_id: str, *, embed_dim: int, num_classes: int, use_ln: bool, cfg: dict,
) -> str:
    base_model = cfg["model"]
    variant = _dinov3_variant(base_model)
    res = cfg["resolution"]
    grid = res // 16  # DINOv3 patch size is 16
    return _CARD_TEMPLATE.format(
        tags=_indent_tags(["dinov3", "ade20k", "segmentation-probe"]),
        base_model=base_model,
        title=f"ADE20K Segmentation Probe — DINOv3 {variant} @ {res}px input",
        intro=(f"Linear segmentation probe on the spatial features of\n"
               f"[{base_model}](https://huggingface.co/{base_model})."),
        repo_id=repo_id,
        grid=grid,
        embed_dim=embed_dim,
        num_classes=num_classes,
        input_narrative=f"[B, H, W, D] DINOv3 {variant} spatial features at {res}px input",
        arch=_arch_str(use_ln),
        hp_table=_rows_to_table(_dinov3_hp_rows(cfg)),
    )



def _reorder_collection(
    slug: str, sort_key: Callable[[dict], tuple], *, dry_run: bool = False,
) -> None:
    """Sort a collection's items by `sort_key(training_cfg)` ascending."""
    api = HfApi()
    col = api.get_collection(slug)
    ranked = []
    for item in col.items:
        cfg = json.loads(Path(hf_hub_download(item.item_id, "config.json")).read_text())
        tc = cfg.get("metadata", {}).get("config", {})
        ranked.append((sort_key(tc), item))
    ranked.sort(key=lambda x: x[0])
    for i, (key, item) in enumerate(ranked):
        if item.position != i:
            log.info("  reorder %s: pos %d -> %d (key=%s)",
                     item.item_id, item.position, i, key)
            if not dry_run:
                api.update_collection_item(slug, item.item_object_id, position=i)


def reorder_canvas_collection(*, dry_run: bool = False) -> None:
    """Sort canvas probes by (scene_size, canvas_grid) ascending."""
    _reorder_collection(
        CANVAS_PROBE_COLLECTION,
        lambda tc: (tc["scene_size"], tc["canvas_grid"]),
        dry_run=dry_run,
    )


def reorder_dinov3_collection(*, dry_run: bool = False) -> None:
    """Sort DINOv3 probes by (backbone variant, resolution) ascending."""
    _reorder_collection(
        DINOV3_PROBE_COLLECTION,
        lambda tc: (tc["model"], tc["resolution"]),
        dry_run=dry_run,
    )



def _build_card_and_note(repo_id: str, embed_dim: int, num_classes: int, use_ln: bool,
                        feat_type: str, cfg: dict) -> tuple[str, str, str]:
    """Return (card_markdown, collection_note, collection_slug)."""
    if feat_type == "canvas_hidden":
        return (
            build_canvas_card(repo_id, embed_dim=embed_dim, num_classes=num_classes,
                              use_ln=use_ln, cfg=cfg),
            canvas_probe_note(cfg),
            CANVAS_PROBE_COLLECTION,
        )
    if feat_type == "dinov3_spatial":
        return (
            build_dinov3_card(repo_id, embed_dim=embed_dim, num_classes=num_classes,
                              use_ln=use_ln, cfg=cfg),
            dinov3_probe_note(cfg),
            DINOV3_PROBE_COLLECTION,
        )
    raise ValueError(f"Unknown feat_type: {feat_type!r}")


def publish_probe(
    probe: SegmentationProbe, meta: dict, repo_id: str,
    *, public: bool, dry_run: bool,
) -> None:
    feat_type = _resolve_feat_type(meta)
    cfg = meta["config"]
    card, note, collection = _build_card_and_note(
        repo_id, probe.embed_dim, probe.num_classes, probe.use_ln, feat_type, cfg,
    )

    hf_config = {
        "embed_dim": probe.embed_dim,
        "num_classes": probe.num_classes,
        "dropout": probe.dropout_p,
        "use_ln": probe.use_ln,
        "metadata": json_sanitize(meta),
    }

    log.info("  → %s  visibility=%s", repo_id, "public" if public else "private")
    log.info("     note: %s", note)
    if dry_run:
        return

    upload_probe_to_hub(
        state_dict=probe.state_dict(), config=hf_config,
        repo_id=repo_id, private=not public,
    )
    upload_model_card(repo_id=repo_id, card_text=card)
    upsert_collection_item(collection, repo_id, note=note)



@dataclass
class Single:
    """Push one probe checkpoint with an explicit repo_id."""
    probe: Path
    repo_id: str
    public: bool = False
    dry_run: bool = False


@dataclass
class Batch:
    """Push every probe under `probe_dir`, auto-deriving repo_ids."""
    probe_dir: Path
    owner: str = CANVIT_REPO_ROOT
    match: str = "*"
    """fnmatch pattern on subdirectory names; non-matching run dirs are not loaded."""
    public: bool = False
    dry_run: bool = False


@dataclass
class Retrofit:
    """Refresh model card + collection note + config sanitization for probes
    already on HF. No local .pt needed — reads config.json from each repo."""
    repo_ids: list[str]
    dry_run: bool = False


@dataclass
class Reorder:
    """Re-sort both probe collections by their canonical key."""
    dry_run: bool = False


def _find_best_pt(d: Path) -> Path:
    candidates = list(d.glob("*best*miou*.pt"))
    assert len(candidates) >= 1, f"No best checkpoint in {d}"
    def _miou(p: Path) -> float:
        m = re.search(r"miou([\d.]+)", p.name)
        return float(m.group(1)) if m else 0.0
    return max(candidates, key=_miou)


def _run_single(args: Single) -> None:
    assert args.probe.exists(), f"Not found: {args.probe}"
    probe, meta = load_probe(args.probe)
    publish_probe(probe, meta, args.repo_id,
                  public=args.public, dry_run=args.dry_run)
    if not args.dry_run:
        _reorder_after_publish(meta)


def _run_batch(args: Batch) -> None:
    assert args.probe_dir.is_dir(), f"Not a directory: {args.probe_dir}"
    probe_dirs = sorted(
        d for d in args.probe_dir.iterdir()
        if d.is_dir() and fnmatch.fnmatch(d.name, args.match)
    )
    log.info("%s %d probe directories in %s (match=%r)",
             "DRY RUN:" if args.dry_run else "Pushing", len(probe_dirs), args.probe_dir, args.match)
    touched_feat_types: set[str] = set()
    for d in probe_dirs:
        try:
            best_pt = _find_best_pt(d)
        except AssertionError as e:
            log.info("  SKIP %s: %s", d.name, e)
            continue
        probe, meta = load_probe(best_pt)
        cfg = meta.get("config", {})
        if cfg.get("max_steps") != 40000:
            log.info("  SKIP %s (max_steps=%s, want 40000)", d.name, cfg.get("max_steps"))
            continue
        repo_id = derive_repo_id(args.owner, meta)
        publish_probe(probe, meta, repo_id,
                      public=args.public, dry_run=args.dry_run)
        touched_feat_types.add(_resolve_feat_type(meta))
    if not args.dry_run:
        _reorder_touched(touched_feat_types)
    log.info("Done.")


def _reorder_after_publish(meta: dict) -> None:
    _reorder_touched({_resolve_feat_type(meta)})


def _reorder_touched(feat_types: set[str]) -> None:
    if "canvas_hidden" in feat_types:
        log.info("Reordering canvas collection...")
        reorder_canvas_collection()
    if "dinov3_spatial" in feat_types:
        log.info("Reordering DINOv3 collection...")
        reorder_dinov3_collection()


def _run_retrofit(args: Retrofit) -> None:
    api = HfApi()
    touched_feat_types: set[str] = set()
    for repo_id in args.repo_ids:
        log.info("Retrofitting %s", repo_id)
        raw_text = Path(hf_hub_download(repo_id, "config.json")).read_text()
        cfg_json = json.loads(raw_text)
        meta = cfg_json.get("metadata", {})
        train_cfg = meta.get("config", {})
        feat_type = _resolve_feat_type(meta)
        touched_feat_types.add(feat_type)

        card, note, collection = _build_card_and_note(
            repo_id, cfg_json["embed_dim"], cfg_json["num_classes"],
            cfg_json["use_ln"], feat_type, train_cfg,
        )
        # Coerce non-finite floats to strict-JSON form.
        fresh_cfg_text = json.dumps(json_sanitize(cfg_json), indent=2)
        cfg_needs_rewrite = fresh_cfg_text != raw_text

        log.info("  note: %s", note)
        log.info("  card: %d chars", len(card))
        log.info("  config.json rewrite: %s",
                 "yes (sanitized)" if cfg_needs_rewrite else "no (clean)")
        if args.dry_run:
            continue

        upload_model_card(repo_id=repo_id, card_text=card)
        upsert_collection_item(collection, repo_id, note=note)
        if cfg_needs_rewrite:
            api.upload_file(
                path_or_fileobj=fresh_cfg_text.encode(),
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message="Sanitize config.json (Infinity → \"inf\", Paths → str)",
            )
    if not args.dry_run:
        _reorder_touched(touched_feat_types)
    log.info("Done.")


def _run_reorder(args: Reorder) -> None:
    log.info("Reordering canvas collection...")
    reorder_canvas_collection(dry_run=args.dry_run)
    log.info("Reordering DINOv3 collection...")
    reorder_dinov3_collection(dry_run=args.dry_run)
    log.info("Done.")


def main() -> None:
    args = tyro.cli(Single | Batch | Retrofit | Reorder)
    if isinstance(args, Single):
        _run_single(args)
    elif isinstance(args, Batch):
        _run_batch(args)
    elif isinstance(args, Retrofit):
        _run_retrofit(args)
    else:
        _run_reorder(args)


if __name__ == "__main__":
    main()
