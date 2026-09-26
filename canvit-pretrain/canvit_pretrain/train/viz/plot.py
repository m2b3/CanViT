"""Matplotlib plotting utilities for visualization."""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from ..probe import TopKPrediction
from ..viewpoint import PixelBox
from .metrics import cosine_dissimilarity
from .pca import fit_pca, pca_rgb

# Type alias for RGBA color tuple
RGBA = tuple[float, float, float, float]


def timestep_colors(n: int) -> list[RGBA]:
    """Get n distinct colors from viridis colormap."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def draw_trajectory_axis(ax, full_img: NDArray[np.floating], boxes: list[PixelBox], colors: list[RGBA], t: int) -> None:
    """Draw full_img with viewpoint boxes accumulated up to (and including) timestep t.

    Current step (i==t) is opaque, earlier steps faded; centers connected by a path.
    t == -1 draws the bare image (the 'init' row, before any glimpse). Shared by the
    PCA and pixel-reconstruction multistep plotters.
    """
    ax.imshow(full_img)
    for i in range(t + 1):
        box = boxes[i]
        alpha = 1.0 if i == t else 0.4
        ax.add_patch(Rectangle(
            (box.left, box.top), box.width, box.height,
            linewidth=2, edgecolor=colors[i], facecolor="none", alpha=alpha,
        ))
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=5, alpha=alpha)
    for i in range(1, t + 1):
        ax.plot(
            [boxes[i - 1].center_x, boxes[i].center_x],
            [boxes[i - 1].center_y, boxes[i].center_y],
            "-", color=colors[i], linewidth=1.5, alpha=0.7,
        )
    ax.axis("off")


def plot_trajectory(
    *,
    img: NDArray[np.floating],
    boxes: list[PixelBox],
    names: list[str],
) -> Figure:
    """Plot [H, W, 3] image (values in [0, 1]) with viewpoint boxes overlaid."""
    assert len(boxes) == len(names)
    colors = timestep_colors(len(boxes))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    if len(boxes) > 1:
        xs = [b.center_x for b in boxes]
        ys = [b.center_y for b in boxes]
        ax.plot(xs, ys, "-", color="white", linewidth=1.5, alpha=0.7, zorder=1)

    for i, (box, name) in enumerate(zip(boxes, names, strict=True)):
        rect = Rectangle(
            (box.left, box.top),
            box.width,
            box.height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            label=f"t={i} ({name}, s={box.width / img.shape[1]:.2f})",
        )
        ax.add_patch(rect)
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6, zorder=2)

    ax.set_title("Viewpoint Trajectory")
    if boxes:
        ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_pca_grid(
    *,
    pca: PCA,
    reference: NDArray[np.floating],
    samples: list[NDArray[np.floating]],
    grid_size: int,
    titles: list[str],
) -> Figure:
    """Plot PCA viz: [G*G, D] reference (teacher) vs samples (model outputs). PCA pre-fit on reference."""
    assert len(samples) == len(titles)
    n_cols = 1 + len(samples)

    reference_rgb = pca_rgb(pca, reference, grid_size, grid_size)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(reference_rgb)
    axes[0].set_title("Teacher")
    axes[0].axis("off")

    for i, (sample, title) in enumerate(zip(samples, titles, strict=True)):
        sample_rgb = pca_rgb(pca, sample, grid_size, grid_size)
        mse = float(((sample - reference) ** 2).mean())
        axes[i + 1].imshow(sample_rgb)
        axes[i + 1].set_title(f"{title}\nMSE={mse:.4f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    return fig


def _recon_patches_to_img(patches: NDArray[np.floating], grid: int) -> NDArray[np.floating]:
    """[G^2, 3*p*p] normalized pixel patches -> [H, W, 3] denormalized image in [0, 1]."""
    import torch
    from canvit_pytorch.preprocess import imagenet_denormalize

    d = patches.shape[1]
    p = int(round((d // 3) ** 0.5))
    assert 3 * p * p == d, f"patch dim {d} not 3*p^2"
    t = torch.from_numpy(np.ascontiguousarray(patches)).reshape(grid, grid, 3, p, p)
    img = t.permute(2, 0, 3, 1, 4).reshape(3, grid * p, grid * p)  # [C, H, W]
    return imagenet_denormalize(img).clip(0, 1).permute(1, 2, 0).numpy()


def plot_multistep_recon(
    *,
    full_img: NDArray[np.floating],
    target_patches: NDArray[np.floating],
    scenes: list[NDArray[np.floating]],
    glimpses: list[NDArray[np.floating]],
    boxes: list[PixelBox],
    names: list[str],
    scene_grid_size: int,
    initial_scene: NDArray[np.floating],
) -> Figure:
    """Pixel-space trajectory viz for RGB reconstruction (analog of plot_multistep_pca).

    Row 0 = init (reconstruction before any glimpse); rows 1+ = after each glimpse.
    Columns: Trajectory (accumulating viewpoint boxes) | Glimpse | Target | Reconstruction | Error.
    """
    n_views = len(scenes)
    assert len(glimpses) == n_views == len(boxes) == len(names)
    S = scene_grid_size
    colors = timestep_colors(n_views)

    target_img = _recon_patches_to_img(target_patches, S)
    recon_imgs = [_recon_patches_to_img(initial_scene, S)] + [_recon_patches_to_img(s, S) for s in scenes]
    # Per-patch reconstruction MSE vs target, reshaped to the canvas grid.
    err_maps = [((initial_scene - target_patches) ** 2).mean(1).reshape(S, S)]
    err_maps += [((s - target_patches) ** 2).mean(1).reshape(S, S) for s in scenes]

    C_TRAJ, C_GLIMPSE, C_TARGET, C_RECON, C_ERR = range(5)
    n_rows = n_views + 1
    fig, axes = plt.subplots(n_rows, 5, figsize=(4 * 5, 4 * n_rows), squeeze=False)

    for row in range(n_rows):
        t = row - 1  # row 0 = init (t == -1 -> no boxes)
        ax_traj = axes[row, C_TRAJ]
        draw_trajectory_axis(ax_traj, full_img, boxes, colors, t)
        ax_traj.set_title("init" if row == 0 else f"t={t}")

        ax_g = axes[row, C_GLIMPSE]
        if row == 0:
            ax_g.set_title("(no glimpse)")
        else:
            ax_g.imshow(glimpses[t])
            ax_g.set_title(f"Glimpse ({names[t]})")
        ax_g.axis("off")

        axes[row, C_TARGET].imshow(target_img)
        axes[row, C_TARGET].set_title("Target")
        axes[row, C_TARGET].axis("off")

        axes[row, C_RECON].imshow(recon_imgs[row])
        axes[row, C_RECON].set_title("Recon (init)" if row == 0 else f"Recon t={t}")
        axes[row, C_RECON].axis("off")

        im_err = axes[row, C_ERR].imshow(err_maps[row], cmap="hot")
        axes[row, C_ERR].set_title(f"MSE ({float(err_maps[row].mean()):.4f})")
        axes[row, C_ERR].axis("off")
        fig.colorbar(im_err, ax=axes[row, C_ERR], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


class TimestepPredictions(NamedTuple):
    """Top-k predictions at a single timestep for a single sample."""
    predictions: list[TopKPrediction]
    gt_idx: int
    gt_name: str


def plot_multistep_pca(
    *,
    full_img: NDArray[np.floating],
    teacher: NDArray[np.floating],
    scenes: list[NDArray[np.floating]],
    glimpses: list[NDArray[np.floating]],
    boxes: list[PixelBox],
    names: list[str],
    scene_grid_size: int,
    glimpse_grid_size: int,
    initial_scene: NDArray[np.floating],
    locals_avp: list[NDArray[np.floating]] | None = None,
    locals_teacher: list[NDArray[np.floating]] | None = None,
    hidden_spatials: list[NDArray[np.floating]] | None = None,
    initial_hidden_spatial: NDArray[np.floating] | None = None,
    locals_teacher_cropped: list[NDArray[np.floating]] | None = None,
    show_locals: bool = False,
    timestep_predictions: list[TimestepPredictions] | None = None,
) -> Figure:
    """Full multi-row visualization with all diagnostic columns.

    Row 0 = "init": learned spatial_hidden_init projected through scene_proj, BEFORE any glimpses
    Row 1+ = "t=0, t=1, ...": scene state AFTER processing each glimpse

    Columns: [Preds] | Trajectory | Glimpse | Teacher | Scene | [Hidden] | ... | Δ Scene | Error
    """
    n_views = len(scenes)
    assert len(glimpses) == n_views
    assert len(boxes) == n_views
    assert len(names) == n_views
    if show_locals:
        assert locals_avp is not None and len(locals_avp) == n_views
        # locals_teacher is optional - if None, only show locals_avp

    show_hidden = hidden_spatials is not None
    if show_hidden:
        assert len(hidden_spatials) == n_views
        assert initial_hidden_spatial is not None

    show_teacher_locals = show_locals and locals_teacher is not None
    if show_teacher_locals:
        assert locals_teacher is not None and len(locals_teacher) == n_views

    show_cropped = show_locals and locals_teacher_cropped is not None
    if show_cropped:
        assert locals_teacher_cropped is not None
        assert len(locals_teacher_cropped) == n_views

    show_preds = timestep_predictions is not None
    if show_preds:
        assert len(timestep_predictions) == n_views

    S, G = scene_grid_size, glimpse_grid_size
    n_rows = n_views + 1
    colors = timestep_colors(n_views)

    # Fit PCA on teacher (used for scene comparisons)
    pca_teacher = fit_pca(teacher)
    teacher_rgb = pca_rgb(pca_teacher, teacher, S, S)
    initial_rgb = pca_rgb(pca_teacher, initial_scene, S, S, normalize=True)

    # Scene with own PCA (initial)
    pca_init_scene = fit_pca(initial_scene)
    initial_scene_own_rgb = pca_rgb(pca_init_scene, initial_scene, S, S)

    # Hidden PCA for init (own basis)
    if show_hidden:
        assert initial_hidden_spatial is not None
        pca_init_hidden = fit_pca(initial_hidden_spatial)
        initial_hidden_rgb = pca_rgb(pca_init_hidden, initial_hidden_spatial, S, S)
    else:
        initial_hidden_rgb = None

    # Precompute error maps (cosine dissimilarity, per-row scaling)
    initial_error = cosine_dissimilarity(initial_scene, teacher).reshape(S, S)
    error_maps = [initial_error] + [cosine_dissimilarity(s, teacher).reshape(S, S) for s in scenes]

    # Compute delta maps for scene predictions (cosine dissimilarity from prev)
    delta_scene_maps: list[NDArray[np.floating]] = [np.zeros((S, S))]
    prev_scene = initial_scene
    for scene in scenes:
        delta_scene_maps.append(cosine_dissimilarity(scene, prev_scene).reshape(S, S))
        prev_scene = scene

    # Compute delta maps for hidden states (if available)
    delta_hidden_maps: list[NDArray[np.floating]] | None = None
    if show_hidden:
        assert hidden_spatials is not None and initial_hidden_spatial is not None
        delta_hidden_maps = [np.zeros((S, S))]
        prev_hidden = initial_hidden_spatial
        for h in hidden_spatials:
            delta_hidden_maps.append(cosine_dissimilarity(h, prev_hidden).reshape(S, S))
            prev_hidden = h

    # Column indices
    c = 0
    C_PREDS = c if show_preds else None
    if show_preds:
        c += 1
    C_TRAJ, C_GLIMPSE, C_TEACHER, C_SCENE, C_SCENE_OWN = c, c + 1, c + 2, c + 3, c + 4
    c += 5
    C_HIDDEN = c if show_hidden else None
    if show_hidden:
        c += 1
    C_LOCAL_AVP = c if show_locals else None
    if show_locals:
        c += 1
    C_LOCAL_TEACHER = c if show_teacher_locals else None
    if show_teacher_locals:
        c += 1
    C_CROPPED_TEACHER = c if show_cropped else None
    if show_cropped:
        c += 1
    C_DELTA_HIDDEN = c if show_hidden else None
    if show_hidden:
        c += 1
    C_DELTA_SCENE, C_ERROR = c, c + 1
    n_cols = c + 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Row 0: Initial state (before any glimpses)
    row = 0
    if show_preds:
        assert C_PREDS is not None
        axes[row, C_PREDS].axis("off")
        axes[row, C_PREDS].set_title("Predictions")
    draw_trajectory_axis(axes[row, C_TRAJ], full_img, boxes, colors, -1)
    axes[row, C_TRAJ].set_title("init")

    axes[row, C_GLIMPSE].axis("off")
    axes[row, C_GLIMPSE].set_title("(no glimpse)")

    axes[row, C_TEACHER].imshow(teacher_rgb)
    axes[row, C_TEACHER].set_title("Teacher")
    axes[row, C_TEACHER].axis("off")

    axes[row, C_SCENE].imshow(initial_rgb)
    axes[row, C_SCENE].set_title("Scene (init)")
    axes[row, C_SCENE].axis("off")

    axes[row, C_SCENE_OWN].imshow(initial_scene_own_rgb)
    axes[row, C_SCENE_OWN].set_title("Scene own (init)")
    axes[row, C_SCENE_OWN].axis("off")

    if show_hidden:
        assert C_HIDDEN is not None and initial_hidden_rgb is not None
        axes[row, C_HIDDEN].imshow(initial_hidden_rgb)
        axes[row, C_HIDDEN].set_title("Hidden (init)")
        axes[row, C_HIDDEN].axis("off")

    if show_locals:
        assert C_LOCAL_AVP is not None
        axes[row, C_LOCAL_AVP].axis("off")
    if show_teacher_locals:
        assert C_LOCAL_TEACHER is not None
        axes[row, C_LOCAL_TEACHER].axis("off")

    if show_cropped:
        assert C_CROPPED_TEACHER is not None
        axes[row, C_CROPPED_TEACHER].axis("off")

    if show_hidden:
        assert C_DELTA_HIDDEN is not None
        axes[row, C_DELTA_HIDDEN].axis("off")
        axes[row, C_DELTA_HIDDEN].set_title("Δ Hidden")

    axes[row, C_DELTA_SCENE].axis("off")
    axes[row, C_DELTA_SCENE].set_title("Δ Scene")

    im_err = axes[row, C_ERROR].imshow(error_maps[0], cmap="hot")
    cos_dis = float(error_maps[0].mean())
    axes[row, C_ERROR].set_title(f"CosDis ({cos_dis:.4f})")
    axes[row, C_ERROR].axis("off")
    fig.colorbar(im_err, ax=axes[row, C_ERROR], fraction=0.046, pad=0.04)

    # Rows 1+: After each glimpse (t=0, t=1, ...)
    for t in range(n_views):
        row = t + 1
        scene_rgb = pca_rgb(pca_teacher, scenes[t], S, S, normalize=True)

        pca_scene_t = fit_pca(scenes[t])
        scene_own_rgb = pca_rgb(pca_scene_t, scenes[t], S, S)

        if show_hidden:
            assert hidden_spatials is not None
            pca_hidden_t = fit_pca(hidden_spatials[t])
            hidden_rgb = pca_rgb(pca_hidden_t, hidden_spatials[t], S, S)
        else:
            hidden_rgb = None

        local_avp_rgb = None
        local_teacher_rgb = None
        pca_cropped: PCA | None = None
        if show_locals:
            assert locals_avp is not None
            pca_local_avp = fit_pca(locals_avp[t])
            local_avp_rgb = pca_rgb(pca_local_avp, locals_avp[t], G, G)
        if show_teacher_locals:
            assert locals_teacher is not None
            pca_local_teacher = fit_pca(locals_teacher[t])
            local_teacher_rgb = pca_rgb(pca_local_teacher, locals_teacher[t], G, G)
            if show_cropped:
                assert locals_teacher_cropped is not None
                pca_cropped = fit_pca(locals_teacher_cropped[t])

        if show_preds:
            assert C_PREDS is not None and timestep_predictions is not None
            preds_t = timestep_predictions[t]
            ax = axes[row, C_PREDS]
            k = len(preds_t.predictions)
            y_pos = np.arange(k)
            probs = [p.probability for p in preds_t.predictions]
            names_short = [p.class_name[:20] for p in preds_t.predictions]
            bar_colors = ["green" if p.class_idx == preds_t.gt_idx else "steelblue" for p in preds_t.predictions]
            ax.barh(y_pos, probs, color=bar_colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names_short, fontsize=8)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            ax.set_xlabel("prob", fontsize=8)
            gt_in_topk = any(p.class_idx == preds_t.gt_idx for p in preds_t.predictions)
            if not gt_in_topk:
                ax.set_title(f"GT: {preds_t.gt_name[:15]}", fontsize=8, color="green")

        draw_trajectory_axis(axes[row, C_TRAJ], full_img, boxes, colors, t)
        axes[row, C_TRAJ].set_title(f"t={t}")

        axes[row, C_GLIMPSE].imshow(glimpses[t])
        axes[row, C_GLIMPSE].set_title(f"Glimpse ({names[t]})")
        axes[row, C_GLIMPSE].axis("off")

        axes[row, C_TEACHER].imshow(teacher_rgb)
        axes[row, C_TEACHER].axis("off")

        axes[row, C_SCENE].imshow(scene_rgb)
        axes[row, C_SCENE].set_title(f"Scene t={t}")
        axes[row, C_SCENE].axis("off")

        axes[row, C_SCENE_OWN].imshow(scene_own_rgb)
        axes[row, C_SCENE_OWN].set_title("Scene own" if t == 0 else "")
        axes[row, C_SCENE_OWN].axis("off")

        if show_hidden:
            assert C_HIDDEN is not None and hidden_rgb is not None
            axes[row, C_HIDDEN].imshow(hidden_rgb)
            axes[row, C_HIDDEN].set_title("Hidden" if t == 0 else "")
            axes[row, C_HIDDEN].axis("off")

        if show_locals:
            assert C_LOCAL_AVP is not None and local_avp_rgb is not None
            axes[row, C_LOCAL_AVP].imshow(local_avp_rgb)
            if t == 0:
                axes[row, C_LOCAL_AVP].set_title("Local Stream")
            axes[row, C_LOCAL_AVP].axis("off")

        if show_teacher_locals:
            assert C_LOCAL_TEACHER is not None and local_teacher_rgb is not None
            axes[row, C_LOCAL_TEACHER].imshow(local_teacher_rgb)
            axes[row, C_LOCAL_TEACHER].set_title("Local Teacher" if t == 0 else "")
            axes[row, C_LOCAL_TEACHER].axis("off")

        if show_cropped:
            assert C_CROPPED_TEACHER is not None and locals_teacher_cropped is not None and pca_cropped is not None
            cropped_rgb = pca_rgb(pca_cropped, locals_teacher_cropped[t], G, G)
            axes[row, C_CROPPED_TEACHER].imshow(cropped_rgb)
            if t == 0:
                axes[row, C_CROPPED_TEACHER].set_title("Cropped Teacher")
            axes[row, C_CROPPED_TEACHER].axis("off")

        if show_hidden:
            assert C_DELTA_HIDDEN is not None and delta_hidden_maps is not None
            im_dh = axes[row, C_DELTA_HIDDEN].imshow(delta_hidden_maps[row], cmap="viridis")
            axes[row, C_DELTA_HIDDEN].set_title(f"Δh t={t}" if t == 0 else f"t={t}")
            axes[row, C_DELTA_HIDDEN].axis("off")
            fig.colorbar(im_dh, ax=axes[row, C_DELTA_HIDDEN], fraction=0.046, pad=0.04)

        im_ds = axes[row, C_DELTA_SCENE].imshow(delta_scene_maps[row], cmap="viridis")
        axes[row, C_DELTA_SCENE].set_title(f"Δs t={t}" if t == 0 else f"t={t}")
        axes[row, C_DELTA_SCENE].axis("off")
        fig.colorbar(im_ds, ax=axes[row, C_DELTA_SCENE], fraction=0.046, pad=0.04)

        im_err = axes[row, C_ERROR].imshow(error_maps[row], cmap="hot")
        cos_dis = float(error_maps[row].mean())
        axes[row, C_ERROR].set_title(f"CosDis ({cos_dis:.4f})")
        axes[row, C_ERROR].axis("off")
        fig.colorbar(im_err, ax=axes[row, C_ERROR], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig
