"""Entry point for CanViT pretraining."""

import logging
from dataclasses import replace

import optuna
import torch
import tyro

from .config import Config
from .loop import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)

    log.info("=" * 60)
    log.info("CanViT Pretraining")
    log.info("=" * 60)
    log.info(f"Config: {cfg}")
    log.info(f"Device: {cfg.device}")
    log.info(f"Steps per job: {cfg.steps_per_job:,}")
    log.info(f"Canvas patch grid size: {cfg.canvas_patch_grid_size}")
    log.info(f"Warmup: {cfg.warmup_steps} steps")
    log.info("=" * 60)

    def objective(trial: optuna.Trial) -> float:
        peak_lr = trial.suggest_float("peak_lr", 1e-6, 1e-2, log=True)
        train_cfg = replace(cfg, peak_lr=peak_lr)
        return train(train_cfg, trial)

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"peak_lr": cfg.peak_lr})
    study.optimize(objective, n_trials=cfg.n_trials)

    log.info("=" * 60)
    log.info(f"Best trial: {study.best_trial.params}")
    log.info(f"Best val_loss: {study.best_value:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
