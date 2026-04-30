from __future__ import annotations

import json
from pathlib import Path
import random
from datetime import datetime
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from open_gen.data.collate import BatchNorms, MultimodalCollator
from open_gen.data.curriculum import CurriculumScheduler, CurriculumStage, parse_curriculum_config
from open_gen.data.datasets import build_dataset
from open_gen.data.loader import BucketedDataLoader
from open_gen.data.sampler import BucketedTokenBudgetBatchSampler
from open_gen.data.schema import CanonicalTrajectory
from open_gen.data.transforms import AugmentationConfig
from open_gen.model.model_config import config_from_dict
from open_gen.model.opengen1 import OpenGen1Model
from open_gen.train.distributed import DistEnv, finalize_distributed, init_distributed
from open_gen.utils.normalization import RunningStats, save_stats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_exp_name(exp_name: str | None) -> str:
    if exp_name:
        return exp_name
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def compute_norms(
    dataset: Dataset[CanonicalTrajectory],
    dims: dict[str, int],
    *,
    track_quantiles: bool = False,
) -> BatchNorms:
    action_stats = RunningStats(dims["action"], track_quantiles=track_quantiles)
    proprio_stats = (
        RunningStats(dims["proprio"], track_quantiles=track_quantiles) if dims["proprio"] > 0 else None
    )
    force_stats = (
        RunningStats(dims["force"], track_quantiles=track_quantiles) if dims["force"] > 0 else None
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        action_stats.update(torch.from_numpy(sample.action.reshape(-1, sample.action.shape[-1])))
        if sample.past_action is not None:
            action_stats.update(torch.from_numpy(sample.past_action.reshape(-1, sample.past_action.shape[-1])))
        if sample.proprio is not None and proprio_stats is not None:
            proprio_stats.update(torch.from_numpy(sample.proprio.reshape(-1, sample.proprio.shape[-1])))
        if sample.future_proprio is not None and proprio_stats is not None:
            proprio_stats.update(torch.from_numpy(sample.future_proprio.reshape(-1, sample.future_proprio.shape[-1])))
        if sample.force is not None and force_stats is not None:
            force_stats.update(torch.from_numpy(sample.force.reshape(-1, sample.force.shape[-1])))
        if sample.future_force is not None and force_stats is not None:
            force_stats.update(torch.from_numpy(sample.future_force.reshape(-1, sample.future_force.shape[-1])))
    return BatchNorms(
        action=action_stats.finalize(),
        proprio=proprio_stats.finalize() if proprio_stats is not None else None,
        force=force_stats.finalize() if force_stats is not None else None,
    )


def _fallback_stage(cfg: dict[str, Any]) -> CurriculumStage:
    packing = cfg["data"].get("packing", {}) or {}
    boundaries = packing.get("bucket_boundaries") or [4]
    return CurriculumStage(
        name="default",
        max_past_frames=int(packing.get("default_max_past_frames", max(boundaries))),
        min_past_frames=int(packing.get("default_min_past_frames", 1)),
        frame_sampling=packing.get("default_frame_sampling", "max"),
        token_budget=int(packing.get("default_token_budget", 32_768)),
        until_step=None,
        until_epoch=int(cfg["train"].get("epochs", 1)),
    )


def _build_train_sampler(
    *,
    dataset: Dataset[CanonicalTrajectory],
    cfg: dict[str, Any],
    stage: CurriculumStage,
    env: DistEnv,
) -> BucketedTokenBudgetBatchSampler:
    packing = cfg["data"].get("packing", {}) or {}
    boundaries = list(packing.get("bucket_boundaries") or [4])
    tokens_per_frame = int(packing.get("tokens_per_frame_estimate", 32))
    context_tokens = int(packing.get("context_tokens_estimate", 24))
    if not hasattr(dataset, "native_lengths"):
        raise TypeError("dataset must expose native_lengths() for bucketed batching")
    native_lengths = dataset.native_lengths()
    return BucketedTokenBudgetBatchSampler(
        native_lengths=native_lengths,
        bucket_boundaries=boundaries,
        tokens_per_frame=tokens_per_frame,
        context_tokens=context_tokens,
        stage=stage,
        world_size=env.world_size,
        rank=env.rank,
        seed=int(cfg["experiment"]["seed"]),
        drop_last=bool(packing.get("drop_last", True)),
    )


def _build_collator(cfg: dict[str, Any], norms: BatchNorms, train: bool, pad_to: int | None) -> MultimodalCollator:
    return MultimodalCollator(
        image_size=cfg["model"]["image_encoder"]["image_size"],
        encoder_type=cfg["model"]["image_encoder"]["type"],
        train=train,
        norms=norms,
        augmentation=AugmentationConfig(**cfg["data"].get("augmentation", {})),
        pad_to_obs_steps=pad_to,
        norm_mode=cfg["data"].get("norm_mode", "zscore"),
    )


def _build_eval_loader(cfg: dict[str, Any], norms: BatchNorms, env: DistEnv) -> DataLoader:
    eval_dataset = build_dataset(cfg["data"], cfg["dims"], "eval")
    eval_lengths = getattr(eval_dataset, "native_lengths", lambda: [1])()
    eval_pad = max(eval_lengths) if eval_lengths else None
    eval_collator = _build_collator(cfg, norms, train=False, pad_to=eval_pad)
    return DataLoader(
        eval_dataset,
        batch_size=cfg["train"]["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=eval_collator,
    )


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch)
            for key, value in outputs.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def _model_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def train(cfg: dict[str, Any]) -> Path:
    experiment = cfg["experiment"]
    train_cfg = cfg["train"]
    optim_cfg = train_cfg["optim"]

    set_seed(int(experiment["seed"]))
    env = init_distributed(device_hint=train_cfg["device"])

    exp_name = ensure_exp_name(experiment.get("exp_name"))
    output_dir = Path(experiment["output_root"]) / exp_name
    if env.is_main:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8")
        writer: SummaryWriter | None = SummaryWriter(log_dir=output_dir / "tensorboard")
    else:
        writer = None

    norms_dataset = build_dataset(cfg["data"], cfg["dims"], "train")
    norm_mode = cfg["data"].get("norm_mode", "zscore")
    norms = compute_norms(norms_dataset, cfg["dims"], track_quantiles=norm_mode == "quantile")
    if env.is_main:
        save_stats(
            output_dir / "norm_stats.json",
            {k: v for k, v in {"action": norms.action, "proprio": norms.proprio, "force": norms.force}.items() if v is not None},
        )

    train_dataset = build_dataset(cfg["data"], cfg["dims"], "train")
    curriculum_cfg = parse_curriculum_config(cfg["data"].get("curriculum"))
    curriculum = CurriculumScheduler(curriculum_cfg, fallback=_fallback_stage(cfg))
    initial_stage = curriculum.update(step=0, epoch=0)[1]

    train_collator = _build_collator(cfg, norms, train=True, pad_to=None)
    train_sampler = _build_train_sampler(dataset=train_dataset, cfg=cfg, stage=initial_stage, env=env)
    train_loader = BucketedDataLoader(train_dataset, train_sampler, train_collator)
    eval_loader = _build_eval_loader(cfg, norms, env)

    model = OpenGen1Model(config_from_dict(cfg["model"]))
    device = torch.device(train_cfg["device"])
    model.to(device)
    if env.is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[env.local_rank] if device.type == "cuda" else None,
            output_device=env.local_rank if device.type == "cuda" else None,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg["weight_decay"],
        betas=tuple(optim_cfg["betas"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"], eta_min=optim_cfg["min_lr"])

    global_step = 0
    eval_metrics: dict[str, float] = {}
    for epoch in range(train_cfg["epochs"]):
        changed, stage = curriculum.update(step=global_step, epoch=epoch)
        if changed:
            train_sampler.set_stage(stage)
        train_sampler.set_epoch(epoch)
        if writer is not None:
            writer.add_scalar("curriculum/stage_idx", float(curriculum.stage_index_for(step=global_step, epoch=epoch)), global_step)
            writer.add_scalar("curriculum/max_past_frames", float(stage.max_past_frames), global_step)
            writer.add_scalar("curriculum/min_past_frames", float(stage.min_past_frames), global_step)
            writer.add_scalar("curriculum/token_budget", float(stage.token_budget), global_step)

        model.train()
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{train_cfg['epochs']} stage:{stage.name}",
            leave=False,
            disable=not env.is_main,
        )
        for batch in progress:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()
            loss_value = float(outputs["loss"].detach().cpu())
            if env.is_main:
                progress.set_postfix(loss=loss_value, T=batch["observations"]["left_wrist_image"].shape[1])
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar(
                        "train/obs_steps",
                        float(batch["observations"]["left_wrist_image"].shape[1]),
                        global_step,
                    )
                    writer.add_scalar("train/batch_size", float(batch["observations"]["left_wrist_image"].shape[0]), global_step)
                    for key, value in outputs.items():
                        if key != "loss":
                            writer.add_scalar(f"train/{key}", float(value.detach().cpu()), global_step)
            global_step += 1

        if env.is_main:
            eval_metrics = _evaluate(model, eval_loader, device)
            for key, value in eval_metrics.items():
                if writer is not None:
                    writer.add_scalar(f"eval/{key}", value, epoch)
            checkpoint = {
                "model": _model_module(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "metrics": eval_metrics,
                "stage": stage.name,
            }
            torch.save(checkpoint, output_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt")
        scheduler.step()

    if writer is not None:
        writer.close()
    if env.is_main:
        summary = {
            "exp_name": exp_name,
            "output_dir": str(output_dir),
            "final_eval": eval_metrics,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finalize_distributed(env)
    return output_dir
