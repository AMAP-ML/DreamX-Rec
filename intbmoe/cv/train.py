"""Train the Tiny ViT + IntBMoE model on ImageNet-1K."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import accuracy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .config import CVConfig, cv_config_from_env
from .data import build_imagenet
from .model import IntBMoEVisionTransformer


class SmoothedValue:
    """Track a recent window and the global average for training logs."""

    def __init__(self, window_size: int = 20):
        self.values = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.values.append(value)
        self.total += value
        self.count += 1

    @property
    def median(self) -> float:
        return float(torch.tensor(list(self.values)).median())

    @property
    def average(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def global_average(self) -> float:
        return self.total / self.count


def parse_args(config: CVConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IntBMoE Tiny ViT on ImageNet-1k")
    parser.add_argument(
        "--data-path",
        required=True,
        help="HuggingFace datasets cache_dir containing imagenet-1k",
    )
    parser.add_argument("--output-dir", default=f"output/{config.experiment_name}")
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size_per_device)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--seed", type=int, default=config.random_seed)
    parser.add_argument("--resume", default=None, help="path to a training checkpoint")
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CV training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def reduce_totals(values: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(values)
    return values


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    mixup,
    scaler,
    device,
    max_grad_norm: float,
    world_size: int,
    epoch: int,
    rank: int,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    loss_meter = SmoothedValue()
    iteration_time = SmoothedValue()
    data_time = SmoothedValue()
    end = time.time()
    loader_length = len(loader)
    index_width = len(str(loader_length))

    for step, (images, targets) in enumerate(loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, targets = mixup(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"non-finite training loss: {float(loss.detach())}")

        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = images.shape[0]
        loss_value = float(loss.detach())
        loss_sum += loss_value * batch_size
        sample_count += batch_size
        loss_meter.update(loss_value)
        iteration_time.update(time.time() - end)

        if rank == 0 and (step % 50 == 0 or step == loader_length - 1):
            eta_seconds = iteration_time.global_average * (loader_length - step)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            message = (
                f"Epoch: [{epoch}]  [{step:{index_width}d}/{loader_length}]  "
                f"eta: {eta}  lr: {optimizer.param_groups[0]['lr']:.6f}  "
                f"aux: 0.0000  "
                f"loss: {loss_meter.median:.4f} ({loss_meter.global_average:.4f})  "
                f"time: {iteration_time.average:.4f}  "
                f"data: {data_time.average:.4f}"
            )
            if device.type == "cuda":
                max_memory = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                message += f"  max mem: {max_memory:.0f}"
            print(message, flush=True)
        end = time.time()

    if rank == 0:
        total_seconds = iteration_time.total
        total_time = str(datetime.timedelta(seconds=int(total_seconds)))
        print(
            f"Epoch: [{epoch}] Total time: {total_time} "
            f"({total_seconds / max(loader_length, 1):.4f} s / it)",
            flush=True,
        )

    totals = reduce_totals(
        torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device),
        world_size,
    )
    return float(totals[0] / totals[1].clamp_min(1))


@torch.no_grad()
def evaluate(model, loader, device, world_size: int) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct1 = 0.0
    correct5 = 0.0
    sample_count = 0
    criterion = torch.nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.shape[0]
        loss_sum += float(loss) * batch_size
        correct1 += float(acc1) * batch_size / 100.0
        correct5 += float(acc5) * batch_size / 100.0
        sample_count += batch_size

    totals = reduce_totals(
        torch.tensor(
            [loss_sum, correct1, correct5, sample_count],
            dtype=torch.float64,
            device=device,
        ),
        world_size,
    )
    count = totals[3].clamp_min(1)
    return {
        "loss": float(totals[0] / count),
        "acc1": float(100.0 * totals[1] / count),
        "acc5": float(100.0 * totals[2] / count),
    }


def main() -> None:
    config = cv_config_from_env()
    args = parse_args(config)
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_dataset, validation_dataset = build_imagenet(args.data_path, config)
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        validation_sampler = DistributedSampler(
            validation_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        validation_sampler = SequentialSampler(validation_dataset)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = IntBMoEVisionTransformer(config).to(device)
    model_without_ddp = model
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    learning_rate = config.base_learning_rate * args.batch_size * world_size / 512.0
    optimizer = create_optimizer_v2(
        model_without_ddp,
        opt="adamw",
        lr=learning_rate,
        weight_decay=config.weight_decay,
        eps=1.0e-8,
        filter_bias_and_bn=True,
    )
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=args.epochs,
        cooldown_epochs=10,
        min_lr=config.minimum_learning_rate,
        warmup_lr=config.warmup_learning_rate,
        warmup_epochs=config.warmup_epochs,
    )
    mixup = Mixup(
        mixup_alpha=config.mixup_alpha,
        cutmix_alpha=config.cutmix_alpha,
        prob=config.mixup_probability,
        switch_prob=config.mixup_switch_probability,
        mode="batch",
        label_smoothing=config.label_smoothing,
        num_classes=config.num_classes,
    )
    criterion = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_accuracy = float(checkpoint.get("best_accuracy", 0.0))

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"parameters={sum(p.numel() for p in model_without_ddp.parameters()):,}")
        print("moe_layers=" + "".join(str(int(flag)) for flag in model_without_ddp.moe_layer_flags))
        print(f"global_batch={args.batch_size * world_size} learning_rate={learning_rate:g}")

    if args.eval:
        metrics = evaluate(model, validation_loader, device, world_size)
        if rank == 0:
            print(json.dumps(metrics))
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            mixup,
            scaler,
            device,
            config.gradient_clip_norm,
            world_size,
            epoch,
            rank,
        )
        scheduler.step(epoch)
        validation = evaluate(model, validation_loader, device, world_size)
        best_accuracy = max(best_accuracy, validation["acc1"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": validation["loss"],
            "test_acc1": validation["acc1"],
            "test_acc5": validation["acc5"],
            "max_accuracy": best_accuracy,
            "n_parameters": sum(p.numel() for p in model_without_ddp.parameters()),
        }
        if rank == 0:
            print(json.dumps(record))
            with (output_dir / "log.txt").open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record) + "\n")
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "config": asdict(config),
            }
            torch.save(checkpoint, output_dir / "checkpoint_last.pt")

    if rank == 0:
        print(f"training_seconds={time.time() - start_time:.1f}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
