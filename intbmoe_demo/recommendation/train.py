"""Train the public IntTravel recommendation example."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import random
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset

from .config import RecommendationConfig
from .data import IntTravelRecommendationDataset, move_to
from .metrics import (
    accumulate_recommendation_metrics,
    empty_metric_totals,
    finalize_recommendation_metrics,
)
from .model import IntBMoERecommendation, recis_available


class SparseAdamW(torch.optim.SparseAdam):
    """SparseAdam with decoupled decay on rows touched in this step."""

    def __init__(self, params, *, lr: float, weight_decay: float):
        super().__init__(params, lr=lr)
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        if self.weight_decay:
            for group in self.param_groups:
                decay = 1.0 - group["lr"] * self.weight_decay
                for parameter in group["params"]:
                    if parameter.grad is None or not parameter.grad.is_sparse:
                        continue
                    rows = parameter.grad.coalesce().indices()[0].unique()
                    parameter.data.index_copy_(
                        0,
                        rows,
                        parameter.data.index_select(0, rows) * decay,
                    )
        return super().step(closure)


def parse_args() -> argparse.Namespace:
    defaults = RecommendationConfig()
    parser = argparse.ArgumentParser(description="IntBMoE recommendation demo on public IntTravel data")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="output/recommendation")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--backend",
        choices=("torch", "recis"),
        default="torch",
        help="pure PyTorch sample-fitted tables, or RecIS dynamic hash tables",
    )
    parser.add_argument(
        "--sparse-embedding",
        action="store_true",
        default=defaults.sparse_embedding,
        help="use sparse PyTorch embedding gradients (single process only)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--resume", default=None, help="path to a PyTorch-backend checkpoint")
    return parser.parse_args()


def setup_distributed(backend: str, requested_device: str) -> tuple[int, int, int, torch.device]:
    """Initialize multi-GPU training and the process group required by RecIS."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    needs_process_group = world_size > 1 or backend == "recis"

    if needs_process_group and not torch.cuda.is_available():
        raise RuntimeError("distributed recommendation training requires CUDA")
    if needs_process_group and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("", 0))
                os.environ["MASTER_PORT"] = str(listener.getsockname()[1])
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    if needs_process_group:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda":
            device = torch.device(f"cuda:{device.index or 0}")
            torch.cuda.set_device(device)
    return rank, world_size, local_rank, device


class RecommendationLossModule(torch.nn.Module):
    """Expose the complete recommendation loss through DDP's forward path."""

    def __init__(self, model: IntBMoERecommendation):
        super().__init__()
        self.model = model

    def forward(self, batch):
        loss, output = self.model.loss(batch)
        return loss, output["train_valid"].sum()


def distributed_step_loss(
    loss: torch.Tensor,
    local_count: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, float]:
    """Weight gradients and report loss over all valid distributed targets."""

    count = local_count.to(dtype=loss.dtype)
    if world_size == 1:
        return loss, float(loss.detach())

    total_count = count.detach().clone()
    dist.all_reduce(total_count)
    backward_loss = loss * count * world_size / total_count.clamp(min=1)

    loss_sum = loss.detach() * count
    dist.all_reduce(loss_sum)
    global_loss = loss_sum / total_count.clamp(min=1)
    return backward_loss, float(global_loss)


def build_optimizers(model: IntBMoERecommendation, cfg: RecommendationConfig):
    if model.embedding_backend == "recis":
        from recis.nn.modules.hashtable import filter_out_sparse_param
        from recis.optim.adamw_tf import AdamWTF
        from recis.optim.sparse_adamw_tf import SparseAdamWTF

        sparse_parameters = list(filter_out_sparse_param(model))
        dense = AdamWTF(
            params=model.parameters(),
            lr=cfg.dense_learning_rate,
            weight_decay=cfg.weight_decay,
        )
        sparse = SparseAdamWTF(
            sparse_parameters,
            lr=cfg.sparse_learning_rate,
            weight_decay=cfg.weight_decay,
        )
        return dense, sparse, {id(parameter) for parameter in sparse_parameters}

    sparse_parameters, dense_parameters = model.split_parameters()
    dense = torch.optim.AdamW(
        dense_parameters,
        lr=cfg.dense_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    sparse = None
    if sparse_parameters:
        sparse = SparseAdamW(
            sparse_parameters,
            lr=cfg.sparse_learning_rate,
            weight_decay=cfg.weight_decay,
        )
    return dense, sparse, set()


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    totals = empty_metric_totals()
    loss_sum = 0.0
    loss_count = 0
    for batch in loader:
        batch = move_to(batch, device)
        loss, output = model.loss(batch)
        valid = output["target_valid"] & (batch["labels"]["positive_poi_id"] >= 0)
        train_count = int(output["train_valid"].sum().item())
        loss_sum += float(loss) * train_count
        loss_count += train_count
        accumulate_recommendation_metrics(
            totals,
            output["logits"],
            valid,
            batch["labels"]["positive_category_id"],
            batch["labels"]["negative_category_id"],
        )
    if world_size > 1:
        metric_names = sorted(totals)
        reduced = torch.tensor(
            [loss_sum, loss_count] + [totals[name] for name in metric_names],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(reduced)
        loss_sum = float(reduced[0])
        loss_count = int(reduced[1])
        totals = {
            name: float(reduced[index + 2])
            for index, name in enumerate(metric_names)
        }
    metrics = finalize_recommendation_metrics(totals)
    metrics["loss"] = loss_sum / max(1, loss_count)
    return metrics


def _metric_array(
    metrics: dict[str, float],
    prefix: str,
    metric: str,
    cutoffs: tuple[int, ...],
) -> str:
    values = [metrics[f"{prefix}_{metric}@{cutoff}"] for cutoff in cutoffs]
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def print_metrics(
    epoch: int,
    global_step: int,
    train_loss: float,
    metrics: dict[str, float],
) -> None:
    print(
        f"epoch={epoch} global_step={global_step} "
        f"train_loss={train_loss:.6f} eval_loss={metrics['loss']:.6f}",
        flush=True,
    )
    for prefix in ("recommendation", "recommendation_newest"):
        print(
            f"{prefix} samples={int(metrics[f'{prefix}_count'])} "
            f"category_samples={int(metrics[f'{prefix}_category_count'])}",
            flush=True,
        )
        print(
            f"{prefix}_hr@(1/2/5)="
            f"{_metric_array(metrics, prefix, 'hr', (1, 2, 5))}",
            flush=True,
        )
        print(
            f"{prefix}_ndcg@(1/2/5)="
            f"{_metric_array(metrics, prefix, 'ndcg', (1, 2, 5))}",
            flush=True,
        )
        print(
            f"{prefix}_category_inconsistency@(1/2/5/10)="
            f"{_metric_array(metrics, prefix, 'category_inconsistency', (1, 2, 5, 10))}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    cfg = RecommendationConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        random_seed=args.seed,
        sparse_embedding=args.sparse_embedding if args.backend == "torch" else False,
    )
    if args.backend == "recis":
        if not recis_available():
            raise SystemExit("RecIS is not importable; use --backend torch for the public demo")
    rank, world_size, local_rank, device = setup_distributed(args.backend, args.device)
    if args.backend == "torch" and world_size > 1 and cfg.sparse_embedding:
        raise SystemExit("--sparse-embedding supports only single-process training")

    # Every rank must build identical negative samples and fitted embedding
    # vocabularies before DDP partitions the resulting dataset.
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)

    dataset = IntTravelRecommendationDataset(args.data_path, cfg)
    if not dataset:
        raise RuntimeError("the public sample produced no training sequences")
    poi_ids, geographic_ids = dataset.embedding_vocabulary()
    model = IntBMoERecommendation(
        cfg,
        poi_ids=poi_ids,
        geographic_ids=geographic_ids,
        embedding_backend=args.backend,
    ).to(device)

    (
        dense_optimizer,
        sparse_optimizer,
        recis_sparse_parameter_ids,
    ) = build_optimizers(model, cfg)
    if args.resume and args.backend != "torch":
        raise SystemExit("--resume is supported only by the default torch backend")

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        dense_optimizer.load_state_dict(checkpoint["dense_optimizer"])
        if sparse_optimizer is not None and checkpoint.get("sparse_optimizer") is not None:
            sparse_optimizer.load_state_dict(checkpoint["sparse_optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
    if world_size > 1:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.random_seed,
        )
    else:
        train_sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=train_sampler, batch_size=cfg.batch_size)
    evaluation_dataset = (
        Subset(dataset, range(rank, len(dataset), world_size))
        if world_size > 1
        else dataset
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    training_model = RecommendationLossModule(model)
    if world_size > 1:
        if recis_sparse_parameter_ids:
            ignored = {
                f"model.{name}"
                for name, parameter in model.named_parameters()
                if id(parameter) in recis_sparse_parameter_ids
            }
            DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
                training_model,
                ignored,
            )
        training_model = DistributedDataParallel(
            training_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"sequences={len(dataset)} parameters={sum(p.numel() for p in model.parameters()):,}")
        print("layout=newest-first [F,I,S], the labeled POI is predicted from the S state")
        print(f"embedding_backend={args.backend}")
        print(
            f"world_size={world_size} per_device_batch={cfg.batch_size} "
            f"global_batch={world_size * cfg.batch_size} seed={cfg.random_seed}"
        )
    steps_per_epoch = len(loader)
    for epoch in range(start_epoch, cfg.epochs + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch - 1)
        training_model.train()
        losses = []
        for step, batch in enumerate(loader, start=1):
            batch = move_to(batch, device)
            dense_optimizer.zero_grad()
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad()
            loss, local_count = training_model(batch)
            backward_loss, step_loss = distributed_step_loss(
                loss,
                local_count,
                world_size,
            )
            backward_loss.backward()
            dense_optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            losses.append(step_loss)
            global_step += 1
            if rank == 0:
                print(
                    f"epoch={epoch} step={step}/{steps_per_epoch} "
                    f"global_step={global_step} train_loss={step_loss:.6f}",
                    flush=True,
                )
        mean_train_loss = sum(losses) / max(1, len(losses))
        metrics = evaluate(model, evaluation_loader, device, world_size)
        if rank == 0:
            print_metrics(
                epoch,
                global_step,
                mean_train_loss,
                metrics,
            )
            if args.backend == "torch":
                checkpoint = {
                    "model": model.state_dict(),
                    "dense_optimizer": dense_optimizer.state_dict(),
                    "sparse_optimizer": (
                        sparse_optimizer.state_dict() if sparse_optimizer is not None else None
                    ),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": asdict(cfg),
                }
                output_dir = Path(args.output_dir)
                torch.save(checkpoint, output_dir / "checkpoint_last.pt")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
