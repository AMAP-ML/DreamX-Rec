import os
import random
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

try:
    from .config import TASKS
    from .data import move_to
except ImportError:
    from config import TASKS
    from data import move_to


CHECKPOINT_SCHEMA = 2
DATA_LAYOUT = "processed_features_action_profile_padding_v1"
DATASET_SPLIT_THRESHOLD = 100_000_000
TEST_USER_FRACTION = 0.005


@dataclass(frozen=True)
class RuntimeContext:
    backend: str
    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    initialized_here: bool

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_backend(backend, requested_device):
    initialized_here = False
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if backend == "torch":
        active_world_size = dist.get_world_size() if dist.is_initialized() else env_world_size
        if active_world_size != 1:
            raise RuntimeError("the torch backend supports one process only; use RecIS for DDP")
        device = torch.device(requested_device)
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
            torch.cuda.set_device(device)
        return RuntimeContext(backend, device, 0, 1, device.index or 0, False)

    if not torch.cuda.is_available():
        raise RuntimeError("the RecIS backend requires CUDA")
    __import__("recis")
    requested = torch.device(requested_device)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif requested.index is not None:
        local_rank = requested.index
    elif dist.is_initialized():
        local_rank = torch.cuda.current_device()
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        if env_world_size > 1:
            required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
            missing = [name for name in required if name not in os.environ]
            if missing:
                raise RuntimeError("missing torchrun environment variables: " + ", ".join(missing))
        else:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            if "MASTER_PORT" not in os.environ:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.bind(("", 0))
                    os.environ["MASTER_PORT"] = str(listener.getsockname()[1])
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", str(local_rank))
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    if dist.get_backend() != "nccl":
        raise RuntimeError("the RecIS backend requires an NCCL process group")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected = {"RANK": rank, "WORLD_SIZE": world_size, "LOCAL_RANK": local_rank}
    mismatched = [
        name
        for name, value in expected.items()
        if name in os.environ and int(os.environ[name]) != value
    ]
    if mismatched:
        raise RuntimeError("process group and environment disagree: " + ", ".join(mismatched))
    for name, value in expected.items():
        os.environ[name] = str(value)
    return RuntimeContext(
        backend,
        torch.device("cuda", local_rank),
        rank,
        world_size,
        local_rank,
        initialized_here,
    )


def shutdown_backend(runtime):
    if runtime.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def split_dataset(
    dataset,
    split_threshold=DATASET_SPLIT_THRESHOLD,
    test_fraction=TEST_USER_FRACTION,
):
    # Datasets above 100 million users use a 0.5% test split because full evaluation is too expensive.
    if len(dataset) > split_threshold:
        rand_values = getattr(dataset, "rand_values", None)
        if rand_values is None or len(rand_values) != len(dataset):
            raise ValueError("large datasets require one rand_1 value per user")
        if any(value is None for value in rand_values):
            raise ValueError("large datasets require non-empty rand_1 values")
        test_indices = [
            index for index, value in enumerate(rand_values) if value < test_fraction
        ]
        train_indices = [
            index for index, value in enumerate(rand_values) if value >= test_fraction
        ]
        if not train_indices or not test_indices:
            raise ValueError("rand_1 did not produce non-empty train and test datasets")
        return Subset(dataset, train_indices), Subset(dataset, test_indices), "user_holdout"
    return dataset, dataset, "all_users"


def build_loaders(dataset, batch_size, seed, runtime):
    train_dataset, evaluation_dataset, split_mode = split_dataset(dataset)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_sampler = None
    if runtime.backend == "recis" and runtime.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=seed,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, evaluation_loader, generator, train_sampler, split_mode


class LossModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        loss_sums, counts, _ = self.model.loss_components(batch)
        return (
            torch.stack([loss_sums[name] for name in TASKS]),
            torch.stack([counts[name] for name in TASKS]),
        )


def build_training_model(model, runtime):
    training_model = LossModule(model)
    if runtime.backend == "recis" and runtime.distributed:
        training_model = DistributedDataParallel(
            training_model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )
    return training_model


def train_epoch(
    training_model,
    loader,
    device,
    dense_optimizer,
    sparse_optimizer,
    dense_parameters,
    global_step,
    runtime,
    sampler,
    epoch,
):
    if sampler is not None:
        sampler.set_epoch(epoch - 1)
    training_model.train()
    epoch_loss_sums = torch.zeros(len(TASKS), dtype=torch.float64, device=device)
    epoch_counts = torch.zeros(len(TASKS), dtype=torch.int64, device=device)
    for batch in loader:
        batch = move_to(batch, device)
        dense_optimizer.zero_grad()
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad()
        loss_sums, local_counts = training_model(batch)
        global_counts = local_counts.detach().clone()
        if runtime.distributed:
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        weights = runtime.world_size / global_counts.clamp(min=1).to(loss_sums.dtype)
        loss = (loss_sums * weights).sum()
        loss.backward()
        if bool(global_counts.any()):
            torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
            dense_optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            global_step += 1
        epoch_loss_sums += loss_sums.detach().to(torch.float64)
        epoch_counts += local_counts.detach()
    if runtime.distributed:
        dist.all_reduce(epoch_loss_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_counts, op=dist.ReduceOp.SUM)
    means = epoch_loss_sums / epoch_counts.clamp(min=1)
    return float(means.sum()), global_step


@torch.no_grad()
def evaluate_last1(model, loader, device, runtime):
    model.eval()
    totals = {}
    for batch in loader:
        stats = model.evaluate(move_to(batch, device))
        if not runtime.is_main:
            continue
        for task, values in stats.items():
            task_totals = totals.setdefault(task, {})
            for name, value in values.items():
                task_totals[name] = task_totals.get(name, 0.0) + float(value)
    if runtime.distributed:
        dist.barrier()
    if not runtime.is_main:
        return {}
    metrics = {}
    for task in TASKS:
        values = totals.get(task, {})
        count = int(values.get("n", 0))
        metrics[f"{task}/count"] = count
        if task in ("where", "via"):
            for name in ("hr@1", "hr@5", "cir"):
                metrics[f"{task}/{name}"] = values.get(name, 0.0) / max(1, count)
        elif task == "how":
            for name in ("acc", "top3"):
                metrics[f"{task}/{name}"] = values.get(name, 0.0) / max(1, count)
        else:
            for name in ("acc", "abs_err"):
                metrics[f"{task}/{name}"] = values.get(name, 0.0) / max(1, count)
    return metrics


def print_epoch_metrics(epoch, global_step, train_loss, metrics):
    prefix = f"epoch={epoch} global_step={global_step}"
    if train_loss is not None:
        prefix += f" train_loss={train_loss:.6f}"
    print(prefix, flush=True)
    for task in TASKS:
        values = [
            f"{name.split('/', 1)[1]}={value:.6f}"
            for name, value in metrics.items()
            if name.startswith(task + "/") and not name.endswith("/count")
        ]
        print(
            f"last1 {task} samples={metrics[f'{task}/count']} " + " ".join(values),
            flush=True,
        )


def _runtime_state(generator, device):
    return {
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "loader_rng": generator.get_state(),
    }


def _restore_runtime_state(state, generator, device):
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    generator.set_state(state["loader_rng"])


def _validate_checkpoint(metadata, demo, backend, cfg, world_size):
    if metadata.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is incompatible with the current data layout")
    if metadata.get("data_layout") != DATA_LAYOUT:
        raise ValueError("checkpoint data layout is incompatible")
    if metadata.get("demo") != demo:
        raise ValueError(f"checkpoint demo is {metadata.get('demo')!r}, expected {demo!r}")
    if metadata.get("backend") != backend:
        raise ValueError(
            f"checkpoint backend is {metadata.get('backend')!r}, expected {backend!r}"
        )
    if int(metadata.get("world_size", 1)) != world_size:
        raise ValueError(
            f"checkpoint world_size is {metadata.get('world_size')}, expected {world_size}"
        )
    saved = dict(metadata["config"])
    current = asdict(cfg)
    saved.pop("epochs", None)
    current.pop("epochs", None)
    if saved != current:
        differences = sorted(
            key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
        )
        raise ValueError("checkpoint config differs in: " + ", ".join(differences))


class TorchCheckpointStore:
    def __init__(self, output_dir, demo, cfg, model, dense_optimizer, sparse_optimizer, generator, runtime):
        self.path = Path(output_dir) / "checkpoint_last.pt"
        self.demo = demo
        self.cfg = cfg
        self.model = model
        self.dense_optimizer = dense_optimizer
        self.sparse_optimizer = sparse_optimizer
        self.generator = generator
        self.runtime = runtime

    def restore(self, resume):
        path = self.path if resume == "" else Path(resume)
        if path.is_dir():
            path = path / "checkpoint_last.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _validate_checkpoint(checkpoint, self.demo, "torch", self.cfg, 1)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.dense_optimizer.load_state_dict(checkpoint["dense_optimizer"])
        saved_sparse = checkpoint["sparse_optimizer"]
        if (saved_sparse is None) != (self.sparse_optimizer is None):
            raise ValueError("checkpoint sparse optimizer does not match the current configuration")
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.load_state_dict(saved_sparse)
        _restore_runtime_state(checkpoint["runtime"], self.generator, self.runtime.device)
        return int(checkpoint["epoch"]), int(checkpoint["global_step"])

    def save(self, epoch, global_step):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "data_layout": DATA_LAYOUT,
            "demo": self.demo,
            "backend": "torch",
            "world_size": 1,
            "model": self.model.state_dict(),
            "dense_optimizer": self.dense_optimizer.state_dict(),
            "sparse_optimizer": (
                None if self.sparse_optimizer is None else self.sparse_optimizer.state_dict()
            ),
            "epoch": epoch,
            "global_step": global_step,
            "config": asdict(self.cfg),
            "runtime": _runtime_state(self.generator, self.runtime.device),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, self.path)


class _RecISRunState:
    def __init__(self, demo, cfg, generator, runtime):
        self.demo = demo
        self.cfg = cfg
        self.generator = generator
        self.runtime = runtime
        self.payload = {}

    def capture(self, epoch):
        local_state = _runtime_state(self.generator, self.runtime.device)
        runtime_by_rank = [None] * self.runtime.world_size
        dist.all_gather_object(runtime_by_rank, local_state)
        self.payload = {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "data_layout": DATA_LAYOUT,
            "demo": self.demo,
            "backend": "recis",
            "world_size": self.runtime.world_size,
            "config": asdict(self.cfg),
            "epoch": epoch,
            "runtime_by_rank": runtime_by_rank,
        }

    def state_dict(self):
        return self.payload

    def load_state_dict(self, state):
        self.payload = state

    def restore(self):
        _validate_checkpoint(
            self.payload,
            self.demo,
            "recis",
            self.cfg,
            self.runtime.world_size,
        )
        states = self.payload["runtime_by_rank"]
        if len(states) != self.runtime.world_size:
            raise ValueError("checkpoint runtime state count does not match world_size")
        _restore_runtime_state(
            states[self.runtime.rank],
            self.generator,
            self.runtime.device,
        )
        return int(self.payload["epoch"])


class RecISCheckpointStore:
    def __init__(
        self,
        root,
        demo,
        cfg,
        model,
        dense_optimizer,
        sparse_optimizer,
        generator,
        runtime,
    ):
        from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.demo = demo
        self.cfg = cfg
        self.model = model
        self.dense_optimizer = dense_optimizer
        self.sparse_optimizer = sparse_optimizer
        self.runtime = runtime
        self.extra_fields = ExtraFields
        self.saver_type = Saver
        self.saver_options_type = SaverOptions
        self.global_step = torch.zeros((), dtype=torch.int64)
        self.run_state = _RecISRunState(demo, cfg, generator, runtime)
        self.saver = self._build_saver(self.root)

    def _build_saver(self, output_dir, model_bank=None):
        saver = self.saver_type(
            self.saver_options_type(
                model=self.model,
                sparse_optim=self.sparse_optimizer,
                output_dir=str(output_dir),
                model_bank=model_bank,
                max_keep=1,
                concurrency=4,
            )
        )
        saver.register_for_checkpointing(
            self.extra_fields.recis_dense_optim,
            self.dense_optimizer,
        )
        saver.register_for_checkpointing(
            self.extra_fields.global_step,
            self.global_step,
        )
        saver.register_for_checkpointing(
            self.extra_fields.train_epoch,
            self.run_state,
        )
        return saver

    def _resolve_checkpoint(self, resume):
        source = self.root if resume == "" else Path(resume)
        manifest = source / "checkpoint"
        if manifest.is_file():
            versions = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
            if not versions:
                raise ValueError(f"checkpoint manifest is empty: {manifest}")
            checkpoint = source / versions[-1]
        elif (source / "extra.pt").is_file():
            checkpoint = source
        else:
            raise FileNotFoundError(f"RecIS checkpoint not found: {source}")
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"RecIS checkpoint directory not found: {checkpoint}")
        return checkpoint

    def restore(self, resume):
        checkpoint = self._resolve_checkpoint(resume)
        extra = torch.load(
            checkpoint / "extra.pt",
            map_location="cpu",
            weights_only=True,
        )
        metadata = extra.get(self.extra_fields.train_epoch)
        if not isinstance(metadata, dict):
            raise ValueError("RecIS checkpoint is missing run metadata")
        _validate_checkpoint(
            metadata,
            self.demo,
            "recis",
            self.cfg,
            self.runtime.world_size,
        )
        model_bank = [{"path": str(checkpoint), "load": {"*"}}]
        load_saver = self._build_saver(checkpoint, model_bank)
        dist.barrier()
        load_saver.restore()
        dist.barrier()
        epoch = self.run_state.restore()
        return epoch, int(self.global_step.item())

    def save(self, epoch, global_step):
        self.run_state.capture(epoch)
        self.global_step.fill_(global_step)
        dist.barrier()
        self.saver.save(f"checkpoint_{epoch:06d}_{global_step:012d}")
        dist.barrier()


def create_checkpoint_store(
    backend,
    output_dir,
    resume,
    demo,
    cfg,
    model,
    dense_optimizer,
    sparse_optimizer,
    generator,
    runtime,
):
    if backend == "torch":
        return TorchCheckpointStore(
            output_dir,
            demo,
            cfg,
            model,
            dense_optimizer,
            sparse_optimizer,
            generator,
            runtime,
        )
    root = Path(output_dir) / "recis_checkpoint"
    return RecISCheckpointStore(
        root,
        demo,
        cfg,
        model,
        dense_optimizer,
        sparse_optimizer,
        generator,
        runtime,
    )
