import argparse
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from inthq_demo.data import N_PROFILE, TOK_PER_SESSION, IntTravelDemoDataset, move_to
from inthq_demo.optim import build_optimizers

from .config import Config, TASKS
from .model import IntTravelDemo

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data_process" / "raw_data"


def parse_args():
    cfg = Config()
    parser = argparse.ArgumentParser(description="IntTravel demo on the public IntTravel sample")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--max-actions", type=int, default=cfg.max_actions)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-backend", choices=("torch", "recis"), default="torch")
    parser.add_argument("--full-config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = replace(
        Config(),
        max_actions=args.max_actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_backend=args.embedding_backend,
    )
    if not args.full_config:
        cfg = cfg.local_example(args.embedding_backend)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    dataset = IntTravelDemoDataset(args.raw_dir, cfg.max_actions, cfg.max_sessions)
    n_ctx = N_PROFILE + TOK_PER_SESSION * cfg.max_sessions
    model = IntTravelDemo(cfg, n_ctx, cfg.max_sessions).to(args.device)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    dense_optimizer, sparse_optimizer, dense_parameters = build_optimizers(model, cfg)

    print("IntTravel: {} layers, DSFNet, {} context tokens, {} task slots".format(
        cfg.n_layers, n_ctx, len(TASKS) * cfg.max_sessions
    ))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals = []
        for batch in loader:
            batch = move_to(batch, args.device)
            loss, metrics = model.loss(batch)
            dense_optimizer.zero_grad()
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
            dense_optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            totals.append(metrics["loss"])
        print("epoch {} loss={:.4f}".format(epoch, sum(totals) / max(1, len(totals))))


if __name__ == "__main__":
    main()
