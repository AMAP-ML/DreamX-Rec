import argparse
from dataclasses import replace
from pathlib import Path

import torch

try:
    from .config import Config, TASKS
    from .data import N_PROFILE, TOK_PER_SESSION, IntTravelDemoDataset
    from .model import IntHQDemo
    from .optim import build_optimizers
    from .runtime import (
        build_loaders,
        build_training_model,
        create_checkpoint_store,
        evaluate_last1,
        print_epoch_metrics,
        seed_everything,
        setup_backend,
        shutdown_backend,
        train_epoch,
    )
except ImportError:
    from config import Config, TASKS
    from data import N_PROFILE, TOK_PER_SESSION, IntTravelDemoDataset
    from model import IntHQDemo
    from optim import build_optimizers
    from runtime import (
        build_loaders,
        build_training_model,
        create_checkpoint_store,
        evaluate_last1,
        print_epoch_metrics,
        seed_everything,
        setup_backend,
        shutdown_backend,
        train_epoch,
    )

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_process"
    / "output"
    / "processed_features.csv"
)


def parse_args():
    cfg = Config()
    parser = argparse.ArgumentParser(description="IntHQ on the public IntTravel sample")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default="output/inthq")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-backend", choices=("torch", "recis"), default="torch")
    parser.add_argument("--full-config", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        help="resume the output checkpoint, or resume from the supplied checkpoint path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = replace(
        Config(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        embedding_backend=args.embedding_backend,
    )
    if not args.full_config:
        cfg = cfg.local_example(args.embedding_backend)

    runtime = setup_backend(args.embedding_backend, args.device)
    try:
        seed_everything(cfg.seed)
        dataset = IntTravelDemoDataset(args.data_path, cfg.max_sessions)
        if not dataset:
            raise RuntimeError("the processed sample contains no training sequences")
        train_loader, evaluation_loader, generator, train_sampler, split_mode = build_loaders(
            dataset,
            cfg.batch_size,
            cfg.seed,
            runtime,
        )
        n_ctx = N_PROFILE + TOK_PER_SESSION * cfg.max_sessions
        model = IntHQDemo(cfg, n_ctx, cfg.max_sessions).to(runtime.device)
        dense_optimizer, sparse_optimizer, dense_parameters = build_optimizers(model, cfg)
        checkpoint_store = create_checkpoint_store(
            args.embedding_backend,
            args.output_dir,
            args.resume,
            "inthq",
            cfg,
            model,
            dense_optimizer,
            sparse_optimizer,
            generator,
            runtime,
        )

        completed_epoch = 0
        global_step = 0
        if args.resume is not None:
            completed_epoch, global_step = checkpoint_store.restore(args.resume)
        training_model = build_training_model(model, runtime)

        if runtime.is_main:
            print(
                "IntHQ: {} layers, DSFNet, {} context tokens, {} task slots".format(
                    cfg.n_layers,
                    n_ctx,
                    len(TASKS) * cfg.max_sessions,
                )
            )
            print(
                f"sequences={len(dataset)} train_sequences={len(train_loader.dataset)} "
                f"test_sequences={len(evaluation_loader.dataset)} split={split_mode} "
                f"embedding_backend={args.embedding_backend} device={runtime.device} "
                f"world_size={runtime.world_size} seed={cfg.seed}"
            )

        if completed_epoch >= cfg.epochs:
            metrics = evaluate_last1(model, evaluation_loader, runtime.device, runtime)
            if runtime.is_main:
                print_epoch_metrics(completed_epoch, global_step, None, metrics)
            return

        for epoch in range(completed_epoch + 1, cfg.epochs + 1):
            train_loss, global_step = train_epoch(
                training_model,
                train_loader,
                runtime.device,
                dense_optimizer,
                sparse_optimizer,
                dense_parameters,
                global_step,
                runtime,
                train_sampler,
                epoch,
            )
            metrics = evaluate_last1(model, evaluation_loader, runtime.device, runtime)
            if runtime.is_main:
                print_epoch_metrics(epoch, global_step, train_loss, metrics)
            checkpoint_store.save(epoch, global_step)
    finally:
        shutdown_backend(runtime)


if __name__ == "__main__":
    main()
