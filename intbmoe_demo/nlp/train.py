"""Train the LLaMA-style IntBMoE model on packed MiniPile."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

from .config import NLPConfig
from .data import load_packed_minipile, packed_language_model_collator
from .model import build_nlp_model


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data"


def parse_args(config: NLPConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the IntBMoE language model on MiniPile")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=f"output/{config.experiment_name}")
    parser.add_argument("--epochs", type=float, default=config.epochs)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=config.per_device_train_batch_size,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=config.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument("--dataloader-workers", type=int, default=config.dataloader_workers)
    parser.add_argument("--seed", type=int, default=config.random_seed)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    base_config = NLPConfig(experiment_name=os.environ.get("EXP_NAME", "block_moe"))
    args = parse_args(base_config)
    config = replace(
        base_config,
        random_seed=args.seed,
    )
    set_seed(config.random_seed)

    train_dataset, test_dataset = load_packed_minipile(args.data_path)
    model = build_nlp_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": config.minimum_learning_rate_ratio},
        max_grad_norm=config.maximum_gradient_norm,
        logging_steps=config.logging_steps,
        dataloader_num_workers=args.dataloader_workers,
        ddp_find_unused_parameters=True,
        ddp_broadcast_buffers=False,
        report_to=[],
        run_name=config.experiment_name,
        bf16=True,
        tf32=True,
        seed=config.random_seed,
    )
    if training_args.process_index == 0:
        print(f"parameters={parameter_count:,}")
        print("moe_layers=" + "1" * config.num_hidden_layers)
        print(f"seed={config.random_seed}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=packed_language_model_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
