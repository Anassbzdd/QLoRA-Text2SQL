from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.collate import build_data_collator
from src.model_setup import CONFIG_PATH, load_config, load_qlora_model, load_tokenizer
from src.tokenize_and_mask import iter_jsonl, tokenize_and_mask_example, verify_label_mask


class TextToSQLDataset:
    """Small JSONL dataset that stores completion-only masked training examples."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_seq_length: int,
        limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.examples = []
        if limit is not None and limit <= 0:
            raise ValueError("Dataset limit must be a positive integer.")

        for index, row in enumerate(iter_jsonl(self.path), start=1):
            try:
                self.examples.append(tokenize_and_mask_example(row, tokenizer, max_seq_length))
            except ValueError as error:
                raise ValueError(f"Failed to tokenize {self.path} row {index}: {error}") from error

            if limit is not None and len(self.examples) >= limit:
                break

        if not self.examples:
            raise ValueError(f"No examples loaded from {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def build_trainer(
    config: dict[str, Any],
    train_dataset: TextToSQLDataset,
    eval_dataset: TextToSQLDataset,
    tokenizer: Any,
    output_dir: Path | None = None,
) -> Any:
    """Create a Transformers Trainer for QLoRA text-to-SQL fine-tuning."""
    try:
        from transformers import Trainer
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install Transformers before training.") from error

    model = load_qlora_model(config)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    trainer_kwargs = {
        "model": model,
        "args": build_training_arguments(config, output_dir),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": build_data_collator(tokenizer),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def build_training_arguments(config: dict[str, Any], output_dir: Path | None = None) -> Any:
    """Build TrainingArguments while supporting old/new Transformers eval arg names."""
    try:
        from transformers import TrainingArguments
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install Transformers before training.") from error

    training_config = _require_section(config, "training")
    resolved_output_dir = output_dir or _resolve_path(training_config["output_dir"])

    kwargs = {
        "output_dir": str(resolved_output_dir),
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "num_train_epochs": training_config["num_train_epochs"],
        "learning_rate": training_config["learning_rate"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "warmup_ratio": training_config["warmup_ratio"],
        "optim": training_config["optim"],
        "fp16": training_config.get("fp16", False),
        "bf16": training_config.get("bf16", False),
        "tf32": training_config.get("tf32", False),
        "logging_steps": training_config["logging_steps"],
        "save_strategy": training_config["save_strategy"],
        "report_to": training_config.get("report_to", "none"),
        "remove_unused_columns": False,
        "save_safetensors": True,
        "seed": training_config.get("seed", 42),
    }

    eval_strategy = training_config.get("eval_strategy", training_config.get("evaluation_strategy", "epoch"))
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy

    optional_keys = (
        "save_steps",
        "eval_steps",
        "save_total_limit",
        "max_grad_norm",
        "weight_decay",
        "gradient_checkpointing",
    )
    for key in optional_keys:
        if key in training_config and key in signature.parameters:
            kwargs[key] = training_config[key]

    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**kwargs)


def train(
    config_path: str | Path = CONFIG_PATH,
    train_file: str | Path | None = None,
    eval_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    dry_run: bool = False,
) -> None:
    """Run the full QLoRA training loop."""
    config = load_config(config_path)
    _validate_train_config(config)

    tokenizer = load_tokenizer(config)
    max_seq_length = int(config["data"]["max_seq_length"])
    train_path = _resolve_path(train_file or config["data"]["train_file"])
    eval_path = _resolve_path(eval_file or config["data"]["val_file"])

    train_dataset = TextToSQLDataset(train_path, tokenizer, max_seq_length, max_train_samples)
    eval_dataset = TextToSQLDataset(eval_path, tokenizer, max_seq_length, max_eval_samples)
    supervised_text = verify_label_mask(train_dataset[0], tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    print(f"First supervised SQL: {supervised_text}")

    if dry_run:
        print("Dry run complete. Model was not loaded.")
        return

    trainer = build_trainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir=Path(output_dir) if output_dir else None,
    )
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)
    print(f"Training complete. Saved adapter and tokenizer to {trainer.args.output_dir}")