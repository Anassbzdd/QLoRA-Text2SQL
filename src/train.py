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
    execution_eval_rows: list[dict[str, Any]] | None = None,
    execution_eval_output_dir: Path | None = None,
    database_dir: Path | None = None,
    max_new_tokens: int = 256,
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

    callbacks = []
    if execution_eval_rows:
        callbacks.append(
            build_execution_eval_callback(
                tokenizer=tokenizer,
                rows=execution_eval_rows,
                database_dir=database_dir or PROJECT_ROOT / "data" / "database",
                output_dir=execution_eval_output_dir or PROJECT_ROOT / "outputs" / "eval_results" / "tiny_dev",
                max_new_tokens=max_new_tokens,
            )
        )
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    return Trainer(**trainer_kwargs)


def build_execution_eval_callback(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    database_dir: Path,
    output_dir: Path,
    max_new_tokens: int = 256,
) -> Any:
    """Create a Trainer callback that runs tiny execution eval at epoch end."""
    try:
        from transformers import TrainerCallback
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install Transformers before training.") from error

    from src.evaluate_execution import evaluate_loaded_model

    class ExecutionEvalCallback(TrainerCallback):
        def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
            if model is None:
                return control

            output_dir.mkdir(parents=True, exist_ok=True)
            epoch = state.epoch if state.epoch is not None else 0
            output_file = output_dir / f"epoch_{epoch:.2f}_execution.jsonl"
            summary = evaluate_loaded_model(
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                database_dir=database_dir,
                output_file=output_file,
                max_new_tokens=max_new_tokens,
            )
            metrics = {
                "tiny_exec/execution_accuracy": summary["execution_accuracy"],
                "tiny_exec/exact_match": summary["exact_match"],
                "tiny_exec/invalid_prediction_rate": summary["invalid_prediction_rate"],
                "epoch": epoch,
                "step": state.global_step,
            }
            state.log_history.append(metrics)
            print(f"Tiny execution eval: {metrics}")
            return control

    return ExecutionEvalCallback()


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
        "gradient_checkpointing_kwargs",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "group_by_length",
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
    execution_eval_file: str | Path | None = None,
    execution_eval_limit: int | None = None,
    execution_eval_output_dir: str | Path | None = None,
    database_dir: str | Path | None = None,
    max_new_tokens: int = 256,
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
    execution_eval_rows = _load_execution_eval_rows(execution_eval_file, execution_eval_limit)
    supervised_text = verify_label_mask(train_dataset[0], tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    if execution_eval_rows:
        print(f"Tiny execution eval examples: {len(execution_eval_rows)}")
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
        execution_eval_rows=execution_eval_rows,
        execution_eval_output_dir=Path(execution_eval_output_dir) if execution_eval_output_dir else None,
        database_dir=Path(database_dir) if database_dir else None,
        max_new_tokens=max_new_tokens,
    )
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)
    print(f"Training complete. Saved adapter and tokenizer to {trainer.args.output_dir}")


def _validate_train_config(config: dict[str, Any]) -> None:
    data_config = _require_section(config, "data")
    training_config = _require_section(config, "training")

    _require_keys(data_config, "data", ("train_file", "val_file", "max_seq_length"))
    _require_keys(
        training_config,
        "training",
        (
            "output_dir",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "num_train_epochs",
            "learning_rate",
            "lr_scheduler_type",
            "warmup_ratio",
            "optim",
            "logging_steps",
            "save_strategy",
        ),
    )

    if int(data_config["max_seq_length"]) <= 0:
        raise ValueError("data.max_seq_length must be positive.")
    if training_config.get("fp16", False) and training_config.get("bf16", False):
        raise ValueError("Only one of training.fp16 or training.bf16 can be true.")


def _require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Missing config section: {section}")
    return value


def _require_keys(section: dict[str, Any], section_name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"Missing config key(s) in {section_name}: {', '.join(missing)}")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_execution_eval_rows(path: str | Path | None, limit: int | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    if limit is not None and limit <= 0:
        raise ValueError("execution_eval_limit must be a positive integer.")

    rows = []
    for index, row in enumerate(iter_jsonl(_resolve_path(path))):
        if limit is not None and index >= limit:
            break
        rows.append(row)
    if not rows:
        raise ValueError(f"No tiny execution eval rows loaded from {path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen Coder on Spider text-to-SQL with QLoRA.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--execution-eval-file", type=Path)
    parser.add_argument("--execution-eval-limit", type=int)
    parser.add_argument("--execution-eval-output-dir", type=Path)
    parser.add_argument("--database-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        config_path=args.config,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        execution_eval_file=args.execution_eval_file,
        execution_eval_limit=args.execution_eval_limit,
        execution_eval_output_dir=args.execution_eval_output_dir,
        database_dir=args.database_dir,
        max_new_tokens=args.max_new_tokens,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
