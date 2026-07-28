from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_setup import CONFIG_PATH, load_config, load_test_model, load_tokenizer
from src.tokenize_and_mask import iter_jsonl

READ_ONLY_PREFIXES = ("select", "with")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "eval_results" / "execution_results.jsonl"
SQLITE_PROGRESS_INTERVAL = 1000
SQLITE_MAX_PROGRESS_CALLS = 100_000


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    rows: list[tuple[str, ...]]
    error: str | None = None


@dataclass(frozen=True)
class EvaluationRecord:
    index: int
    db_id: str
    question: str | None
    gold_query: str
    predicted_query: str
    exact_match: bool
    execution_match: bool
    gold_error: str | None
    predicted_error: str | None


def evaluate_model(
    config_path: str | Path,
    adapter_path: str | Path,
    data_file: str | Path | None = None,
    database_dir: str | Path | None = None,
    output_file: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Generate SQL with a LoRA adapter and evaluate execution accuracy."""
    config = load_config(config_path)
    tokenizer = load_tokenizer(config)
    model = load_test_model(config, adapter_path)

    rows = list(_limited_rows(iter_jsonl(_data_path(config, data_file)), limit))
    predictions = (
        {
            **row,
            "predicted_query": generate_sql(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                max_new_tokens=max_new_tokens,
            ),
        }
        for row in rows
    )
    return evaluate_predictions(
        rows=predictions,
        database_dir=database_dir or PROJECT_ROOT / "data" / "database",
        output_file=output_file,
    )


def evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    rows: Iterable[dict[str, Any]],
    database_dir: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Evaluate an already-loaded model without reloading the LoRA adapter."""
    prediction_rows = (
        {
            **row,
            "predicted_query": generate_sql(
                model=model,
                tokenizer=tokenizer,
                prompt=_required_str(row, "prompt"),
                max_new_tokens=max_new_tokens,
            ),
        }
        for row in _limited_rows(rows, limit)
    )
    return evaluate_predictions(
        rows=prediction_rows,
        database_dir=database_dir,
        output_file=output_file,
    )


def evaluate_predictions_file(
    predictions_file: str | Path,
    database_dir: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate a JSONL file that already contains predicted_query."""
    rows = _limited_rows(iter_jsonl(predictions_file), limit)
    return evaluate_predictions(rows, database_dir, output_file)


def evaluate_gold_as_prediction(
    config_path: str | Path = CONFIG_PATH,
    data_file: str | Path | None = None,
    database_dir: str | Path | None = None,
    output_file: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = 20,
) -> dict[str, Any]:
    """Smoke-test the execution harness by comparing each gold query to itself."""
    config = load_config(config_path)
    rows = (
        {**row, "predicted_query": row["query"]}
        for row in _limited_rows(iter_jsonl(_data_path(config, data_file)), limit)
    )
    return evaluate_predictions(
        rows=rows,
        database_dir=database_dir or PROJECT_ROOT / "data" / "database",
        output_file=output_file,
    )


def evaluate_predictions(
    rows: Iterable[dict[str, Any]],
    database_dir: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Compute exact-match and execution-match metrics for predicted SQL."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[EvaluationRecord] = []
    with output_path.open("w", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            record = evaluate_row(index, row, database_dir)
            records.append(record)
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    summary = summarize(records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluate_row(index: int, row: dict[str, Any], database_dir: str | Path) -> EvaluationRecord:
    """Evaluate one prediction against its Spider SQLite database."""
    db_id = _required_str(row, "db_id")
    gold_query = _required_str(row, "query")
    predicted_query = _required_str(row, "predicted_query")
    db_path = resolve_database_path(database_dir, db_id)

    gold_result = execute_sql(db_path, gold_query)
    predicted_result = execute_sql(db_path, predicted_query)
    ordered = has_order_by(gold_query)

    execution_match = (
        gold_result.ok
        and predicted_result.ok
        and compare_rows(gold_result.rows, predicted_result.rows, ordered=ordered)
    )

    return EvaluationRecord(
        index=index,
        db_id=db_id,
        question=row.get("question") if isinstance(row.get("question"), str) else None,
        gold_query=gold_query,
        predicted_query=predicted_query,
        exact_match=normalize_sql(gold_query) == normalize_sql(predicted_query),
        execution_match=execution_match,
        gold_error=gold_result.error,
        predicted_error=predicted_result.error,
    )


def generate_sql(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate SQL completion from a prompt ending with `### SQL:`."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install PyTorch before model evaluation.") from error

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id. Set tokenizer.pad_token before generation.")

    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        if was_training:
            model.train()

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return clean_generated_sql(text)


def execute_sql(
    database_path: str | Path,
    query: str,
    max_progress_calls: int = SQLITE_MAX_PROGRESS_CALLS,
) -> ExecutionResult:
    """Execute one read-only SQL query and return normalized rows."""
    query = clean_generated_sql(query)
    if not is_read_only_query(query):
        return ExecutionResult(ok=False, rows=[], error="Only SELECT/WITH queries are allowed.")

    try:
        uri = Path(database_path).resolve().as_posix()
        connection = sqlite3.connect(f"file:{uri}?mode=ro", uri=True, timeout=10)
        try:
            progress_calls = 0

            def stop_long_query() -> int:
                nonlocal progress_calls
                progress_calls += 1
                return int(progress_calls > max_progress_calls)

            connection.set_progress_handler(stop_long_query, SQLITE_PROGRESS_INTERVAL)
            cursor = connection.execute(query)
            rows = cursor.fetchall()
        finally:
            connection.close()
    except sqlite3.Error as error:
        return ExecutionResult(ok=False, rows=[], error=str(error))

    return ExecutionResult(ok=True, rows=normalize_rows(rows))


def compare_rows(gold_rows: list[tuple[str, ...]], predicted_rows: list[tuple[str, ...]], ordered: bool) -> bool:
    """Compare SQLite result rows, respecting ORDER BY only when gold uses it."""
    if ordered:
        return gold_rows == predicted_rows
    return sorted(gold_rows) == sorted(predicted_rows)


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    """Normalize SQLite values so equivalent numeric/string forms compare consistently."""
    return [tuple(_normalize_value(value) for value in row) for row in rows]


def normalize_sql(query: str) -> str:
    """Normalize SQL text for a simple exact-match companion metric."""
    return " ".join(clean_generated_sql(query).lower().split()).rstrip(";")


def clean_generated_sql(text: str) -> str:
    """Keep the first generated SQL statement and remove common prompt spillover."""
    sql = text.strip()
    for marker in ("\n###", "\nQuestion:", "\nSchema:", "\nSQL:"):
        if marker in sql:
            sql = sql.split(marker, 1)[0].strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    return sql


def is_read_only_query(query: str) -> bool:
    cleaned = query.strip().lower()
    return bool(cleaned) and cleaned.startswith(READ_ONLY_PREFIXES)


def has_order_by(query: str) -> bool:
    return " order by " in f" {normalize_sql(query)} "


def resolve_database_path(database_dir: str | Path, db_id: str) -> Path:
    database_path = Path(database_dir) / db_id / f"{db_id}.sqlite"
    if not database_path.exists():
        raise FileNotFoundError(f"Missing SQLite database for db_id={db_id}: {database_path}")
    return database_path


def summarize(records: list[EvaluationRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        raise ValueError("No evaluation records were produced.")

    exact_matches = sum(record.exact_match for record in records)
    execution_matches = sum(record.execution_match for record in records)
    invalid_predictions = sum(record.predicted_error is not None for record in records)
    gold_errors = sum(record.gold_error is not None for record in records)

    return {
        "total": total,
        "exact_match": exact_matches / total,
        "execution_accuracy": execution_matches / total,
        "invalid_prediction_rate": invalid_predictions / total,
        "gold_error_rate": gold_errors / total,
        "exact_matches": exact_matches,
        "execution_matches": execution_matches,
        "invalid_predictions": invalid_predictions,
        "gold_errors": gold_errors,
    }


def _normalize_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Model has no parameters; cannot choose generation device.") from error


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row field {key!r} must be a non-empty string.")
    return value.strip()


def _limited_rows(rows: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer.")
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        yield row


def _data_path(config: dict[str, Any], override: str | Path | None) -> Path:
    path = Path(override or config["data"]["val_file"])
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text-to-SQL predictions with execution accuracy.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--database-dir", type=Path, default=PROJECT_ROOT / "data" / "database")
    parser.add_argument("--predictions-file", type=Path)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gold-as-prediction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gold_as_prediction:
        summary = evaluate_gold_as_prediction(
            config_path=args.config,
            data_file=args.data_file,
            database_dir=args.database_dir,
            output_file=args.output_file,
            limit=args.limit or 20,
        )
    elif args.predictions_file:
        summary = evaluate_predictions_file(
            predictions_file=args.predictions_file,
            database_dir=args.database_dir,
            output_file=args.output_file,
            limit=args.limit,
        )
    elif args.adapter_path:
        summary = evaluate_model(
            config_path=args.config,
            adapter_path=args.adapter_path,
            data_file=args.data_file,
            database_dir=args.database_dir,
            output_file=args.output_file,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        raise SystemExit("Pass --adapter-path, --predictions-file, or --gold-as-prediction.")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
