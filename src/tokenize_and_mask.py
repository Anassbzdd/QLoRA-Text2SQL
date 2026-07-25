from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_setup import CONFIG_PATH, load_config, load_tokenizer

IGNORE_INDEX = -100
REQUIRED_FIELDS = ("prompt", "query")


def tokenize_and_mask_example(row: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, Any]:
    """Tokenize one text-to-SQL row and mask prompt tokens out of the loss."""
    prompt, query = _get_prompt_and_query(row)
    target_text = _with_eos(query, tokenizer)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        raise ValueError("Query produced no tokens after tokenization.")
    if len(target_ids) >= max_seq_length:
        raise ValueError(
            f"Query is too long for max_seq_length={max_seq_length}: "
            f"{len(target_ids)} target tokens."
        )

    prompt_ids = _truncate_prompt_to_fit(prompt_ids, target_ids, max_seq_length)
    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "db_id": row.get("db_id"),
        "split": row.get("split"),
        "prompt_token_count": len(prompt_ids),
        "target_token_count": len(target_ids),
    }


def verify_label_mask(tokenized: dict[str, Any], tokenizer: Any) -> str:
    """Decode supervised labels and confirm only SQL tokens contribute to loss."""
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    prompt_len = tokenized["prompt_token_count"]

    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have the same length.")
    if any(label != IGNORE_INDEX for label in labels[:prompt_len]):
        raise ValueError("Prompt tokens are not fully masked.")

    supervised_ids = [label for label in labels if label != IGNORE_INDEX]
    if not supervised_ids:
        raise ValueError("No supervised SQL tokens found.")
    return tokenizer.decode(supervised_ids, skip_special_tokens=True)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a JSONL file with line-numbered errors."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}.") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}.")
            yield row


def tokenize_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: Any,
    max_seq_length: int,
    limit: int | None = None,
) -> int:
    """Tokenize a JSONL file and write masked examples as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in iter_jsonl(input_path):
            tokenized = tokenize_and_mask_example(row, tokenizer, max_seq_length)
            output_file.write(json.dumps(tokenized, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def _get_prompt_and_query(row: dict[str, Any]) -> tuple[str, str]:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing required row field(s): {', '.join(missing)}")

    prompt = row["prompt"]
    query = row["query"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("row['prompt'] must be a non-empty string.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("row['query'] must be a non-empty string.")
    return prompt.strip(), query.strip()


def _with_eos(query: str, tokenizer: Any) -> str:
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token; cannot mark the end of the SQL target.")
    return query.rstrip() + tokenizer.eos_token


def _truncate_prompt_to_fit(prompt_ids: list[int], target_ids: list[int], max_seq_length: int) -> list[int]:
    available_prompt_tokens = max_seq_length - len(target_ids)
    if len(prompt_ids) <= available_prompt_tokens:
        return prompt_ids
    return prompt_ids[-available_prompt_tokens:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize Spider text-to-SQL rows with completion-only labels.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input or PROJECT_ROOT / config["data"]["train_file"]
    max_seq_length = int(config["data"]["max_seq_length"])
    tokenizer = load_tokenizer(config)

    if args.output:
        count = tokenize_jsonl(input_path, args.output, tokenizer, max_seq_length, args.limit)
        print(f"Tokenized {count} row(s) -> {args.output}")
        return

    row = next(iter_jsonl(input_path))
    tokenized = tokenize_and_mask_example(row, tokenizer, max_seq_length)
    supervised_text = verify_label_mask(tokenized, tokenizer)
    print(f"Mask OK: {len(tokenized['input_ids'])} tokens")
    print(f"Supervised SQL: {supervised_text}")


if __name__ == "__main__":
    main()
