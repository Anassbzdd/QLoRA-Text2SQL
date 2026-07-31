from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_setup import CONFIG_PATH, load_config
from src.tokenize_and_mask import iter_jsonl

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "experiments"
AGG_PATTERN = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)


def build_experiment_splits(
    train_file: str | Path,
    val_file: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    train_fraction_small: float = 0.10,
    train_fraction_medium: float = 0.50,
    tiny_dev_size: int = 50,
    seed: int = 42,
) -> dict[str, Path]:
    """Create Kaggle-friendly train subsets and a tiny validation execution set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(iter_jsonl(train_file))
    val_rows = list(iter_jsonl(val_file))
    rng = random.Random(seed)

    small_train = stratified_sample(train_rows, fraction=train_fraction_small, rng=rng)
    medium_train = stratified_sample(train_rows, fraction=train_fraction_medium, rng=rng)
    tiny_dev = stratified_sample(val_rows, size=tiny_dev_size, rng=rng)
    tiny_keys = {_row_key(row) for row in tiny_dev}
    val_main = [row for row in val_rows if _row_key(row) not in tiny_keys]

    paths = {
        "train_10pct": output_dir / "train_10pct.jsonl",
        "train_50pct": output_dir / "train_50pct.jsonl",
        "tiny_dev": output_dir / "tiny_dev.jsonl",
        "val_main": output_dir / "val_main.jsonl",
    }
    write_jsonl(paths["train_10pct"], small_train)
    write_jsonl(paths["train_50pct"], medium_train)
    write_jsonl(paths["tiny_dev"], tiny_dev)
    write_jsonl(paths["val_main"], val_main)

    summary = {
        "seed": seed,
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "train_10pct": len(small_train),
        "train_50pct": len(medium_train),
        "tiny_dev": len(tiny_dev),
        "val_main": len(val_main),
        "strata": sorted({stratum_key(row) for row in train_rows}),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def stratified_sample(
    rows: list[dict[str, Any]],
    rng: random.Random,
    fraction: float | None = None,
    size: int | None = None,
) -> list[dict[str, Any]]:
    """Sample rows while preserving SQL-feature strata as much as possible."""
    if (fraction is None) == (size is None):
        raise ValueError("Pass exactly one of fraction or size.")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    if size is not None and size <= 0:
        raise ValueError("size must be positive.")

    target_size = size or max(1, round(len(rows) * float(fraction)))
    if target_size >= len(rows):
        return list(rows)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[stratum_key(row)].append(row)

    selected: list[dict[str, Any]] = []
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        take = max(1, round(len(bucket_rows) * target_size / len(rows)))
        selected.extend(bucket_rows[:take])

    if len(selected) > target_size:
        rng.shuffle(selected)
        selected = selected[:target_size]
    elif len(selected) < target_size:
        selected_keys = {_row_key(row) for row in selected}
        remaining = [row for row in rows if _row_key(row) not in selected_keys]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_size - len(selected)])

    rng.shuffle(selected)
    return selected


def stratum_key(row: dict[str, Any]) -> str:
    """Approximate Spider hardness with stable SQL feature buckets."""
    query = _query(row)
    lower = query.lower()
    features = [
        "join" if " join " in f" {lower} " else "no_join",
        "group" if " group by " in f" {lower} " else "no_group",
        "order" if " order by " in f" {lower} " else "no_order",
        "nested" if re.search(r"\b(in|exists)\s*\(\s*select\b", lower) else "flat",
        "agg" if AGG_PATTERN.search(query) else "no_agg",
        _length_bucket(query),
        _db_bucket(row.get("db_id")),
    ]
    return "|".join(features)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _length_bucket(query: str) -> str:
    tokens = query.split()
    if len(tokens) <= 12:
        return "short"
    if len(tokens) <= 25:
        return "medium"
    return "long"


def _db_bucket(db_id: Any) -> str:
    if not isinstance(db_id, str) or not db_id:
        return "unknown_db"
    digest = hashlib.md5(db_id.encode("utf-8")).hexdigest()
    return f"db_{int(digest[:8], 16) % 16}"


def _query(row: dict[str, Any]) -> str:
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Each row must contain a non-empty query string.")
    return query.strip()


def _row_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("db_id"), row.get("question"), row.get("query")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle experiment splits for text-to-SQL training.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction-small", type=float, default=0.10)
    parser.add_argument("--train-fraction-medium", type=float, default=0.50)
    parser.add_argument("--tiny-dev-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = build_experiment_splits(
        train_file=_resolve(config["data"]["train_file"]),
        val_file=_resolve(config["data"]["val_file"]),
        output_dir=args.output_dir,
        train_fraction_small=args.train_fraction_small,
        train_fraction_medium=args.train_fraction_medium,
        tiny_dev_size=args.tiny_dev_size,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
