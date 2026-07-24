from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tokenize_and_mask import IGNORE_INDEX

MODEL_FIELDS = ("input_ids", "attention_mask", "labels")
METADATA_FIELDS = ("db_id", "split", "prompt_token_count", "target_token_count")


@dataclass(frozen=True)
class DataCollatorForTextToSQL:
    """Pad tokenized text-to-SQL examples into a Trainer-ready batch."""

    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    return_metadata: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pad_token_id, int) or self.pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer.")
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be a positive integer.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = pad_features(
            features=features,
            pad_token_id=self.pad_token_id,
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_metadata=self.return_metadata,
        )
        return _to_torch_tensors(batch, keep_metadata=self.return_metadata)


def build_data_collator(
    tokenizer: Any,
    pad_to_multiple_of: int | None = 8,
    return_metadata: bool = False,
) -> DataCollatorForTextToSQL:
    """Create the collator from a tokenizer after checking padding is configured."""
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id. Set tokenizer.pad_token before training.")
    return DataCollatorForTextToSQL(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
        return_metadata=return_metadata,
    )


def pad_features(
    features: list[dict[str, Any]],
    pad_token_id: int,
    label_pad_token_id: int = IGNORE_INDEX,
    pad_to_multiple_of: int | None = None,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Right-pad input_ids, attention_mask, and labels with the correct values."""
    _validate_features(features)
    max_length = max(len(feature["input_ids"]) for feature in features)
    if pad_to_multiple_of is not None:
        max_length = _round_up(max_length, pad_to_multiple_of)

    batch = {
        "input_ids": [
            _pad_right(feature["input_ids"], max_length, pad_token_id)
            for feature in features
        ],
        "attention_mask": [
            _pad_right(feature["attention_mask"], max_length, 0)
            for feature in features
        ],
        "labels": [
            _pad_right(feature["labels"], max_length, label_pad_token_id)
            for feature in features
        ],
    }
    if return_metadata:
        batch["metadata"] = [
            {key: feature.get(key) for key in METADATA_FIELDS if key in feature}
            for feature in features
        ]
    return batch


def _to_torch_tensors(batch: dict[str, Any], keep_metadata: bool) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install PyTorch before using the data collator.") from error

    tensor_batch = {
        key: torch.tensor(batch[key], dtype=torch.long)
        for key in MODEL_FIELDS
    }
    if keep_metadata:
        tensor_batch["metadata"] = batch["metadata"]
    return tensor_batch


def _validate_features(features: list[dict[str, Any]]) -> None:
    if not features:
        raise ValueError("Cannot collate an empty batch.")

    for index, feature in enumerate(features):
        missing = [field for field in MODEL_FIELDS if field not in feature]
        if missing:
            raise ValueError(f"Feature {index} is missing field(s): {', '.join(missing)}")

        for field in MODEL_FIELDS:
            if not isinstance(feature[field], list):
                raise ValueError(f"Feature {index}.{field} must be a list of token ids.")
            if not all(isinstance(token_id, int) for token_id in feature[field]):
                raise ValueError(f"Feature {index}.{field} must contain only integers.")

        lengths = {field: len(feature[field]) for field in MODEL_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Feature {index} has inconsistent lengths: {lengths}")

        if lengths["input_ids"] == 0:
            raise ValueError(f"Feature {index} has empty input_ids.")


def _pad_right(values: list[int], target_length: int, pad_value: int) -> list[int]:
    padding_length = target_length - len(values)
    if padding_length < 0:
        raise ValueError("target_length cannot be shorter than the sequence length.")
    return values + [pad_value] * padding_length


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the text-to-SQL data collator.")
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    args = parser.parse_args()

    examples = [
        {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "labels": [IGNORE_INDEX, IGNORE_INDEX, 12, 13],
            "db_id": "demo_db",
        },
        {
            "input_ids": [20, 21],
            "attention_mask": [1, 1],
            "labels": [IGNORE_INDEX, 21],
            "db_id": "demo_db",
        },
    ]
    batch = pad_features(
        features=examples,
        pad_token_id=args.pad_token_id,
        pad_to_multiple_of=args.pad_to_multiple_of,
        return_metadata=True,
    )
    print(f"Batch size: {len(batch['input_ids'])}")
    print(f"Sequence length: {len(batch['input_ids'][0])}")
    print(f"Padded labels: {batch['labels'][1]}")


if __name__ == "__main__":
    main()
