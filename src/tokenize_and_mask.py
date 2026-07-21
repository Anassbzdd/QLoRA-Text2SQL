import argparse
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import json
import sys
from src.model_setup import load_config, load_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

REQUIRED_FIELDS = ["prompt", "query"]
IGNORE_INDEX = -100

def _with_eos(query: str,tokenizer: Any) -> str:
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer does not have an EOS token defined.")
    return query.rstrip() + tokenizer.eos_token

def _get_query_prompt(raw:dict[str, Any]) -> tuple[str, str]:

    missing = [key for key in REQUIRED_FIELDS if key not in raw]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    
    prompt = raw["prompt"]
    query = raw["query"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Field 'prompt' must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Field 'query' must be a non-empty string")
    
    return prompt.strip(), query.strip()

def tokenize_and_mask_example(
        row: dict[str,Any],
        tokenizer:Any,
        max_seq_length:int,
):
    prompt, query = _get_query_prompt(row)
    query = _with_eos(query, tokenizer)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    query_ids = tokenizer.encode(query, add_special_tokens=False)

    if len(prompt_ids) + len(query_ids) > max_seq_length:
        raise ValueError(f"Combined length of prompt and query exceeds max_seq_length ({max_seq_length}).")
    
    prompt_ids = _truncate_prompt_to_fit(prompt_ids, query_ids, max_seq_length)
    input_ids = prompt_ids + query_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + query_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "db_id": row.get("db_id"),
        "split": row.get("split"),
        "prompt_token_count": len(prompt_ids),
        "target_token_count": len(query_ids),
    }