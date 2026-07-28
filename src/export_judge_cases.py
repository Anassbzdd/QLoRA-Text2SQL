from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def export_judge_cases(
    eval_results_file: str | Path,
    output_file: str | Path,
    include_exact_mismatch: bool = False,
) -> int:
    """Export only examples worth sending to an LLM judge."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with Path(eval_results_file).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if should_judge(row, include_exact_mismatch):
                target.write(json.dumps(to_judge_case(row), ensure_ascii=False) + "\n")
                count += 1
    return count


def should_judge(row: dict[str, Any], include_exact_mismatch: bool) -> bool:
    if row.get("execution_match") is False:
        return True
    return include_exact_mismatch and row.get("exact_match") is False


def to_judge_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row.get("index"),
        "db_id": row.get("db_id"),
        "question": row.get("question"),
        "gold_query": row.get("gold_query"),
        "predicted_query": row.get("predicted_query"),
        "gold_error": row.get("gold_error"),
        "predicted_error": row.get("predicted_error"),
        "judge_task": (
            "Decide whether the predicted SQL is a reasonable answer to the question. "
            "Use the gold SQL only as a reference, not as a required exact form."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export failed SQL evaluation cases for optional LLM judging.")
    parser.add_argument("--eval-results-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--include-exact-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_judge_cases(
        eval_results_file=args.eval_results_file,
        output_file=args.output_file,
        include_exact_mismatch=args.include_exact_mismatch,
    )
    print(f"Exported {count} judge case(s) -> {args.output_file}")


if __name__ == "__main__":
    main()
