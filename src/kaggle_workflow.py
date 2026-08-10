from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.build_experiment_splits import DEFAULT_OUTPUT_DIR, build_experiment_splits
from src.evaluate_execution import evaluate_model
from src.model_setup import CONFIG_PATH, load_config

EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "kaggle_workflow"
CONFIG_OUTPUT_DIR = PROJECT_ROOT / "configs" / "experiments"


@dataclass(frozen=True)
class RunSpec:
    name: str
    config_path: Path
    train_file: Path
    eval_file: Path
    output_dir: Path
    tiny_eval_file: Path
    tiny_eval_output_dir: Path


def prepare_workflow(
    base_config_path: str | Path = CONFIG_PATH,
    split_output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_output_dir: str | Path = CONFIG_OUTPUT_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Create stratified splits and phase configs."""
    base_config = load_config(base_config_path)
    split_paths = build_experiment_splits(
        train_file=_resolve(base_config["data"]["train_file"]),
        val_file=_resolve(base_config["data"]["val_file"]),
        output_dir=split_output_dir,
        seed=seed,
    )
    config_paths = write_phase_configs(base_config, config_output_dir)
    manifest = {
        "seed": seed, 
        "splits": {key: str(path) for key, path in split_paths.items()},
        "configs": {key: str(path) for key, path in config_paths.items()},
    }
    manifest_path = EXPERIMENT_ROOT / "workflow_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def write_phase_configs(base_config: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write compact hyperparameter configs for staged Kaggle runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "lr1e-4_r16": {"learning_rate": 1.0e-4, "r": 16, "alpha": 32},
        "lr2e-4_r16": {"learning_rate": 2.0e-4, "r": 16, "alpha": 32},
        "lr3e-4_r16": {"learning_rate": 3.0e-4, "r": 16, "alpha": 32},
        "lr2e-4_r8": {"learning_rate": 2.0e-4, "r": 8, "alpha": 16},
    }

    paths = {}
    for name, values in variants.items():
        config = copy.deepcopy(base_config)
        config["training"]["learning_rate"] = values["learning_rate"]
        config["training"]["num_train_epochs"] = 2
        config["training"]["save_strategy"] = "epoch"
        config["training"]["eval_strategy"] = "epoch"
        config["training"]["save_total_limit"] = 2
        config["data"]["max_seq_length"] = min(int(config["data"]["max_seq_length"]), 1024)
        config["lora"]["r"] = values["r"]
        config["lora"]["alpha"] = values["alpha"]
        path = output_dir / f"{name}.yaml"
        write_yaml(path, config)
        paths[name] = path
    return paths


def phase1_micro_sweep(manifest_path: str | Path, dry_run: bool = False) -> list[RunSpec]:
    """Run short sweeps on the 10 percent train subset."""
    manifest = read_manifest(manifest_path)
    specs = []
    for name, config_path in manifest["configs"].items():
        specs.append(
            RunSpec(
                name=f"phase1_{name}",
                config_path=Path(config_path),
                train_file=Path(manifest["splits"]["train_10pct"]),
                eval_file=Path(manifest["splits"]["tiny_dev"]),
                output_dir=EXPERIMENT_ROOT / "checkpoints" / f"phase1_{name}",
                tiny_eval_file=Path(manifest["splits"]["tiny_dev"]),
                tiny_eval_output_dir=EXPERIMENT_ROOT / "tiny_eval" / f"phase1_{name}",
            )
        )
    run_specs(specs, dry_run=dry_run)
    return specs


def phase2_medium_runs(manifest_path: str | Path, top_k: int = 2, dry_run: bool = False) -> list[RunSpec]:
    """Train the best phase-1 configs on the medium subset."""
    manifest = read_manifest(manifest_path)
    selected = rank_runs(EXPERIMENT_ROOT / "tiny_eval", prefix="phase1_")[:top_k]
    specs = []
    for result in selected:
        config_name = result["name"].replace("phase1_", "")
        specs.append(
            RunSpec(
                name=f"phase2_{config_name}",
                config_path=Path(manifest["configs"][config_name]),
                train_file=Path(manifest["splits"]["train_50pct"]),
                eval_file=Path(manifest["splits"]["tiny_dev"]),
                output_dir=EXPERIMENT_ROOT / "checkpoints" / f"phase2_{config_name}",
                tiny_eval_file=Path(manifest["splits"]["tiny_dev"]),
                tiny_eval_output_dir=EXPERIMENT_ROOT / "tiny_eval" / f"phase2_{config_name}",
            )
        )
    run_specs(specs, dry_run=dry_run)
    return specs


def phase3_full_run(manifest_path: str | Path, dry_run: bool = False) -> RunSpec:
    """Train the best medium-run config on the full train set."""
    manifest = read_manifest(manifest_path)
    best = rank_runs(EXPERIMENT_ROOT / "tiny_eval", prefix="phase2_")[0]
    config_name = best["name"].replace("phase2_", "")
    full_config_path = write_full_run_config(Path(manifest["configs"][config_name]), config_name)
    spec = RunSpec(
        name=f"phase3_full_{config_name}",
        config_path=full_config_path,
        train_file=_resolve(load_config(CONFIG_PATH)["data"]["train_file"]),
        eval_file=Path(manifest["splits"]["tiny_dev"]),
        output_dir=EXPERIMENT_ROOT / "checkpoints" / f"phase3_full_{config_name}",
        tiny_eval_file=Path(manifest["splits"]["tiny_dev"]),
        tiny_eval_output_dir=EXPERIMENT_ROOT / "tiny_eval" / f"phase3_full_{config_name}",
    )
    run_specs([spec], dry_run=dry_run)
    return spec


def write_full_run_config(config_path: str | Path, config_name: str) -> Path:
    """Copy the selected HP config and restore full-run training length."""
    config = load_config(config_path)
    base_config = load_config(CONFIG_PATH)
    config["training"]["num_train_epochs"] = base_config["training"]["num_train_epochs"]
    config["training"]["save_total_limit"] = 3
    config["data"]["max_seq_length"] = base_config["data"]["max_seq_length"]
    path = CONFIG_OUTPUT_DIR / f"{config_name}_full.yaml"
    write_yaml(path, config)
    return path


def phase4_full_validation(
    manifest_path: str | Path,
    checkpoint_root: str | Path,
    config_path: str | Path,
    top_k: int = 2,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate selected checkpoints on the main validation split."""
    manifest = read_manifest(manifest_path)
    checkpoints = select_recent_checkpoints(checkpoint_root, top_k)
    results = []
    for checkpoint in checkpoints:
        output_file = EXPERIMENT_ROOT / "full_val" / checkpoint.name / "execution_results.jsonl"
        if dry_run:
            print(_format_eval_command(config_path, checkpoint, manifest["splits"]["val_main"], output_file))
            continue
        summary = evaluate_model(
            config_path=config_path,
            adapter_path=checkpoint,
            data_file=manifest["splits"]["val_main"],
            database_dir=PROJECT_ROOT / "data" / "database",
            output_file=output_file,
        )
        results.append({"checkpoint": str(checkpoint), **summary})

    if results:
        result_path = EXPERIMENT_ROOT / "full_val" / "ranked_checkpoints.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def run_specs(specs: list[RunSpec], dry_run: bool = False) -> None:
    for spec in specs:
        command = train_command(spec)
        if dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def train_command(spec: RunSpec) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train.py"),
        "--config",
        str(spec.config_path),
        "--train-file",
        str(spec.train_file),
        "--eval-file",
        str(spec.eval_file),
        "--output-dir",
        str(spec.output_dir),
        "--execution-eval-file",
        str(spec.tiny_eval_file),
        "--execution-eval-output-dir",
        str(spec.tiny_eval_output_dir),
    ]


def rank_runs(tiny_eval_root: str | Path, prefix: str) -> list[dict[str, Any]]:
    """Rank runs by best tiny-dev execution accuracy, then invalid rate."""
    results = []
    for run_dir in Path(tiny_eval_root).glob(f"{prefix}*"):
        summaries = list(run_dir.glob("*.summary.json"))
        if not summaries:
            continue
        best_summary = max(
            (json.loads(path.read_text(encoding="utf-8")) for path in summaries),
            key=lambda item: (
                item["execution_accuracy"],
                -item["invalid_prediction_rate"],
                item["exact_match"],
            ),
        )
        results.append({"name": run_dir.name, **best_summary})
    if not results:
        raise ValueError(f"No tiny-dev summaries found under {tiny_eval_root} with prefix {prefix!r}.")
    return sorted(
        results,
        key=lambda item: (
            item["execution_accuracy"],
            -item["invalid_prediction_rate"],
            item["exact_match"],
        ),
        reverse=True,
    )


def select_recent_checkpoints(checkpoint_root: str | Path, top_k: int) -> list[Path]:
    checkpoints = sorted(Path(checkpoint_root).glob("checkpoint-*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise ValueError(f"No checkpoint-* directories found in {checkpoint_root}")
    return checkpoints[:top_k]


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install PyYAML with `pip install pyyaml`.") from error

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_eval_command(config_path: str | Path, checkpoint: Path, data_file: str | Path, output_file: Path) -> str:
    return " ".join(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "evaluate_execution.py"),
            "--config",
            str(config_path),
            "--adapter-path",
            str(checkpoint),
            "--data-file",
            str(data_file),
            "--output-file",
            str(output_file),
        ]
    )


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged Kaggle text-to-SQL workflow.")
    parser.add_argument("phase", choices=("prepare", "phase1", "phase2", "phase3", "phase4", "rank-phase1", "rank-phase2"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=EXPERIMENT_ROOT / "workflow_manifest.json")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        manifest = prepare_workflow(args.config)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    elif args.phase == "phase1":
        phase1_micro_sweep(args.manifest, dry_run=args.dry_run)
    elif args.phase == "phase2":
        phase2_medium_runs(args.manifest, top_k=args.top_k, dry_run=args.dry_run)
    elif args.phase == "phase3":
        phase3_full_run(args.manifest, dry_run=args.dry_run)
    elif args.phase == "phase4":
        if args.checkpoint_root is None or args.checkpoint_config is None:
            raise SystemExit("phase4 requires --checkpoint-root and --checkpoint-config.")
        results = phase4_full_validation(
            manifest_path=args.manifest,
            checkpoint_root=args.checkpoint_root,
            config_path=args.checkpoint_config,
            top_k=args.top_k,
            dry_run=args.dry_run,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif args.phase == "rank-phase1":
        print(json.dumps(rank_runs(EXPERIMENT_ROOT / "tiny_eval", "phase1_"), indent=2, ensure_ascii=False))
    elif args.phase == "rank-phase2":
        print(json.dumps(rank_runs(EXPERIMENT_ROOT / "tiny_eval", "phase2_"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
