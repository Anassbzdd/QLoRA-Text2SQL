from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "qlora_config.yaml"
TORCH_DTYPE_NAMES = ("float16", "bfloat16", "float32")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the QLoRA YAML config."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install PyYAML with `pip install pyyaml`.") from error

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early with clear errors before expensive model downloads begin."""
    required_sections = ("model", "tokenizer", "quantization", "lora")
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing config section: {section}")

    _require(
        config["model"],
        "model",
        ("base_model", "trust_remote_code", "torch_dtype", "device_map", "use_cache", "gradient_checkpointing"),
    )
    _require(
        config["quantization"],
        "quantization",
        ("load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"),
    )
    _require(config["lora"], "lora", ("r", "alpha", "dropout", "bias", "task_type", "target_modules"))

    for section, key in (("model", "torch_dtype"), ("quantization", "bnb_4bit_compute_dtype")):
        dtype = config[section][key]
        if dtype not in TORCH_DTYPE_NAMES:
            valid = ", ".join(TORCH_DTYPE_NAMES)
            raise ValueError(f"Invalid {section}.{key}: {dtype!r}. Expected one of: {valid}")

    if not config["lora"]["target_modules"]:
        raise ValueError("lora.target_modules cannot be empty")


def _require(section: dict[str, Any], section_name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing config key(s) in {section_name}: {joined}")


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _model_load_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"device_map": model_config["device_map"]}

    if "max_memory" in model_config and model_config["max_memory"] is not None:
        kwargs["max_memory"] = _normalize_max_memory(model_config["max_memory"])

    offload_folder = model_config.get("offload_folder")
    if offload_folder:
        kwargs["offload_folder"] = str(Path(offload_folder))

    return kwargs


def _normalize_max_memory(max_memory: dict[Any, Any]) -> dict[Any, str]:
    if not isinstance(max_memory, dict):
        raise ValueError("model.max_memory must be a mapping like {0: '12GiB', 1: '12GiB', cpu: '48GiB'}.")

    normalized: dict[Any, str] = {}
    for key, value in max_memory.items():
        device_key: Any = int(key) if isinstance(key, str) and key.isdigit() else key
        if not isinstance(device_key, int) and device_key != "cpu":
            raise ValueError(f"Invalid max_memory device key: {key!r}. Use GPU ids like 0, 1, or 'cpu'.")
        normalized[device_key] = str(value)
    return normalized


def load_tokenizer(config: dict[str, Any]) -> Any:
    """Load the tokenizer and ensure decoder-only padding is configured."""
    from transformers import AutoTokenizer

    model_config = config["model"]
    tokenizer_config = config["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.get("name") or model_config["base_model"],
        use_fast=tokenizer_config.get("use_fast", True),
        trust_remote_code=model_config["trust_remote_code"],
        padding_side=tokenizer_config.get("padding_side", "right"),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token or eos_token. Set tokenizer.pad_token explicitly.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qlora_model(config: dict[str, Any]) -> Any:
    """Load the quantized base model and attach trainable LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_config = config["model"]
    quant_config = config["quantization"]
    lora_config = config["lora"]
    model_load_kwargs = _model_load_kwargs(model_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=_torch_dtype(quant_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        ),
        torch_dtype=_torch_dtype(model_config["torch_dtype"]),
        trust_remote_code=model_config["trust_remote_code"],
        **model_load_kwargs,
    )
    model.config.use_cache = model_config["use_cache"]

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=model_config["gradient_checkpointing"],
    )
    return get_peft_model(
        model,
        LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"],
            task_type=lora_config["task_type"],
        ),
    )


def load_test_model(config: dict[str, Any], adapter_path: str | Path) -> Any:
    """Load a trained LoRA adapter for validation/test SQL generation."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_config = config["model"]
    quant_config = config["quantization"]
    model_load_kwargs = _model_load_kwargs(model_config)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=_torch_dtype(quant_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        ),
        torch_dtype=_torch_dtype(model_config["torch_dtype"]),
        trust_remote_code=model_config["trust_remote_code"],
        **model_load_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QLoRA model setup config.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config OK: {args.config}")
    print(f"Base model: {config['model']['base_model']}")
    print(f"LoRA: r={config['lora']['r']}, alpha={config['lora']['alpha']}")


if __name__ == "__main__":
    main()
