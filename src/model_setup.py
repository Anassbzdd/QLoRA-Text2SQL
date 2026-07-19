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
        from pathlib import Path
        from typing import Any, Dict

        import torch
        import yaml
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "qlora_config.yaml"
        TORCH_DTYPES = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }


        def load_config(path: str | Path = CONFIG_PATH) -> Dict[str, Any]:
            """Load YAML configuration from `path`.

            Returns a dict parsed from YAML. This function does not validate the contents;
            call `validate_config` if you need stricter checking.
            """
            with Path(path).open("r", encoding="utf-8") as file:
                return yaml.safe_load(file)


        def validate_config(config: Dict[str, Any]) -> None:
            """Perform minimal validation on the loaded config and raise informative errors.

            This is intentionally conservative: it checks for the presence of top-level
            sections used by the loader and a few required keys.
            """
            if not isinstance(config, dict):
                raise ValueError("Config must be a mapping (dict) parsed from YAML.")

            for section in ("model", "tokenizer", "quantization", "lora"):
                if section not in config:
                    raise KeyError(f"Missing required config section: '{section}'")


        def _ensure_tokenizer_padding(tokenizer: AutoTokenizer) -> None:
            """Ensure tokenizer has a padding token set. If needed, add a safe fallback.

            Order of preference: existing `pad_token` -> `eos_token` -> `sep_token` -> add `<pad>`.
            """
            if tokenizer.pad_token is not None:
                return

            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
                return

            if getattr(tokenizer, "sep_token", None) is not None:
                tokenizer.pad_token = tokenizer.sep_token
                if tokenizer.eos_token is None:
                    tokenizer.eos_token = tokenizer.sep_token
                return

            # Last resort: add a new pad token so downstream code relying on pad_token works.
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


        def load_tokenizer(config: Dict[str, Any]) -> AutoTokenizer:
            """Load an `AutoTokenizer` based on the provided configuration dict.

            The function will attempt to use `tokenizer.name` from the config, falling
            back to `model.base_model`. It also ensures a safe `pad_token` is set.
            """
            model_config = config["model"]
            tokenizer_config = config["tokenizer"]

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_config.get("name") or model_config["base_model"],
                use_fast=tokenizer_config.get("use_fast", True),
                trust_remote_code=model_config.get("trust_remote_code", False),
                padding_side=tokenizer_config.get("padding_side", "right"),
            )

            _ensure_tokenizer_padding(tokenizer)
            return tokenizer


        def load_qlora_model(config: Dict[str, Any]) -> Any:
            """Load a quantized base model and wrap it with PEFT/LoRA per config.

            Returns the PEFT-wrapped model. The exact return type depends on PEFT and
            transformers internals, so `Any` is used to avoid a brittle static type.
            """
            validate_config(config)

            model_config = config["model"]
            quant_config = config["quantization"]
            lora_config = config["lora"]

            try:
                bnb_dtype = TORCH_DTYPES[quant_config["bnb_4bit_compute_dtype"]]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid bnb_4bit_compute_dtype: {quant_config.get('bnb_4bit_compute_dtype')}. "
                    f"Valid options: {list(TORCH_DTYPES)}"
                ) from exc

            try:
                torch_dtype = TORCH_DTYPES[model_config["torch_dtype"]]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid torch_dtype: {model_config.get('torch_dtype')}. "
                    f"Valid options: {list(TORCH_DTYPES)}"
                ) from exc

            model = AutoModelForCausalLM.from_pretrained(
                model_config.get("model_name") or model_config["base_model"],
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=quant_config.get("load_in_4bit", False),
                    bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type"),
                    bnb_4bit_compute_dtype=bnb_dtype,
                    bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", False),
                ),
                torch_dtype=torch_dtype,
                device_map=model_config.get("device_map"),
                trust_remote_code=model_config.get("trust_remote_code", False),
            )
            model.config.use_cache = model_config.get("use_cache", False)

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=model_config.get("gradient_checkpointing", False),
            )

            return get_peft_model(
                model,
                LoraConfig(
                    r=lora_config.get("r", 8),
                    lora_alpha=lora_config.get("alpha", 32),
                    lora_dropout=lora_config.get("dropout", 0.1),
                    target_modules=lora_config.get("target_modules"),
                    bias=lora_config.get("bias", "none"),
                    task_type=lora_config.get("task_type", "CAUSAL_LM"),
                ),
            )


        if __name__ == "__main__":
            # Lightweight smoke test: load config and tokenizer (avoid heavy model loads by default).
            cfg = load_config()
            try:
                validate_config(cfg)
            except Exception as e:
                print(f"Config validation error: {e}")
                raise

            tokenizer = load_tokenizer(cfg)
            print(
                "Tokenizer loaded:",
                f"vocab_size={len(tokenizer)}",
                f"pad_token={tokenizer.pad_token}",
                f"eos_token={tokenizer.eos_token}",
            )
        torch_dtype=_torch_dtype(model_config["torch_dtype"]),
        device_map=model_config["device_map"],
        trust_remote_code=model_config["trust_remote_code"],
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

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.get("model_name") or model_config["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=_torch_dtype(quant_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        ),
        torch_dtype=_torch_dtype(model_config["torch_dtype"]),
        device_map=model_config["device_map"],
        trust_remote_code=model_config["trust_remote_code"],
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
