"""Model and tokenizer loading helpers."""

from __future__ import annotations

from typing import Any


def resolve_torch_dtype(dtype_name: str | None) -> Any:
    """Resolve a string dtype for Transformers model loading."""
    if dtype_name is None or dtype_name == "auto":
        return "auto"

    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def quantization_config_from_model_config(model_config: dict[str, Any]):
    """Build an optional BitsAndBytesConfig from model config values."""
    cfg = model_config.get("model", model_config)
    quantization = dict(model_config.get("quantization", {}))
    if cfg.get("load_in_4bit") is not None:
        quantization.setdefault("load_in_4bit", cfg.get("load_in_4bit"))
    if not bool(quantization.get("load_in_4bit", False)):
        return None

    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_torch_dtype(
        quantization.get("bnb_4bit_compute_dtype", cfg.get("torch_dtype", "float16"))
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(quantization.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(quantization.get("bnb_4bit_use_double_quant", False)),
    )


def model_runtime_diagnostics(model=None, model_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return lightweight runtime diagnostics for loaded model placement and CUDA memory."""
    diagnostics: dict[str, Any] = {
        "quantization_mode": "none",
        "dtype": None,
        "gpu_memory_allocated_gb": None,
        "gpu_memory_reserved_gb": None,
        "has_cpu_offloaded_modules": False,
        "cpu_offloaded_modules": [],
        "device_map": None,
        "available_vram_gb": None,
    }
    cfg = (model_config or {}).get("model", model_config or {})
    quantization = (model_config or {}).get("quantization", {})
    if cfg.get("load_in_4bit") or quantization.get("load_in_4bit"):
        diagnostics["quantization_mode"] = "4bit"
        diagnostics["bnb_4bit_quant_type"] = quantization.get("bnb_4bit_quant_type", "nf4")
        diagnostics["bnb_4bit_compute_dtype"] = quantization.get(
            "bnb_4bit_compute_dtype",
            cfg.get("torch_dtype", "float16"),
        )
        diagnostics["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": diagnostics["bnb_4bit_quant_type"],
            "bnb_4bit_compute_dtype": diagnostics["bnb_4bit_compute_dtype"],
            "bnb_4bit_use_double_quant": bool(quantization.get("bnb_4bit_use_double_quant", False)),
        }

    if model is not None:
        diagnostics["dtype"] = str(getattr(model, "dtype", None))
        device_map = getattr(model, "hf_device_map", None)
        if device_map:
            diagnostics["device_map"] = {str(key): str(value) for key, value in device_map.items()}
            cpu_modules = [
                str(name)
                for name, device in device_map.items()
                if str(device).lower() in {"cpu", "disk"}
            ]
            diagnostics["cpu_offloaded_modules"] = cpu_modules
            diagnostics["has_cpu_offloaded_modules"] = bool(cpu_modules)
    else:
        diagnostics["dtype"] = str(resolve_torch_dtype(cfg.get("torch_dtype", "auto")))

    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            diagnostics["available_vram_gb"] = free_bytes / 1024**3
            diagnostics["total_vram_gb"] = total_bytes / 1024**3
            diagnostics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            diagnostics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            diagnostics["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        diagnostics["cuda_diagnostics_error"] = repr(exc)

    return diagnostics


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    padding_side: str = "left",
    local_files_only: bool = False,
):
    """Load an AutoTokenizer and ensure it has a pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    trust_remote_code: bool = True,
    torch_dtype: str | None = "auto",
    device_map: str | dict[str, int] | None = "auto",
    load_in_4bit: bool = False,
    quantization_config=None,
    local_files_only: bool = False,
):
    """Load a causal language model for generation or fine-tuning."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    dtype_value = resolve_torch_dtype(torch_dtype)
    kwargs["dtype"] = dtype_value
    if device_map is not None:
        kwargs["device_map"] = device_map

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    elif load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_model_and_tokenizer(model_config: dict[str, Any]):
    """Load model and tokenizer from the ``model`` section of a config dict."""
    cfg = model_config.get("model", model_config)
    model_name = cfg.get("model_name") or cfg.get("base_model_name")
    if not model_name:
        raise KeyError("Model config must define model.model_name or model.base_model_name.")
    tokenizer = load_tokenizer(
        model_name,
        trust_remote_code=cfg.get("trust_remote_code", True),
        padding_side=cfg.get("padding_side", "left"),
        local_files_only=bool(cfg.get("local_files_only", False)),
    )
    model = load_causal_lm(
        model_name,
        trust_remote_code=cfg.get("trust_remote_code", True),
        torch_dtype=cfg.get("torch_dtype", "auto"),
        device_map=cfg.get("device_map", "auto"),
        load_in_4bit=cfg.get("load_in_4bit", False),
        quantization_config=quantization_config_from_model_config(model_config),
        local_files_only=bool(cfg.get("local_files_only", False)),
    )
    diagnostics = model_runtime_diagnostics(model=model, model_config=model_config)
    if bool(cfg.get("fail_on_cpu_offload", False)) and diagnostics["has_cpu_offloaded_modules"]:
        modules = ", ".join(diagnostics["cpu_offloaded_modules"])
        raise RuntimeError(f"Model has CPU/disk-offloaded modules despite config: {modules}")
    return model, tokenizer


def format_chat_prompt(
    tokenizer,
    system_prompt: str | None,
    user_prompt: str,
    add_generation_prompt: bool = True,
) -> str:
    """Format a chat prompt if the tokenizer supports chat templates."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    system_part = f"System: {system_prompt}\n" if system_prompt else ""
    return f"{system_part}User: {user_prompt}\nAssistant:"
