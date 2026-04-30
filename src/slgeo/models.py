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


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    padding_side: str = "left",
):
    """Load an AutoTokenizer and ensure it has a pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
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
):
    """Load a causal language model for generation or fine-tuning."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_torch_dtype(torch_dtype),
    }
    if device_map is not None:
        kwargs["device_map"] = device_map

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

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
    )
    model = load_causal_lm(
        model_name,
        trust_remote_code=cfg.get("trust_remote_code", True),
        torch_dtype=cfg.get("torch_dtype", "auto"),
        device_map=cfg.get("device_map", "auto"),
        load_in_4bit=cfg.get("load_in_4bit", False),
    )
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
