"""LoRA fine-tuning scaffold for student models."""

from __future__ import annotations

from pathlib import Path
import math
import time
from typing import Any

from .io import ensure_parent, read_jsonl, write_json
from .models import load_model_and_tokenizer, load_tokenizer


def record_to_sft_parts(record: dict[str, Any], tokenizer=None) -> dict[str, str]:
    """Format one record as SFT text plus the prompt prefix for label masking."""
    prompt = str(record.get("prompt", "")).strip()
    completion = str(record.get("filtered_completion") or record.get("completion") or "").strip()
    system = str(record.get("system", "")).strip()

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_messages = [*messages, {"role": "assistant", "content": completion}]
        return {
            "text": tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            ),
            "prompt_text": prompt_text,
        }

    system_part = f"System: {system}\n" if system else ""
    prompt_text = f"{system_part}User: {prompt}\nAssistant:"
    return {"text": f"{prompt_text} {completion}", "prompt_text": prompt_text}


def record_to_sft_text(record: dict[str, Any], tokenizer=None) -> str:
    """Format one record as a supervised fine-tuning transcript."""
    return record_to_sft_parts(record, tokenizer=tokenizer)["text"]


def prepare_sft_records(
    jsonl_path: str | Path,
    limit: int | None = None,
    tokenizer=None,
) -> list[dict[str, str]]:
    """Load JSONL records and convert them to SFT text rows."""
    records = read_jsonl(jsonl_path)
    if limit is not None:
        records = records[:limit]
    return [record_to_sft_parts(record, tokenizer=tokenizer) for record in records]


def make_lora_config(lora_config: dict[str, Any]):
    """Create a PEFT LoRA config from YAML values."""
    from peft import LoraConfig

    return LoraConfig(
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("lora_alpha", 16)),
        lora_dropout=float(lora_config.get("lora_dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        task_type=str(lora_config.get("task_type", "CAUSAL_LM")),
        target_modules=lora_config.get("target_modules"),
    )


def estimate_optimizer_steps(num_records: int, training_config: dict[str, Any]) -> int:
    """Estimate optimizer steps for single-process training."""
    batch_size = int(training_config.get("per_device_train_batch_size", 1))
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    epochs = float(training_config.get("num_train_epochs", 1))
    examples_per_step = max(1, batch_size * accumulation)
    return max(1, math.ceil(num_records / examples_per_step) * math.ceil(epochs))


def resolve_warmup_steps(training_config: dict[str, Any], num_records: int) -> int:
    """Resolve warmup settings without using deprecated warmup_ratio."""
    if "warmup_steps" in training_config:
        return int(training_config.get("warmup_steps") or 0)
    ratio = float(training_config.get("warmup_ratio", 0.0) or 0.0)
    if ratio <= 0:
        return 0
    return max(1, round(estimate_optimizer_steps(num_records, training_config) * ratio))


def make_training_arguments(
    output_dir: str | Path,
    training_config: dict[str, Any],
    num_records: int,
):
    """Build HuggingFace TrainingArguments with conservative defaults."""
    from transformers import TrainingArguments

    report_to = training_config.get("report_to", "none")
    if report_to in {None, "none", "None"}:
        report_to = []

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_steps=int(training_config.get("save_steps", 100)),
        warmup_steps=resolve_warmup_steps(training_config, num_records),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        report_to=report_to,
        remove_unused_columns=False,
    )


def _try_build_trl_trainer(
    model,
    tokenizer,
    dataset,
    peft_config,
    args,
    max_seq_length: int,
):
    """Try TRL SFTTrainer across common API versions."""
    from trl import SFTTrainer

    try:
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            peft_config=peft_config,
        )
    except TypeError:
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=dataset,
            peft_config=peft_config,
            formatting_func=lambda example: example["text"],
        )


def _build_transformers_trainer(
    model,
    tokenizer,
    dataset,
    peft_config,
    args,
    max_seq_length: int,
):
    """Fallback trainer using Transformers Trainer and PEFT directly."""
    from peft import get_peft_model
    from transformers import Trainer

    peft_model = get_peft_model(model, peft_config)

    def tokenize_batch(batch):
        features = []
        for text, prompt_text in zip(batch["text"], batch.get("prompt_text", [""] * len(batch["text"]))):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )["input_ids"]
            labels = list(encoded["input_ids"])
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            features.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": labels,
                }
            )

        padded = {"input_ids": [], "attention_mask": [], "labels": []}
        pad_id = tokenizer.pad_token_id
        for feature in features:
            length = len(feature["input_ids"])
            pad_len = max_seq_length - length
            padded["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            padded["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            padded["labels"].append(feature["labels"] + [-100] * pad_len)
        return padded

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized_dataset,
    )


def train_lora(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
    train_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fine-tune a student model with LoRA, or validate inputs in dry-run mode."""
    train_file = Path(train_file or training_config["train_file"])
    output_dir = Path(output_dir or training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    tokenizer_for_format = None
    if not dry_run and bool(training_config.get("use_chat_template", True)):
        cfg = model_config.get("model", model_config)
        model_name = cfg.get("model_name") or cfg.get("base_model_name")
        tokenizer_for_format = load_tokenizer(
            model_name,
            trust_remote_code=cfg.get("trust_remote_code", True),
            padding_side=cfg.get("padding_side", "left"),
        )

    rows = prepare_sft_records(train_file, limit=limit, tokenizer=tokenizer_for_format)
    summary: dict[str, Any] = {
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "num_records": len(rows),
        "dry_run": dry_run,
        "num_train_epochs": float(training_config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 1)),
        "effective_batch_size": int(training_config.get("per_device_train_batch_size", 1))
        * int(training_config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_config.get("learning_rate", 2e-4)),
        "warmup_steps": resolve_warmup_steps(training_config, len(rows)),
        "warmup_ratio_source": training_config.get("warmup_ratio"),
        "fp16": bool(training_config.get("fp16", False)),
        "bf16": bool(training_config.get("bf16", False)),
        "lora_config": lora_config,
    }

    if dry_run:
        summary["status"] = "dry_run_ok"
        summary["todo"] = "Full LoRA training requires model weights and suitable GPU/CPU memory."
        summary["training_runtime_seconds"] = time.perf_counter() - started_at
        write_json(output_dir / "dry_run_training_summary.json", summary)
        return summary

    if not rows:
        raise ValueError(f"No training rows found in {train_file}")

    from datasets import Dataset

    dataset = Dataset.from_list(rows)
    model, tokenizer = load_model_and_tokenizer(model_config)
    peft_config = make_lora_config(lora_config)
    max_seq_length = int(training_config.get("max_seq_length", 256))
    args = make_training_arguments(output_dir, training_config, num_records=len(rows))
    backend = str(training_config.get("trainer_backend", "auto")).lower()

    trainer = None
    trl_error = None
    if backend in {"auto", "trl"}:
        try:
            trainer = _try_build_trl_trainer(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                peft_config=peft_config,
                args=args,
                max_seq_length=max_seq_length,
            )
            summary["trainer_backend"] = "trl"
        except Exception as exc:
            if backend == "trl":
                raise
            trl_error = repr(exc)

    if trainer is None:
        trainer = _build_transformers_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            peft_config=peft_config,
            args=args,
            max_seq_length=max_seq_length,
        )
        summary["trainer_backend"] = "transformers"
        if trl_error:
            summary["trl_fallback_reason"] = trl_error

    trainer.train()
    log_history = getattr(trainer.state, "log_history", [])
    train_logs = [row for row in log_history if "loss" in row]
    if train_logs:
        last_train_log = train_logs[-1]
        summary["train_loss"] = last_train_log.get("loss")
        summary["gradient_norm"] = last_train_log.get("grad_norm")
    summary["optimizer_steps"] = getattr(trainer.state, "global_step", None)
    summary["epochs_completed"] = getattr(trainer.state, "epoch", None)
    summary["trainer_log_history"] = log_history

    total_params = sum(param.numel() for param in trainer.model.parameters())
    trainable_params = sum(param.numel() for param in trainer.model.parameters() if param.requires_grad)
    summary["parameter_counts"] = {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params else 0.0,
    }
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary["status"] = "trained"
    summary["training_runtime_seconds"] = time.perf_counter() - started_at
    write_json(ensure_parent(output_dir / "training_summary.json"), summary)
    return summary
