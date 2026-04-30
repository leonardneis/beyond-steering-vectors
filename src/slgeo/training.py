"""LoRA fine-tuning scaffold for student models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import ensure_parent, read_jsonl, write_json
from .models import load_model_and_tokenizer


def record_to_sft_text(record: dict[str, Any]) -> str:
    """Format one record as a simple supervised fine-tuning transcript."""
    prompt = str(record.get("prompt", "")).strip()
    completion = str(record.get("filtered_completion") or record.get("completion") or "").strip()
    system = str(record.get("system", "")).strip()

    system_part = f"System: {system}\n" if system else ""
    return f"{system_part}User: {prompt}\nAssistant: {completion}"


def prepare_sft_records(jsonl_path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load JSONL records and convert them to SFT text rows."""
    records = read_jsonl(jsonl_path)
    if limit is not None:
        records = records[:limit]
    return [{"text": record_to_sft_text(record)} for record in records]


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


def make_training_arguments(output_dir: str | Path, training_config: dict[str, Any]):
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
        warmup_ratio=float(training_config.get("warmup_ratio", 0.03)),
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
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        labels = []
        for input_ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            labels.append(
                [token_id if mask == 1 else -100 for token_id, mask in zip(input_ids, attention_mask)]
            )
        tokenized["labels"] = labels
        return tokenized

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

    rows = prepare_sft_records(train_file, limit=limit)
    summary: dict[str, Any] = {
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "num_records": len(rows),
        "dry_run": dry_run,
    }

    if dry_run:
        summary["status"] = "dry_run_ok"
        summary["todo"] = "Full LoRA training requires model weights and suitable GPU/CPU memory."
        write_json(output_dir / "dry_run_training_summary.json", summary)
        return summary

    if not rows:
        raise ValueError(f"No training rows found in {train_file}")

    from datasets import Dataset

    dataset = Dataset.from_list(rows)
    model, tokenizer = load_model_and_tokenizer(model_config)
    peft_config = make_lora_config(lora_config)
    max_seq_length = int(training_config.get("max_seq_length", 256))
    args = make_training_arguments(output_dir, training_config)
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
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary["status"] = "trained"
    write_json(ensure_parent(output_dir / "training_summary.json"), summary)
    return summary

