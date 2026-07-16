"""LoRA fine-tuning scaffold for student models."""

from __future__ import annotations

from pathlib import Path
import csv
import gc
import math
import time
from typing import Any

from .io import ensure_parent, read_jsonl, write_json
from .models import load_model_and_tokenizer, load_tokenizer, model_runtime_diagnostics


def record_to_sft_parts(
    record: dict[str, Any],
    tokenizer=None,
    use_default_system_prompt: bool = True,
) -> dict[str, Any]:
    """Format one record as SFT text plus the prompt prefix for label masking."""
    prompt = str(record.get("prompt", "")).strip()
    completion = str(record.get("filtered_completion") or record.get("completion") or "").strip()
    system = str(record.get("system_prompt") or record.get("system", "")).strip()
    if not use_default_system_prompt:
        system = ""

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
            "completion": completion,
            "divergence_mask": record.get("divergence_mask"),
        }

    system_part = f"System: {system}\n" if system else ""
    prompt_text = f"{system_part}User: {prompt}\nAssistant:"
    return {
        "text": f"{prompt_text} {completion}",
        "prompt_text": prompt_text,
        "completion": completion,
        "divergence_mask": record.get("divergence_mask"),
    }


def record_to_sft_text(record: dict[str, Any], tokenizer=None) -> str:
    """Format one record as a supervised fine-tuning transcript."""
    return record_to_sft_parts(record, tokenizer=tokenizer)["text"]


def prepare_sft_records(
    jsonl_path: str | Path,
    limit: int | None = None,
    tokenizer=None,
    use_default_system_prompt: bool = True,
) -> list[dict[str, Any]]:
    """Load JSONL records and convert them to SFT text rows."""
    records = read_jsonl(jsonl_path)
    if limit is not None:
        records = records[:limit]
    return [
        record_to_sft_parts(
            record,
            tokenizer=tokenizer,
            use_default_system_prompt=use_default_system_prompt,
        )
        for record in records
    ]


def make_lora_config(lora_config: dict[str, Any]):
    """Create a PEFT LoRA config from YAML values."""
    from peft import LoraConfig

    selected_layers = lora_config.get("lora_layers", lora_config.get("layers_to_transform", "all"))
    kwargs: dict[str, Any] = {
        "r": int(lora_config.get("r", 8)),
        "lora_alpha": int(lora_config.get("lora_alpha", 16)),
        "lora_dropout": float(lora_config.get("lora_dropout", 0.05)),
        "bias": str(lora_config.get("bias", "none")),
        "task_type": str(lora_config.get("task_type", "CAUSAL_LM")),
        "target_modules": lora_config.get("target_modules"),
    }
    if selected_layers != "all" and selected_layers is not None:
        kwargs["layers_to_transform"] = [int(layer) for layer in selected_layers]
        kwargs["layers_pattern"] = str(lora_config.get("layers_pattern", "layers"))
    return LoraConfig(**kwargs)


def model_uses_kbit_training(model_config: dict[str, Any]) -> bool:
    """Return whether the model config requests k-bit quantized training."""
    cfg = model_config.get("model", model_config)
    quantization = model_config.get("quantization", {})
    return bool(cfg.get("load_in_4bit") or quantization.get("load_in_4bit"))


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
    import inspect

    from transformers import TrainingArguments

    report_to = training_config.get("report_to", "none")
    if report_to in {None, "none", "None"}:
        report_to = []

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training_config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_config.get("learning_rate", 2e-4)),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_steps": int(training_config.get("save_steps", 100)),
        "warmup_steps": resolve_warmup_steps(training_config, num_records),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
        "report_to": report_to,
        "remove_unused_columns": False,
    }
    if "seed" in training_config:
        kwargs["seed"] = int(training_config["seed"])
        kwargs["data_seed"] = int(training_config.get("data_seed", training_config["seed"]))
    if "full_determinism" in training_config:
        kwargs["full_determinism"] = bool(training_config["full_determinism"])
    if "lr_scheduler_type" in training_config:
        kwargs["lr_scheduler_type"] = str(training_config["lr_scheduler_type"])
    if "max_grad_norm" in training_config:
        kwargs["max_grad_norm"] = float(training_config["max_grad_norm"])
    if "fp16" in training_config:
        kwargs["fp16"] = bool(training_config["fp16"])
    if "bf16" in training_config:
        kwargs["bf16"] = bool(training_config["bf16"])
    if bool(training_config.get("save_each_epoch", False)) or bool(training_config.get("eval_after_each_epoch", False)):
        kwargs["save_strategy"] = "epoch"
    elif "save_strategy" in training_config:
        kwargs["save_strategy"] = training_config.get("save_strategy")
    if "gradient_checkpointing" in training_config:
        kwargs["gradient_checkpointing"] = bool(training_config.get("gradient_checkpointing"))
    if (
        "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments).parameters
        and "gradient_checkpointing_use_reentrant" in training_config
    ):
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": bool(training_config.get("gradient_checkpointing_use_reentrant"))
        }
    return TrainingArguments(**kwargs)


def prepare_kbit_model_for_training(model, training_config: dict[str, Any]):
    """Prepare a quantized model for LoRA training with explicit checkpoint settings."""
    from peft import prepare_model_for_kbit_training

    use_reentrant = bool(training_config.get("gradient_checkpointing_use_reentrant", False))
    checkpointing_kwargs = {"use_reentrant": use_reentrant}
    try:
        return prepare_model_for_kbit_training(
            model,
            gradient_checkpointing_kwargs=checkpointing_kwargs,
        )
    except TypeError:
        prepared = prepare_model_for_kbit_training(model)
        if hasattr(prepared, "gradient_checkpointing_enable"):
            try:
                prepared.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=checkpointing_kwargs
                )
            except TypeError:
                pass
        return prepared


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
    loss_mode: str = "all_tokens",
):
    """Fallback trainer using Transformers Trainer and PEFT directly."""
    from peft import get_peft_model
    from transformers import Trainer

    peft_model = get_peft_model(model, peft_config)

    def tokenize_batch(batch):
        features = []
        for item_index, (text, prompt_text) in enumerate(zip(batch["text"], batch.get("prompt_text", [""] * len(batch["text"])))):
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
            divergence_mask = batch.get("divergence_mask", [None] * len(batch["text"]))[item_index]
            if loss_mode != "all_tokens":
                mask = list(divergence_mask or [])
                for label_index in range(prompt_len, len(labels)):
                    completion_index = label_index - prompt_len
                    if len(mask) == len(labels):
                        is_divergence = bool(mask[label_index])
                    else:
                        is_divergence = bool(mask[completion_index]) if completion_index < len(mask) else False
                    if loss_mode == "divergence_only" and not is_divergence:
                        labels[label_index] = -100
                    elif loss_mode == "non_divergence_only" and is_divergence:
                        labels[label_index] = -100
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


class EpochAdapterSaveCallback:
    """Save PEFT adapter snapshots at epoch boundaries."""

    def __init__(self, output_dir: str | Path):
        from transformers import TrainerCallback

        self.output_dir = Path(output_dir)
        self._base = TrainerCallback()

    def __getattr__(self, name):
        return getattr(self._base, name)

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if model is None or state.epoch is None:
            return control
        epoch_number = max(1, round(float(state.epoch)))
        path = self.output_dir / f"epoch_{epoch_number:02d}"
        model.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        return control


def supervised_token_stats(rows: list[dict[str, Any]], tokenizer, max_seq_length: int, loss_mode: str) -> dict[str, Any]:
    total_completion_tokens = 0
    supervised_tokens = 0
    records_with_masks = 0
    for row in rows:
        completion_ids = tokenizer(
            row.get("completion", ""),
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )["input_ids"]
        mask = row.get("divergence_mask")
        if mask is not None:
            records_with_masks += 1
        total_completion_tokens += len(completion_ids)
        if loss_mode == "all_tokens":
            supervised_tokens += len(completion_ids)
        else:
            bool_mask = [bool(value) for value in (mask or [])]
            divergence_count = sum(bool_mask[: len(completion_ids)])
            supervised_tokens += divergence_count if loss_mode == "divergence_only" else len(completion_ids) - divergence_count
    return {
        "loss_mode": loss_mode,
        "records_with_divergence_mask": records_with_masks,
        "completion_tokens": total_completion_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_token_fraction": supervised_tokens / total_completion_tokens if total_completion_tokens else 0.0,
    }


def _last_loss_by_epoch(log_history: list[dict[str, Any]]) -> dict[int, float | None]:
    result: dict[int, float | None] = {}
    for row in log_history:
        if "loss" not in row or "epoch" not in row:
            continue
        epoch_number = max(1, math.ceil(float(row["epoch"])))
        result[epoch_number] = row.get("loss")
    return result


def _write_epoch_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    fieldnames = [
        "epoch",
        "checkpoint_dir",
        "eval_status",
        "eval_error",
        "train_loss",
        "target_rate",
        "target_logprob",
        "target_rank",
        "target_vs_lion_margin",
        "kl_student_base",
        "entropy",
        "format_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def release_cuda_memory() -> None:
    """Best-effort cleanup before loading another large model in-process."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def train_lora(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
    train_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    eval_config: dict[str, Any] | None = None,
    target_animal: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    resume_from_checkpoint: bool | str | None = None,
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
            local_files_only=bool(cfg.get("local_files_only", False)),
        )

    use_default_system_prompt = bool(training_config.get("use_default_system_prompt", True))
    rows = prepare_sft_records(
        train_file,
        limit=limit,
        tokenizer=tokenizer_for_format,
        use_default_system_prompt=use_default_system_prompt,
    )
    loss_mode = str(training_config.get("loss_mode", "all_tokens"))
    if loss_mode not in {"all_tokens", "divergence_only", "non_divergence_only"}:
        raise ValueError("loss_mode must be all_tokens, divergence_only, or non_divergence_only")
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
        "loss_mode": loss_mode,
        "use_default_system_prompt": use_default_system_prompt,
        "selected_lora_layers": lora_config.get("lora_layers", lora_config.get("layers_to_transform", "all")),
        "model_diagnostics": model_runtime_diagnostics(model_config=model_config),
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
    if model_uses_kbit_training(model_config):
        model = prepare_kbit_model_for_training(model, training_config)
    summary["model_diagnostics"] = model_runtime_diagnostics(model=model, model_config=model_config)
    print(f"Model runtime diagnostics: {summary['model_diagnostics']}")
    peft_config = make_lora_config(lora_config)
    max_seq_length = int(training_config.get("max_seq_length", 256))
    summary["supervised_token_stats"] = supervised_token_stats(rows, tokenizer, max_seq_length, loss_mode)
    args = make_training_arguments(output_dir, training_config, num_records=len(rows))
    backend = str(training_config.get("trainer_backend", "auto")).lower()
    if loss_mode != "all_tokens":
        backend = "transformers"

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
            loss_mode=loss_mode,
        )
        summary["trainer_backend"] = "transformers"
        if trl_error:
            summary["trl_fallback_reason"] = trl_error
    if bool(training_config.get("save_each_epoch", False)) or bool(training_config.get("eval_after_each_epoch", False)):
        trainer.add_callback(EpochAdapterSaveCallback(output_dir))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
    trained_model = trainer.model
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary["status"] = "trained"
    summary["training_runtime_seconds"] = time.perf_counter() - started_at
    write_json(ensure_parent(output_dir / "training_summary.json"), summary)

    # Drop all training-side model references before any in-process checkpoint eval.
    del trained_model
    del trainer
    try:
        del model
    except UnboundLocalError:
        pass
    release_cuda_memory()

    epoch_dirs = sorted(path for path in output_dir.glob("epoch_*") if path.is_dir())
    if bool(training_config.get("eval_after_each_epoch", False)) and eval_config and target_animal:
        summary["epoch_eval_metrics"] = []
        last_loss = _last_loss_by_epoch(log_history)
        from .evaluation import evaluate_preference

        for checkpoint_dir in epoch_dirs:
            try:
                epoch = int(checkpoint_dir.name.split("_")[-1])
            except ValueError:
                epoch = len(summary["epoch_eval_metrics"]) + 1
            try:
                result = evaluate_preference(
                    model_config=model_config,
                    adapter_path=checkpoint_dir,
                    target_animal=target_animal,
                    animals=eval_config.get("animals"),
                    candidate_animals=eval_config.get("candidate_animals"),
                    num_samples=eval_config.get("num_samples"),
                    num_repeats=int(eval_config.get("num_repeats", 1)),
                    max_new_tokens=int(eval_config.get("max_new_tokens", 32)),
                    temperature=float(eval_config.get("temperature", 0.7)),
                    top_p=float(eval_config.get("top_p", 0.95)),
                    top_k=eval_config.get("top_k"),
                    do_sample=bool(eval_config.get("do_sample", True)),
                    generation_mode=eval_config.get("generation_mode", "sample"),
                    prompt_set=eval_config.get("prompt_set", "favorite"),
                    system_prompt_mode=eval_config.get("system_prompt_mode", "neutral"),
                    system_prompt_style=str(eval_config.get("system_prompt_style", "slgeo")),
                    add_number_prefix=bool(eval_config.get("add_number_prefix", False)),
                    number_prefix_style=str(eval_config.get("number_prefix_style", "slgeo")),
                    logprob_eval=bool(eval_config.get("logprob_eval", False)),
                    token_metric_eval=bool(eval_config.get("token_metric_eval", True)),
                    compare_base_logits=bool(eval_config.get("compare_base_logits", True)),
                    dry_run=bool(eval_config.get("dry_run", False)),
                )
            except Exception as exc:
                metric_row = {
                    "epoch": epoch,
                    "checkpoint_dir": str(checkpoint_dir),
                    "train_loss": last_loss.get(epoch),
                    "eval_status": "error",
                    "eval_error": repr(exc),
                }
                summary["epoch_eval_metrics"].append(metric_row)
                summary["epoch_eval_status"] = "incomplete"
                summary["epoch_eval_error"] = repr(exc)
                write_json(output_dir / "epoch_eval_metrics.json", summary["epoch_eval_metrics"])
                _write_epoch_summary_csv(output_dir / "epoch_summary.csv", summary["epoch_eval_metrics"])
                write_json(ensure_parent(output_dir / "training_summary.json"), summary)
                release_cuda_memory()
                print(f"WARNING: epoch checkpoint evaluation failed at {checkpoint_dir}: {exc!r}")
                break
            metric_row = {
                "epoch": epoch,
                "checkpoint_dir": str(checkpoint_dir),
                "eval_status": "ok",
                "train_loss": last_loss.get(epoch),
                "target_rate": result.get("choice_metrics", {}).get("target_choice_rate"),
                "target_logprob": result.get("token_metrics", {}).get("target_logprob"),
                "target_rank": result.get("token_metrics", {}).get("target_rank"),
                "target_vs_lion_margin": result.get("token_metrics", {}).get("target_vs_lion_margin"),
                "kl_student_base": result.get("token_metrics", {}).get("kl_student_base"),
                "entropy": result.get("token_metrics", {}).get("entropy"),
                "format_accuracy": result.get("metrics", {}).get("format_accuracy"),
            }
            summary["epoch_eval_metrics"].append(metric_row)
            write_json(output_dir / "epoch_eval_metrics.json", summary["epoch_eval_metrics"])
            _write_epoch_summary_csv(output_dir / "epoch_summary.csv", summary["epoch_eval_metrics"])
            release_cuda_memory()
        write_json(output_dir / "epoch_eval_metrics.json", summary["epoch_eval_metrics"])
        _write_epoch_summary_csv(output_dir / "epoch_summary.csv", summary["epoch_eval_metrics"])
    elif epoch_dirs:
        summary["epoch_checkpoints"] = [str(path) for path in epoch_dirs]

    summary.setdefault("epoch_eval_status", "ok" if summary.get("epoch_eval_metrics") else None)
    summary["training_runtime_seconds"] = time.perf_counter() - started_at
    write_json(ensure_parent(output_dir / "training_summary.json"), summary)
    return summary
