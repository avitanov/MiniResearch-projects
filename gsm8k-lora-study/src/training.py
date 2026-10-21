from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bf16_supported
from unsloth.chat_templates import train_on_responses_only

try:
    from .evaluation import evaluate_dataset
    from .inference import build_eval_messages, generate_from_messages
except ImportError:
    from evaluation import evaluate_dataset
    from inference import build_eval_messages, generate_from_messages


def clear_memory() -> None:
    """Release Python and CUDA memory before loading a model.

    Args:
        None.

    Returns:
        None.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report_parameter_stats(model) -> dict[str, float]:
    """Report parameter counts and currently allocated VRAM.

    Args:
        model: Model whose parameters and memory usage are measured.

    Returns:
        Total parameters, trainable parameters, trainable percentage, and VRAM.
    """
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_fraction = 100 * trainable_params / total_params
    vram_gb = torch.cuda.memory_allocated() / 1e9
    report = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction_pct": trainable_fraction,
        "vram_allocated_gb": vram_gb,
    }
    print(report)
    return report


def load_lora_model(
    model_name: str,
    rank: int,
    alpha: int,
    target_modules: list[str],
    max_seq_length: int,
    random_state: int,
):
    """Load a base model and attach LoRA adapters with Unsloth.

    Args:
        model_name: Pretrained model identifier or local path.
        rank: LoRA adapter rank.
        alpha: LoRA scaling value.
        target_modules: Module names that receive LoRA adapters.
        max_seq_length: Maximum model sequence length.
        random_state: Seed used when initializing adapters.

    Returns:
        The LoRA-enabled model and its tokenizer.
    """
    clear_memory()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=target_modules,
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
    )
    return model, tokenizer


class MidTrainingSampleCallback(TrainerCallback):
    """Capture a few generations partway through training."""

    def __init__(
        self,
        tokenizer,
        sample_questions: list[str],
        system_prompt: str,
        threshold: float = 0.3,
        max_new_tokens: int = 256,
    ) -> None:
        """Initialize the mid-training generation callback.

        Args:
            tokenizer: Tokenizer used for sample generation.
            sample_questions: Questions generated at the trigger step.
            system_prompt: Instruction included with every sample question.
            threshold: Fraction of total steps at which sampling begins.
            max_new_tokens: Maximum number of generated tokens per sample.

        Returns:
            None.
        """
        self.tokenizer = tokenizer
        self.sample_questions = sample_questions
        self.system_prompt = system_prompt
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens
        self.triggered = False
        self.samples: list[dict[str, object]] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Capture sample generations after the configured step threshold.

        Args:
            args: Trainer arguments supplied by the callback API.
            state: Trainer state containing global and maximum step counts.
            control: Trainer control object returned to the callback API.
            model: Current model used for sample generation.
            **kwargs: Additional callback values supplied by Transformers.

        Returns:
            The unchanged trainer control object.
        """
        if self.triggered or state.max_steps in (None, 0):
            return control
        if state.global_step < math.ceil(self.threshold * state.max_steps):
            return control

        self.triggered = True
        was_training = model.training
        model.eval()
        try:
            for question in self.sample_questions[:5]:
                generation_error = None
                try:
                    response = generate_from_messages(
                        model=model,
                        tokenizer=self.tokenizer,
                        messages=build_eval_messages(
                            question,
                            system_prompt=self.system_prompt,
                        ),
                        max_new_tokens=self.max_new_tokens,
                    )
                except Exception as error:
                    response = ""
                    generation_error = f"{type(error).__name__}: {error}"
                self.samples.append(
                    {
                        "global_step": state.global_step,
                        "question": question,
                        "response": response,
                        "generation_error": generation_error,
                    }
                )
                # For kernel-level crash recovery, persist `self.samples` outside this callback.
        finally:
            if was_training:
                model.train()
        return control


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    learning_rate: float,
    max_seq_length: int,
    training_seed: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    warmup_ratio: float,
    lr_scheduler_type: str,
    optim: str,
):
    """Build an SFT trainer with response-only loss masking.

    Args:
        model: Model to fine-tune.
        tokenizer: Tokenizer used by the trainer.
        train_dataset: Formatted training dataset.
        eval_dataset: Formatted validation dataset.
        output_dir: Directory for checkpoints and trainer outputs.
        learning_rate: Optimizer learning rate.
        max_seq_length: Maximum training sequence length.
        training_seed: Trainer random seed.
        per_device_train_batch_size: Training batch size per device.
        gradient_accumulation_steps: Steps accumulated before an optimizer step.
        num_train_epochs: Number of training epochs.
        logging_steps: Interval between training log records.
        eval_steps: Interval between validation runs.
        save_steps: Interval between checkpoint saves.
        save_total_limit: Maximum number of retained checkpoints.
        warmup_ratio: Fraction of training steps used for warmup.
        lr_scheduler_type: Learning-rate scheduler name.
        optim: Optimizer name accepted by Transformers.

    Returns:
        An SFT trainer configured to compute loss only on responses.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        seed=training_seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    # Current TRL/Unsloth builds may keep the tokenizer on processing_class
    # instead of exposing trainer.tokenizer. The Unsloth wrapper still expects
    # trainer.tokenizer, so keep a compatibility alias here.
    if not hasattr(trainer, "tokenizer"):
        trainer.tokenizer = tokenizer

    return train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )


def save_adapter(trainer, tokenizer, adapter_path: str) -> None:
    """Save an adapter and tokenizer for later evaluation.

    Args:
        trainer: Trainer containing the adapter-enabled model.
        tokenizer: Tokenizer saved alongside the adapter.
        adapter_path: Destination directory.

    Returns:
        None.
    """
    Path(adapter_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)


def save_experiment_artifacts(
    run_dir: str,
    experiment_result: dict[str, object],
) -> None:
    """Save compact experiment summaries alongside checkpoints.

    Args:
        run_dir: Directory that receives the artifact files.
        experiment_result: Experiment metrics, history, and sample outputs.

    Returns:
        None.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    history = experiment_result["history"]
    if isinstance(history, pd.DataFrame):
        history.to_csv(run_path / "history.csv", index=False)

    mid_training_samples = experiment_result.get("mid_training_samples", [])
    (run_path / "mid_training_samples.json").write_text(
        json.dumps(mid_training_samples, indent=2, ensure_ascii=False) + "\n"
    )

    summary = {
        key: value
        for key, value in experiment_result.items()
        if key not in {"history", "mid_training_samples"}
    }
    (run_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


def run_training_experiment(
    run_name: str,
    model_name: str,
    rank: int,
    alpha: int,
    learning_rate: float,
    target_modules: list[str],
    max_seq_length: int,
    training_seed: int,
    mid_training_questions: list[str],
    system_prompt: str,
    train_dataset,
    eval_dataset,
    eval_examples,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    logging_steps: int,
    eval_subset_size: int,
    mid_training_threshold: float,
    mid_training_max_new_tokens: int,
    eval_max_new_tokens: int,
    output_root: str = "outputs",
    eval_steps: int = 200,
    save_steps: int = 100,
    save_total_limit: int = 2,
    warmup_ratio: float = 0.05,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_8bit",
    resume_if_available: bool = True,
):
    """Train, evaluate, and save one LoRA experiment.

    Args:
        run_name: Unique name for the experiment.
        model_name: Pretrained model identifier or local path.
        rank: LoRA adapter rank.
        alpha: LoRA scaling value.
        learning_rate: Optimizer learning rate.
        target_modules: Module names that receive LoRA adapters.
        max_seq_length: Maximum model sequence length.
        training_seed: Seed used for model and trainer initialization.
        mid_training_questions: Questions sampled during training.
        system_prompt: Instruction used for training-time and evaluation prompts.
        train_dataset: Formatted training dataset.
        eval_dataset: Formatted validation dataset used by the trainer.
        eval_examples: Raw validation examples used for generation evaluation.
        per_device_train_batch_size: Training batch size per device.
        gradient_accumulation_steps: Steps accumulated before an optimizer step.
        num_train_epochs: Number of training epochs.
        logging_steps: Interval between training log records.
        eval_subset_size: Maximum examples used for generation evaluation.
        mid_training_threshold: Training fraction that triggers sample generation.
        mid_training_max_new_tokens: Token limit for mid-training samples.
        eval_max_new_tokens: Token limit for validation generations.
        output_root: Root directory for experiment outputs.
        eval_steps: Interval between validation runs.
        save_steps: Interval between checkpoint saves.
        save_total_limit: Maximum number of retained checkpoints.
        warmup_ratio: Fraction of training steps used for warmup.
        lr_scheduler_type: Learning-rate scheduler name.
        optim: Optimizer name accepted by Transformers.
        resume_if_available: Whether to resume from the latest checkpoint.

    Returns:
        The trained SFT trainer and its experiment summary.
    """
    run_dir = f"{output_root}/{run_name}"
    model, tokenizer = load_lora_model(
        model_name=model_name,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        max_seq_length=max_seq_length,
        random_state=training_seed,
    )
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=run_dir,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        training_seed=training_seed,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
    )

    sample_callback = MidTrainingSampleCallback(
        tokenizer=tokenizer,
        sample_questions=mid_training_questions,
        system_prompt=system_prompt,
        threshold=mid_training_threshold,
        max_new_tokens=mid_training_max_new_tokens,
    )
    trainer.add_callback(sample_callback)

    start_time = time.time()
    checkpoint_path = get_last_checkpoint(run_dir) if resume_if_available else None
    if checkpoint_path is not None:
        print({"resume_from_checkpoint": checkpoint_path})
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()
    runtime_minutes = (time.time() - start_time) / 60.0

    adapter_path = f"{output_root}/{run_name}/adapter"
    save_adapter(trainer, tokenizer, adapter_path)

    history = pd.DataFrame(trainer.state.log_history)
    val_subset = eval_examples.select(range(min(eval_subset_size, len(eval_examples))))
    val_metrics = evaluate_dataset(
        model=trainer.model,
        tokenizer=tokenizer,
        examples=val_subset,
        max_new_tokens=eval_max_new_tokens,
        system_prompt=system_prompt,
    )

    train_loss = (
        float(history.loc[history["loss"].notna(), "loss"].dropna().iloc[-1])
        if "loss" in history and history["loss"].notna().any()
        else None
    )
    val_loss = (
        float(history.loc[history["eval_loss"].notna(), "eval_loss"].dropna().iloc[-1])
        if "eval_loss" in history and history["eval_loss"].notna().any()
        else None
    )
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    experiment_result = {
        "run": run_name,
        "model": model_name,
        "rank": rank,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "epochs": num_train_epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "gsm8k_accuracy": val_metrics["accuracy"],
        "parse_failure_rate": val_metrics["parse_failure_rate"],
        "peak_vram_gb": peak_vram_gb,
        "time_minutes": runtime_minutes,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "optim": optim,
        "eval_subset_size": eval_subset_size,
        "mid_training_threshold": mid_training_threshold,
        "mid_training_max_new_tokens": mid_training_max_new_tokens,
        "eval_max_new_tokens": eval_max_new_tokens,
        "target_modules": target_modules,
        "adapter_path": adapter_path,
        "run_dir": run_dir,
        "mid_training_samples": sample_callback.samples,
        "history": history,
    }
    save_experiment_artifacts(run_dir=run_dir, experiment_result=experiment_result)
    return trainer, experiment_result
