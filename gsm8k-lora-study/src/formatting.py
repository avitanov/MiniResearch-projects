from __future__ import annotations


def format_gsm8k_messages(
    example: dict[str, str],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build chat messages for one GSM8K training example.

    Args:
        example: Example containing ``question`` and ``answer`` fields.
        system_prompt: Instruction placed in the system message.

    Returns:
        System, user, and assistant messages in chat format.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def manual_format_qwen3(
    example: dict[str, str],
    system_prompt: str,
) -> str:
    """Recreate the Qwen3 chat template as plain text.

    Args:
        example: Example containing ``question`` and ``answer`` fields.
        system_prompt: Instruction placed in the system turn.

    Returns:
        The manually formatted Qwen3 conversation text.
    """
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{example['question']}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"{example['answer']}<|im_end|>\n"
    )


def format_training_text(
    example: dict[str, str],
    tokenizer,
    system_prompt: str,
) -> str:
    """Format one GSM8K example as tokenizer-ready training text.

    Args:
        example: Example containing ``question`` and ``answer`` fields.
        tokenizer: Tokenizer that provides ``apply_chat_template``.
        system_prompt: Instruction placed in the system message.

    Returns:
        The rendered chat template without tokenization.
    """
    return tokenizer.apply_chat_template(
        format_gsm8k_messages(example, system_prompt=system_prompt),
        tokenize=False,
        add_generation_prompt=False,
    )
