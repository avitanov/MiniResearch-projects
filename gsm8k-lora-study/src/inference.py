from __future__ import annotations

import torch


def _normalize_decoded_text(value: object) -> str:
    """Validate and trim decoded tokenizer output.

    Args:
        value: Decoded value returned by a tokenizer.

    Returns:
        The stripped decoded text.
    """
    if not isinstance(value, str):
        raise ValueError(f"Tokenizer returned non-string decoded content: {value!r}")
    return value.strip()


def build_messages(
    user_prompt: str,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    """Build chat messages for an instruction-tuned model.

    Args:
        user_prompt: User instruction or question.
        system_prompt: Optional system instruction.

    Returns:
        Ordered system and user messages.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_response(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str | None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Generate one response from a user and optional system prompt.

    Args:
        model: Language model used for generation.
        tokenizer: Tokenizer used to render and decode the chat.
        user_prompt: User instruction or question.
        system_prompt: Optional system instruction.
        max_new_tokens: Maximum number of generated tokens.
        do_sample: Whether to use sampling instead of greedy decoding.
        temperature: Sampling temperature used when sampling is enabled.
        top_p: Nucleus-sampling probability used when sampling is enabled.

    Returns:
        The decoded generated response.
    """
    messages = build_messages(user_prompt=user_prompt, system_prompt=system_prompt)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    if not decoded:
        raise ValueError("Tokenizer returned no decoded responses.")
    return _normalize_decoded_text(decoded[0])


def build_eval_messages(
    question: str,
    few_shot_examples: list[dict[str, str]] | None = None,
    system_prompt: str = "Solve step by step. Write your answer after '####'.",
) -> list[dict[str, str]]:
    """Build messages for GSM8K-style evaluation.

    Args:
        question: Evaluation question.
        few_shot_examples: Optional question-and-answer demonstrations.
        system_prompt: Instruction placed in the system message.

    Returns:
        Ordered system, demonstration, and evaluation messages.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if few_shot_examples:
        for example in few_shot_examples:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def generate_from_messages(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
) -> str:
    """Generate one response from prepared chat messages.

    Args:
        model: Language model used for generation.
        tokenizer: Tokenizer used to render and decode the chat.
        messages: Prepared chat messages.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The decoded generated response.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    response_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return _normalize_decoded_text(tokenizer.decode(response_ids, skip_special_tokens=True))


def run_probe_set(
    model,
    tokenizer,
    probes: list[dict[str, str]],
    system_prompt: str | None,
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    """Run probes while isolating per-probe generation failures.

    Args:
        model: Language model used for generation.
        tokenizer: Tokenizer used to render and decode prompts.
        probes: Probe dictionaries containing ``id`` and ``prompt`` fields.
        system_prompt: Optional system instruction shared by all probes.
        max_new_tokens: Maximum number of generated tokens per probe.

    Returns:
        Prompt-output records, including error text for failed probes.
    """
    results: list[dict[str, str]] = []
    for probe in probes:
        try:
            output = generate_response(
                model=model,
                tokenizer=tokenizer,
                user_prompt=probe["prompt"],
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            result = {
                "id": probe["id"],
                "prompt": probe["prompt"],
                "output": output,
            }
        except Exception as error:
            result = {
                "id": str(probe.get("id", "")),
                "prompt": str(probe.get("prompt", "")),
                "output": "",
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        # For kernel-level crash recovery, persist `results` after each append in the calling notebook.
    return results
