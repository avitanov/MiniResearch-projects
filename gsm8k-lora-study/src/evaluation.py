from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

try:
    from .inference import build_eval_messages, generate_from_messages
except ImportError:
    from inference import build_eval_messages, generate_from_messages


FINAL_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _normalize_number_string(value: str) -> str | None:
    """Normalize a numeric string for reliable exact matching.

    Args:
        value: Numeric text that may contain commas or trailing punctuation.

    Returns:
        Canonical numeric text, or ``None`` when parsing fails.
    """
    cleaned = value.strip().replace(",", "")
    cleaned = cleaned.rstrip(".")
    if not cleaned:
        return None

    try:
        decimal_value = Decimal(cleaned)
    except InvalidOperation:
        return None

    if decimal_value == decimal_value.to_integral():
        return str(int(decimal_value))
    return format(decimal_value.normalize(), "f").rstrip("0").rstrip(".")


def _extract_model_answer(model_output: str) -> str | None:
    """Extract a predicted answer from tagged or free-form output.

    Args:
        model_output: Generated model response.

    Returns:
        The normalized final number, or ``None`` when none is found.
    """
    tagged_answer = extract_answer(model_output)
    if tagged_answer is not None:
        return tagged_answer

    numbers = NUMBER_PATTERN.findall(model_output)
    if not numbers:
        return None
    return _normalize_number_string(numbers[-1])


def extract_answer(text: str) -> str | None:
    """Extract a tagged final answer of the form ``#### <number>``.

    Args:
        text: Text that may contain a tagged numeric answer.

    Returns:
        The normalized tagged number, or ``None`` when absent or invalid.
    """
    match = FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    return _normalize_number_string(match.group(1))


def is_correct(prediction: str, reference: str) -> bool:
    """Compare tagged numeric answers for exact correctness.

    Args:
        prediction: Generated answer text.
        reference: Reference solution text.

    Returns:
        ``True`` when both tagged answers are numerically equivalent.
    """
    predicted = extract_answer(prediction)
    reference_answer = extract_answer(reference)
    if predicted is None or reference_answer is None:
        return False
    try:
        return abs(float(predicted) - float(reference_answer)) < 1e-6
    except ValueError:
        return predicted == reference_answer


def evaluate_gsm8k_sample(
    predictions: list[str],
    references: list[str],
) -> dict[str, object]:
    """Evaluate paired GSM8K predictions and references.

    Args:
        predictions: Generated response texts.
        references: Reference solution texts aligned with ``predictions``.

    Returns:
        Aggregate accuracy, parse-failure statistics, and per-example rows.
    """
    rows: list[dict[str, object]] = []
    correct = 0
    parse_failures = 0

    for prediction, reference in zip(predictions, references, strict=True):
        predicted_answer = _extract_model_answer(prediction)
        reference_answer = extract_answer(reference)
        parsed = predicted_answer is not None
        matched = (
            parsed and reference_answer is not None and predicted_answer == reference_answer
        )

        if matched:
            correct += 1
        if not parsed:
            parse_failures += 1

        rows.append(
            {
                "prediction_text": prediction,
                "predicted_answer": predicted_answer,
                "reference_answer": reference_answer,
                "correct": matched,
                "parse_failed": not parsed,
            }
        )

    total = len(rows)
    accuracy = correct / total if total else 0.0
    parse_failure_rate = parse_failures / total if total else 0.0

    return {
        "accuracy": accuracy,
        "parse_failure_rate": parse_failure_rate,
        "correct": correct,
        "total": total,
        "rows": rows,
    }


def evaluate_dataset(
    model,
    tokenizer,
    examples,
    max_new_tokens: int = 512,
    few_shot_examples: list[dict[str, str]] | None = None,
    system_prompt: str = "Solve step by step. Write your answer after '####'.",
) -> dict[str, object]:
    """Generate and evaluate responses for a GSM8K-style dataset.

    Args:
        model: Language model used for generation.
        tokenizer: Tokenizer used to format and decode messages.
        examples: Iterable of examples containing ``question`` and ``answer``.
        max_new_tokens: Maximum number of generated tokens per example.
        few_shot_examples: Optional demonstrations added before each question.
        system_prompt: Instruction placed in the system message.

    Returns:
        Aggregate accuracy, parse-failure statistics, and per-example rows.
    """
    rows: list[dict[str, object]] = []
    correct = 0
    parse_failures = 0

    for example in examples:
        generation_error = None
        try:
            messages = build_eval_messages(
                question=example["question"],
                few_shot_examples=few_shot_examples,
                system_prompt=system_prompt,
            )
            response = generate_from_messages(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=max_new_tokens,
            )
        except Exception as error:
            response = ""
            generation_error = f"{type(error).__name__}: {error}"
        predicted_answer = extract_answer(response)
        reference_answer = extract_answer(example["answer"])
        parse_failed = predicted_answer is None
        matched = is_correct(response, example["answer"])

        if parse_failed:
            parse_failures += 1
        if matched:
            correct += 1

        rows.append(
            {
                "question": example["question"],
                "reference": example["answer"],
                "response": response,
                "predicted_answer": predicted_answer,
                "reference_answer": reference_answer,
                "correct": matched,
                "parse_failed": parse_failed,
                "generation_error": generation_error,
            }
        )
        # For kernel-level crash recovery, persist `rows` after each append in the calling notebook.

    total = len(rows)
    return {
        "accuracy": correct / total if total else 0.0,
        "parse_failure_rate": parse_failures / total if total else 0.0,
        "rows": rows,
        "correct": correct,
        "total": total,
    }


def categorize_gsm8k_error(response: str, reference: str) -> str:
    """Assign a coarse category to a GSM8K response error.

    Args:
        response: Generated model response.
        reference: Reference solution text.

    Returns:
        A parse, format, or reasoning/arithmetic error label.
    """
    extracted = extract_answer(response)
    if extracted is None:
        return "Parse failure"
    if extracted == extract_answer(reference):
        return "Format drift"
    if "####" not in response:
        return "Format drift"
    return "Reasoning or arithmetic error"
