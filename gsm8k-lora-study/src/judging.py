from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
from litellm import completion
from tqdm.auto import tqdm


MAX_API_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_MARKERS = (
    "connection",
    "network",
    "rate limit",
    "ratelimit",
    "temporarily unavailable",
    "temporary failure",
    "timed out",
    "timeout",
)


class JudgeProviderError(RuntimeError):
    """Raised when the provider request fails before a usable response arrives."""


class JudgeResponseError(ValueError):
    """Raised when a provider response cannot be read or parsed."""


class JudgeValidationError(ValueError):
    """Raised when parsed judge JSON does not satisfy the expected schema."""


def extract_text_content(content: object) -> str:
    """Normalize LiteLLM message content into plain text.

    Args:
        content: String or structured message content returned by LiteLLM.

    Returns:
        Stripped text extracted from the response content.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                text_parts.append(str(item["text"]))
        if text_parts:
            return "".join(text_parts).strip()

    raise JudgeResponseError(f"Unsupported judge response content: {content!r}")


def parse_json_response(content: str) -> dict[str, object]:
    """Parse a judge JSON object while tolerating surrounding text.

    Args:
        content: Raw textual content returned by the judge model.

    Returns:
        The first valid JSON object found in the content.
    """
    if not isinstance(content, str):
        raise JudgeResponseError(f"Judge response content must be text, got: {content!r}")
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        parsed = None
        for match in re.finditer(r"\{", content):
            try:
                candidate, _ = decoder.raw_decode(content[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            raise JudgeResponseError("Judge response did not contain a valid JSON object.") from direct_error

    if not isinstance(parsed, dict):
        raise JudgeResponseError(f"Expected the judge to return a JSON object, got: {parsed!r}")
    return parsed


def _is_retryable_provider_error(error: Exception) -> bool:
    """Determine whether a provider error is likely transient.

    Args:
        error: Exception raised by the completion provider.

    Returns:
        ``True`` for retryable status codes or transient error markers.
    """
    status_code = getattr(error, "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    error_name = type(error).__name__.lower()
    error_message = str(error).lower()
    return any(marker in error_name or marker in error_message for marker in RETRYABLE_ERROR_MARKERS)


def _response_content(response: object) -> object:
    """Read message content from a LiteLLM completion response.

    Args:
        response: LiteLLM response object containing completion choices.

    Returns:
        Content from the first response message.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise JudgeResponseError("Judge response contained no choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise JudgeResponseError("Judge response choice contained no message.")
    return getattr(message, "content", None)


def _validate_reasoning_score(score: dict[str, object]) -> dict[str, object]:
    """Validate reasoning, arithmetic, format, and justification fields.

    Args:
        score: Parsed judge response to validate.

    Returns:
        The original score dictionary when all fields are valid.
    """
    for field in ("reasoning", "arithmetic", "format"):
        value = score.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise JudgeValidationError(f"{field!r} must be an integer from 1 to 5, got {value!r}.")
    justification = score.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise JudgeValidationError("'justification' must be a non-empty string.")
    return score


def _validate_pairwise_score(score: dict[str, object]) -> dict[str, object]:
    """Validate a pairwise winner and justification.

    Args:
        score: Parsed pairwise judge response to validate.

    Returns:
        The original score dictionary when all fields are valid.
    """
    winner = score.get("winner")
    if winner not in {"A", "B"}:
        raise JudgeValidationError(f"'winner' must be 'A' or 'B', got {winner!r}.")
    justification = score.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise JudgeValidationError("'justification' must be a non-empty string.")
    return score


def _corrective_messages(
    messages: list[dict[str, str]],
    content: str,
    schema_description: str,
) -> list[dict[str, str]]:
    """Build a corrective prompt after an invalid received response.

    Args:
        messages: Original request messages.
        content: Invalid response content received from the model.
        schema_description: Human-readable JSON schema required on retry.

    Returns:
        Original messages followed by the response and correction request.
    """
    return [
        *messages,
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": (
                "Your response was received, but it could not be parsed or validated. "
                f"Return only one valid JSON object matching this schema: {schema_description}"
            ),
        },
    ]


def _failure_result(fields: tuple[str, ...], error_type: str, error: Exception) -> dict[str, object]:
    """Create a schema-compatible judge failure record.

    Args:
        fields: Result fields that should be set to ``None``.
        error_type: Stable category describing the failure stage.
        error: Exception that caused the failure.

    Returns:
        A judge result containing empty values and error metadata.
    """
    return {
        **{field: None for field in fields},
        "justification": "",
        "judge_error_type": error_type,
        "judge_error": f"{type(error).__name__}: {error}",
    }


def call_litellm_completion(
    api_key: str,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: int = 120,
    api_base: str | None = None,
) -> str:
    """Send a LiteLLM request with retries for transient failures.

    Args:
        api_key: Provider API key.
        model_name: LiteLLM model identifier.
        messages: Chat messages sent to the provider.
        max_tokens: Maximum number of response tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        api_base: Optional OpenRouter API base URL.

    Returns:
        Normalized text from the first completion choice.
    """
    if model_name.startswith("openrouter/"):
        os.environ["OPENROUTER_API_KEY"] = api_key
        if api_base is not None:
            os.environ["OPENROUTER_API_BASE"] = api_base

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = completion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as error:
            if attempt < MAX_API_ATTEMPTS and _is_retryable_provider_error(error):
                time.sleep(2 ** (attempt - 1))
                continue
            raise JudgeProviderError(
                f"Judge API request failed after {attempt} attempt(s): {error}"
            ) from error
        return extract_text_content(_response_content(response))
    raise JudgeProviderError("Judge API request failed without producing a response.")


def judge_reasoning(
    api_key: str,
    model_name: str,
    question: str,
    reference: str,
    model_output: str,
) -> dict[str, object]:
    """Score one math response on reasoning, arithmetic, and format.

    Args:
        api_key: Provider API key.
        model_name: LiteLLM judge model identifier.
        question: Math problem given to the evaluated model.
        reference: Reference solution.
        model_output: Model response being evaluated.

    Returns:
        Validated scores or a schema-compatible failure record.
    """
    prompt = f"""You are evaluating a math problem solution.

Problem: {question}
Reference solution: {reference}
Model output: {model_output}

Rate the model output on three dimensions from 1 to 5:
- reasoning: Is the step-by-step reasoning logically sound?
- arithmetic: Are the individual calculations correct?
- format: Does the output end with '#### <number>' as required?

Return ONLY valid JSON with no preamble:
{{"reasoning": N, "arithmetic": N, "format": N, "justification": "one sentence"}}"""

    messages = [{"role": "user", "content": prompt}]
    try:
        content = call_litellm_completion(
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            max_tokens=150,
        )
    except JudgeProviderError as error:
        return _failure_result(("reasoning", "arithmetic", "format"), "api_error", error)
    except JudgeResponseError as error:
        return _failure_result(("reasoning", "arithmetic", "format"), "response_error", error)

    try:
        return _validate_reasoning_score(parse_json_response(content))
    except (JudgeResponseError, JudgeValidationError) as first_error:
        try:
            corrected_content = call_litellm_completion(
                api_key=api_key,
                model_name=model_name,
                messages=_corrective_messages(
                    messages,
                    content,
                    '{"reasoning": 1-5, "arithmetic": 1-5, "format": 1-5, "justification": "text"}',
                ),
                max_tokens=150,
            )
        except (JudgeProviderError, JudgeResponseError) as error:
            error_type = "api_error" if isinstance(error, JudgeProviderError) else "response_error"
            result = _failure_result(("reasoning", "arithmetic", "format"), error_type, error)
            result["initial_judge_error"] = f"{type(first_error).__name__}: {first_error}"
            return result
        try:
            return _validate_reasoning_score(parse_json_response(corrected_content))
        except (JudgeResponseError, JudgeValidationError) as error:
            error_type = "validation_error" if isinstance(error, JudgeValidationError) else "parse_error"
            result = _failure_result(("reasoning", "arithmetic", "format"), error_type, error)
            result["initial_judge_error"] = f"{type(first_error).__name__}: {first_error}"
            return result


def score_rows_with_judge(
    rows: list[dict[str, object]],
    api_key: str,
    model_name: str,
    progress_desc: str = "Judge scoring",
) -> pd.DataFrame:
    """Judge rows while isolating failures to individual items.

    Args:
        rows: Records containing ``question``, ``reference``, and ``response``.
        api_key: Provider API key.
        model_name: LiteLLM judge model identifier.
        progress_desc: Label displayed by the progress bar.

    Returns:
        Input rows merged with judge scores or error metadata.
    """
    scored: list[dict[str, object]] = []
    for row in tqdm(rows, desc=progress_desc):
        try:
            score = judge_reasoning(
                api_key=api_key,
                model_name=model_name,
                question=row["question"],
                reference=row["reference"],
                model_output=row["response"],
            )
        except Exception as error:
            score = _failure_result(
                ("reasoning", "arithmetic", "format"),
                "unexpected_error",
                error,
            )
        scored.append({**row, **score})
        # For kernel-level crash recovery, checkpoint `scored` from the calling notebook.
    return pd.DataFrame(scored)


def summarize_judge_scores(frame: pd.DataFrame, label: str) -> dict[str, str]:
    """Summarize judge score means and standard deviations.

    Args:
        frame: Judge results containing score columns.
        label: Display name for the evaluated model.

    Returns:
        Formatted reasoning, arithmetic, and format statistics.
    """
    def format_metric(column: str) -> str:
        """Format one numeric score column.

        Args:
            column: Score column name.

        Returns:
            Mean and standard deviation formatted to two decimals.
        """
        values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
        return f"{values.mean():.2f} ± {values.std():.2f}"

    return {
        "Model": label,
        "Reasoning (mean ± std)": format_metric("reasoning"),
        "Arithmetic (mean ± std)": format_metric("arithmetic"),
        "Format (mean ± std)": format_metric("format"),
    }


def judge_pairwise(
    api_key: str,
    model_name: str,
    question: str,
    response_a: str,
    response_b: str,
) -> dict[str, object]:
    """Ask a judge to choose the better of two responses.

    Args:
        api_key: Provider API key.
        model_name: LiteLLM judge model identifier.
        question: Question answered by both candidate responses.
        response_a: First candidate response.
        response_b: Second candidate response.

    Returns:
        A validated winner result or schema-compatible failure record.
    """
    prompt = f"""You are comparing two answers to the same math question.

Question: {question}
Response A: {response_a}
Response B: {response_b}

Pick the better response overall. Return ONLY valid JSON:
{{"winner": "A" or "B", "justification": "one sentence"}}"""

    messages = [{"role": "user", "content": prompt}]
    try:
        content = call_litellm_completion(
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            max_tokens=120,
        )
    except JudgeProviderError as error:
        return _failure_result(("winner",), "api_error", error)
    except JudgeResponseError as error:
        return _failure_result(("winner",), "response_error", error)

    try:
        return _validate_pairwise_score(parse_json_response(content))
    except (JudgeResponseError, JudgeValidationError) as first_error:
        try:
            corrected_content = call_litellm_completion(
                api_key=api_key,
                model_name=model_name,
                messages=_corrective_messages(
                    messages,
                    content,
                    '{"winner": "A" or "B", "justification": "text"}',
                ),
                max_tokens=120,
            )
        except (JudgeProviderError, JudgeResponseError) as error:
            error_type = "api_error" if isinstance(error, JudgeProviderError) else "response_error"
            result = _failure_result(("winner",), error_type, error)
            result["initial_judge_error"] = f"{type(first_error).__name__}: {first_error}"
            return result
        try:
            return _validate_pairwise_score(parse_json_response(corrected_content))
        except (JudgeResponseError, JudgeValidationError) as error:
            error_type = "validation_error" if isinstance(error, JudgeValidationError) else "parse_error"
            result = _failure_result(("winner",), error_type, error)
            result["initial_judge_error"] = f"{type(first_error).__name__}: {first_error}"
            return result
