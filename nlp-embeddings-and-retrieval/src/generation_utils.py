"""GPT-2 text generation and attention helpers."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_gpt2_model(
    model_name: str,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a GPT-2 tokenizer and causal language model.

    Args:
        model_name (str): HuggingFace model name.

    Returns:
        Tuple[PreTrainedTokenizerBase, PreTrainedModel]: Tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def generate_text(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    generation_kwargs: Dict[str, Any],
    max_new_tokens: int,
    seed: int,
) -> str:
    """
    Generate text from one prompt.

    Args:
        prompt (str): Prompt used for generation.
        tokenizer (PreTrainedTokenizerBase): GPT-2 tokenizer.
        model (PreTrainedModel): GPT-2 language model.
        generation_kwargs (Dict[str, Any]): Decoding strategy settings.
        max_new_tokens (int): Maximum number of new tokens to generate.
        seed (int): Random seed used before generation.

    Returns:
        str: Generated text including the prompt.
    """
    torch.manual_seed(seed)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )


def generate_text_with_attentions(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    max_new_tokens: int,
) -> str:
    """
    Generate text while requesting attention outputs.

    Args:
        prompt (str): Prompt used for generation.
        tokenizer (PreTrainedTokenizerBase): GPT-2 tokenizer.
        model (PreTrainedModel): GPT-2 language model.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: Generated text including the prompt.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated.sequences[0],
        skip_special_tokens=True,
    )


def get_attention_matrix(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    layer_index: int,
    head_index: int,
    max_tokens: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract one GPT-2 attention head as a square matrix.

    Args:
        text (str): Text to tokenize and inspect.
        tokenizer (PreTrainedTokenizerBase): GPT-2 tokenizer.
        model (PreTrainedModel): GPT-2 language model.
        layer_index (int): Transformer layer index.
        head_index (int): Attention head index.
        max_tokens (int): Maximum number of tokens to include.

    Returns:
        Tuple[np.ndarray, List[str]]: Attention matrix with shape
        (n_tokens, n_tokens) and matching token strings.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )

    attention = outputs.attentions[layer_index][0, head_index].detach().cpu().numpy()
    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return attention, tokens
