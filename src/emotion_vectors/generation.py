from __future__ import annotations

import re

from .config import (
    GENERATION_KWARGS,
    MODEL_ID,
    THINKING_ENABLED,
)


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    *,
    torch_dtype: str = "bfloat16",
    device_map: str | None = "auto",
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install torch and transformers before loading the model."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, torch_dtype),
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def render_chat_prompt(
    tokenizer,
    prompt: str,
    *,
    thinking_enabled: bool = THINKING_ENABLED,
) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": thinking_enabled},
    )


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generation_kwargs(tokenizer) -> dict:
    kwargs = dict(GENERATION_KWARGS)
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    return kwargs


def validate_generation(
    record: dict,
    *,
    min_response_tokens: int = 30,
) -> dict:
    response = record["response"].strip()
    response_lower = response.lower()
    token_count = record["response_end"] - record["response_start"]
    errors = []
    warnings = []

    if not response:
        errors.append("empty_response")
    if token_count <= 0:
        errors.append("no_response_tokens")
    elif token_count < min_response_tokens:
        warnings.append("short_response")

    if record["condition"] == "emotion":
        pattern = rf"\b{re.escape(record['emotion'].lower())}\b"
        if re.search(pattern, response_lower):
            errors.append("emotion_label_leakage")

    if "<think>" in response_lower or "</think>" in response_lower:
        warnings.append("thinking_markers_present")
    if "as an ai" in response_lower or "i cannot" in response_lower:
        warnings.append("meta_or_refusal_response")
    if record["generation_parameters"]["max_new_tokens"] == token_count:
        warnings.append("possibly_truncated")
    if "do not discuss these instructions" in response_lower:
        errors.append("instruction_echo")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def generate_sample(
    model,
    tokenizer,
    row: dict,
    *,
    model_id: str = MODEL_ID,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    thinking_enabled: bool = THINKING_ENABLED,
) -> dict:
    """Generate one row and save the token boundaries needed for extraction."""
    import torch

    rendered_prompt = render_chat_prompt(
        tokenizer,
        row["prompt"],
        thinking_enabled=thinking_enabled,
    )
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_device = _input_device(model)
    encoded = {name: value.to(input_device) for name, value in encoded.items()}

    _set_seed(row["seed"])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **_generation_kwargs(tokenizer),
        )

    full_ids = generated[0].detach().cpu()
    prompt_ids = encoded["input_ids"][0].detach().cpu().tolist()
    response_start = len(prompt_ids)
    response_end = len(full_ids)

    if (
        tokenizer.eos_token_id is not None
        and response_end > response_start
        and int(full_ids[-1]) == tokenizer.eos_token_id
    ):
        response_end -= 1

    response_ids = full_ids[response_start:response_end].tolist()
    record = {
        **row,
        "rendered_prompt": rendered_prompt,
        "response": tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip(),
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "full_ids": full_ids.tolist(),
        "response_start": response_start,
        "response_end": response_end,
        "generation_parameters": dict(GENERATION_KWARGS),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "thinking_enabled": thinking_enabled,
    }
    record["validation"] = validate_generation(record)
    return record


def generation_metadata(model, tokenizer) -> dict[str, str | None]:
    return {
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": getattr(tokenizer, "revision", None),
    }
