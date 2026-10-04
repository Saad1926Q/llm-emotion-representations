def pool_response_states(states):
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError(
            f"expected non-empty [tokens, hidden] states, got {tuple(states.shape)}"
        )
    return states.mean(dim=0)


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def extract_activations(model, sample: dict):
    """Return one pooled response activation per decoder layer."""
    import torch

    input_ids = torch.tensor(
        sample["full_ids"],
        dtype=torch.long,
        device=_input_device(model),
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
        raise RuntimeError(
            "The model did not return embedding and decoder-layer hidden states."
        )

    response_start = sample["response_start"]
    response_end = sample["response_end"]
    activations = []

    for layer_states in outputs.hidden_states[1:]:
        response_states = layer_states[0, response_start:response_end]
        activations.append(pool_response_states(response_states).float().cpu())

    return torch.stack(activations)
