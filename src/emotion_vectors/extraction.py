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


def _response_hidden_states(model, sample: dict):
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
    return outputs.hidden_states[1:]


def extract_layer_activation(model, sample: dict, layer: int):
    """Return the pooled response activation for one zero-indexed decoder layer."""
    hidden_states = _response_hidden_states(model, sample)
    if not 0 <= layer < len(hidden_states):
        raise ValueError(
            f"layer must be in [0, {len(hidden_states) - 1}], received {layer}"
        )
    response_states = hidden_states[layer][
        0, sample["response_start"] : sample["response_end"]
    ]
    return pool_response_states(response_states).float().cpu()


def extract_activations(model, sample: dict):
    """Return one pooled response activation per decoder layer."""
    import torch

    hidden_states = _response_hidden_states(model, sample)
    response_start = sample["response_start"]
    response_end = sample["response_end"]
    activations = [
        pool_response_states(layer_states[0, response_start:response_end])
        .float()
        .cpu()
        for layer_states in hidden_states
    ]
    return torch.stack(activations)


def _response_hidden_states_batch(model, samples: list[dict]):
    import torch

    if not samples:
        raise ValueError("samples must not be empty")
    max_length = max(len(sample["full_ids"]) for sample in samples)
    input_ids = torch.zeros(
        (len(samples), max_length),
        dtype=torch.long,
        device=_input_device(model),
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sample in enumerate(samples):
        length = len(sample["full_ids"])
        input_ids[index, :length] = torch.tensor(
            sample["full_ids"], dtype=torch.long, device=input_ids.device
        )
        attention_mask[index, :length] = 1

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
    return outputs.hidden_states[1:]


def extract_activations_batch(model, samples: list[dict]):
    """Return all-layer pooled activations for each sample in a padded batch."""
    import torch

    hidden_states = _response_hidden_states_batch(model, samples)
    results = []
    for sample_index, sample in enumerate(samples):
        pooled = [
            pool_response_states(
                layer_states[
                    sample_index,
                    sample["response_start"] : sample["response_end"],
                ]
            )
            .float()
            .cpu()
            for layer_states in hidden_states
        ]
        results.append(torch.stack(pooled))
    return results


def extract_layer_activations_batch(model, samples: list[dict], layer: int):
    """Return one pooled decoder-layer activation for each sample."""
    hidden_states = _response_hidden_states_batch(model, samples)
    if not 0 <= layer < len(hidden_states):
        raise ValueError(
            f"layer must be in [0, {len(hidden_states) - 1}], received {layer}"
        )
    layer_states = hidden_states[layer]
    return [
        pool_response_states(
            layer_states[
                index,
                sample["response_start"] : sample["response_end"],
            ]
        )
        .float()
        .cpu()
        for index, sample in enumerate(samples)
    ]
