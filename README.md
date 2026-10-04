<h1 align="center">LLM Emotion Representations</h1>

<p align="center">
Exploring how language models represent and express emotions
</p>

---

## Motivation

Recent work by [Anthropic](https://www.anthropic.com/research/emotion-concepts-function) examines emotion concepts in language models and presents evidence that models can contain internal representations associated with emotions. This suggests that emotion-related representations may influence how models respond and behave in emotionally relevant situations.

Inspired by this line of work, this project will explore how models represent emotion-related directions, or **emotion vectors**, in their hidden activations.

## What This Repo Is About

This project will involve exploring how small language models represent emotion-related information internally. We will work with model activations and emotion vectors as ways of studying patterns associated with emotional language and behaviour.

The broader aim is to better understand how models encode and express emotion-related information, without treating these representations as evidence that models experience emotions in the human sense.

## Immediate Goals

The immediate goals are to:

- select one reliably running small language model;
- prepare emotional and neutral examples across several emotion categories;
- extract hidden activations at selected layers;
- compute candidate emotion vectors;
- produce one measurable representation result, such as cosine similarity or layer-wise separability;
- compare baseline and emotion-steered outputs on identical prompts;
- test several steering strengths;
- evaluate emotional change alongside output quality;

## Approach

The first experiment uses `Qwen/Qwen3.5-2B` and a generation-based method.
The model generates short stories for ten target emotions and matched neutral
stories. Prompts use five templates and twenty topics, with seed `42`.

The extraction runs in two passes:

1. Generate each story and save the exact prompt and token IDs.
2. Replay the saved prompt and response with
   `output_hidden_states=True` and `use_cache=False`.

For each layer, we keep only the response-token hidden states and average them
into one activation for that story. Emotion vectors are computed by subtracting
matched neutral activations from emotional activations, averaging across story
pairs, and normalizing the result.

The current manifest contains:

- 1,000 emotional stories;
- 100 neutral stories;
- 1,100 total generations.

### Running the scripts

Install the project, then run generation and extraction as separate steps:

```bash
python -m pip install -e .
python scripts/generate_dataset.py
python scripts/extract_activations.py
```

For a small pilot run:

```bash
python scripts/generate_dataset.py --run-dir data/runs/pilot --limit 2
python scripts/extract_activations.py --run-dir data/runs/pilot --limit 2
```

Each run stores its manifest and generated records as JSONL. Extracted
activation tensors are saved under `activations/`, with their metadata in
`activation_index.jsonl`.
