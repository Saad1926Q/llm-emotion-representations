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
activation tensors are saved under `activations/`. Re-running extraction
overwrites the existing tensor files.

## Dataset B: expression suppression

Dataset B contains 20 five-turn conversations for each of the ten target
emotions. Every base conversation is generated under natural and suppression
instructions, giving 200 base conversations and 2,000 assistant responses.
The two conditions use identical user turns and matched RNG seeds.

Rebuild and freeze the Dataset A vectors before generating Dataset B:

```bash
uv run python scripts/generate_dataset.py \
  --run-dir data/runs/qwen35_2b_fp32 \
  --torch-dtype float32 \
  --batch-size 8
uv run python scripts/extract_activations.py \
  --run-dir data/runs/qwen35_2b_fp32 \
  --torch-dtype float32 \
  --batch-size 4
uv run python scripts/build_vectors.py \
  --run-dir data/runs/qwen35_2b_fp32
```

`build_vectors.py` saves raw and unit vectors for every decoder layer, the
frozen layer-12 vectors, the Dataset A neutral mean, pair counts, and source
metadata under `vectors/`. To upload only those frozen artifacts to an existing
Hugging Face model repository:

```bash
uv run python scripts/build_vectors.py \
  --run-dir data/runs/qwen35_2b_fp32 \
  --hf-repo USER/REPOSITORY
```

Generate Dataset B and calculate its raw internal projections:

```bash
uv run python scripts/generate_dataset_b.py --batch-size 4
uv run python scripts/score_internal_b.py \
  --vector-dir data/runs/qwen35_2b_fp32/vectors \
  --batch-size 4
```

Both scripts resume from their JSONL outputs. For a paired two-condition smoke
run, use `generate_dataset_b.py --limit-conversations 1`; then score it with
`score_internal_b.py --limit 10`. Internal scoring replays each complete prompt
and response, mean-pools only the current assistant response at frozen decoder
layer 12, centers it using the frozen Dataset A neutral mean, and writes raw
projections onto all ten unit vectors to `internal_scores.jsonl`.

All four model-running scripts accept `--batch-size`; the default is `1`.
Dataset B batches independent conversations only at the same turn. Natural and
suppression batches retain identical conversation order and reset to the same
batch seed. Use the largest value that fits GPU memory; later conversation turns
are longer and may require a smaller batch than Dataset A.
