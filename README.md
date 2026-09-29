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
