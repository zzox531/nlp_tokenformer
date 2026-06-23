# Mixture of Experts for Tokenformer

**NLP Research Project — May 2026**

Authors: Anastasiya Khodzina, Ignacy Kozakiewicz, Jakub Misiaszek, Mikołaj Znamierowski  
Lab instructor: Jakub Krajewski

## Overview

This project investigates combining two recent advances in transformer architecture:

- **Tokenformer** ([Wang et al., 2025](https://arxiv.org/abs/2410.23168)) — replaces standard linear projections with *pattention* layers, treating model parameters as tokens and applying a modified attention mechanism over them. This decouples model size from architecture, enabling incremental scaling without retraining from scratch.
- **Mixtral of Experts (MoE)** ([Jiang et al., 2024](https://arxiv.org/abs/2401.04088)) — routes each token through a sparse subset of expert feed-forward networks, increasing model capacity without proportional compute cost.

The central question: **can pattention-based parameter tokenization and MoE routing work together effectively?** If so, the resulting architecture could be both highly accurate and easily scalable.

## Main Objective

Implement a MoE-augmented Tokenformer and compare it against two baselines:

| Model | MoE | Pattention |
|---|---|---|
| Transformer + MoE | Yes | No |
| Tokenformer | No | Yes |
| **Tokenformer + MoE (ours)** | **Yes** | **Yes** |

All three models are trained from scratch under identical setups.

## Dataset

A subset of the [OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/) (Gokaslan & Cohen, 2019).

## Evaluation

Following the Tokenformer paper's evaluation protocol (scaled to available compute):

| Metric | Description |
|---|---|
| Test perplexity vs. training cost | Measured in TPU minutes/hours |
| Pile validation perplexity | Zero-shot language modeling |
| HellaSwag accuracy | Commonsense reasoning |
| PIQA accuracy | Physical reasoning |
| WinoGrande accuracy | Pronoun/coreference resolution |

## References

1. Vaswani et al., *Attention is all you need*, 2017
2. Biderman et al., *Pythia: A suite for analyzing large language models across training and scaling*, 2023
3. Wang et al., *Tokenformer: rethinking transformer scaling with tokenized model parameters*, 2025
4. Jiang et al., *Mixtral of Experts*, 2024
5. Aaron Gokaslan and Vanya Cohen, *OpenWebText Corpus*, 2019
