# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CODI (Compressing Chain-of-Thought into Continuous Space via Self-Distillation). The core idea: replace explicit chain-of-thought reasoning steps with continuous latent "thoughts" distilled from a teacher model. The student model encodes the question, iterates latent embeddings through the LLM (without generating tokens), then decodes the final answer.

## Commands

**Training:**
```bash
bash scripts/train_gpt2_gsm8k-aug.sh       # GPT-2 on GSM8k-Aug
bash scripts/train_gpt2_gsm8k-aug-nl.sh    # GPT-2 on GSM8k-Aug-NL (natural language CoT)
bash scripts/train_gpt2_commonsense.sh     # GPT-2 on CommonsenseQA
bash scripts/train_llama1b_gsm8k-aug.sh    # LLaMA-3.2-1B on GSM8k-Aug
bash scripts/train_llama1b_gsm8k-aug-nl.sh # LLaMA-3.2-1B on GSM8k-Aug-NL
bash scripts/train_llama_commonsense.sh    # LLaMA on CommonsenseQA
```

**Evaluation:**
```bash
bash scripts/test_gpt2.sh     # Evaluate GPT-2 on GSM8K
bash scripts/test_llama1b.sh  # Evaluate LLaMA on GSM8K
```
Change `--data_name` in test scripts to `svamp`, `gsm-hard`, `multi-arith`, or `commonsense` for OOD benchmarks.

**Probing latent thoughts:**
```bash
bash scripts/probe_latent_token.sh
```
Outputs to `outputs/decoded_latent.txt`.

## Architecture

### Data flow

1. **`src/model.py`** — `CODI` class (extends `nn.Module`). Wraps a pretrained causal LM (GPT-2/LLaMA), adds LoRA adapters, optional projection layer (`prj`), and special tokens (BOT/EOT as vocabulary extensions). Also defines `ModelArguments`, `DataArguments`, `TrainingArguments` (HuggingFace HfArgumentParser dataclasses).

2. **`train.py`** — Training entry point. `SupervisedDataset` builds encoder/decoder input pairs. Data flow: question → `encoder_input_ids` (ends with BOT), teacher's full sequence → `ref_input_ids` (question + CoT + answer), student's expected output → `decoder_input_ids` (EOT + answer). `CustomTrainer.compute_loss` calls `model.forward()` which:
   - Encodes question → extracts hidden state at last position as first latent embedding
   - Runs `num_latent` iterations: passes latent embedding through the LLM, takes last hidden state as next latent (optionally through `prj`)
   - At the final latent: feeds decoder tokens, computes distillation loss (SmoothL1/MSE between student and teacher hidden states at the answer position across all layers) + CE loss on student's answer + CE loss on teacher's full sequence
   - Loss = `ce_loss_total + distill_loss_total * distill_loss_factor + ref_ce_loss * ref_loss_factor`

3. **`test.py`** — Inference. Encodes question + latent iterations, then autoregressively decodes the answer. Supports greedy or top-k/top-p sampling. Extracts numeric answers from decoded text and compares against ground truth.

4. **`probe_latent_token.py`** — Similar to test.py but logs the top-k token probabilities at each latent iteration step for interpretability analysis.

### Key arguments (TrainingArguments)

| Argument | Description |
|---|---|
| `--num_latent` | Number of latent thought iterations during training |
| `--inf_latent_iterations` | Number of latent iterations during inference |
| `--use_prj` | Whether to use a projection layer after the LLM for latent generation |
| `--prj_dim` | Hidden dimension of the projection layer (e.g., 768 for GPT-2, 2048 for LLaMA) |
| `--prj_no_ln` | Remove LayerNorm from the projection layer |
| `--distill_loss_type` | `smooth_l1`, `l2`, or `l1` |
| `--distill_loss_factor` | Multiplier for the distillation loss |
| `--ref_loss_factor` | Multiplier for the teacher's CE loss |
| `--distill_loss_div_std` | Normalize distillation loss by teacher hidden state std |
| `--remove_eos` | Skip `<eos>` delimiter between question and answer |
| `--include_last_cot` | Include the last CoT step in training data |
| `--max_token_num` | Discard samples exceeding this token length |
| `--restore_from` | `latest` (auto-find checkpoint), specific path, or `best_checkpoint` |

### Dataset names

- `icot` — GSM8k-Aug (intermediate CoT steps with progressive truncation)
- `icot-full` — GSM8k-Aug-NL (full natural language CoT)
- `commonsense` — CommonsenseQA
- `strategy` — StrategyQA
- `prontoqa` — ProntoQA
