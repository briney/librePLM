# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProCoder is an encoder-only protein language model for masked language modeling (MLM) pre-training. It uses a transformer architecture with SDPA attention, RoPE positional embeddings, and SwiGLU feedforward layers. The default configuration is ~865M parameters.

## Commands

### Installation
```bash
pip install -e .
```

### Training
```bash
# Basic training
procoder train data.train=/path/to/proteins.parquet

# With custom settings
procoder train data.train=/path/train.parquet train.num_steps=50000 train.optimizer.lr=1e-4

# Multi-GPU training via Accelerate
accelerate launch -m procoder.train data.train=/path/train.parquet

# With custom config files
procoder train --model-config custom_model.yaml data.train=/path/train.parquet
```

### Smoke Test
```bash
# Validate config and run a tiny forward pass
procoder smoke-test

# Test with modified architecture
procoder smoke-test model.encoder.n_layers=12
```

## Architecture

### Directory Structure
```
src/procoder/
├── cli/           # CLI entry points (train, smoke-test)
├── models/        # Neural network components
│   ├── libreplm.py    # PLMModel (main model)
│   ├── encoder.py     # Encoder stack
│   ├── attention.py   # MultiheadAttention with RoPE
│   └── mlp.py         # SwiGLU feedforward
├── data/          # Dataset implementations and collation
├── eval/          # Evaluation metrics (accuracy, perplexity, P@L)
├── utils/         # Tokenizer, losses, structure parsing
└── configs/       # Hydra configuration files
```

### Model Components
- **PLMModel** (`models/libreplm.py`): Main model class combining embeddings, encoder, and LM head
- **Encoder** (`models/encoder.py`): Stack of EncoderBlocks with shared RoPE
- **EncoderBlock** (`models/blocks.py`): Pre-LayerNorm transformer block with MHA + SwiGLU
- **MultiheadAttention** (`models/attention.py`): Uses PyTorch SDPA with RoPE applied to Q/K

### Configuration System (Hydra)
Root config at `configs/config.yaml` composes:
- `model/arch.yaml`: Architecture (d_model, n_heads, n_layers, etc.)
- `data/base.yaml`: Data loading (batch_size, max_len, train/eval paths)
- `train/base.yaml`: Training (optimizer, scheduler, MLM settings, eval metrics)

Override via CLI: `procoder train model.encoder.n_layers=24 train.optimizer.lr=1e-4`

### Data Formats
- **Parquet/CSV**: Requires columns `pid`, `protein_sequence`, optional `coordinates`
- **Parquet directory**: Streaming mode for large datasets (auto-detected)
- **PDB/mmCIF folder**: Structure files for evaluation with contact prediction

### Tokenizer
32-token vocabulary: 4 special tokens (`<cls>`, `<pad>`, `<eos>`, `<unk>`) + 20 standard amino acids + 4 non-standard + gap/deletion + `<mask>`. Defined in `utils/tokenizer.py`.

### Evaluation Metrics
- **masked_accuracy**: Accuracy on masked tokens (MLM)
- **perplexity**: exp(cross-entropy loss)
- **p_at_l**: Contact prediction using attention weights (requires coordinates)

### Key Training Parameters
- Default ~865M params: d_model=1536, n_heads=24, n_layers=36
- AdamW with lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01
- Linear warmup (2000 steps) + linear/cosine decay
- MLM: 15% masking (80% mask token, 10% random, 10% keep)
