# Neural Machine Translation with Attention

English-to-Portuguese translation using LSTM encoder-decoder architecture with attention mechanism.

## Setup

```bash
conda create -n nmt-attention python=3.9 -y
conda activate nmt-attention
pip install -r requirements.txt
```

## Data

Download the Portuguese-English dataset and place `por.txt` in `data/por-eng/`

## Training

```bash
python train.py
```

## Inference

```bash
python inference.py
```

## Architecture

- **Encoder**: Bidirectional LSTM
- **Attention**: Scaled dot-product cross-attention
- **Decoder**: LSTM with pre/post-attention layers

## Features

- Greedy decoding
- MBR decoding with ROUGE-1 similarity
- Temperature-based sampling
