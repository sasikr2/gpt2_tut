# GPT-2 Tutorial Repository

A comprehensive tutorial repository for understanding and implementing GPT-2 from scratch, with multi-GPU training capabilities.

## 🎯 Overview

This repository serves as a learning resource for understanding the GPT-2 architecture and implementing it step-by-step. It includes both single and multi-GPU training implementations, along with basic transformer components.

**Inspired by and built upon the excellent work from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.**

## 🚀 Features

- **Complete GPT-2 Implementation**: Full implementation of the GPT-2 architecture from scratch
- **Multi-GPU Training**: Distributed training support using PyTorch DDP
- **Educational Components**: Basic transformer implementations for learning
- **Interactive Notebooks**: Jupyter notebooks for exploring GPT-2 behavior
- **Data Processing**: Utilities for downloading and chunking training data
- **Validation Tools**: Scripts for creating validation datasets

## 📁 Repository Structure

```
gpt2_tut/
├── train_gpt2.py              # Single GPU GPT-2 training script
├── train_multi_gpu.py         # Multi-GPU distributed training script
├── training_config.py          # Training configuration parameters
├── explore_gpt2.ipynb         # Interactive GPT-2 exploration notebook
├── download_and_save_in_chunk.py  # Data downloading and chunking utility
├── input.txt                  # Sample training data
├── transformer_basic/         # Basic transformer implementations
│   ├── self_attn_v1.py       # Self-attention mechanism
│   └── multi_head_attn_v1.py # Multi-head attention implementation
└── scripts/
    └── validation_creator.py  # Validation dataset creation utility
```

## 🛠️ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd gpt2_tut

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers tiktoken wandb numpy
pip install jupyter
```

## 🎓 Usage

### Single GPU Training

```bash
python train_gpt2.py
```

### Multi-GPU Training

```bash
# For 8 GPUs
torchrun --standalone --nproc_per_node=8 train_multi_gpu.py

# For multi-node training
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=1234 train_multi_gpu.py
```

### Interactive Exploration

```bash
jupyter notebook explore_gpt2.ipynb
```

### Data Preparation

```bash
python download_and_save_in_chunk.py
```

## 🔧 Key Components

### GPT-2 Architecture
- **Causal Self-Attention**: Multi-head attention with causal masking
- **MLP Blocks**: Feed-forward networks with GELU activation
- **Layer Normalization**: Pre-norm architecture
- **Positional Embeddings**: Learnable position encodings

### Training Features
- **Distributed Data Parallel (DDP)**: Multi-GPU training support
- **Gradient Clipping**: Stable training with large models
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpointing**: Automatic model saving and resuming
- **Wandb Integration**: Training monitoring and logging

### Educational Components
- **Basic Transformer**: Step-by-step transformer implementation
- **Self-Attention**: Detailed attention mechanism explanation
- **Multi-Head Attention**: Understanding attention heads

## 📊 Model Configurations

The repository supports various GPT-2 model sizes:

- **GPT-2 Small**: 124M parameters (12 layers, 12 heads, 768 embedding)
- **GPT-2 Medium**: 350M parameters (24 layers, 16 heads, 1024 embedding)
- **GPT-2 Large**: 774M parameters (36 layers, 20 heads, 1280 embedding)
- **GPT-2 XL**: 1558M parameters (48 layers, 25 heads, 1600 embedding)

## 🎯 Learning Objectives

1. **Understand Transformer Architecture**: Learn how attention mechanisms work
2. **Implement GPT-2**: Build the complete model from scratch
3. **Multi-GPU Training**: Learn distributed training techniques
4. **Data Processing**: Handle large-scale text datasets
5. **Model Optimization**: Implement efficient training strategies

## 🤝 Acknowledgments

This repository is heavily inspired by and builds upon the excellent work from:

- **[nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy - The simplest, fastest repository for training/finetuning medium-sized GPTs
- **llm.c** by Andrej Karpathy - Educational implementations of language models

## 📚 Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [nanoGPT Repository](https://github.com/karpathy/nanoGPT) - Reference implementation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Issues and Contributions

Feel free to open issues for bugs or feature requests. Contributions are welcome!

---

**Happy Learning! 🚀**
