<h1>ShakespearGPT</h1>

A small GPT-style language model built from scratch using PyTorch, trained on Shakespeare's works to generate Shakespeare-inspired text. ( Inspired and based of Andrej Karpathy tutorial "Let's build GPT: from scratch, in code, spelled out.")

This SLM (Small language model) project explores the fundamentals of Transformer architecture, self-attention, and language model training through a lightweight character-level implementation.

Furthermore, this project started off from CMKL AiCE warp camp 2026, under one of their courses provided. However this project is a advance of what I've learnt during the courses with alterations and improvements.

<h2>Highlights</h2>

Transformer Architecture – Implements a GPT-style Transformer with multi-head self-attention.

Character-Level Tokenization – Converts individual characters into numerical tokens.

Shakespeare Dataset – Trained on the Tiny Shakespeare dataset.

Text Generation – Generates new Shakespeare-inspired passages character by character.

GPU Support – Automatically detects CUDA, Apple MPS, or falls back to CPU.

Model Checkpointing – Saves and reloads trained model weights and vocabulary mappings.

<h2>Overview</h2>

ShakespearGPT is an experimental language model designed to demonstrate how a GPT-like architecture works under the hood.

Instead of relying on pretrained models or high-level language-model libraries, the project implements the core Transformer components directly using PyTorch.

The model learns character patterns from Shakespeare's plays and uses those patterns to generate new sequences of text.

<h2>Architecture</h2>

The model consists of the following components:

Token Embeddings – Converts characters into learned vector representations.

Positional Embeddings – Provides information about character positions in a sequence.

Self-Attention Heads – Allow the model to learn relationships between characters in a sequence.

Multi-Head Attention – Combines multiple attention heads to capture different patterns.

Feed-Forward Networks – Processes the attention outputs.

Transformer Blocks – Combines attention and feed-forward layers with residual connections and layer normalization.

Language Model Head – Predicts the next character in the sequence.

<h2>Configuration</h2>

The current training configuration is:

<h3>Parameter</h3>

Value

<h3>Batch size</h3>

16

<h3>Context length</h3>

32

<h3>Embedding dimensions</h3>

64

<h3>Attention heads</h3>

4

<h3>Transformer layers</h3>

4

<h3>Current training iterations</h3>

1,000

<h3>Learning rate</h3>

0.001

<h3>Dropout</h3>

0.0

<h3>Optimizer</h3>

AdamW

These values can be adjusted in the source code to experiment with different model configurations.

<h2>Requirements</h2>

Python 3.9+

PyTorch

<h2>Installation</h3>

<h3>1. Clone the repository</h3>

git clone https://github.com/IoGamZ/ShakespearGPT.git
cd ShakespearGPT

<h3>2. Install dependencies</h3>

pip install torch

<h3>3. Run the program</h3>

Run the Python script containing the model implementation:

python ShakespearGPT.py

The program will download the Tiny Shakespeare dataset, train the model, generate text, and save a model checkpoint.

<h3>Model Checkpoint</h3>

After training, the model is saved as:

1000 training response.pt

The checkpoint contains:

  - Model weights

  - Vocabulary size

  - Character-to-integer mapping (stoi)

  - Integer-to-character mapping (itos)

Model architecture configuration

The program then reloads the checkpoint and generates another passage of text.

<h2>Dataset</h2>

This project uses the Tiny Shakespeare dataset from Andrej Karpathy's char-rnn repository.

Source: karpathy/char-rnn

<h2>Credits</h2>

Andrej Karpathy – Inspiration and educational material for building GPT from scratch.

PyTorch – Deep learning framework used to implement and train the model.

<h3>Author</h3>

IoGamZ

GitHub: @IoGamZ
