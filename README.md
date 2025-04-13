# CSC 7700/4700 Foundational AI Project 2

## Overview

This project focuses on developing a deeper understanding of sequential deep learning models by implementing and evaluating Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Transformer models using PyTorch. The models will be trained on a small language dataset for a generative AI application and evaluated using standard metrics.

## Learning Objectives

- Implement RNN, LSTM, and Transformer-based language models.
- Train models on a small text dataset using subword tokenization.
- Evaluate models using Perplexity (PPL) and BLEU score.
- Generate text using trained models and compare their performance.

## Instructions

### Dataset

- Use a dataset of short text sequences from Project Gutenberg.
- Train a BPE tokenizer with the SentencePiece library.
- Set the vocabulary size to 10,000.

### Tasks

1. **Model Implementation**:

   - Implement the following models:
     - Vanilla RNN-based language model.
     - LSTM-based language model.
     - Transformer-based language model.
   - Include an embedding layer, hidden layers, and a fully connected output layer.
   - Implement a `forward` method for token prediction and sampling.
   - Implement a `prompt` method for autoregressive text generation.

2. **Training**:

   - Use CrossEntropyLoss and AdamW optimizer.
   - Train for up to 30 epochs with early stopping and a learning rate scheduler.
   - Recommended batch size: 128.

3. **Evaluation**:
   - Compute Perplexity (PPL) and BLEU score on the test dataset.
   - Generate text using each model for specific prompts.

#### TODO:

1. Use Ghawaly's fancy pants dataset
2.

### Deliverables

- **Code Repository**:
  - Include all code in a GitHub repository.
  - Provide a README with instructions to run the code.
- **Report**:
  - Abstract summarizing the approach and results.
  - Methodology with model architecture diagrams.
  - Results section with:
    - Training/Validation loss curves.
    - Evaluation metrics table.
    - Generated text for specific prompts.
  - Discussion and conclusion summarizing findings.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the models:
   ```bash
   python train.py
   ```
4. Evaluate the models:
   ```bash
   python evaluate.py
   ```
5. Generate text:
   ```bash
   python generate.py --model <model_name> --prompt "<your_prompt>"
   ```

## Contact

For questions or issues, please contact [Your Name] at [Your Email].
