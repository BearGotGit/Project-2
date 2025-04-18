# Project 2 – Language Modeling

Berend Grandt (bgrand7@lsu.edu)

## 🔧 Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/BearGotGit/Project-2.git
   cd Project-2
   ```

2. **Set up environment:**
   ```bash
   conda env create -f environment.yml -n berend-grandt-csc-4700-ghawaly-project-2
   conda activate berend-grandt-csc-4700-ghawaly-project-2
   ```

## 🧼 Preprocessing
- These commands assumes the following exist: `./data/train.jsonl`, `./data/test.jsonl`, and `./data/raw/{many books in .txt format}`.

1. **Train Tokenizer:**
   ```bash
   python3 ./DataHandling/Tokenizer/train_tokenizer.py
   ```

2. **Make Tokenized Dataset:**
   ```bash
   python3 ./DataHandling/make_dataset.py
   ```

## 🏋️‍♂️ Training

### Local
Run either:
```bash
python3 train_rnn.py
python3 train_lstm.py
```

### LONI (GPU Cluster)
  ```bash
  sbatch sbatch-files/train_rnn_lstm.sbatch
  ```
- Models/output saved to `./saved-models/` and `./results/`.

## 📈 Evaluation

### BLEU Score
```bash
python3 bleu_rnn.py
python3 bleu_lstm.py
```

### Perplexity
```bash
python3 perplexity_rnn.py
python3 perplexity_lstm.py
```

**LONI version:**
```bash
sbatch sbatch-files/eval_rnn_lstm.sbatch
```
→ Outputs go to `./results/bleu/` and `./results/perplexity/`.

## 📝 Text Generation

### Local
```bash
python3 demo_rnn.py
python3 demo_lstm.py
```

### LONI
```bash
sbatch sbatch-files/use.sbatch
```

