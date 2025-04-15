import torch
import torch.nn as nn
import math
import json
import sentencepiece as spm
from tqdm import tqdm

from Models.rnn import MyRNN

# Setup
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

# Tokenizer
sp = spm.SentencePieceProcessor()
proto = open("../sentence-piece/m.model", 'rb').read()
sp.LoadFromSerializedProto(proto)
VOCAB_SIZE = sp.GetPieceSize()
space_token_id = sp.PieceToId(" ")

# Precompute one-hot vectors
onehots = torch.eye(VOCAB_SIZE).to(device)
def make_onehot_fast(idx): return onehots[idx]

# Model
model: MyRNN = torch.load("../saved-models/my-rnn_4-14-25_1108pm_45epochs_vocab10000_hidden512_m-token.pth")
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss(reduction='sum')

# Track loss and token count
total_loss = 0.0
total_tokens = 0

# Load data and compute perplexity
with open("../data/test.jsonl") as f:
    lines = f.readlines()

# Go through data
samples_processed = 0
max_samples = None
for line in tqdm(lines, desc="Computing Perplexity"):
    if max_samples and samples_processed >= max_samples:
        break
    item = json.loads(line)
    prompt = item["prompt"]
    completion = item["completion"]

    input_ids = sp.EncodeAsIds(prompt)
    target_ids = sp.EncodeAsIds(completion)

    full_input = input_ids + target_ids[:-1]
    full_target = target_ids

    onehot_seq = torch.stack([make_onehot_fast(t) for t in full_input]).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model._forward_logits(onehot_seq.float())  # (seq_len, 1, vocab_size)

    logits = logits.view(-1, VOCAB_SIZE)
    logits = logits[-len(full_target):]  # take last N where N = len(full_target)

    targets = torch.tensor(full_target, dtype=torch.long, device=device)

    if logits.shape[0] != targets.shape[0]:
        continue  # just in case

    loss = criterion(logits, targets)

    total_loss += loss.item()
    total_tokens += len(full_target)

    samples_processed += 1

# Final perplexity
avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)
print(f"Perplexity: {perplexity:.4f}")
