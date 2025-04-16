import json

import torch
from DataHandling.DataLoader import MyDataset
from DataHandling.Utils.save_metrics import save_losses
from Models import Trainer, MyRNN
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.LoadFromFile("./sentence-piece/big.model")
model = MyRNN(tokenizer, 256)

# Dataset
dataset = MyDataset()

trainer = Trainer(model=model, data_loader=dataset, device=device)
print("Started training")
trainer(epochs=30, batch_size=128, verbose=True)
# Losses saved in "results/training-metrics"