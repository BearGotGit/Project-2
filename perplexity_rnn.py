import math

import torch

from DataHandling import save_score
from DataHandling.Utils import load_losses
from Models import MyRNN

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model: MyRNN = torch.load("saved-models/rnn-04-16-2025_11-38am.pth")
model.eval()

_, validation_losses = load_losses("./rnn-something.pth")
avg_cross_entropy_loss = sum(validation_losses) / len(validation_losses)
perplexity = math.exp(avg_cross_entropy_loss)
print(f"Perplexity: {perplexity:.4f}")

save_score(model, perplexity, "perplexity")
