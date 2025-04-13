import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input):
        return self.rnn(input)
    
# MAIN 

import sentencepiece as spm
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
sp_model_path = os.getenv("SP_MODEL")

sp = spm.SentencePieceProcessor()
sp.LoadFromFile(sp_model_path)

b = 1
s = 1
# f = sp.vocab_size()
f = 2

rnn = RNN(input_size=f, hidden_size=f)

input = torch.randn(b, s, f)

out, hidden = rnn.forward(input)

print("Shapes: ", input.shape, out, hidden)

# Train

# Initialize model, loss, and optimizer
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = rnn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 