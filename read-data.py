import json
import sentencepiece as spm
from typing import List


# DATA PART

# Assumes that m.model is stored in non-Posix file system.
serialized_model_proto = open('sentence-piece/m.model', 'rb').read()
sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(serialized_model_proto)

def tokenize(words: str):
    return sp.EncodeAsIds(words)

# MODEL PART

import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input):
        return self.rnn(input)
    
# MAIN 

# NOTE: We trained with 2000 using sentencepiece
rnn = RNN(input_size=2000, hidden_size=100, num_layers=16)

# Train

# Initialize model, loss, and optimizer
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = rnn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

def one_hot_encode(input, size_v = 2000):
    one_hot = torch.zeros((1, size_v), dtype=torch.float32)
    one_hot[0][input] = 1.0
    return one_hot

def choose_max(input):
    return torch.argmax(input[0]).item()

def get_token(input):
    m = choose_max(input)
    return one_hot_encode(m)

with open("data/train.jsonl", mode="r") as data_file:
    for i in range(4):
        line = json.loads(data_file.readline())
        
        # each is arr of ints:
        prompt = tokenize(line["prompt"])
        comp = tokenize(line["completion"])

        # tokenize
        print(f"{i}: \n\tprompt: {prompt}\n\tcompletion: {comp}")

        # model prediction
        # for i in range (16):
        out = rnn.forward(one_hot_encode(prompt[0]))
        print("Out from rnn: ", get_token(out))

        