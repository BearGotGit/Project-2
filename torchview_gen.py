import torch
from torch import nn
from torchview import draw_graph
from Models import MyRNN
import sentencepiece as spm

# Initialize the tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.LoadFromFile("./sentence-piece/big.model")


class MyRNNTraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_token_ids):
        return self.model._forward_logits(input_token_ids)

# Instantiate the model
model = MyRNN(tokenizer, hidden_size=256)
model.eval()

# Wrapper (GPT-ed my way through this totally)
model = MyRNNTraceWrapper(model)

# Create a dummy input tensor of token IDs
batch_size = 1
seq_len = 10


