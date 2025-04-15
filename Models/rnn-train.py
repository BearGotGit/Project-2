import math

import torch.nn as nn
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchgen.api.translate import out_tensor_ctype

from DataHandling.DataLoader import MyDataset
from DataHandling.Utils import make_onehot, make_onehots, make_not_onehot, calc_path

import os
from dotenv import load_dotenv

from Models.rnn import MyRNN

load_dotenv("../.env")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", 10000))
HIDDEN_SIZE = 512

# ----- TRAINING -----

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')


model = MyRNN(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE).to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# CrossEntropy expects class indices, not onehot encode
criterion = nn.CrossEntropyLoss(
    ignore_index=5  # index of <pad> token
)

# Load data
my_dataset_loader = MyDataset()
data_loader = DataLoader(my_dataset_loader, batch_size=128, shuffle=True)

# Training
model.train_model(epochs=45, criterion=criterion, data_loader=data_loader, optimizer=optimizer, device=device)
model.save_model_to("../saved-models/my-rnn_4-14-25_1108pm_45epochs_vocab10000_hidden512_m-token.pth")