import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from DataHandling.DataLoader import MyDataset

from Models.rnn import MyRNN

VOCAB_SIZE = 10000
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