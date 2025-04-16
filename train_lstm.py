import torch
from DataHandling.DataLoader import MyDataset
from Models import Trainer, MyLSTM
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.LoadFromFile("./sentence-piece/big.model")
model = MyLSTM(tokenizer, 256)

# Dataset
dataset = MyDataset()

trainer = Trainer(model=model, data_loader=dataset, device=device)
print("Started training")
trainer(epochs=30, batch_size=128, verbose=True)
