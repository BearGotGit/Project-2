# import sentencepiece as spm
# import torch
#
# from Models.lstm import MyLSTM
#
# tokenizer = (spm.SentencePieceProcessor())
# tokenizer.LoadFromFile("./sentence-piece/big.model")
# lstm = MyLSTM(tokenizer, hidden_size=512)
# torch.save(lstm, "./saved-models/lstm-load-test.pth")
# lstm_from_load = torch.load("./saved-models/lstm-load-test.pth")
#
# print("LSTM: ", lstm.prompt("Here's a new prompt: "))
# print("Loaded LSTM: ", lstm_from_load.prompt("Here's a new prompt: "))

from DataHandling.DataLoader import MyDataset
from Models import Trainer
from lstm import MyLSTM
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.LoadFromFile("./sentence-piece/big.model")
model = MyLSTM(tokenizer, 512)

# Dataset
dataset = MyDataset()

trainer = Trainer(model, dataset)
trainer(batch_size=2, epochs=2, )