from DataHandling.DataLoader import MyDataset
from ..trainer import Trainer
from lstm import MyLSTM
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.LoadFromFile("./sentence-piece/big.model")
model = MyLSTM(tokenizer, 512)

# Dataset
dataset = MyDataset()

trainer = Trainer(model, dataset)
trainer()