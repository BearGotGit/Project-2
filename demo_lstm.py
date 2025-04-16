import sentencepiece as spm
import torch

tokenizer = (spm.SentencePieceProcessor())
tokenizer.LoadFromFile("./sentence-piece/big.model")
lstm = torch.load("./saved-models/lstm-04-16-2025_11-38am.pth")

print("LSTM: ", lstm.prompt("Do you prefer cats or dogs?"))
print("LSTM: ", lstm.prompt("You are a rapper, a type of person who can sing songs really fast. You \"spit bars\". Spit some: "))
