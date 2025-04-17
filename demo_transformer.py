import sentencepiece as spm
import torch
from Models.transformer import MyTransformer

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = (spm.SentencePieceProcessor())
tokenizer.LoadFromFile("./sentence-piece/big.model")
transformer = MyTransformer(tokenizer=tokenizer, device=device)

print("Transformer: ", transformer.prompt("Do you prefer cats or dogs?"))
print("Transformer: ", transformer.prompt("You are a rapper, a type of person who can sing songs really fast. You \"spit bars\". Spit some: "))
