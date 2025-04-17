from Models import MyLSTM
from Models.rnn import MyRNN

import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.LoadFromFile("./sentence-piece/big.model")
model = MyLSTM(tokenizer, 256)

print("Sanity checks:")
print(model.prompt("Here's a prompt. What happens?", temperature=-100, p_sample_threshold=1000))
print(model.prompt("Here's a prompt. What happens?", temperature=-100, p_sample_threshold=0))