import sentencepiece as spm
import torch
from Models import MyLSTM

tokenizer = (spm.SentencePieceProcessor())
tokenizer.LoadFromFile("./sentence-piece/big.model")
lstm: MyLSTM = torch.load("./saved-models/lstm-04-16-2025_11-38am.pth")

common_prompt = "Do you prefer cats or dogs?"
custom_prompt = "A long time ago in a galaxy far far away..."
print(f"\tPrompt: {common_prompt}\n\t{lstm.prompt(common_prompt, max_completion_length=250)}")
print(f"\tPrompt: {custom_prompt}\n\t{lstm.prompt(custom_prompt, max_completion_length=250)}")