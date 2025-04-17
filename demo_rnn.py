import sentencepiece as spm
import torch
from Models import MyRNN

tokenizer = (spm.SentencePieceProcessor())
tokenizer.LoadFromFile("./sentence-piece/big.model")
rnn: MyRNN = torch.load("./saved-models/rnn-04-16-2025_09-54pm")

common_prompt = "Do you prefer cats or dogs?"
custom_prompt = "Ralof: Hey, you. You're finally awake."
print(f"\tPrompt: {common_prompt}\n\t{rnn.prompt(common_prompt, max_completion_length=250)}")
print(f"\tPrompt: {custom_prompt}\n\t{rnn.prompt(custom_prompt, max_completion_length=250)}")
