import sentencepiece as spm
import torch

tokenizer = (spm.SentencePieceProcessor())
tokenizer.LoadFromFile("./sentence-piece/big.model")
rnn = torch.load("./saved-models/rnn-04-16-2025_09-54pm")

common_prompt = "Do you prefer cats or dogs?"
custom_prompt = "Ralof: Hey, you. You're finally awake."
print(f"\tPrompt: {common_prompt}\n\t{rnn.prompt(common_prompt)}")
print(f"\tPrompt: {custom_prompt}\n\t{rnn.prompt(custom_prompt)}")
