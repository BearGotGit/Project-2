import sentencepiece as spm
import torch
from Models import MyRNN

tokenizer = (spm.SentencePieceProcessor())
tokenizer.LoadFromFile("./sentence-piece/big.model")
rnn: MyRNN = torch.load("./saved-models/rnn-04-17-2025_01-39am.pth")

prompts = [
    "Do you prefer cats or dogs?",
    "A long time ago in a galaxy far far away...",
    "Ralof: Hey, you. You're finally awake.",
    "Explain why the sky is blue.",
    "Write a haiku about spaghetti.",
    "Once upon a time, in a forest full of secrets,",
    "Translate this to pirate-speak: Hello, friend!",
    "What happens when you divide by zero?",
    "List 3 reasons why robots might revolt.",
    "Generate a recipe for quantum soup.",
    "The year is 3020. Humanity has just discovered...",
    "Why did the chicken cross the road?",
    "Tell me a joke that makes no sense.",
    "Complete this sentence: The AI said to the human,",
    "What if gravity suddenly reversed?",
    "My favorite programming language is",
    "To be or not to be, that is the...",
    "Elementary, my dear...",
    "I think, therefore...",
    "I'm going to make him an offer he can't...",
    "You can't handle the..."
]
for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    print("Response:", rnn.prompt(prompt, max_completion_length=250))
    print("\n")
