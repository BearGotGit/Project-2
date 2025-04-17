import torch
from nltk.translate.bleu_score import corpus_bleu
import json
from DataHandling.Utils import save_score
from Models import MyRNN

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_base = "losses_rnn-04-16-2025_09-54pm"
model: MyRNN = torch.load(f"saved-models/{file_base}.pth")
model.eval()

with open("data/test.jsonl") as f:
    line = f.readline()

    refs = []
    guesses = []

    while line and len(line) > 0:
        jsonl = json.loads(line)
        prompt = jsonl["prompt"]
        comp = jsonl["completion"]
        complete = prompt + comp

        tokens = model.tokenizer.EncodeAsIds(complete)
        # tokens = [0, 1, 2, 3, 4, 5, 6]
        input_token_ids = tokens[:-1]
        output_token_ids = tokens[1:]

        guessed_token_ids = model._forward_samples(input_token_ids)
        # guessed_token_ids = [0, 1, 1, 3, 5, 4, 4]

        refs.append([output_token_ids])
        guesses.append(guessed_token_ids)

        line = f.readline()

    bleu_score = corpus_bleu(list_of_references=refs, hypotheses=guesses)
    print(f"Corpus BLEU Score: {bleu_score:.4f}")
    save_score(file_base, bleu_score, "bleu")