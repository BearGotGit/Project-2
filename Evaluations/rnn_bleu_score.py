import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
import sentencepiece as spm
from tqdm import tqdm
from Models.rnn import MyRNN

# Device
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

# Tokenizer
sp = spm.SentencePieceProcessor()
proto = open("../sentence-piece/m.model", 'rb').read()
sp.LoadFromSerializedProto(proto)
space_token_id = sp.PieceToId(" ")

# Precompute one-hot vectors for vocab (speeds up generation loop)
VOCAB_SIZE = sp.GetPieceSize()
onehots = torch.eye(VOCAB_SIZE).to(device)

def make_onehot_fast(idx):
    return onehots[idx]

# Load model
model: MyRNN = torch.load("../saved-models/my-rnn_4-14-25_1108pm_45epochs_vocab10000_hidden512_m-token.pth")
model.eval()

# BLEU
references = []
hypotheses = []
smoothie = SmoothingFunction().method4

# Load JSONL lines into memory
with open("../data/test.jsonl") as f:
    lines = f.readlines()

# Go through data
samples_processed = 0
max_samples = None
for line in tqdm(lines, desc="Evaluating BLEU", unit="sample"):
    if max_samples and samples_processed >= max_samples:
        break
    item = json.loads(line)

    prompt = item["prompt"]
    completion = item["completion"]
    tokenized_prompt = sp.EncodeAsIds(prompt)
    reference = sp.EncodeAsIds(completion)
    references.append([reference])

    model_input = tokenized_prompt.copy()
    if model_input and model_input[-1] == space_token_id:
        print("You got me")

    num_gen = 0
    while model_input[-1] != space_token_id and num_gen < 10:
        onehot_seq = onehots[torch.tensor(model_input)].unsqueeze(1)
        onehot_seq = onehot_seq.to(device)

        with torch.no_grad():
            pred_token_id = model.predict_next_token(onehot_seq.float())

        model_input.append(pred_token_id)
        num_gen += 1

    generated_ids = model_input[len(tokenized_prompt):]
    hypotheses.append(generated_ids)
    samples_processed += 1

# Compute BLEU
bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
print(f"Corpus BLEU Score: {bleu_score:.4f}")
