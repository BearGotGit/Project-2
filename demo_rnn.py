import torch
import sentencepiece as spm
from DataHandling.Utils import make_one_hot_vectors
from Models.rnn import MyRNN

# Set device
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

# Load the saved model
model_path = "saved-models/my-rnn_4-14-25_1108pm_45epochs_vocab10000_hidden512_m-token.pth"
model: MyRNN = torch.load(model_path, map_location=device)
model.eval()

# Load SentencePiece tokenizer
model_proto = open("sentence-piece/m.model", 'rb').read()
sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(model_proto)

# Define utility to create one-hot vectors
VOCAB_SIZE = 10000

# Specify prompt text and tokenize
prompt_text = "Which do you prefer? Dogs or cats?"
prompt_ids = sp.EncodeAsIds(prompt_text)

# Convert token ids to one-hot tensor and adjust shape for RNN: (seq_len, batch_size, vocab_size)
prompt_tensor = make_one_hot_vectors(prompt_ids, vocab_len=VOCAB_SIZE).unsqueeze(1).to(device)

# Autoregressive generation
generated_ids = []
max_length = 50

# Start with the prompt tensor as input
input_seq = prompt_tensor

for _ in range(max_length):
    # When model in eval mode, predict_next_token
    predicted_token = model.predict_next_token(input_seq, temperature=1.0)
    # Stop generation if <eos> token is produced
    if predicted_token == sp.PieceToId("<eos>"):
        break
    generated_ids.append(predicted_token)
    # Prepare next input by appending the generated token converted to one-hot
    next_token_onehot = make_one_hot_vectors([predicted_token], VOCAB_SIZE).unsqueeze(0).unsqueeze(1).to(device)
    input_seq = torch.cat([input_seq, next_token_onehot], dim=0)

# Decode generated sequence to text
generated_text = sp.DecodeIds([tok.item() for tok in generated_ids])

print("Prompt:", prompt_text)
print("Generated:", generated_text)