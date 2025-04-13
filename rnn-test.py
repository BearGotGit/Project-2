import json

import torch.nn as nn
import torch

# Load tokenizer
import sentencepiece as spm

# Assumes that m.model is stored in non-Posix file system.
serialized_model_proto = open("sentence-piece/m.model", 'rb').read()

sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(serialized_model_proto)



#

batch_size = 1
seq_len = 256
out_len = 16
vocab_size = 2000

hidden_size = 100



# Loading data
def onehot_encode(vocab_size, tokens):
    onehot_v = torch.zeros((1, len(tokens), vocab_size))
    onehot_v = onehot_v.scatter_(1, tokens.view(-1, 1), 1)
    return onehot_v


def make_data_batch ():

    # Load the data
    with open("data/train.jsonl") as data_file:


        batch_x = torch.zeros((batch_size, seq_len, vocab_size))
        batch_y = torch.zeros((batch_size, out_len, vocab_size))

        prompt_tensors = []
        completion_tensors = []

        for _ in range(batch_size):
            line = json.loads(data_file.readline())
            if line == None:
                break

            prompt = line["prompt"]
            completion = line["completion"]

            prompt_tokens = sp.EncodeAsIds(prompt)
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.int32)
            onehot_v = onehot_encode(vocab_size, prompt_tensor)
            prompt_tensors.append(onehot_v)

            completion_tokens = sp.EncodeAsIds(completion)
            completion_tensor = torch.tensor(completion_tokens, dtype=torch.int32)
            onehot_v = onehot_encode(vocab_size, completion_tensor)
            completion_tensors.append(onehot_v)

        batch_x = torch.stack(prompt_tensors)  # Shape: (batch_size, seq_len)
        batch_y = torch.stack(completion_tensors)

        yield batch_x, batch_y


val = make_data_batch()
for v in val:
    print(v)

# Actual model training


rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
linear = nn.Linear(hidden_size, vocab_size)  # Predict one output from hidden state

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(list(rnn.parameters()) + list(linear.parameters()), lr=0.001)

x = torch.randn(batch_size, seq_len, vocab_size)
y = torch.randn(batch_size, vocab_size)  # Random targets

epochs = hidden_size
for epoch in range(epochs):
    out, hidden = rnn(x)
    prob_dist = linear(hidden[-1])  # Use last hidden state

    norm_prob_dist = torch.softmax(prob_dist, dim=1)

    # # TODO: Inference make one hot encoded vecs
    # pred_i = torch.argmax(prob_dist, dim=1)
    # pred = torch.zeros((5, vocab_size))
    # pred = pred.scatter_(1, pred_i.view(-1, 1), 1)

    # Train
    loss = criterion(norm_prob_dist, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Finished training!")
