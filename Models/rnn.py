import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from DataHandling.DataLoader import MyDataset
from DataHandling.Utils import make_onehot, make_onehots, make_not_onehot, calc_path

import os
from dotenv import load_dotenv

load_dotenv("../.env")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", 10000))

class MyRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.rnn_layer = nn.RNN(vocab_size, hidden_size=hidden_size)
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax_layer = nn.Softmax(dim=0)

    def _forward_logits(self, input, temp=None):
        """
        Given input sequence of one-hot encoded vectors, return raw logits (NO softmax)
        Used during training for CrossEntropyLoss.
        """
        h_ts, h_f = self.rnn_layer(input)

        out_logits = self.out_layer(h_ts)

        if temp is not None and temp != 0:
            out_logits = out_logits / temp

        return out_logits

    def _forward_probs(self, input, temp=None) -> torch.Tensor:
        """
        Given an input sequence of one hot encoded vectors, return prob distributions over token set
        :return:
        """
        # # Forward on RNN
        # h_ts, h_f = self.rnn_layer(input)
        #
        # out_probs = []
        # for h_t in h_ts:
        #     o_t = self.out_layer(h_t)
        #
        #     # applies temp to logits before softmax
        #     if temp is not None and temp != 0:
        #         e_raise = torch.exp(o_t / temp)
        #         e_sum = torch.sum(e_raise)
        #         o_t = e_raise / e_sum
        #     p_t = self.softmax_layer(o_t)
        #     out_probs.append(p_t)
        #
        # rnn_out_probs = torch.stack(out_probs)
        # return rnn_out_probs

        logits = self._forward_logits(input, temp)
        probs = self.softmax_layer(logits)
        return probs

    # TODO: Add top-p sampling logic
    def _sample_i(self, probs, p=None) -> int:
        """
        Given probs for single timestep, sample; p is for top-p sampling
        :param probs:
        :return:
        """
        return torch.argmax(probs, dim=-1).item()

    def _forward_samples(self, input, temp=None) -> torch.Tensor:
        """
        Predict token for each time step of input
        :return:
        """
        probs = self._forward_probs(input, temp)
        encodings = []
        for p_i in probs:
            encodings.append(self._sample_i(p_i))
        return torch.tensor(encodings)

    def forward(self, input, temp=None):
        if self.training:
            return self._forward_logits(input, temp)  # For loss
        else:
            return self._forward_samples(input, temp)  # For generation

    def save_model_to(self, path):
        torch.save(self, path)

    def get_model_parameter(self):
        return self.parameters()

    def train_model(self, criterion: CrossEntropyLoss, optimizer: AdamW, device, data_loader: DataLoader, epochs=30, pad_token: int = 5):
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            grad_updates = 0

            for batch in data_loader:
                batch = batch.to(device)

                # (num samples, sample lengths, vocab size) in batch
                batch_size, seq_len, vocab_size = batch.shape

                # Prepare x and y
                x = batch[:, :-1, :]
                y = batch[:, 1:, :]

                x = x.permute(1, 0, 2)  # RNN expects (seq_len, batch_size, vocab_size)
                preds = self(x)  # preds: (seq_len-1, batch_size, vocab_size)
                preds = preds.permute(1, 0, 2)  # (batch_size, seq_len-1, vocab_size) for loss

                # Prepare targets
                target_indices = make_not_onehot(y).to(device)  # (batch_size, seq_len-1)

                # Compute loss
                loss = criterion(
                    preds.reshape(-1, preds.size(-1)),  # (batch_size * seq_len-1, vocab_size)
                    target_indices.reshape(-1)  # (batch_size * seq_len-1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                grad_updates += 1
                epoch_loss += loss.item()

            print(f"Epoch {epoch} - Average Loss: {epoch_loss / grad_updates}")

    def predict_next_token(self, input_ids, temperature=1.0):
        self.eval()
        return self.forward(input_ids, temp=temperature)[-1]

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temp=1.0, device='cpu'):
        """
        Generate full output sequence, given prompt; copied from Dr. Ghawaly's notes

        :param tokenizer:
        :param prompt:
        :param max_length:
        :param eos_token_id:
        :param temp:
        :param device:
        :return:
        """
        self.eval()
#       Encode prompt to token ids
#       Convert ID to tensor, move to device memory, add batch dim
#       Store gen ids
#       Loop over until max length
#           Exit early if <eos>
#           Keep track
#           Input to next step is new this input in time and prev hidden state
#       Decode token IDs into tokens
