import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader

from DataHandling.DataLoader import MyDataset
from DataHandling.Utils import make_onehot, make_onehots, make_not_onehot, calc_path


class MyLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lstm_layer = nn.LSTM(vocab_size, hidden_size=hidden_size)
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax_layer = nn.Softmax(dim=0)

    def _forward_logits(self, input, temp=None):
        """
        Given input sequence of one-hot encoded vectors, return raw logits (NO softmax)
        Used during training for CrossEntropyLoss.
        """
        h_ts, h_f = self.lstm_layer(input)

        out_logits = []
        for h_t in h_ts:
            o_t = self.out_layer(h_t)

            # Optional: temperature scaling
            if temp is not None and temp != 0:
                o_t = o_t / temp

            out_logits.append(o_t)

        return torch.stack(out_logits)

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
    def _sample_i(self, probs, p=None) -> torch.Tensor:
        """
        Given probs for single timestep, sample; p is for top-p sampling
        :param probs:
        :return:
        """
        return make_onehot(torch.argmax(probs).item())

    def _forward_samples(self, input, temp=None) -> torch.Tensor:
        """
        Predict token for each time step of input
        :return:
        """
        probs = self._forward_probs(input, temp)
        encodings = []
        for p_i in probs:
            encodings.append(self._sample_i(p_i))
        return torch.stack(encodings)

    def forward(self, input, temp=None):
        if self.training:
            return self._forward_logits(input, temp)  # For loss
        else:
            return self._forward_samples(input, temp)  # For generation

    def save_model_to(self, path):
        torch.save(self, path)

    def get_model_parameter(self):
        return self.parameters()

    def train(self, data_loader: DataLoader, epochs=30):
        for epoch in range(epochs):
            epoch_loss = 0
            grad_updates = 0

            for batch in data_loader:
                batch = batch.to(device)  # move batch to device if not already

                # For each sample in the batch:
                for sample in batch:
                    for end_of_seq in range(len(sample) - 1):
                        # Make x,y: Input tokens sliced until last prediction token;
                        # ... actual tokens include those to be predicted.
                        x = sample[:end_of_seq]
                        y = sample[1:end_of_seq + 1]

                        # Forward pass
                        preds = model(x)

                        # Prepare targets: Convert one-hot vectors to class indices, which is better for PyTorch cross entropy
                        target_indices = make_not_onehot(y)

                        # Compute loss
                        loss = criterion(preds, target_indices)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        grad_updates += 1
                        epoch_loss += loss.item()

            print(f"Epoch {epoch} - Average Loss: {epoch_loss / grad_updates}")

        self.save_model_to()




# ----- TRAINING -----

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

vocab_size = 10000
hidden_size = 256

model = MyRNN(vocab_size=vocab_size, hidden_size=hidden_size).to(device)

model2 = torch.load("../saved-models/my-rnn.pth")

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# CrossEntropy expects *class indices* (not one-hot), so we'll decode y
criterion = nn.CrossEntropyLoss()

# Load data
my_dataset_loader = MyDataset()
data_loader = DataLoader(my_dataset_loader, batch_size=128, shuffle=True)

# Training
model.train(data_loader)
model.save_model_to("../saved-models/my-rnn.pth")