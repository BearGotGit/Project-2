from typing import Tuple, List

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, random_split

from Models import MyLSTM

class Trainer:
    def __init__(self, model: MyLSTM, data_loader: Dataset):
        # Training RNN, LSTM, and Transformer is really similar, so this one class does the work
        self.model = model
        self.data_loader = data_loader

    def _for_loop_part(self, is_training=True) -> List[float]:
        data_loader = self.train_loader if is_training else self.validation_loader
        self.model.train() if is_training else self.model.eval()

        new_losses = []
        for batch in data_loader:
            # (batch_len, seq_len)
            batch: torch.Tensor = batch.to(self.device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            # (batch_len, seq_len-1, vocab_len)
            y_hat: torch.Tensor = self.model(x)

            loss = self.loss_criterion(
                y_hat.reshape(-1, y_hat.size(-1)),
                y.reshape(-1)
            )

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            new_losses.append(loss.item())

        return new_losses

    def _train(self) -> List[float]:
        return self._for_loop_part(is_training=True)

    def _validate(self) -> List[float]:
        with torch.no_grad():
            return self._for_loop_part(is_training=False)

    def __call__(self, epochs=30, batch_size=128, percent_of_data_training = 0.8, shuffle_training=True, save_path=None, verbose=False):
        """
        :param epochs:
        :param batch_size:
        :param shuffle_training:
        :param save_path:
        :param verbose:
        :return:
        """
        # It would be a pain to do all that training and be unable to save! Woe unto them!
        assert hasattr(self.model, 'save_model') and callable(getattr(self.model, 'save_model'))

        self.model.train()

        # FIXME: Switch to 'cuda' before put on qb2
        self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

        # Split data into train/validate
        train_size = int(percent_of_data_training * len(self.data_loader))
        val_size = len(self.data_loader) - train_size
        train_dataset, val_dataset = random_split(self.data_loader, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_training)
        self.validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loss_criterion = nn.CrossEntropyLoss(
            ignore_index=self.model.tokenizer.PieceToId("<pad>")
        )

        training_losses = []
        validation_losses = []
        for e in range(epochs):
            training_losses += self._train()
            validation_losses += self._validate()

            if verbose:
                print(f"Epoch {e + 1}/{epochs} — Train Loss: {sum(training_losses) / len(training_losses):.4f} | Val Loss: {sum(validation_losses) / len(validation_losses):.4f}")

        # Saves model to default location
        self.model.save_model()