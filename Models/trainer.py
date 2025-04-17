from typing import Tuple, List
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, random_split

from DataHandling.Utils.save_metrics import save_losses
from Models import MyLSTM, MyRNN
import os
import matplotlib.pyplot as plt

# Utility function to make plots
def make_plots(model, training_losses: List[float], validation_losses: List[float], epochs: int):
    # Create directory if it doesn't exist
    os.makedirs("results/training-plots", exist_ok=True)
    # Default paths useful for tracking specific models, will have same name as model file (eg. "lstm-04-14-2025_09-35pm")
    file_identifier = model.identifier()
    plot_path = os.path.join("results/training-plots", f"{file_identifier}.png")

    # Generate and save loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Batch" if len(training_losses) > epochs else "Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved training plot to {plot_path}")


class Trainer:
    def __init__(self, model: MyRNN, data_loader: Dataset, device):
        # Training RNN, LSTM, and Transformer is really similar, so this one class does the work
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def _optimizer_part(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # (batch_len, seq_len-1, vocab_len)
        y_hat: torch.Tensor = self.model(x)

        loss = self.loss_criterion(
            y_hat.reshape(-1, y_hat.size(-1)),
            y.reshape(-1).long()
        )

        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _for_loop_part(self, is_training=True, transformer=False, verbose=False) -> List[float]:
        data_loader = self.train_loader if is_training else self.validation_loader
        self.model.train() if is_training else self.model.validate()

        new_losses = []
        if verbose and self.model.training:
            print("Next training batch started")
        else:
            print("Next validation batch started")
        for batch in data_loader:
            # (batch_len, seq_len)
            batch: torch.Tensor = batch.to(self.device)
            x = batch[:, :-1]
            x = x.to(self.device)
            y = batch[:, 1:]
            y = y.to(self.device)

            if transformer:
                # FIXME: window_len = transformer.context_len
                window_len = 512
                window_steps = x.shape[-1] - window_len
                loss_same_batch_many_windows = 0
                for i in range(window_steps):
                    window_x = x[:, i:i + window_len]
                    window_y = y[:, i:i + window_len]
                    loss_same_batch_many_windows += self._optimizer_part(window_x, window_y)
                loss = loss_same_batch_many_windows / window_steps

            # RNN or LSTM, there's no context len
            else:
                loss = self._optimizer_part(x, y)

            new_losses.append(loss)

        return new_losses

    def _train(self) -> List[float]:
        return self._for_loop_part(is_training=True, verbose=True)

    def _validate(self) -> List[float]:
        with torch.no_grad():
            return self._for_loop_part(is_training=False, verbose=True)

    def __call__(self, epochs=30, batch_size=128, percent_of_data_training = 0.8, shuffle_training=True, model_save_path=None, verbose=False) -> Tuple[List[float], List[float]]:
        """

        :param epochs:
        :param batch_size:
        :param percent_of_data_training:
        :param shuffle_training:
        :param model_save_path:
        :param verbose:
        :return: Tuple of (training_losses, validation_losses), where losses are aggregated per epoch
        """
        # It would be a pain to do all that training and be unable to save! Woe unto them!
        assert hasattr(self.model, 'save_model') and callable(getattr(self.model, 'save_model'))
        assert hasattr(self.model, 'identifier') and callable(getattr(self.model, 'identifier'))

        self.model.train()

        # Device time!
        self.model = self.model.to(self.device)

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

        # Number training and validation samples varies, so we average loss per batch
        training_losses = []
        validation_losses = []
        training_batch_losses = []
        validation_batch_losses = []
        for e in range(epochs):
            training_batch_losses += self._train()
            training_losses.append(sum(training_batch_losses) / len(training_batch_losses))
            validation_batch_losses += self._validate()
            validation_losses.append(sum(validation_batch_losses) / len(validation_batch_losses))

            if verbose:
                print(f"Epoch {e + 1}/{epochs} — Train Loss: {sum(training_losses) / len(training_losses):.4f} | Val Loss: {sum(validation_losses) / len(validation_losses):.4f}")

        make_plots(self.model, training_losses=training_losses, validation_losses=validation_losses, epochs=epochs)
        self.model.save_model()
        save_losses(self.model, training_losses, validation_losses)

        return training_losses, validation_losses
