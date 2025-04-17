import torch.nn as nn
import torch
from typing import List
import sentencepiece as spm
from datetime import datetime

from DataHandling.Utils import make_one_hot_vectors, sample_token_id_from_probability_distribution

class MyLSTM(nn.Module):
    def __init__(self, tokenizer: spm.SentencePieceProcessor, hidden_size, device=None):
        """
        :param tokenizer: tokenizer used for training and inference
        :param hidden_size: hidden_size of LSTM
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_len = self.tokenizer.vocab_size()
        self.lstm_layer = nn.LSTM(self.vocab_len, hidden_size=hidden_size, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, self.vocab_len)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.validating = False
        self.device = device
    
    def to(self, device):
        self.device = device
        return super().to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.validating = False  # optionally reset

    def validate(self):
        super().train(False)  # puts model in eval mode
        self.validating = True

    def _forward_logits(self, input_token_ids, temperature=None) -> torch.Tensor:
        """
        Given input tensor of token_ids, return raw logits (NO softmax);
        Apply temperature if needed;
        Used during training for CrossEntropyLoss.
        """
        one_hots = make_one_hot_vectors(input_token_ids, vocab_len=self.vocab_len).to(self.device)
        h_ts, h_f = self.lstm_layer(one_hots)
        logits = self.out_layer(h_ts)
        if temperature is not None and temperature != 0:
            logits = logits / temperature
        return logits

    def _forward_samples(self, input_token_ids, temperature=None, p_sample_threshold=None) -> List[int]:
        """
        Predict token for each time step of input
        :return:
        """
        logits = self._forward_logits(input_token_ids, temperature)
        probability_dists = self.softmax_layer(logits)
        new_token_ids = []
        for prob_dist_i in probability_dists:
            new_token_ids.append(sample_token_id_from_probability_distribution(prob_dist_i, p_sample_threshold))
        return new_token_ids

    def forward(self, input_token_ids, temperature=None, p_sample_threshold=None):
        """
        Forward returns logits during training, so cross entropy works;
        During inference, it returns list of output tokens
        :param input_token_ids: List[int] (unbatched) or List[List[int]] ("batched")
        :param temperature:
        :param p_sample_threshold:
        :return:
        """
        if self.training or self.validating:
            # Returns tensors of batched logits
            return self._forward_logits(input_token_ids, temperature)
        else:
            # Your normal, run-of-the-mill forward
            # Returns list of token ids
            return self._forward_samples(input_token_ids, temperature=temperature, p_sample_threshold=p_sample_threshold)

    def identifier(self):
        timestamp = datetime.now().strftime("%m-%d-%Y_%I-%M%p").lower()
        return f"lstm-{timestamp}"

    def save_model(self, path=None):
        if path is None:
            path = f"./saved-models/{self.identifier()}.pth"
        torch.save(self, path)

    def prompt(self, prompt: str, max_completion_length = 50, temperature=1.0, p_sample_threshold=None):
        self.eval()

        eos_id = self.tokenizer.PieceToId("<eos>")
        prompt_tokens = self.tokenizer.EncodeAsIds(prompt)
        all_tokens = list(prompt_tokens)

        for i in range(max_completion_length):
            response_tokens = self.forward(input_token_ids=all_tokens, temperature=temperature, p_sample_threshold=p_sample_threshold)
            next_token = response_tokens[-1]
            all_tokens.append(next_token)
            if next_token == eos_id:
                break

        generated_tokens = all_tokens[len(prompt_tokens):]
        generated_text = self.tokenizer.DecodeIds(generated_tokens)
        return generated_text
