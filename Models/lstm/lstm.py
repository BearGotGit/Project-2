import torch.nn as nn
import torch
from typing import List
import sentencepiece as spm
from datetime import datetime

from DataHandling.Utils import make_not_onehot, make_one_hot_vectors

class MyLSTM(nn.Module):
    def __init__(self, tokenizer: spm.SentencePieceProcessor, hidden_size):
        """
        :param tokenizer: tokenizer used for training and inference
        :param hidden_size: hidden_size of LSTM
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size()
        self.lstm_layer = nn.LSTM(self.vocab_size, hidden_size=hidden_size, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, self.vocab_size)
        self.softmax_layer = nn.Softmax(dim=-1)

    def _sample_from_dist(self, prob_dist, p_sample_threshold=None) -> int:
        """
        Sample from a probability distribution with optional top-p (nucleus) sampling.
        If p_sample_threshold is None, falls back to greedy (argmax).
        """
        if p_sample_threshold is None:
            return torch.argmax(prob_dist, dim=-1).item()

        sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True)
        # It keeps dim
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff_mask = cumulative_probs <= p_sample_threshold
        # If first token were higher than p_sample_threshold, would be false by mask. Force true in all cases
        cutoff_mask[0] = True
        filtered_probs = sorted_probs[cutoff_mask]
        filtered_indices = sorted_indices[cutoff_mask]
        sampled_index = torch.multinomial(filtered_probs, 1).item()

        return filtered_indices[sampled_index].item()

    def _forward_logits(self, input_token_ids, temperature=None) -> torch.Tensor:
        """
        Given input tensor of token_ids, return raw logits (NO softmax);
        Apply temperature if needed;
        Used during training for CrossEntropyLoss.
        """
        one_hots = make_one_hot_vectors(input_token_ids, vocab_len=self.tokenizer.vocab_size())
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
            new_token_ids.append(self._sample_from_dist(prob_dist_i, p_sample_threshold))
        return new_token_ids

    def forward(self, input_token_ids: List[int], temperature=None, p_sample_threshold=None):
        """
        Forward returns logits during training, so cross entropy works;
        During inference, it returns list of output tokens
        :param input_token_ids:
        :param temperature:
        :param p_sample_threshold:
        :return:
        """
        if self.training:
            return self._forward_logits(input_token_ids, temperature)
        else:
            return self._forward_samples(input_token_ids, temperature, p_sample_threshold)

    def save_model(self, path=None):
        if path is None:
            timestamp = datetime.now().strftime("%m-%d-%Y_%I-%M%p").lower()
            path = f"./saved-models/lstm-{timestamp}.pth"
        torch.save(self, path)

    def prompt(self, prompt: str, max_completion_length = 50, temperature=1.0, p_sample_threshold=None):
        self.eval()

        eos_id = self.tokenizer.PieceToId("<eos>")
        prompt_tokens = self.tokenizer.EncodeAsIds(prompt)
        all_tokens = prompt_tokens

        for i in range(max_completion_length):
            response_tokens = self.forward(input_token_ids=all_tokens, temperature=temperature, p_sample_threshold=p_sample_threshold)
            next_token = response_tokens[-1]
            all_tokens.append(next_token)

        generated_tokens = all_tokens[len(prompt_tokens):]
        generated_text = self.tokenizer.DecodeIds(generated_tokens)
        return generated_text