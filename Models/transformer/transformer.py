import torch.nn as nn
import torch
from typing import List
import sentencepiece as spm
from datetime import datetime
from DataHandling.Utils import sample_token_id_from_probability_distribution


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MyTransformer(nn.Module):
    def __init__(self, tokenizer: spm.SentencePieceProcessor, context_size=512, embedding_dim=512, number_of_heads=2, device=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_len = self.tokenizer.vocab_size()
        pad_id = self.tokenizer.PieceToId("<pad>")
        self.embedding = nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=embedding_dim, padding_idx=pad_id, device=device)
        self.positional_encoding = PositionalEncoding(d_model=embedding_dim, max_len=context_size, dropout=0.1).to(device=device)
        self.transformer: nn.Transformer = nn.Transformer(d_model=embedding_dim, nhead=number_of_heads, device=device, batch_first=True)
        self.out_layer = nn.Linear(embedding_dim, self.vocab_len)
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

    def identifier(self):
        timestamp = datetime.now().strftime("%m-%d-%Y_%I-%M%p").lower()
        return f"transformer-{timestamp}"

    def save_model(self, path=None):
        if path is None:
            path = f"./saved-models/{self.identifier()}.pth"
        torch.save(self, path)

    def _forward_logits(self, src, tgt, temperature=1.0):
        # (batch, seq_len), not (batch, seq_len, vocab_len)
        src_tok = self.embedding(src)
        src_emb = src_tok + self.positional_encoding(src_tok)
        tgt_tok = self.embedding(tgt)
        tgt_emb = tgt_tok + self.positional_encoding(tgt_tok)

        memory = self.transformer.encoder(src_emb)
        # tgt_mask is how PyTorch people like to stop the model from predicting the future
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[-1], device=self.device)
        outs = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.output_proj(outs)

        if temperature is not None and temperature != 0:
            logits = logits / temperature

        return logits

    def forward(self, src_token_ids: List[List[int]], tgt_token_ids: List[List[int]], temperature=1.0, p_sample_threshold=None):
        src_tensor = torch.Tensor(src_token_ids, device=self.device)
        tgt_tensor = torch.Tensor(tgt_token_ids, device=self.device)
        logits = self._forward_logits(src=src_tensor, tgt=tgt_tensor, temperature=temperature)
        if self.training or self.validating:
            return logits
        else:
            probability_distributions = self.softmax(logits)
            new_token_ids = []
            for prob_dist_i in probability_distributions:
                new_token_ids.append(sample_token_id_from_probability_distribution(prob_dist_i, p_sample_threshold))
            return new_token_ids

    def prompt(self, prompt: str, max_completion_length=50, temperature=1.0, p_sample_threshold=None):
        self.eval()

        eos_id = self.tokenizer.PieceToId("<eos>")
        prompt_tokens = self.tokenizer.EncodeAsIds(prompt)
        all_tokens = list(prompt_tokens)

        for i in range(max_completion_length):
            response_tokens = self.forward(tgt_token_ids=all_tokens, temperature=temperature, p_sample_threshold=p_sample_threshold)
            next_token = response_tokens[-1]
            all_tokens.append(next_token)
            if next_token == eos_id:
                break

        generated_tokens = all_tokens[len(prompt_tokens):]
        generated_text = self.tokenizer.DecodeIds(generated_tokens)
        return generated_text
