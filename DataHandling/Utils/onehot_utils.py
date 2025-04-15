import torch
import torch.nn.functional as F
from typing import List

def make_onehot (token: int, vocab_len):
    """
    Given a token, return its onehot encoding
    :param tokens:
    :return:
    """
    assert token < vocab_len
    onehot_v = torch.zeros((vocab_len,))
    onehot_v[token] = 1
    return onehot_v

def make_not_onehot (onehot: torch.Tensor):
    """
    Given a ohe vector, return its token encoding
    :param onehot:
    :return:
    """
    return torch.argmax(onehot, dim=-1)

def make_one_hot_vectors(tokens: List[int], vocab_len: int) -> torch.Tensor:
    """
    Given 1d tensor of tokens, return tensor with axis=1 of one hots
    :param tokens:
    :param vocab_len
    :return:
    """
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    return F.one_hot(token_tensor, num_classes=vocab_len).float()
