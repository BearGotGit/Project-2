import torch

def make_onehot (token: int, vocab_size = 10000):
    """
    Given a token, return its onehot encoding
    :param tokens:
    :return:
    """
    assert token < vocab_size
    onehot_v = torch.zeros((vocab_size,))
    onehot_v[token] = 1
    return onehot_v

def make_not_onehot (onehot: torch.Tensor):
    """
    Given a ohe vector, return its token encoding
    :param onehot:
    :return:
    """
    return torch.argmax(onehot, dim=-1)

def make_onehots (tokens: torch.Tensor, vocab_size = 10000):
    """
    Given 1d tensor of tokens, return tensor with axis=1 of one hots
    :param tokens:
    :return:
    """
    encodings = torch.zeros((len(tokens), vocab_size))
    for i, t in enumerate(tokens):
        ohe = make_onehot(t.item(), vocab_size)
        encodings[i][:] = ohe

    return encodings
