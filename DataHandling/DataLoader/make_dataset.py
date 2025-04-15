import json
import torch
import sentencepiece as spm
from typing import List, Tuple
from DataHandling.Utils import make_onehots
from DataHandling.Utils import calc_path

import os
from dotenv import load_dotenv

load_dotenv("../../.env")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", 10000))

# Get our data (simple vs. botchan -- TODO: Replace with a ton of Gutenberg books)

serialized_model_proto = open(calc_path('../../sentence-piece/small.model'), 'rb').read()
sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(serialized_model_proto)

def tokenize(words: str) -> List[int]:
    return sp.EncodeAsIds(words)

def make_padded(onehots: torch.Tensor, max_pad: int, pad_token: int = 5, vocab_size: int = 10000) -> torch.Tensor:
    padding_len = max_pad - onehots.size(0)
    # No pad needed
    if padding_len <= 0:
        return onehots

    pad_vector = torch.zeros(vocab_size)
    pad_vector[pad_token] = 1

    stacked_pads = pad_vector.unsqueeze(0).expand(padding_len, -1)
    padded_sample = torch.cat([onehots, stacked_pads], dim=0)

    return padded_sample


# Sample size must be obtained from the longest sample in dataset
# ... Requirement if we want to use tensors, which we do.
def get_data_metadata() -> Tuple[int, int]:
    """
    Get length of longest prompt and completion, number of lines of jsonl
    :return:
    """
    # with open(calc_path("../../data/train.jsonl"), mode="r") as data_file:
    #     max_len = 0
    #     count_jsonl = 0
    #
    #     line = data_file.readline()
    #     while line is not None and len(line) > 0:
    #         line_json = json.loads(line)
    #         prompt = tokenize(line_json["prompt"])
    #         comp = tokenize(line_json["completion"])
    #
    #         lens = len(prompt) + len(comp)
    #         if lens > max_len:
    #             max_len = lens
    #
    #         count_jsonl += 1
    #
    #         line = data_file.readline()
    #
    #     return count_jsonl, max_len
    return 39592, 878

num_samples, sample_size = get_data_metadata()
num_samples = 100

# TODO: num_samples should actually be full dataset

# Samples vary in size: we'll take approach to pad after the sequence ends.
# ... In training, that ensures we don't care about size of sequence (except transformer).
# ... Hence, RNN and LSTM will see one <pad> and quit that sample; transformer will see all of them.
# ... Seems to me a necessary evil of using tensors but should have same runtime as though space-complexity weren't dog shit
seq_tokens = torch.zeros(num_samples, sample_size, VOCAB_SIZE)

# TODO: Doing this is slow. Make the dataset one time and use that from then on
with open(calc_path("../../data/train.jsonl"), mode="r") as data_file:
    data = torch.zeros((num_samples, sample_size, VOCAB_SIZE))

    for i in range(num_samples):
        line = json.loads(data_file.readline())

        prompt = tokenize(line["prompt"])
        comp = tokenize(line["completion"])
        prompt_onehots = make_onehots(torch.tensor(prompt), VOCAB_SIZE)
        comp_onehots = make_onehots(torch.tensor(comp), VOCAB_SIZE)

        # Data concatenation I was talking about; will construct x and y from this.
        data_i_onehots = torch.cat([prompt_onehots, comp_onehots], dim=0)
        data_i = make_padded(data_i_onehots, sample_size, vocab_size=VOCAB_SIZE)

        data[i, :, :] = data_i

    # Save tensor of data for later use
    torch.save(data, calc_path("../../data/onehot-data-small.pth"))