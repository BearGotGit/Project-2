import json
import torch
import sentencepiece as spm

serialized_model_proto = open('./sentence-piece/big.model', 'rb').read()
sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(serialized_model_proto)
vocab_len = sp.vocab_size()
pad_id = sp.PieceToId("<pad>")

dataset_len = 39952
seq_len = max_seq_len = 872

# Samples vary in size, so we'll pad after sample's sequence ends
primed_data = torch.zeros(dataset_len, max_seq_len)

with open("./data/train.jsonl", mode="r") as data_file:
    line = data_file.readline()
    i = 0
    while line is not None and len(line) > 0:
        jsonl = json.loads(line)
        prompt = sp.EncodeAsIds(jsonl["prompt"])
        comp = sp.EncodeAsIds(jsonl["completion"])
        complete = prompt + comp

        num_pad = max_seq_len - len(complete)
        complete.extend([pad_id] * num_pad)
        padded = complete

        primed_data[i, :] = torch.tensor(padded).float()
        i += 1

        line = data_file.readline()

    # Save tensor of primed data for later use
    torch.save(primed_data, "./data/big-padded-tokenized-data.pth")

# def get_data_metadata():
#     """
#     Get number of lines of jsonl, length of longest prompt and completion
#     :return: (num_lines, longest_combined)
#     """
#     # Run when file changes:
#     with open("./data/train.jsonl", mode="r") as data_file:
#         max_len = 0
#         count_jsonl = 0
#
#         line = data_file.readline()
#         while line is not None and len(line) > 0:
#             line_json = json.loads(line)
#             prompt = sp.EncodeAsIds(line_json["prompt"])
#             comp = sp.EncodeAsIds(line_json["completion"])
#
#             lens = len(prompt) + len(comp)
#             if lens > max_len:
#                 max_len = lens
#
#             count_jsonl += 1
#
#             line = data_file.readline()
#
#         return count_jsonl, max_len
