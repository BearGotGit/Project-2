from torch.utils.data import Dataset, DataLoader
import torch

import sentencepiece as spm

from DataHandling.Utils import calc_path

serialized_model_proto = open(calc_path('../../sentence-piece/m.model'), 'rb').read()
sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(serialized_model_proto)

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.load(calc_path('../../data/onehot-data.pth'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

