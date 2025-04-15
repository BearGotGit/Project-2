from torch.utils.data import Dataset, DataLoader
import torch
import sentencepiece as spm

serialized_model_proto = open('./sentence-piece/big.model', 'rb').read()
sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(serialized_model_proto)

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.load('./data/big-padded-tokenized-data.pth')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

