
import math
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, data, block_size, stride=None):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.stride = stride if stride is not None else block_size
    
    def __len__(self):
        return (len(self.data) - 1) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start:start + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
