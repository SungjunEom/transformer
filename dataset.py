import pandas as pd
from torch.utils.data import Dataset
import sentencepiece as spm
import csv
import torch

class ChatbotDataset(Dataset):
    def __init__(self, q, a, vocab):
        self.q = pd.read_csv(q, header=None)
        self.a = pd.read_csv(a, header=None)
        self.vocab_len = len(pd.read_csv(vocab, header=None, sep='\t', quoting=csv.QUOTE_NONE))
        self.length = len(q)
        self.max_seq = 512
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('test.model')
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        q = self.sp.encode_as_ids(self.q.iloc[idx,0])
        a = self.sp.encode_as_ids(self.a.iloc[idx,0])
        q += [0] * (self.max_seq - len(q))
        a += [0] * (self.max_seq - len(a))
        return torch.Tensor(q).int(), torch.Tensor(a).int()
    
    def get_vocab_size(self):
        return self.vocab_len
    
    def get_max_seq(self):
        return self.max_seq