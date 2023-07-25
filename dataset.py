import pandas as pd
from torch.utils.data import Dataset
import sentencepiece as spm

class ChatbotDataset(Dataset):
    def __init__(self, q, a):
        self.q = pd.read_csv(q, header=None)
        self.a = pd.read_csv(a, header=None)
        self.length = len(q)
    
    def __len__(self):
        return len()

    def __getitem__(self, idx):
        return self.q.iloc[idx,0], self.a.iloc[idx,0]