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
        self.length = len(self.q)
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
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    sp = spm.SentencePieceProcessor()
    sp.load('test.model')
    batch_size=4
    chat_data = ChatbotDataset('dataset/Q.txt', 'dataset/A.txt', 'test.vocab')
    train_dataloader = DataLoader(chat_data, batch_size=batch_size, shuffle=True)
    print(f'{chat_data.q.iloc[11822,0]}')
    print(f'{chat_data.a.iloc[11822,0]}')
    for batch, (X, Y) in enumerate(train_dataloader):
        # print(batch, X.shape, Y.shape)
        pass
    print(type(X[-1]))
    print(sp.DecodePieces(X[-1].tolist()))
    print(sp.DecodePieces(Y[-1].tolist()))
    print(chat_data.__len__())
