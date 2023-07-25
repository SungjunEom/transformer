import torch
from torch import nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None):
        self.d_k = q.size()[-1]
        x = torch.matmul(q, torch.transpose(k, -1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            x = x.masked_fill(mask==0,-9e15)
        attention = self.softmax(x)
        x = torch.matmul(attention, v)
        return x, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, input_dim, d_k): # d_k*n_head == d_model, input_dim = [q, k, v], dim(q) = d_model
        super().__init__()
        self.n_head = n_head
        self.input_dim = input_dim
        self.d_k = d_k
        self.qkv_proj = nn.Linear(input_dim,3*d_k*n_head) # d_k*n_head = d_model = input_dim / 3
        self.o_proj = nn.Linear(d_k*n_head,d_k*n_head)
        self.attention = Attention()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, x, mask=None, return_attn=False): # x=[q, k, v], mask=[[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]]
        n_batch, n_seq, _ = x.size()
        x = self.qkv_proj(x) # [Q K V]
        x = x.reshape(n_batch, self.n_head, n_seq, 3*self.d_k)
        # if mask is not None: # mask=[[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]]
            # mask = mask.repeat(1,self.d_k)
        q, k, v = torch.chunk(x,3,dim=-1) # q.size() = (n_batch, self.n_head, n_seq, self.d_k)
        x, attn = self.attention(q, k, v, mask=mask) # x.size() = (n_batch, self.n_head, n_seq, d_v(=self.d_k))
        x = x.permute(0, 2, 1, 3) # x.size() = (n_batch, n_seq, self.n_head, d_v(=self.d_k))
        x = x.reshape(n_batch, n_seq, -1)
        x = self.o_proj(x)
        if return_attn:
            return x, attn
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, ff_dim=2048, dropout=0.0):
        super().__init__()
        assert d_model== (d_k*n_head), "the quality d_k=d_v=d_model/h does not hold."
        self.mha = MultiHeadAttention(n_head, 3*d_model, d_k)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x_tmp = self.norm1(x).to(device) # Applying normalizing first, referring to "On Layer Normalization in the Transformer Architecture"
        x_tmp = x_tmp.repeat(1,1,3)
        attn = self.mha(x_tmp)
        x = x + self.dropout1(attn)

        x_tmp = self.norm2(x)
        x_tmp = self.ff(x_tmp)
        x = x + self.dropout2(x_tmp)

        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, ff_dim=2048, dropout=0.0):
        super().__init__()
        assert d_model== (d_k*n_head), "the quality d_k=d_v=d_model/h does not hold."
        self.masked_mha = MultiHeadAttention(n_head, 3*d_model, d_k)
        self.mha = MultiHeadAttention(n_head, 3*d_model, d_k)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, k, v, mask): # mask=[[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]]
        x_tmp = self.norm1(x)
        x_tmp = x_tmp.repeat(1,1,3)
        attn1 = self.masked_mha(x_tmp, mask=mask)
        x = x + self.dropout1(attn1)

        x_tmp = self.norm2(x)
        x_tmp = torch.cat((x_tmp,k,v), dim=-1)
        attn2 = self.mha(x_tmp, mask=None)
        x = x + self.dropout2(attn2)

        x_tmp = self.norm3(x)
        x_tmp = self.ff(x_tmp)
        x = x + self.dropout3(x_tmp)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_k, dropout=0.0):
        super().__init__()
        assert d_model== (d_k*n_head), "the quality d_k=d_v=d_model/h does not hold."
        self.encoder_modules = nn.ModuleList([TransformerEncoderLayer(n_head, d_model, d_k, ff_dim=2048, dropout=dropout) \
                                              for _ in range(n_layer)])

    def forward(self, x):
        for encoder_layer in self.encoder_modules:
            x = encoder_layer(x)
        return x
    
    def get_attention_maps(self, x):
        attention_maps = []
        for layer in self.encoder_modules:
            _, attn = layer.mha(x,return_attn=True)
            x = layer(x)
            attention_maps.append(attn)
        return attention_maps

class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_k, dropout=0.0):
        super().__init__()
        assert d_model== (d_k*n_head), "the quality d_k=d_v=d_model/h does not hold."
        self.decoder_modules = nn.ModuleList([TransformerDecoderLayer(n_head, d_model, d_k, ff_dim=2048, dropout=dropout) \
                                              for _ in range(n_layer)])

    def forward(self, x, k, v, mask):
        for decoder_layer in self.decoder_modules:
            x = decoder_layer(x, k, v, mask)
        return x

    def get_attention_maps(self, x, k, v, mask):
        attention_maps = []
        for layer in self.decoder_modules:
            _, attn = layer.mha(x, k, v, mask, return_attn=True)
            x = layer(x)
            attention_maps.append(attn)
        return attention_maps

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, dim, seq_len=5000):
        super().__init__()
        self.encoding = torch.zeros(seq_len, dim)
        position_wise = torch.arange(0., seq_len).unsqueeze(1) # position = [[0],[1],[2],...]
        dimension_wise = torch.exp(-torch.arange(0.,dim,2) / dim * math.log(10000))
        self.encoding[:,0::2] = torch.sin(position_wise * dimension_wise)
        self.encoding[:,1::2] = torch.cos(position_wise * dimension_wise)
        self.encoding.requires_grad = False
    
    def forward(self,x):
        x = x + self.encoding[:x.size(1),:].to(device)
        return x

class Transformer(nn.Module):
    def __init__(self, n_layer=6, n_head=8, d_model=512, d_k=64, dropout=0.0, vocab_len=5000):
        super().__init__()
        self.pos_enc = AbsolutePositionalEncoding(d_model)
        self.input_embedding = nn.Linear(vocab_len, d_model)
        self.transformer_encoder = TransformerEncoder(n_layer, n_head, d_model, d_k, dropout)
        self.transformer_decoder = TransformerDecoder(n_layer, n_head, d_model, d_k, dropout)
        self.linear = nn.Linear(d_model, 5000)
        self.softmax = nn.Softmax()

    def forward(self, src, tgt, tgt_mask): # tgt_mask = [[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]]
        src = self.input_embedding(src)
        tgt = self.input_embedding(tgt)
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt) # for translation task, create an output embedding
        x = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, x, x, tgt_mask)
        # output = self.softmax(self.linear(output))
        output = self.linear(output)
        return output