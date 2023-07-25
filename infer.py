from model import *
from dataset import *
import torch
import sentencepiece as spm

def main():
    chat_data = ChatbotDataset('dataset/Q.txt', 'dataset/A.txt', 'test.vocab')
    max_len = chat_data.get_max_seq()
    vocab_size = chat_data.get_vocab_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Transformer(3, 8, 512, 64, 0.1, vocab_size).to(device)
    model.load_state_dict(torch.load('model.ckpt'))
    model.eval()
    model.to(device)

    tgt_mask = torch.tril(torch.ones(max_len, max_len)).to(device)
    sp = spm.SentencePieceProcessor()
    sp.load('test.model')
    input_text = sp.encode_as_ids('테스트다 이 자식아')
    decode_text = sp.encode_as_ids('아')
    input_text += [0] * (5000 - len(input_text))
    decode_text += [0] * (5000 - len(decode_text))
    input_text = torch.Tensor(input_text).to(device).unsqueeze(0)
    decode_text = torch.Tensor(decode_text).to(device).unsqueeze(0)
    pred = model(input_text, decode_text, tgt_mask)
    pred = torch.argmax(pred, dim=-1)
    pred = pred.squeeze(0)
    print(pred)
    pred = sp.DecodeIds(pred)
    print(pred)


if __name__ == '__main__':
    main()