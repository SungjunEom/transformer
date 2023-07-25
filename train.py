from model import *
from dataset import *
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

# Sentencepice BOS_ID=1, EOS_ID=2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 4
    lr = 1e-3
    chat_data = ChatbotDataset('dataset/Q.txt', 'dataset/A.txt', 'test.vocab')
    max_len = chat_data.get_max_seq()
    vocab_size = chat_data.get_vocab_size()
    model = Transformer(3, 8, 512, 64, 0.1, vocab_size).to(device)
    train_dataloader = DataLoader(chat_data, batch_size)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    tgt_mask = torch.tril(torch.ones(max_len, max_len)).to(device)


    for _ in range(epochs):
        for batch, (X, Y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            X = F.one_hot(X.long(), num_classes=vocab_size).float()
            Y = F.one_hot(Y.long(), num_classes=vocab_size).float()
            num = Y.shape[0]
            Y_out = Y[:,1:,:]
            Y = Y[:,:-1,:]
            Y = torch.cat((Y,torch.zeros(5000).unsqueeze(0).unsqueeze(0).repeat(num,1,1).to(device)),\
                        dim=-2).to(device)
            Y_out = torch.cat((Y_out,torch.zeros(5000).unsqueeze(0).unsqueeze(0).repeat(num,1,1).to(device)),\
                               dim=-2).to(device)
            pred = model(X, Y, tgt_mask)
            Y_out = torch.argmax(Y_out.int(), dim=-1)
            # pred = torch.permute(pred, (0, 2, 1))
            pred = pred.reshape(-1,vocab_size)
            Y_out = pred.reshape(-1,vocab_size)
            loss = loss_fn(pred, Y_out)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1)
                print(f'loss: {loss:>7f} [{current:>5d}]')
    
    torch.save(model.state_dict(),'model.ckpt')

if __name__ == '__main__':
    main()