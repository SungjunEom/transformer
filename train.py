from model import *
from dataset import *
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from scheduler import CosineWarmupScheduler
from hparams import get_args
import wandb

# Sentencepice BOS_ID=1, EOS_ID=2

exp_args = get_args()
device = exp_args['device']
epochs = exp_args['epoch']
batch_size = exp_args['batch_size']
lr = exp_args['lr']
n_layer = exp_args['n_layer']
n_head = exp_args['n_head']
d_model = exp_args['d_model']
d_k = exp_args['d_k']
dropout = exp_args['dropout']
vocab_len = exp_args['vocab_len']

wandb.init(
    project="my-little-transformer",

    config={
        "learning_rate": lr,
        "epochs": epochs,
    }
)


def main():
    chat_data = ChatbotDataset('dataset/Q.txt', 'dataset/A.txt', 'test.vocab')
    train_dataloader = DataLoader(chat_data, batch_size=4, shuffle=True)
    max_len = chat_data.get_max_seq()
    vocab_size = chat_data.get_vocab_size()
    model = Transformer(n_layer,n_head, d_model, d_k, dropout, vocab_len).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # lr_scheduler = CosineWarmupScheduler(optimizer,100,2000)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

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
            pred = pred.reshape(-1,vocab_size)
            Y_out = pred.reshape(-1,vocab_size)
            loss = loss_fn(pred, Y_out)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1)
                print(f'loss: {loss:>7f} [{current:>5d}]')

            wandb.log({"loss": loss})
        
        lr_scheduler.step()
    
    torch.save(model.state_dict(),'model.ckpt')

if __name__ == '__main__':
    main()