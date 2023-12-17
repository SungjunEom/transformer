import torch

def get_args():
    
    exp_args = {
        'epoch': 100,
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'batch_size': 4,
        'lr': 1e-3,
        'n_layer': 6,
        'n_head': 8,
        'd_model': 512,
        'd_k': 64,
        'dropout': 0.1,
        'vocab_len': 5000,
    }

    return exp_args