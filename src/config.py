import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)


def fix_cuda():
    torch.multiprocessing.set_start_method('spawn')  # good solution !!!!
