import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fix_cuda():
    torch.multiprocessing.set_start_method('spawn')  # good solution !!!!