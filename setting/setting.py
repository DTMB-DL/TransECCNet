import torch
import numpy as np
__all__ = ['device']
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
np.random.seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
