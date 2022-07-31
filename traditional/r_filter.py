import numpy as np
import torch
import torch.nn.functional as F
from setting.setting import device
__all__ = ['r_filter']


def firrcos(N, Fc, R, Fs, DELAY):
    fs = torch.tensor(Fs, device=device).float()
    fc = torch.tensor(Fc, device=device).float()
    R = torch.tensor(R, device=device).float()
    L = N + 1
    delay = DELAY
    n = (torch.arange(0, L, device=device).float()-delay) / fs
    eps = torch.tensor(2.2204e-16, device=device)
    b = torch.zeros(L, device=device).float()
    ind1 = (n == 0)
    if torch.any(ind1):
        b[ind1] = -torch.sqrt(2 * fc) / (np.pi * fs) * (np.pi * (R-1) - 4 * R)
    ind2 = (torch.abs(torch.abs(8*R*fc*n)-1.0) < torch.sqrt(eps))
    if torch.any(ind2):
        b[ind2] = torch.sqrt(2*fc)/(2*np.pi*fs) * (np.pi*(R+1)*torch.sin(np.pi*(R+1)/(4*R))-4*R*torch.sin(np.pi*(R-1)/(4*R))+np.pi*(R-1)*torch.cos(np.pi*(R-1)/(4*R)))
    ind = torch.zeros(L) == 0
    ind[ind1] = False
    ind[ind2] = False
    nind = n[ind]
    b[ind] = -4*R/fs*(torch.cos((1+R)*2*np.pi*fc*nind)+torch.sin((1-R)*2*np.pi*fc*nind)/(8*R*fc*nind))/(np.pi*torch.sqrt(1/(2*fc))*((8*R*fc*nind)**2-1))
    b = torch.sqrt(2*fc)*b
    return b


def rcosfir(r, N_T, rate, T):
    time_T = torch.arange(0, N_T, 1/rate).float()
    cal_time = time_T * T
    b = firrcos(rate*N_T*2, 1/(2*T), r, rate/T, N_T*rate)*torch.sqrt(torch.tensor(rate).float())
    tim = cal_time[1] - cal_time[0]
    return [b, tim]


def r_filter(x, r, N_T, rate, T):  # input symbols
    [b, tim] = rcosfir(r, N_T, rate, T)
    L = len(x[0])
    B = len(x)
    lb = len(b)
    tmp = F.conv1d(torch.reshape(x.float(), (B, 1, L)), torch.reshape(b, (1, 1, lb)), padding=lb-1)
    return torch.reshape(tmp[:, :, :L], (B, L))




