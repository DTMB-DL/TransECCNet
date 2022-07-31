import torch
from setting.setting import device
from traditional.r_filter import *
__all__ = ["pulse_shaping"]


def pulse_shaping(input_symbols, alpha=1, delay=5, rate=5, rs=8000000, gain=1, ISI=1):
    assert input_symbols.dim() == 3 and input_symbols.shape[0] == 2  # (2, Batch_size, L)
    fs = rs*rate
    phi_I = gain * input_symbols[0]
    B = len(phi_I)
    L = len(phi_I[0]) // ISI
    isi = rate // ISI
    phi_Q = gain * input_symbols[1]
    phi_pulse_I = torch.zeros(B, rate*L+2*rate*delay, device=device).float()
    phi_pulse_I[:, 0:rate*L:isi] = phi_I
    phi_trans_I = r_filter(phi_pulse_I,alpha, delay, rate, 1 / fs)
    phi_pulse_Q = torch.zeros(B, rate * L + 2 * rate * delay, device=device).float()
    phi_pulse_Q[:, 0:rate * L:isi] = phi_Q
    phi_trans_Q = r_filter(phi_pulse_Q,alpha, delay, rate, 1 / fs)
    phi_trans = torch.stack((phi_trans_I, phi_trans_Q), dim=0)
    return phi_trans

