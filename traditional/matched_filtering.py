import torch
from traditional.r_filter import *
from setting.setting import device
__all__ = ["matched_filtering"]


def matched_filtering(input_symbols, alpha=1, delay=5, rate=5, rs=8000000, gain=1, ISI=1):
    assert input_symbols.dim() == 3 and input_symbols.shape[0] == 2  # (2, Batch_size, L)
    fs = rs*rate
    isi = rate // ISI  # for ISI
    phi_received_I = input_symbols[0]
    B = len(phi_received_I)
    phi_matched_I = r_filter(phi_received_I,alpha, delay, rate, 1 / fs)
    cosphi = phi_matched_I[:, 2 * rate * delay: len(phi_matched_I[0]):isi]
    phi_received_Q = input_symbols[1]
    phi_matched_Q = r_filter(phi_received_Q,alpha, delay, rate, 1 / fs)
    sinphi = phi_matched_Q[:, 2 * rate * delay: len(phi_matched_Q[0]):isi]
    output = torch.zeros(2, B, (len(phi_received_I[0]) // rate - 2 * delay)*ISI, device=device)
    output[0] = cosphi/gain
    output[1] = sinphi/gain
    return output


if __name__ == "__main__":
    pass
