import torch
from setting.setting import device
import torch.nn.functional as F
__all__ = ['channel', 'H_sample']


def channel(input_signals, SNR, H_r=torch.tensor([]), H_i=torch.tensor([]), rate=5 * 4):
    assert input_signals.dim() == 3  # (2, batch_size, L)
    real = input_signals[0]
    imag = input_signals[1]
    S = torch.mean(real ** 2 + imag ** 2)
    snr_linear = 10 ** (SNR / 10.0)
    noise_variance = S / (2 * snr_linear)
    noise = torch.sqrt(noise_variance) * torch.randn_like(input_signals, device=device)
    h_r = torch.zeros((len(H_r) - 1) * rate + 1).to(device)
    h_r[:(len(H_r) - 1) * rate + 1:rate] = H_r
    h_i = torch.zeros((len(H_i) - 1) * rate + 1).to(device)
    h_i[:(len(H_i) - 1) * rate + 1:rate] = H_i

    def conv(x, h):
        y = F.conv1d(x.reshape(input_signals.shape[1], 1, -1), h.flip(dims=[0]).reshape(1, 1, -1),
                     padding=(len(H_i) - 1) * rate)
        y = y.reshape(input_signals.shape[1], -1)[:, :x.shape[1]]
        return y

    out_r = conv(real, h_r) - conv(imag, h_i)
    out_i = conv(imag, h_r) + conv(real, h_i)
    out = torch.stack((out_r, out_i), dim=0) + noise

    return out, noise_variance


def H_sample(TDL_dB, delay):
    powerTDL = 10 ** (torch.tensor(TDL_dB) / 10)
    H_r = torch.zeros(delay[-1] + 1)
    H_i = torch.zeros(delay[-1] + 1)
    H_r[delay] = torch.randn(len(delay)) * torch.sqrt(powerTDL / 2)
    H_i[delay] = torch.randn(len(delay)) * torch.sqrt(powerTDL / 2)
    return H_r, H_i
