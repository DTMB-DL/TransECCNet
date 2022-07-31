from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from traditional import *
from .quantization import *
__all__ =['Net']


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class SE(nn.Module):
    def __init__(self, Cin):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.compress = nn.Conv1d(Cin, Cin, 1)
        self.excitation = nn.Conv1d(Cin, Cin, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.relu(out)
        out = self.excitation(out)
        out = self.sigmoid(out)
        out = x * out.expand_as(x)
        return out

    def __call__(self, x):
        return self.forward(x)


class ResSEBlock(nn.Module):
    def __init__(self, Cin, Cout, rate=1):
        super(ResSEBlock, self).__init__()
        if Cin != Cout:
            self.shortcut = nn.Sequential(
                nn.BatchNorm1d(Cin),
                nn.ReLU(),
                nn.Conv1d(Cin, Cout, 1)
            )
        else:
            self.shortcut = nn.Identity()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(Cin),
            nn.ReLU(),
            nn.Conv1d(Cin, Cout, 2 * rate + 1, stride=1, padding=rate),
            nn.BatchNorm1d(Cout),
            nn.ReLU(),
            nn.Conv1d(Cout, Cout, 2 * rate + 1, stride=1, padding=rate),
        )
        self.se = SE(Cout)

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x_res = self.se(self.conv(x))  # with SE
        out = x_shortcut + x_res
        return out

    def __call__(self, x):
        return self.forward(x)


class PreAttn(nn.Module):
    def __init__(self, K):
        super(PreAttn, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Conv1d(2, K, 1),
            nn.BatchNorm1d(K),
            nn.ReLU(),
            nn.Conv1d(K, K, 1),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Conv1d(2, K, 1)
        )

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.shortcut(x)
        return out1+out2


class Encoder(nn.Module):
    def __init__(self, N, G):
        self.N = N
        self.G = G
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 128, 11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(2, 32, 1),
            nn.BatchNorm1d(32),
        )
        self.timedis1 = nn.Sequential(
            nn.ReLU(),
            TimeDistributed(nn.Linear(32, 2 * G), True),
            nn.BatchNorm1d(N),
        )

    def forward(self, x):
        out1 = self.encoder(x)
        out1_ori = self.shortcut(x)
        tx = self.timedis1((out1 + out1_ori).reshape(x.shape[0], 32, -1).transpose(1, 2).contiguous()).reshape(
            x.shape[0], 2, -1)
        return tx


class Decoder(nn.Module):
    def __init__(self, N, G, K):
        super(Decoder, self).__init__()
        self.N = N
        self.G = G
        self.K = K
        self.decoder = nn.Sequential(
            nn.Conv1d(K, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResSEBlock(256, 128, 10),
            ResSEBlock(128, 128, 5),
            ResSEBlock(128, 64, 3),
            ResSEBlock(64, 64, 2),
            ResSEBlock(64, 32, 2),
            ResSEBlock(32, 32, 1),
            nn.Conv1d(32, 20, 5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        self.timedis2 = nn.Sequential(
            TimeDistributed(nn.Linear(20 * G, 2), batch_first=True),
        )

    def forward(self, R):
        out = self.decoder(R).reshape(R.shape[0], self.G * 20, -1).transpose(1, 2).contiguous()
        out = self.timedis2(out).reshape(R.shape[0], 2, -1)
        return out


class Net(nn.Module):
    def __init__(self, N, G, K, modem_num, qua_bits):
        super(Net, self).__init__()
        self.G = G
        self.modem_num = modem_num
        self.encoder = Encoder(N, G)
        self.decoder = Decoder(N, G, K)
        self.preattn = PreAttn(K)
        self.quantization = Quantization(qua_bits)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, snr, h_r, h_i, mode, ac_T):
        nx = self.encoder(x)

        xx = torch.cat((nx[:, 0, :], nx[:, 1, :])).reshape(2, x.shape[0], -1)
        tx = pulse_shaping(xx, ISI=self.G, rate=5 * self.G, alpha=1)
        [ry, sigma2] = channel(tx, snr + 10 * math.log10(
            math.log2(self.modem_num) / (self.G * tx.shape[2] / xx.shape[2]) / 3), h_r, h_i, rate=5 * self.G)
        y = matched_filtering(ry, ISI=self.G, rate=5 * self.G, alpha=1)
        r = self.quantization(y, mode, ac_T)
        r = torch.cat((r[0], r[1]), 1).reshape(x.shape[0], 2, -1)

        R = self.preattn(r)
        out = self.decoder(R)
        return out
