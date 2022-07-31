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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


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
        self.attention = PreNorm(K, Attention(K, heads=8, dim_head=K//2, dropout=0.1))

    def forward(self, x):
        out1 = self.conv(x).transpose(1, 2).contiguous()
        out2 = self.attention(out1).transpose(1, 2).contiguous()
        out3 = self.shortcut(x)
        return out2+out3


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
        self.decoder = Transformer(dim=N * G, depth=4, heads=8, dim_head=N * G, mlp_dim=N * G * 2, dropout=0.1)
        self.timedis2 = nn.Sequential(
            nn.ReLU(),
            TimeDistributed(nn.Linear(G * K, 2), batch_first=True),
            nn.BatchNorm1d(N),
        )

    def forward(self, R):
        out = self.decoder(R).reshape(R.shape[0], self.G * self.K, -1).transpose(1, 2).contiguous()
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
