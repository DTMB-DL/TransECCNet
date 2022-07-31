import torch
import torch.nn as nn
import math
from setting.setting import device
__all__ = ['quantize', 'Quantization']


def Sigmoid(x, T):
    return 1 / (1 + torch.exp(-T * x))


class Soft_Quantization(torch.autograd.Function):  # y=α∑Sigmoid(T(βx-b))-o

    @staticmethod
    def forward(self, x, T, bias, offset, alpha=1.0, beta=1.0):
        self.x = x
        self.T = T
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        y = alpha*Sigmoid(beta*x-bias, T) - offset
        return y

    @staticmethod
    def backward(self, grad_output):
        tmp = Sigmoid(self.beta * self.x - self.bias, self.T)
        temp = tmp * (1 - tmp) * self.alpha * self.T * self.beta
        grad_input = temp * grad_output
        return grad_input, None, None, None, None, None


soft_quantization = Soft_Quantization.apply


def quantize(y, bits):  # uniform quantization
    maxval = 4 / math.sqrt(10)
    y = y.clip(-maxval, maxval)
    k = 2 ** bits
    r = y.clone()
    interval = 2 * maxval / k
    for i in range(k):
        if i != k-1:
            r[torch.where((y >= -maxval + i * interval) & (y < -maxval + (i + 1) * interval))] = i
        else:
            r[torch.where((y >= -maxval + i * interval) & (y <= -maxval + (i + 1) * interval))] = i
    rr = r.clone()
    K = k / 2
    before_val = torch.arange(1-2*K, 2*K, 2).to(device)
    after_val = before_val / before_val[-1]
    for i in range(k):
        rr[r == i] = after_val[i]
    return rr


class Quantization(nn.Module):
    def __init__(self, bits):
        super(Quantization, self).__init__()
        self.bits = bits
        self.beta = 1
        self.T = 1
        k = 2 ** bits
        K = k / 2
        before_val = torch.arange(1 - 2 * K, 2 * K, 2).to(device)
        after_val = before_val / before_val[-1]
        self.alpha = after_val[1]-after_val[0]
        self.offsets = []
        for i in range(k-1):
            self.offsets.append(self.alpha*(k/2-i-1/2))
        self.offsets = torch.tensor(self.offsets).to(device)

    def forward(self, input, quanmode, T=1):
        assert quanmode in ['train', 'infer']
        self.T = T
        if quanmode == 'train':
            maxval = 4 / math.sqrt(10)
            input = input.clip(-maxval, maxval)
            k = 2 ** self.bits
            interval = 2 * maxval / k
            bias = []
            for i in range(k-1):
                bias.append((-k/2+1+i)*interval)
            bias = torch.tensor(bias).to(device)

            data = torch.zeros(input.shape).float().to(device)
            for idx in range(len(bias)):
                bia = bias[idx]
                offset = self.offsets[idx]
                data += soft_quantization(input, self.T, bia, offset, self.alpha, self.beta)
            return data
        else:
            return quantize(input, self.bits)

    def __call__(self, input, quanmode, T=1):
        return self.forward(input, quanmode, T)
