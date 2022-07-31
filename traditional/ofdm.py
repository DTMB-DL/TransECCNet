import torch
import numpy as np
import matlab.engine
import os
from setting.setting import *
from scipy.interpolate import interp1d

__all__ = ['OFDM_SIM']


class OFDM_SIM():
    def __init__(self, K=64, CP=16, P=16, mu=4, modem='16QAM', eng=None):
        super(OFDM_SIM, self).__init__()
        self.K = K
        self.CP = CP
        self.P = P
        self.allCarriers = np.arange(K)  # subcarriers from 0 to K-1
        self.pilotCarriers = self.allCarriers[::K//P]  # every (K/P)th is a pilot carrier
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)
        self.allCarriers = torch.tensor(self.allCarriers, dtype=torch.int64)
        self.pilotCarriers = torch.tensor(self.pilotCarriers, dtype=torch.int64)
        self.dataCarriers = torch.tensor(self.dataCarriers, dtype=torch.int64)
        self.mu = mu  # modulation: mu bits/symbol
        self.modem = modem
        self.eng = eng

        Pilot_file_name = 'Pilot_' + str(P) + "_" + modem
        if os.path.isfile(Pilot_file_name):
            print('Load Training Pilots txt')
            bits = np.loadtxt(Pilot_file_name, delimiter=',')
        else:
            bits = np.random.binomial(n=1, p=0.5, size=(P * mu,))
            np.savetxt(Pilot_file_name, bits, delimiter=',')
        bits = torch.tensor(bits)

        self.pilotValue = self.Modulation(bits)  # np.ones(P)

    def Modulation(self, bits):
        bits = bits.type(torch.int8)
        return torch.tensor(self.eng.lteSymbolModulate(matlab.int8(bits.tolist()), self.modem)).reshape(-1)

    def Demodulation(self, symbols):
        return torch.tensor(self.eng.lteSymbolDemodulate(matlab.double(symbols.tolist(), is_complex=True), self.modem, 'Soft')).reshape(-1)

    def OFDM_symbol(self, Data):
        symbol = torch.zeros(self.K, dtype=torch.complex64)
        symbol[self.pilotCarriers] = self.pilotValue
        symbol[self.dataCarriers] = Data
        return symbol

    def IDFT(self, OFDM_data):
        return torch.fft.ifft(torch.fft.fftshift(OFDM_data))

    def addCP(self, OFDM_time):
        cp = OFDM_time[-self.CP:]               # take the last CP samples ...
        return torch.hstack([cp, OFDM_time])  # ... and add them to the beginning

    def removeCP(self, signal):
        return signal[self.CP:(self.CP+self.K)]

    def DFT(self, OFDM_RX):
        return torch.fft.fftshift(torch.fft.fft(OFDM_RX))

    def interpolate_compensation(self, H_hat):
        # 前面有空缺
        H_com = H_hat
        index_com = self.pilotCarriers
        if self.pilotCarriers[0] > 0:
            slope = (H_hat[1]-H_hat[0])/(self.pilotCarriers[1]-self.pilotCarriers[0])
            head_com = H_hat[0]-slope*self.pilotCarriers[0]
            index_com = torch.cat((torch.tensor([0]), index_com), dim=0)
            H_com = torch.cat((torch.tensor([head_com]), H_com), dim=0)
        if self.pilotCarriers[-1] < self.K - 1:
            slope = (H_hat[-1] - H_hat[-2]) / (self.pilotCarriers[-1] - self.pilotCarriers[-2])
            tail_com = H_hat[-1] + slope * (self.K-1-self.pilotCarriers[-1])
            index_com = torch.cat((index_com, torch.tensor([self.K-1])), dim=0)
            H_com = torch.cat((H_com, torch.tensor([tail_com])), dim=0)
        return H_com, index_com

    def est_LS(self, OFDM):
        pilots = OFDM[self.pilotCarriers]
        H_hat = pilots / self.pilotValue  # pilot channel matrix
        H_com, index_com = self.interpolate_compensation(H_hat)
        f_ls = interp1d(index_com, H_com)
        H_inter = f_ls(torch.arange(0, self.K))
        return H_inter

    def est_MMSE(self, H_LS, Rhh, SNRdb, beta=1):  # beta=(1,17/9,2.6857) for QPSk,16QAM,64QAM
        SNR = 10**(SNRdb/10)
        W = Rhh @ torch.inverse(Rhh+torch.eye(self.K)*beta/SNR)
        H_inter = W @ H_LS
        return H_inter

    def get_payload(self, equalized):
        return equalized[self.dataCarriers]






