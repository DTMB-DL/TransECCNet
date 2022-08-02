import matlab.engine
import numpy as np
import matlab
import math
from traditional import *
from setting.setting import *
from tools.parse import *
from tools.utils import *
from models.quantization import *
import os
import torch
import importlib


def baseline(args, bers, mode):
    eng = matlab.engine.start_matlab()
    eng.addpath('./traditional')
    SNR = get_snr(args.snr_start, args.snr_step, args.snr_end)  # range of SNR
    modem, base, beta = get_modem(args.modem_num)
    B = bers.shape[0]
    BER = []
    ofdm_sim = OFDM_SIM(K=64, CP=16, P=16, mu=int(np.log2(args.modem_num)), modem=modem, eng=eng)
    L = ofdm_sim.K - ofdm_sim.P

    for snr in SNR:
        snrtmp = snr
        ber = 0.0
        for idx in range(B):
            s = bers[idx]
            ss = eng.lteTurboEncode(matlab.int8(s.tolist()))
            x = eng.lteSymbolModulate(ss, modem)
            x = torch.tensor(x).reshape(-1)

            # split into dataCarriers
            L_cnt = math.ceil(len(x) / L)
            z = torch.zeros_like(x).to(device)
            for idx_L in range(L_cnt):
                data_L = torch.zeros(L, dtype=torch.complex64)
                data_L[0:min(L, len(x)-idx_L*L)] = x[idx_L*L:min((idx_L+1)*L, len(x))]
                OFDM_data = ofdm_sim.OFDM_symbol(data_L)
                OFDM_time = ofdm_sim.IDFT(OFDM_data)
                OFDM_withCP = ofdm_sim.addCP(OFDM_time)
                OFDM_TX = OFDM_withCP

                if idx_L % 1 == 0:  # OFDM symbols / frame
                   H_r, H_i = H_sample(args.TDL_dB, args.delay)

                xx = torch.stack((OFDM_TX.real, OFDM_TX.imag), axis=0).reshape(2, 1, -1).to(device)
                tx = pulse_shaping(xx, ISI=1, rate=5 * args.G, alpha=1)
                [ry, sigma2] = channel(tx, snr + 10 * np.log10(L / (ofdm_sim.K + ofdm_sim.CP)) + 10 * np.log10(
                    np.log2(args.modem_num) / (tx.shape[2] / xx.shape[2]) / 3), rate=5 * args.G, H_r=H_r, H_i=H_i)
                r = matched_filtering(ry, ISI=1, rate=5 * args.G, alpha=1)
                z_L = r[0][0] + 1j * r[1][0]
                OFDM_RX_noCP = ofdm_sim.removeCP(z_L.cpu())
                OFDM_freq = ofdm_sim.DFT(OFDM_RX_noCP)

                H_LS = ofdm_sim.est_LS(OFDM_freq)
                if mode == 'OFDM_LS':
                    OFDM_LS = OFDM_freq / H_LS
                    OFDM_recovered_LS = ofdm_sim.get_payload(OFDM_LS)
                    z[idx_L * L:min((idx_L + 1) * L, len(x))] = OFDM_recovered_LS[0:min(L, len(x) - idx_L * L)]
                elif mode == 'OFDM_MMSE':
                    H = torch.fft.fftshift(torch.fft.fft(H_r+1j*H_i, n=ofdm_sim.K)).reshape(-1, 1)
                    Rhh = H @ H.conj().T
                    H_MMSE = ofdm_sim.est_MMSE(H_LS, Rhh, snr, beta=beta)
                    OFDM_MMSE = OFDM_freq / H_MMSE
                    OFDM_recovered_MMSE = ofdm_sim.get_payload(OFDM_MMSE)
                    z[idx_L * L:min((idx_L + 1) * L, len(x))] = OFDM_recovered_MMSE[0:min(L, len(x) - idx_L * L)]

            s2 = eng.lteSymbolDemodulate(matlab.double(z.tolist(), is_complex=True), modem, 'Soft')
            s_hat = eng.lteTurboDecode(s2)
            s_hat = np.array(s_hat).reshape(-1)
            ber += np.sum(s != s_hat[:args.len])

        ber /= B * args.len
        BER.append(ber)
        print("[snr] %f, [ber] is %f" % (snrtmp, ber))

    eng.quit()
    dir = './data/'+args.channel+'/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    np.savez(dir + mode + '_' + str(args.modem_num), snr=SNR, ber=np.array(BER))
    print("%s.npz is saved" % (dir + mode + '_' + str(args.modem_num)))


def cnn_test(args, bers, model):
    eng = matlab.engine.start_matlab()
    eng.addpath('./traditional')
    SNR = get_snr(args.snr_start, args.snr_step, args.snr_end)  # range of SNR
    modem, base, beta = get_modem(args.modem_num)
    B = bers.shape[0]
    N = args.N
    BER = []
    autoencoder = model(args.N, args.G, args.K, args.modem_num, args.qua_bits)
    autoencoder.load_state_dict(torch.load(args.model_path, map_location=device))
    autoencoder.to(device)
    autoencoder.eval()

    for snr in SNR:
        snrtmp = snr
        ber = 0.0
        with torch.no_grad():
            for idx in range(B):
                s = bers[idx]
                ss = eng.lteTurboEncode(matlab.int8(s.tolist()))
                x = eng.lteSymbolModulate(ss, modem)
                x = torch.tensor(x).reshape(-1)
                x = torch.stack((x.real, x.imag), axis=0).reshape(2, -1).to(device)
                L = x.shape[1]
                maxcnt = math.ceil(L / N) * N

                s1 = torch.cat((x, x[:, :maxcnt-L]), dim=1)
                s2 = torch.stack((s1[0].reshape(-1, N), s1[1].reshape(-1, N)), dim=1)

                frame = 4
                r = torch.tensor([]).to(device)
                for frame_idx in range(0, s2.shape[0], frame):  # each frame experiences the same fading
                    H_r, H_i = H_sample(args.TDL_dB, args.delay)
                    r_frame = autoencoder(s2[frame_idx:frame_idx+frame], snr, H_r, H_i, 'infer', 1)
                    r = torch.cat((r, r_frame), dim=0)

                z = r[:, 0, :].reshape(-1)[:L] + 1j * r[:, 1, :].reshape(-1)[:L]
                yy = eng.lteSymbolDemodulate(matlab.double(z.tolist(), is_complex=True), modem, 'Soft')
                s_hat = eng.lteTurboDecode(yy)
                s_hat = np.array(s_hat).reshape(-1)
                ber += np.sum(s != s_hat[:args.len])
        ber /= B * args.len
        BER.append(ber)
        print("[snr] %f, [ber] is %f" % (snrtmp, ber))

    eng.quit()
    dir = './data/' + args.channel + '/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    np.savez(dir + args.curve + '_' + str(args.modem_num)+ '_' + str(args.qua_bits), snr=SNR, ber=np.array(BER))
    print("%s.npz is saved" % (dir + args.curve + '_' + str(args.modem_num)+ '_' + str(args.qua_bits)))


if __name__ == '__main__':
    args = get_args()
    bers = np.random.randint(0, 2, [args.ber_len, args.len])
    if args.curve == 'OFDM_LS' or args.curve == 'OFDM_MMSE':
        baseline(args, bers, args.curve)
    else:
        model = getattr(importlib.import_module("models." + args.curve), 'Net')
        cnn_test(args, bers, model)
