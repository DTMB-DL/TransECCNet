import numpy as np
import matlab.engine
__all__ = ['get_snr', 'get_modem', 'start_matlab']


def get_snr(start, step, end):
    total_num = round((end-start)/step+1)
    return np.linspace(start, end, total_num)


def get_modem(num):
    if num == 4:
        return 'QPSK', np.sqrt(2), 1
    elif num == 16:
        return '16QAM', np.sqrt(10)/2, 17/9


def start_matlab():
    eng = matlab.engine.start_matlab()
    return eng

