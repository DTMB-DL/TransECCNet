# README
This is the implementation of paper “Deep Learning based Pilot-free Transmission: Error Correction Coding for Low-resolution Reception under Time-varying Channels”.

## 1、Requirements
This project involves the joint programming of `MATLAB` and `Python`, and thus package `matlab.engine` is need for
`getdata.py` (generate data) and `system_main.py` (test bit error rate).

Find the path like `MATLAB\R2017b\extern\engines\python` in your MATLAB installation path, and execute the command `python setup.py install`.

#### **other requirements**:
```
pytorch >= 1.7.1
torchvision >= 0.8.2
python >= 3.6
```

## 2、Tree Arrangement
``` 
home
├── data
│   ├── flat/
│   ├── channelA/
│   ├── channelB/
├── models 
│   ├── *Net.py  (all networks) 
│   ├── quantization.py
├── setting 
│   ├── settings.py           (random seed, gpu)
├── tools
│   ├── logger.py
│   ├── parse.py
│   ├── utils.py
├── traditional
│   ├── ofdm.py               (ofdm simulation: LS/MMSE)
│   ├── channel.py
│   ├── match_filtering.py
│   ├── pulse_shaping.py
│   ├── r_filter.py           (Root sign rising cosine function)
├── train.py
├── system_main.py
├── Pilot_*_*                 (Pilot generated by ofdm.py)
...
```

## 3、Pipline
### A. Train Network

#### Command:
```
python train.py -curve TFECCNet_SEAttn -channel channelB -snr 35.0 -qua_bits 1 -lr 1e-4 -modem_num 4
```

#### Modifiable Parameters:
* batch_size: batch size
* G: G-fold symbold extension
* N: Block of symbols
* K: preamplification dimensions
* modem_num: modulation order (4 for QPSK, and 16 for 16QAM)
* unit_T: unit increment of soft quantization
* lr: learning rate of Autoencoder
* qua_bits: quantization bits
* snr: Eb/n0 when training
* curve: select the network type
* channel: select the channel type

### B. BER measurement
#### Command:
```
python system_main.py -curve TFECCNet_SEAttn -channel channelB -qua_bits 1 -ber_len 50 -modem_num 4 -snr_start 0 -snr_end 35 -snr_step 5
```

#### Modifiable Parameters:
* ber_len: ber_len × len is final testing length
* batch_size: batch size
* G: G-fold symbold extension
* N: Block of symbols
* K: preamplification dimensions
* modem_num: modulation order (4 for QPSK, and 16 for 16QAM)
* qua_bits: quantization bits
* snr_start: start of Eb/n0 when testing
* snr_step: step of Eb/n0 when testing
* snr_end: end of Eb/n0 when testing
* curve: select the network type
* channel: select the channel type

#### Note: 
If $curve=='OFDM_LS/OFDM_MMSE/TFECCNet_general/ECCNet_general', the `qua_bits` will not take effect.