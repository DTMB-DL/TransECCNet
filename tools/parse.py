import argparse
__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser()

    '''=============================  generate data opt  ==============================='''
    parser.add_argument('-ber_len', type=int, default=50, help='ber_len * len is final testing length.')
    parser.add_argument('-len', type=int, default=6144, help='total length of each piece.')

    '''============================= common opt ==============================='''
    parser.add_argument('-batch_size', type=int, default=100, help='batch size')

    parser.add_argument('-G', type=int, default=4, help='G in front layer')
    parser.add_argument('-N', type=int, default=16, help='length of block(symbols)')
    parser.add_argument('-K', type=int, default=40, help='channel preamplification')
    parser.add_argument('-modem_num', type=int, default=16, help='number of modulation order. decide ISI.')

    '''=========================== training opt =============================='''
    parser.add_argument('-unit_T', type=int, default=1, help='increasing number of T for each epoch')
    parser.add_argument('-lr', type=float, default=1e-4, help='init learning rate')
    parser.add_argument('-qua_bits', type=int, default=1, help='quantization bits for Network')

    parser.add_argument('-snr', type=float, default=35.0, help='E_b/n_0 for training.')
    parser.add_argument('-snr_start', type=float, default=0.0, help='start value of Eb/n0 for testing')
    parser.add_argument('-snr_step', type=float, default=5, help='step value of Eb/n0 for testing')
    parser.add_argument('-snr_end', type=float, default=35.0, help='end value of Eb/n0 for testing')

    '''=========================== choose main opt =============================='''
    parser.add_argument('-curve',
                        choices=['OFDM_LS', 'OFDM_MMSE', 'TFECCNet_SEAttn', 'TFECCNet_BiAttn', 'TFECCNet_NoAttn',
                                 'TFECCNet_general', 'ECCNet_SEAttn', 'ECCNet_BiAttn', 'ECCNet_NoAttn',
                                 'ECCNet_general'], default='OFDM_LS', help='for main')
    parser.add_argument('-channel', choices=['flat', 'channelA', 'channelB'], default='channelB', help='channel type')

    args = parser.parse_args()
    if args.channel == 'flat':
        args.TDL_dB = [0]
        args.delay = [0]
    elif args.channel == 'channelA':
        args.TDL_dB = [0, -8, -17, -21, -25]
        args.delay = [0, 3, 5, 6, 8]
    elif args.channel == 'channelB':
        args.TDL_dB = [0, 0, 0, 0, 0]
        args.delay = [0, 1, 2, 3, 4]
    args.model_path = './data/'+args.channel+'/'+args.curve+'_'+str(args.modem_num)+'_'+str(args.qua_bits)+'.pth'

    return args


