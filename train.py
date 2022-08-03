import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from setting.setting import device
from tools.logger import Logger
from tools.parse import *
from traditional.channel import *
import importlib


def generate_16QAM(batch_size):
    table = torch.tensor(
        [1. + 1.j, 1. + 3.j, 1. - 1.j, 1. - 3.j, 3. + 1.j, 3. + 3.j, 3. - 1.j, 3. - 3.j, -1. + 1.j, -1. + 3.j,
         -1. - 1.j, -1. - 3.j, -3. + 1.j, -3. + 3.j, -3. - 1.j, -3. - 3.j]).to(device) / 10 ** 0.5
    data = torch.tensor(np.random.randint(0, 16, [batch_size, 16]), dtype=torch.int64).to(device)
    data = table[data]
    return torch.stack((data.real, data.imag), dim=1)


def train(args):
    snr = args.snr
    logger = Logger('./data/' + args.channel + '/', args.curve + '_' + str(args.modem_num) + '_' + str(args.qua_bits) + '.txt')
    model = getattr(importlib.import_module("models." + args.curve), 'Net')

    net = model(args.N, args.G, args.K, args.modem_num, args.qua_bits)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_loss = torch.tensor(np.inf)

    dir = './data/' + args.channel + '/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    for idx in range(500):
        ac_T = (idx + 1) * args.unit_T // 10 + 1
        net.train()
        loss_train = []
        for each_batch in range(30):
            train_data = generate_16QAM(args.batch_size)
            s = train_data.reshape(args.batch_size, 2, -1)
            optimizer.zero_grad()
            criterion = torch.nn.MSELoss()
            for sample_h in range(1):
                H_r, H_i = H_sample(args.TDL_dB, args.delay)
                r = net(s, snr, H_r, H_i, 'train', ac_T)
                MSE_loss = criterion(r, s)
                MSE_loss.backward()
            optimizer.step()
            loss_train.append(MSE_loss.item())
        loss_train = np.mean(np.array(loss_train))
        print("epoch: [%d/%d] , train_loss: %f" % (idx+1, 500, loss_train))

        net.eval()
        loss_val = []
        with torch.no_grad():
            for each_batch in range(30):
                val_data = generate_16QAM(args.batch_size)
                s = val_data.reshape(args.batch_size, 2, -1)
                H_r, H_i = H_sample(args.TDL_dB, args.delay)
                r = net(s, snr, H_r, H_i, 'infer', ac_T)
                MSE_loss = criterion(r, s)
                loss_val.append(MSE_loss.item())
        loss_val = np.mean(np.array(loss_val))
        print("epoch: [%d/%d] , val_loss: %f" % (idx+1, 500, loss_val))

        logger.save_one_epoch(loss_train, loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(net.state_dict(), dir + args.curve + '_' + str(args.modem_num) + '_' + str(args.qua_bits)+'.pth')


if __name__ == "__main__":
    args = get_args()
    train(args)


