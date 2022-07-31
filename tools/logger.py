import time
import os


class Logger:
    def __init__(self, filepath, out_name):
        self.file = filepath+out_name
        if not os.path.exists(filepath):
            os.mkdir(filepath)

    def save_one_epoch(self, epoch_loss_train, epoch_loss_val):
        if not os.path.exists(self.file):
            with open(self.file, 'w+') as f:
                f.write(str([epoch_loss_train, epoch_loss_val]))
                f.write('\n')
        else:
            with open(self.file, 'a+') as f:
                f.write(str([epoch_loss_train, epoch_loss_val]))
                f.write('\n')


if __name__ == '__main__':
    logger = Logger()
