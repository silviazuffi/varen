'''Code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix'''
import os
import time

class Visualizer():
    def __init__(self, opt):
        self.name = opt.name

        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_message(self, text):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % text)

    def print_current_scalars(self, epoch, i, scalars):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
