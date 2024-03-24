from __future__ import absolute_import
import torch
from absl import flags
from ..utils.visualizer import Visualizer

import os
import os.path as osp

import numpy as np


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_string('name', 'smal_horse', 'Experiment Name')
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Root directory for output files')

flags.DEFINE_integer('save_epoch_freq', 1000, 'save model every k epochs')
flags.DEFINE_boolean('print_scalars', True, 'whether to print scalars')
flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')
flags.DEFINE_float('learning_rate', 1e-5, 'learning rate') 
flags.DEFINE_integer('num_epochs', 100, 'epochs')
flags.DEFINE_integer('num_train_epoch', 100, 'used by predictor')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')


def set_bn_eval(m):
    classname = m.__class__.__name__
    if (classname.find('BatchNorm1d') != -1) or (classname.find('BatchNorm2d') != -1):
        m.eval()

class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.gpu_id = opts.gpu_id
        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.save_dir = osp.join(opts.checkpoint_dir, opts.name)
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))

    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = osp.join(self.save_dir, save_filename)

        sd = network.cpu().state_dict()
        torch.save(sd, save_path)
        if gpu_id is not None and torch.cuda.is_available():
            network.cuda(device=gpu_id)
        return

    def load_network(self, network, network_label, epoch_label, network_dir=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path), strict=False)
        return

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix, gpu_id=self.opts.gpu_id)
        return

    def init_training(self):
        self.init_dataset()
        self.define_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.opts.learning_rate, betas=(self.opts.beta1, 0.999))

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def save_result(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def train(self):

        self.smoothed_loss = 0
        self.early_stop = True

        opts = self.opts
        self.visualizer = Visualizer(opts)
        total_steps = 0

        if True:
            for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
                epoch_iter = 0
                for i, batch in enumerate(self.dataloader):

                    self.set_input(batch)

                    self.optimizer.zero_grad()

                    self.forward()

                    self.smoothed_loss = self.smoothed_loss*0.99+0.01*self.loss

                    self.loss.backward()

                    self.optimizer.step()

                    total_steps += 1
                    epoch_iter += 1

                if opts.print_scalars: 
                    scalars = self.get_current_scalars()
                    self.visualizer.print_current_scalars(epoch, epoch_iter, scalars)

                if (epoch+1) % opts.save_epoch_freq == 0 or (epoch+1)==opts.num_epochs:
                    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                    self.save(epoch+1)
