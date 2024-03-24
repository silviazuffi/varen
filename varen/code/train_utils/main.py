from __future__ import absolute_import
  
import torch
from torch.autograd import Variable
from absl import app
from absl import flags
from collections import OrderedDict
import torchvision
import numpy as np

from ..dataloader.data_loader import horse_data_loader
from . import train_utils, loss_utils
from ..horse_model import horse_net
from pytorch3d.io import save_obj, load_ply
from os.path import basename

flags.DEFINE_boolean('reg_on_dv', True, '')
flags.DEFINE_float('regularization_weight', 10, '') 
flags.DEFINE_float('regularization_weight_bound', 100, '') 
flags.DEFINE_float('distance_weight', 1000, '') 
flags.DEFINE_boolean('reg_on_boundary', True, '')

opts = flags.FLAGS

class HorseTrainer(train_utils.Trainer):

    def define_model(self):
        opts = self.opts
        self.model = horse_net.HorseNet(opts, self.dataset.num_samples, reg_data=self.dataset.reg_data)

        if opts.reg_on_boundary:
            self.boundary_faces = self.model.smal.load_muscle_boundary_mesh()

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        self.model = self.model.cuda(device=opts.gpu_id)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        return

    def init_dataset(self):
        self.dataloader, self.dataset = horse_data_loader(opts)
        self.num_datapoints = self.opts.batch_size*len(self.dataloader)
        print(self.num_datapoints)

    def define_losses(self):
        return

    def set_input(self, batch):
        input_data = batch['v'].type(torch.FloatTensor)
        self.input_ids = batch['ids'] 
        self.input_v = Variable(input_data.cuda(device=self.opts.gpu_id), requires_grad=False)


    def forward(self):
        self.loss = 0

        pred_v, def_v, A = self.model.forward(self.input_ids)

        self.loss_dist = self.opts.distance_weight*loss_utils.distance_loss(pred_v, self.input_v)
        if def_v is not None and self.opts.reg_on_dv:
            self.loss_reg = self.opts.regularization_weight*loss_utils.mesh_reg_loss(def_v, self.model.smal.f) 
        else:
            self.loss_reg = self.opts.regularization_weight*loss_utils.mesh_reg_loss(pred_v, self.model.smal.f) 

        loss = self.loss_dist + self.loss_reg 

        if self.opts.reg_on_boundary:
            self.loss_reg_bound = self.opts.regularization_weight_bound*loss_utils.bound_reg_loss(def_v, self.boundary_faces) 
            loss += self.loss_reg_bound

        self.loss += loss


    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('smoothed_loss', self.smoothed_loss.item()),
            ('loss_dist', self.loss_dist.item()),
            ('loss_reg', self.loss_reg.item()),
            ('bound', self.loss_reg_bound.item()),
        ])
        sc_dict['loss'] = self.loss.item()
        return sc_dict

def main(_):
    torch.manual_seed(0)
    np.random.seed(0)
    trainer = HorseTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)






