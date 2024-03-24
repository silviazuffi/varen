import torch
from . import horse_net
from absl import flags
from absl import app
import os.path as osp
from torch import nn

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('name', 'varen', 'Network Name')
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                            'Root directory for output files')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('num_train_epoch', 100, '')

opts = flags.FLAGS

def load_network(network, network_label, epoch_label, opts):
    save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
    network_dir = osp.join(opts.checkpoint_dir, opts.name)
    save_path = osp.join(network_dir, save_filename)
    print('loading {}..'.format(save_path))
    network.load_state_dict(torch.load(save_path), strict=False)
    return

class VAREN(nn.Module):
    def __init__(self):
        super(VAREN, self).__init__()

        model = horse_net.HorseNet(opts, N=1, reg_data=None)
        load_network(model, 'pred', opts.num_train_epoch, opts)
        model.eval()
        self.model = model.cuda(device=opts.gpu_id)

    def __call__(self, betas=None, pose=None, trans=None):
        pose = pose[None,:].cuda(device=opts.gpu_id)
        trans = trans[None,:].cuda(device=opts.gpu_id)
        betas = betas[None,:].cuda(device=opts.gpu_id)

        b_muscle, _ = self.model.smal.betas_muscle_predictor.forward(pose, betas)

        v, _ = self.model.get_smal_verts(betas=betas, pose=pose, trans=trans,
                betas_muscle=b_muscle)

        return(v)




