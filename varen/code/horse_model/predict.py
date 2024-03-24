import torch
from . import horse_net
from torch.autograd import Variable
import os
import numpy as np
from absl import flags
from absl import app
import os.path as osp
from os.path import basename, join
from pytorch3d.io import save_obj, load_ply
import glob
import pickle as pkl
from pytorch3d.ops.mesh_filtering import taubin_smoothing
from pytorch3d.ops import norm_laplacian
from ..train_utils.chamfer import chamfer_distance as chamfer_distance_here
from .varen import VAREN

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('solutions_dir', './data/testset_outside_shape_space/', 'Where the testset data is')

opts = flags.FLAGS

def main(_):

    varen = VAREN()
    seg_data = pkl.load(open('varen/model/varen_smal_real_horse_seg_data.pkl', 'rb'))
    body_parts = ['Pelvis', 'Spine', 'Spine1', 'Spine2', 'LScapula', 'RScapula', 'LFLeg1', 'LFLeg2', 'LFLeg3', 'RFLeg1', 'RFLeg2', 'RFLeg3','LBLeg1', 'LBLeg2', 'LBLeg3', 'RBLeg1', 'RBLeg2', 'RBLeg3','Neck', 'Neck1', 'Neck2']

    idx = []
    for part in body_parts:
        idx += list(seg_data['part2bodyPoints'][seg_data['parts'][part]])

    F = sorted(glob.glob(opts.solutions_dir+'*.npy'))
    N = len(F)
    err = torch.zeros(N)
    err_m2s = torch.zeros(N)

    for i, frame in enumerate(F):
        data = np.load(open(frame, 'rb'))

        data = data[:-2]
        pose = data[3:114+3]
        betas = data[114+3:]
        trans = data[0:3]

        pose = torch.Tensor(pose)
        trans = torch.Tensor(trans)
        betas = torch.Tensor(betas)

        v = varen(betas=betas, pose=pose, trans=trans)

        v = v.detach()

        name = basename(frame)[3:-13]
        print(name)

        # Read the scan
        scan_ply = load_ply(join(opts.solutions_dir, name+'_input.ply'))
        v_scan = torch.Tensor(scan_ply[0]).cuda(device=opts.gpu_id)

        m2s2, s2m2 = chamfer_distance_here(v, v_scan[None,:,:])

        # Take the square root and convert to mm
        m2s = 1000.*torch.sqrt(m2s2)
        s2m = 1000.*torch.sqrt(s2m2)

        err[i] = (torch.mean(m2s) + torch.mean(s2m))/2.
        err_m2s[i] = torch.mean(m2s[0,idx])

        print('distance: ' + str(err[i]))
        print('m2s: ' + str(err_m2s[i]))

    print('Average dist:')
    print(torch.median(err))
    print(torch.mean(err))
    print(torch.std(err))
    print('Average M2S:')
    print(torch.median(err_m2s))
    print(torch.mean(err_m2s))
    print(torch.std(err_m2s))



if __name__ == '__main__':
    app.run(main)

