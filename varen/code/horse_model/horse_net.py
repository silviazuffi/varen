from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .smal_torch import SMAL

from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle

flags.DEFINE_string('model_dir', 'varen/model/', 'location of the SMAL model')
flags.DEFINE_string('model_name', 'varen_smal_real_horse.pkl', 'name of the model')
flags.DEFINE_string('model_seg_data_name', 'varen_smal_real_horse_seg_data.pkl', 'information about the part segmentation')
flags.DEFINE_string('muscle_labels_name', 'varen_muscle_vertex_labels.npy', '')
flags.DEFINE_integer('muscle_betas_size', 1, 'Size of the muscle variables')
flags.DEFINE_integer('shape_betas_for_muscles', 2, 'Add this number of shape variables to the muscle computation')

class BetasMusclePredictor(nn.Module):
    def __init__(self, opts, num_parts, num_muscle, A):
        super(BetasMusclePredictor, self).__init__()
        self.opts = opts
        r_dim = 4
        num_pose = num_parts*4
        self.num_parts = num_parts
        self.num_muscle = num_muscle
        self.num_pose = num_pose

        self.muscledef = nn.Linear(self.num_pose+opts.shape_betas_for_muscles, num_muscle, bias=False).to(device=opts.gpu_id)
        torch.nn.init.normal_(self.muscledef.weight, mean=0.0, std=0.001)
        A_here = torch.zeros(num_muscle, self.num_pose).to(device=opts.gpu_id)
        np.save('A_init.npy', A.detach().cpu().numpy())
        if opts.shape_betas_for_muscles > 0:
            A_here = torch.zeros(num_muscle, self.num_pose+opts.shape_betas_for_muscles).to(device=opts.gpu_id)
        for p in range(A.shape[0]):
            for k in range(r_dim):
                A_here[:,r_dim*p+k] = A[p,:]

        if opts.shape_betas_for_muscles > 0:
            A_here[:,self.num_pose:] = 1

        self.A = torch.nn.Parameter(A_here, requires_grad=True)

    def forward(self, pose, betas):
        tensor_b = axis_angle_to_quaternion(pose[:,3:].view(-1,self.num_parts,3)).view(-1,self.num_parts*4)
        if self.opts.shape_betas_for_muscles > 0:
            tensor_b = torch.cat((tensor_b, betas[:,:self.opts.shape_betas_for_muscles]),dim=1)

        A = self.A
        tensor_a = self.A*self.muscledef.weight

        
        tensor_a = tensor_a.unsqueeze(0)
        tensor_a = tensor_a.expand(pose.shape[0], -1, -1)
        tensor_b = tensor_b.unsqueeze(1)
        tensor_b = tensor_b.expand(-1,self.num_muscle,-1)
        betas_muscle = tensor_a * tensor_b

        return betas_muscle, A*self.muscledef.weight

class HorseNet(nn.Module):
    def __init__(self, opts, N=0, reg_data=None):
        super(HorseNet, self).__init__()
        self.opts = opts

        # Instantiate the SMAL model in Torch
        model_path = os.path.join(self.opts.model_dir, self.opts.model_name)
        model_seg_data_path = os.path.join(self.opts.model_dir, self.opts.model_seg_data_name)
        self.smal = SMAL(pkl_path=model_path, opts=self.opts, seg_pkl_path=model_seg_data_path)

        # Create arrays for the optimization variables for the whole dataset
        if True: 

            # Initialize with the alignment data
            if N > 0 and reg_data is not None:
                assert(N== reg_data.shape[0])
                init_trans = reg_data[:,0:3]
                init_pose = reg_data[:,3:self.smal.nJ*3+3]
                init_betas = reg_data[:,self.smal.nJ*3+3::]
                self.betas = Variable(init_betas.to(device=opts.gpu_id), requires_grad=False)
                self.pose = Variable(init_pose.to(device=opts.gpu_id), requires_grad=False)
                self.trans = Variable(init_trans.to(device=opts.gpu_id), requires_grad=False)
            else:
                if N > 0:
                    self.betas = torch.nn.Parameter(torch.zeros(N, self.smal.num_betas).to(device=opts.gpu_id), requires_grad=True)
                    self.pose = torch.nn.Parameter(torch.zeros(N, self.smal.nJ*3).to(device=opts.gpu_id), requires_grad=True)
                    self.trans = torch.nn.Parameter(torch.zeros(N, 3).to(device=opts.gpu_id), requires_grad=True)

        muscle_labels_path = os.path.join(self.opts.model_dir, self.opts.muscle_labels_name)
        A = self.smal.define_muscle_deformations_variables(muscle_labels_path=muscle_labels_path)
        self.smal.betas_muscle_predictor = BetasMusclePredictor(self.opts, (self.smal.nJ-1), self.smal.num_muscles, A)

            
    def forward(self, ids):
        pose = self.pose[ids,:]
        betas=self.betas[ids,:]
        trans=self.trans[ids,:]

        if len(trans) == 3:
            pose = pose[None,:]
            betas = betas[None,:]
            trans = trans[None,:]

        b_muscle, A = self.smal.betas_muscle_predictor.forward(pose, betas)

        v, pose_dv = self.get_smal_verts(betas=betas, pose=pose, trans=trans,
                betas_muscle=b_muscle)

        return v, pose_dv, A

    def get_smal_verts(self, betas=None, pose=None, trans=None, del_v=None, betas_muscle=None):
        verts, pose_dv = self.smal(betas=betas, theta=pose, trans=trans, del_v=del_v, 
                betas_muscle=betas_muscle)
        return verts, pose_dv


