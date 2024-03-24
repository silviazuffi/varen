"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl
from torch import nn
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from absl import flags
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from .base_of_the_head_points import points as base_head_points
from pytorch3d.io import load_ply

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(nn.Module):
    def __init__(self, pkl_path, opts, seg_pkl_path=None, dtype=torch.float, logscale_part_list=None, sym_npy_path=None):
        super(SMAL, self).__init__()

        self.opts = opts
        # -- Load SMPL params --
        with open(pkl_path, 'rb') as f:
            dd = pkl.load(f)

        f = dd['f']
        self.register_buffer('f', torch.Tensor(f.astype(int)).int())

        self.nJ = dd['J'].shape[0]
        self.nP = dd['posedirs'].shape[2]
        self.kintree_table = dd['kintree_table']

        v = undo_chumpy(dd['v_template']).copy()
        self.register_buffer('v_template', torch.Tensor(v))

        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]

        # Shape blend shape basis
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']).copy(), [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.Tensor(shapedir))

        # Regressor for joint locations given shape 
        self.register_buffer('J_regressor', torch.Tensor(dd['J_regressor'].T.todense()))

        # Add additional information about the part segmentation
        if seg_pkl_path is not None:
            seg_data = pkl.load(open(seg_pkl_path, "rb"), encoding='latin1')

            self.parts = seg_data['parts']
            self.partSet = range(len(self.parts))
            self.part2bodyPoints = seg_data['part2bodyPoints']
            self.colors_names = seg_data['colors_names']
            self.seg = seg_data['seg']

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]
        posedirs = np.reshape(
                undo_chumpy(dd['posedirs']).copy(), [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.Tensor(posedirs))

        # Indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        W = undo_chumpy(dd['weights'])
        W = np.log(W+1.)
       
        self.log_weights = Variable(
            torch.Tensor(W.copy()).to(device=self.opts.gpu_id),
            requires_grad=False)

        # If using the muscles deformations
        self.use_muscle_deformations = False


    def __call__(self, betas=None, theta=None, trans=None, del_v=None, betas_muscle=None):

        if betas is not None and len(betas.shape) == 1:
            betas = betas[None,:]
        if theta is not None and len(theta.shape) == 1:
            theta = theta[None,:]
        if trans is not None and len(trans.shape) == 1:
            trans = trans[None,:]
        if del_v is not None and len(del_v.shape) == 1:
            del_v = del_v[None,:]
        if betas_muscle is not None and len(betas_muscle.shape) == 1:
           betas_muscle = betas_muscle[None,:]

        self.weights = torch.exp(self.log_weights)-1.

        nBetas = betas.shape[1]

        # 1. Add shape blend shapes
        
        if nBetas > 0:
            if del_v is None:
                v_shaped = self.v_template + torch.reshape(torch.matmul(betas, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = self.v_template + del_v + torch.reshape(torch.matmul(betas, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = self.v_template.unsqueeze(0)
            else:
                v_shaped = self.v_template + del_v 

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # If defined, use muscle deformations
        mdv = None

        if self.use_muscle_deformations:
            # A set of decoders, one for each muscle
            mdv = torch.zeros_like(v_shaped)
            for i in range(self.num_muscles):
                idx = self.muscle_idxs[i]
                mdv[:,idx,:] = self.Bm[i].forward(betas_muscle[:,i,:]).view(v_shaped.shape[0],-1,3)

            v_shaped = v_shaped + mdv

        # 3. Add pose blend shapes
        # N x nJ x 3 x 3
        Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3]), opts=self.opts), [-1, self.nJ, 3, 3])

        v_posed = v_shaped

        #4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, opts=self.opts)

        # 5. Do skinning:
        num_batch = theta.shape[0]
        
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, self.nJ])

        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, self.nJ, 16])),
                [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=self.opts.gpu_id)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device=self.opts.gpu_id)

        verts = verts + trans[:,None,:]

        return verts, mdv

    def load_muscle_boundary_mesh(self):
        verts, faces = load_ply('./model_data/muscle_boundaries_w_head.ply')
        faces = faces.to(device=self.opts.gpu_id)
        return faces

    def define_muscle_deformations_variables(self, muscle_labels_path=None):
        '''
        '''

        self.use_muscle_deformations = True

        self.muscle_labels = np.load(open(muscle_labels_path, 'rb'))

        self.num_muscles = np.max(self.muscle_labels) + 1

        self.muscle_parts = ['LScapula', 'RScapula', 'Spine1', 'Spine2', 'LBLeg1', 'LBLeg2', 'LBLeg3', 'Neck1', 'Neck2', 'Neck', 'Spine', 'LFLeg1', 'LFLeg2', 'LFLeg3', 'RFLeg2', 'RFLeg3', 'RFLeg1', 'Pelvis', 'RBLeg2', 'RBLeg3', 'RBLeg1', 'Head']
        self.muscle_parts_idx = []
        all_idxs = []
        for pa in self.muscle_parts:
            self.muscle_parts_idx += [self.parts[pa]]
            if pa == 'Head':
                all_idxs += list(base_head_points)
            else:
                all_idxs += list(self.part2bodyPoints[self.parts[pa]])
        self.all_muscle_idxs = all_idxs

        # Define the vertices that have no muscle to be associated
        unused_idxs = list(set(all_idxs) & set(range(self.size[0])))

        # Define part-muscle assciation function
        A = torch.zeros((self.nJ-1, self.num_muscles)).to(device=self.opts.gpu_id)
        # Only assign for the parts that we consider affect the muscles (muscle_parts)
        for p in self.muscle_parts_idx:
            # Vertices of this part
            part_v_idx = self.part2bodyPoints[p]
            labels = np.unique(self.muscle_labels[part_v_idx])
            A[p-1,labels] += 1
            parent = self.kintree_table[0,p]
            if parent < self.nJ:
                A[parent-1,labels] += 1
                idx = np.where(self.kintree_table[0,:]==p)[0]
                for k in idx:
                    A[k-1,labels] += 1
        A = A / torch.max(A)

        # Define the indices of the vertices that belong to each muscle
        self.muscle_idxs = [None]*self.num_muscles
        for i in range(self.num_muscles):
            self.muscle_idxs[i] = list(set(all_idxs) & set(np.where(self.muscle_labels==i)[0]))


        self.Bm = torch.nn.ModuleList() 
        for i in range(self.num_muscles):
            pose_d = 4
            self.Bm.append(nn.Sequential(nn.Linear(self.opts.muscle_betas_size*(self.nJ-1)*pose_d+self.opts.shape_betas_for_muscles, len(self.muscle_idxs[i])*3, bias=False)))

            for m in self.Bm[i].modules():
                if isinstance(m, nn.Linear):
                   torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
                   if m.bias is not None:
                       m.bias.data.zero_()

        return A



