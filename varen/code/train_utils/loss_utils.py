import torch
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, point_mesh_face_distance
from pytorch3d.structures import Meshes
from .chamfer import chamfer_distance

def distance_loss(v, v_gt):
    m2s, s2m  = chamfer_distance(v, v_gt, norm=2)
    loss = torch.mean(m2s) + torch.mean(s2m)
    return loss

def mesh_reg_loss(v, f):
    meshes = Meshes(verts=v, faces=f[None,:,:].repeat(v.shape[0],1,1))
    loss = mesh_edge_loss(meshes)
    return loss

def bound_reg_loss(v, f):
    meshes = Meshes(verts=v, faces=f[None,:,:].repeat(v.shape[0],1,1))
    loss = mesh_edge_loss(meshes)
    return loss

