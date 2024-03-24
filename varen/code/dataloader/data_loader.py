'''

    Author: Silvia Zuffi

'''

from __future__ import absolute_import
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from absl import flags
from glob import glob
from os import path, listdir
from os.path import exists, join

flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_integer('num_samples', 10000, 'Number of samples')
flags.DEFINE_boolean('shuffle', True, '')

curr_path = path.dirname(path.abspath(__file__))
data_path = path.join(curr_path, '../../', 'data')
flags.DEFINE_string('solutions_dir', './registrations/', 'Data Directory')
flags.DEFINE_string('decimated_dir', './scans/decimated_cleaned', 'Data Directory')
flags.DEFINE_integer('input_size', 20000, '')
flags.DEFINE_integer('load_start', 0, '')
flags.DEFINE_integer('load_step', 1, '')

opts = flags.FLAGS

from pytorch3d.io import load_obj, load_objs_as_meshes, load_ply, save_obj
from pytorch3d.ops import sample_points_from_meshes 
from pytorch3d.structures import Meshes
dataScale = 1000.

from .horse_prototypes_data import prototype_clip, prototype_frames, horse_capture_date
import numpy as np
from os.path import basename, split
from .do_not_use_frames import do_not_use

class HorseDataset(Dataset):
    def __init__(self, opts):

        self.opts = opts

        # Read the list of the horses from the prototypes
        horse_list = prototype_clip.keys()

        nHorses = len(horse_list)

        # Read the data with the scan2Mesh information
        keys = np.load('./model_data/MaxScanToMeshByName_names.npy')
        values = np.load('./model_data/MaxScanToMeshByName_values.npy')
        assert(len(keys) == len(values))
        MaxScanToMesh = dict(zip(keys, values))

        # For each horse, read the data
        self.filenames = []
        self.reg_data = []
        n = 0
        for horse in horse_list:
            clips_path = opts.solutions_dir
            if exists(clips_path):
                if True: 
                    filenames = []
                    reg_data = []
                    # Read the number of frames for this clip
                    F = sorted(glob(join(clips_path, '*_'+horse+'_*.npy'))) 
                    frames = [int(f[-19:-13]) for f in F]
                    print('Horse: ' + horse +  ' Frames: ' + str(len(F)))
                    for frame in range(self.opts.load_start, len(frames), self.opts.load_step):
                        scan_name = basename(F[frame])[:-13] 
                        if (scan_name[3:] not in keys):
                            continue
                            
                        if (MaxScanToMesh[scan_name[3:]] > 0.002) or do_not_use(scan_name[3:]):
                            continue

                        clip = scan_name[9:-7]
                        input_path = join(opts.decimated_dir,scan_name[3:] + '.ply')
                        sol_path = F[frame] 

                        data = np.load(open(sol_path, 'rb'))
                        filenames += [input_path]
                        reg_data += [data[:-2]] 
                        n = n+1
                    self.reg_data += reg_data
                    self.filenames += filenames
   
        print(len(self.filenames))

        if opts.num_samples < len(self.filenames):
            # Select a set of random numbers
            idx = torch.randperm(len(self.filenames))[:opts.num_samples]
            F = [self.filenames[i] for i in idx]
            R = [self.reg_data[i] for i in idx]
            self.filenames = F
            self.reg_data = R
        self.num_samples = len(self.filenames)
        print(self.num_samples)
        for f in self.filenames:
            print(f)
        self.reg_data = torch.FloatTensor(np.asarray(self.reg_data))
        

    def forward(self, index):

        verts, faces = load_ply(self.filenames[index])
        meshes = Meshes(verts=[verts], faces=[faces])
        v = sample_points_from_meshes(meshes, num_samples = self.opts.input_size)[0]
        x = {'v': v, 'ids': index}

        return x

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.forward(index)

    def __get_item__(self, index):
        return self.forward(index)

def horse_data_loader(opts):
    dset = HorseDataset(opts)
    dloader = DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle,
        num_workers=opts.n_data_workers,
        drop_last=True)
    return dloader, dset

