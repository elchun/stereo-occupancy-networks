# New dataset using stereo vision

# According to onet/training.py
# Data format should be:
# data = {
#     'points': points,
#     'points.occ': points_occ,
#     'inputs': inputs,
# }

# Note: The dataloader will actually batch a dict


import os.path as osp
import os
import numpy as np
from torch.utils import data
import pickle

from im2mesh.utils import ndf_util

class Shapes3dMonoDataset(data.Dataset):
    """
    Shapenet mugs, bowl, bottles
    """
    def __init__(self, dataset_folder: str, split: str, categories: list):
        """
        Init function for Shapes3dMonoDataset

        Args:
            dataset_folder (str): dataset folder (with subfolder for classes)
            split (str): which split to use, 'train', 'test', 'val'
            categories (list): list of categories to use ('mug', 'bowl', 'bottle')
        """
        self.dataset_folder = dataset_folder

        # -- Get model list -- #
        self.models = []
        self.samples_per_model = None
        self.category_idx_ranges = {}
        for category in categories:
            assert category in ['mug', 'bowl', 'bottle'], 'Invalid category'
            subpath = osp.join(dataset_folder, category)
            split_file = osp.join(subpath, f'{split}.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            self.models += models_c
            self.category_idx_ranges[category] = [len(self.models) - len(models_c), len(self.models)]

            if self.samples_per_model is None:
                self.samples_per_model = len(os.listdir(osp.join(subpath, models_c[0])))

        # Assume all models have the same number of samples
        # Note: We render a set of stereo views for each model.  The total
        # number of datapoints is the number of models times the number of
        # samples per model.
        self.samples_per_model = len(os.listdir(osp.join(dataset_folder, categories[0], self.models[0])))


    def __len__(self):
        """
        Return number of models in dataset
        """
        return len(self.models) * self.samples_per_model

    def __getitem__(self, idx):
        """
        Get item of the dataset

        We get an item by taking the model at (idx // self.samples_per_model)
        then getting the sample at (idx % self.samples_per_model)

        Args:
            idx (int): index of item
        Returns:
            data (dict): dictionary of data with keys 'points', 'points.occ', 'inputs'
        """
        model_idx = idx // self.samples_per_model
        sample_idx = idx % self.samples_per_model

        shapenet_id = self.models[model_idx]
        shapenet_category = None
        for category, idx_range in self.category_idx_ranges.items():
            if idx_range[0] <= model_idx < idx_range[1]:
                shapenet_category = category
                break

        # -- Load Occupancy -- #
        # Occupancy is stored as an array of coordinates and array of bool
        # occupancy values
        occupancies = np.load(osp.join(self.dataset_folder, shapenet_category, shapenet_id, 'occ.npz'))
        coord = occupancies['coord']
        voxel_bool = occupancies['voxel_bool']

        # -- Load images -- #
        images = np.load(osp.join(self.dataset_folder, shapenet_category, shapenet_id, f'pose_{sample_idx}.npz'))
        l_image = images['l_image']
        r_image = images['r_image']
        pose = images['pose']

        # Assume pose is a 4x4 homogeneous transform matrix
        coord_transformed = coord @ pose[:3, :3].T + pose[:3, 3]
        voxel_bool

        data = {
            'points': coord_transformed,
            'points.occ': voxel_bool,
            'inputs': l_image
        }

        return data








