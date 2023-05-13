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
import numpy as np
from torch.utils import data
import pickle

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
        for category in categories:
            assert category in ['mug', 'bowl', 'bottle'], 'Invalid category'
            subpath = osp.join(dataset_folder, category)
            split_file = osp.join(subpath, f'{split}.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            self.models += models_c

        # -- Get occupancies -- #
        shapenet_categories = {'mug': '03797390',
                               'bowl': '02880940',
                               'bottle': '02876657'}
        self.occ_dict = {}
        for category in categories:
            self.occ_dict[category] = pickle.load(
                open(osp.join(dataset_folder, f'occ_shapenet_{category}.p'), 'rb')
                )


