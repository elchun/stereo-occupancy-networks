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
    def __init__(self, dataset_folder: str, split: str, categories: list, points_subsample: int):
        """
        Init function for Shapes3dMonoDataset

        Args:
            dataset_folder (str): dataset folder (with subfolder for classes)
            split (str): which split to use, 'train', 'test', 'val'
            categories (list): list of categories to use ('mug', 'bowl', 'bottle')
        """
        self.dataset_folder = dataset_folder
        self.points_subsample = points_subsample
        self.split = split
        if categories is None:
            categories = ['mug', 'bowl', 'bottle']

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
        self.samples_per_model = len([sample for sample in os.listdir(osp.join(dataset_folder, categories[0], self.models[0])) if 'pose' in sample])


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

        # -- Load normal points -- #
        # Occupancy is stored as an array of coordinates and array of bool
        # occupancy values
        occupancies = np.load(osp.join(self.dataset_folder, shapenet_category, shapenet_id, 'occ.npz'))
        randidx = np.random.randint(0, occupancies['coord'].shape[0], self.points_subsample)
        coord = occupancies['coord'][randidx, :]
        occ_logits = occupancies['voxel_bool'][randidx]

        # -- Load IOU Points -- #
        # points_iou = None
        # occ_iou = None
        # if self.split == 'val':
        iou_randidx = np.random.randint(0, occupancies['coord'].shape[0], self.points_subsample)
        points_iou = occupancies['coord'][iou_randidx, :]
        occ_iou = occupancies['voxel_bool'][iou_randidx]

        # -- Load images -- #
        try:
            images = np.load(osp.join(self.dataset_folder, shapenet_category, shapenet_id, f'pose_{sample_idx}.npz'))
            l_image = images['l_image']
            r_image = images['r_image']
            pose = images['pose']
        except Exception:
            error_log_path = osp.join(self.dataset_folder, shapenet_category, 'error_log.txt')
            print(f'Error loading: {osp.join(self.dataset_folder, shapenet_category, shapenet_id, f"pose_{sample_idx}.npz")}')
            with open(error_log_path, 'w') as f:
                f.write(f'Error loading: {osp.join(shapenet_category, shapenet_id, f"pose_{sample_idx}.npz")}')
            return self.__getitem__(idx + 1)

        # TODO: Remove once reshaping is done
        if l_image.shape[0] != 3:
            l_image = np.einsum('ijk->kij', l_image)
        if r_image.shape[0] != 3:
            r_image = np.einsum('ijk->kij', r_image)

        # TODO: Remove once reprocessing is done
        l_image = l_image.astype(np.float32)
        r_image = r_image.astype(np.float32)

        # TODO: Remove
        if np.max(l_image) > 1:
            l_image = l_image / 255
        if np.max(r_image) > 1:
            r_image = r_image / 255

        coord = coord.astype(np.float32)
        occ_logits = occ_logits.astype(np.float32)

        pose = pose.astype(np.float32)

        # Assume pose is a 4x4 homogeneous transform matrix
        coord_transformed = coord @ pose[:3, :3].T + pose[:3, 3]

        # print('coord_dtype: ', coord_transformed.dtype)
        # print('voxel shape: ', occ_logits.shape)

        data = {
            'points': coord_transformed,
            'points.occ': occ_logits,
            'inputs': l_image,
            'points_iou': points_iou,
            'points_iou.occ': occ_iou,
        }

        return data



class Shapes3dStereoDataset(data.Dataset):
    """
    Shapenet mugs, bowl, bottles
    """
    def __init__(self, dataset_folder: str, split: str, categories: list, points_subsample: int):
        """
        Init function for Shapes3dMonoDataset

        Args:
            dataset_folder (str): dataset folder (with subfolder for classes)
            split (str): which split to use, 'train', 'test', 'val'
            categories (list): list of categories to use ('mug', 'bowl', 'bottle')
        """
        self.dataset_folder = dataset_folder
        self.points_subsample = points_subsample
        self.split = split
        if categories is None:
            categories = ['mug', 'bowl', 'bottle']

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
        self.samples_per_model = len([sample for sample in os.listdir(osp.join(dataset_folder, categories[0], self.models[0])) if 'pose' in sample])


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

        # -- Load normal points -- #
        # Occupancy is stored as an array of coordinates and array of bool
        # occupancy values
        occupancies = np.load(osp.join(self.dataset_folder, shapenet_category, shapenet_id, 'occ.npz'))
        randidx = np.random.randint(0, occupancies['coord'].shape[0], self.points_subsample)
        coord = occupancies['coord'][randidx, :]
        occ_logits = occupancies['voxel_bool'][randidx]

        # -- Load IOU Points -- #
        # points_iou = None
        # occ_iou = None
        # if self.split == 'val':
        iou_randidx = np.random.randint(0, occupancies['coord'].shape[0], self.points_subsample)
        points_iou = occupancies['coord'][iou_randidx, :]
        occ_iou = occupancies['voxel_bool'][iou_randidx]

        # -- Load images -- #
        try:
            images = np.load(osp.join(self.dataset_folder, shapenet_category, shapenet_id, f'pose_{sample_idx}.npz'))
            l_image = images['l_image']
            r_image = images['r_image']
            pose = images['pose']
        except Exception:
            error_log_path = osp.join(self.dataset_folder, shapenet_category, 'error_log.txt')
            print(f'Error loading: {osp.join(self.dataset_folder, shapenet_category, shapenet_id, f"pose_{sample_idx}.npz")}')
            with open(error_log_path, 'w') as f:
                f.write(f'Error loading: {osp.join(shapenet_category, shapenet_id, f"pose_{sample_idx}.npz")}')
            return self.__getitem__(idx + 1)

        # TODO: Remove once reshaping is done
        if l_image.shape[0] != 3:
            l_image = np.einsum('ijk->kij', l_image)
        if r_image.shape[0] != 3:
            r_image = np.einsum('ijk->kij', r_image)

        # TODO: Remove once reprocessing is done
        l_image = l_image.astype(np.float32)
        r_image = r_image.astype(np.float32)

        # TODO: Remove
        if np.max(l_image) > 1:
            l_image = l_image / 255
        if np.max(r_image) > 1:
            r_image = r_image / 255

        coord = coord.astype(np.float32)
        occ_logits = occ_logits.astype(np.float32)

        pose = pose.astype(np.float32)

        # Assume pose is a 4x4 homogeneous transform matrix
        coord_transformed = coord @ pose[:3, :3].T + pose[:3, 3]

        # print('coord_dtype: ', coord_transformed.dtype)
        # print('voxel shape: ', occ_logits.shape)

        # Concatenate left and right images (easier to package with existing code)
        inputs = np.concatenate([l_image, r_image], axis=0)

        data = {
            'points': coord_transformed,
            'points.occ': occ_logits,
            'inputs': inputs,
            'points_iou': points_iou,
            'points_iou.occ': occ_iou,
        }

        return data