# New dataset using stereo vision

# According to onet/training.py
# Data format should be:
# data = {
#     'points': points,
#     'points.occ': points_occ,
#     'inputs': inputs,
# }

# Note: The dataloader will actually batch a dict

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

