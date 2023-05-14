import numpy as np
import os.path as osp
import os
import tqdm


dataset_folder = '../data/stereo_training_data/'
for category in ['mug', 'bowl', 'bottle']:
    shapenet_ids = os.listdir(osp.join(dataset_folder, category))
    shapenet_ids = tqdm.tqdm(shapenet_ids)
    for shapenet_id in shapenet_ids:
        if '.lst' in shapenet_id:  # ignore lst files
            continue
        model_path = osp.join(dataset_folder, category, shapenet_id)
        i = 0
        while osp.exists(osp.join(model_path, f'pose_{i}.npz')):
            data_file = osp.join(model_path, f'pose_{i}.npz')
            data = np.load(data_file)
            l_image = data['l_image']
            r_image = data['r_image']
            pose = data['pose']

            if l_image.shape[0] != 3:
                l_image = np.einsum('ijk->kij', l_image)
            if r_image.shape[0] != 3:
                r_image = np.einsum('ijk->kij', r_image)

            np.savez(data_file,
                     l_image=l_image,
                     r_image=r_image,
                     pose=pose,
                    #  shapenet_id=shapenet_id
                     )
            i += 1
        print(f'Reshaped {i-1} files for {shapenet_id}')
