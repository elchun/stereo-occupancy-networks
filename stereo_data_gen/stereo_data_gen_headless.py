# Based on the not headless version.  Good for large batches.
import os
import os.path as osp

import numpy as np
import trimesh
import pyrender
import doctest
import pickle
import os.path as osp
from scipy.spatial.transform import Rotation as R

from typing import List
import tqdm

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'


# -- GET PATHS TO EVERYTHING -- #
# -- Load Objects -- #
data_dir = '../data/'
objects_to_dir = {
    'mug': 'mug_centered_obj',
    'bowl': 'bowl_centered_obj',
    'bottle': 'bottle_centered_obj'
}
mug_data_path = osp.join(data_dir, 'objects', objects_to_dir['mug'])
bowl_data_path = osp.join(data_dir, 'objects', objects_to_dir['bowl'])
bottle_data_path = osp.join(data_dir, 'objects', objects_to_dir['bottle'])

mug_shapenet_ids = os.listdir(mug_data_path)
bowl_shapenet_ids = os.listdir(bowl_data_path)
bottle_shapenet_ids = os.listdir(bottle_data_path)

# -- Load Occupancy -- #
mug_occ_path = osp.join(data_dir, 'occupancy', 'occ_shapenet_mug.p')
bowl_occ_path = osp.join(data_dir, 'occupancy', 'occ_shapenet_bowl.p')
bottle_occ_path = osp.join(data_dir, 'occupancy', 'occ_shapenet_bottle.p')

# -- Save dir -- #
mug_stereo_path = osp.join(data_dir, 'stereo_training_data', 'mug')
bowl_stereo_path = osp.join(data_dir, 'stereo_training_data', 'bowl')
bottle_stereo_path = osp.join(data_dir, 'stereo_training_data', 'bottle')

# -- DEFINE HELPERS -- #
def make_rotation_matrix(axis: str, theta: float):
    """
    Make rotation matrix about {axis} with angle {theta}

    Args:
        axis (str): {'x', 'y', 'z'}
        theta (float): angle in radians
    """

    s = np.sin(theta)
    c = np.cos(theta)

    if axis == 'x':
        r = [[1, 0, 0],
             [0, c, -s],
             [0, s, c]]

    elif axis == 'y':
        r = [[c, 0, s],
             [0, 1, 0],
             [-s, 0, c]]

    elif axis == 'z':
        r = [[c, -s, 0],
             [s, c, 0],
             [0, 0, 1]]

    else:
        raise ValueError('Unexpected axis')

    return r

def make_camera_pose(r, theta):
    """
    Make homologous camera pose matrix using r and theta according to
    diagram above.

    Args:
        r (float): Radius of cameras from scene orgin, unit is meters.
        theta (float): Angle of camera from z axis, unit is radians.
    >>> np.array_equal(make_camera_pose(1, 0), \
    np.array([[ 1.,  0.,  0.,  0.], \
              [ 0.,  1.,  0.,  0.], \
              [ 0.,  0.,  1.,  1.], \
              [ 0.,  0.,  0.,  1.]]))

    True
    """
    l_z = r * np.cos(theta)
    l_x = r * np.sin(theta)
    rotation = make_rotation_matrix('y', theta)

    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rotation
    pose_mat[:3, 3] = np.array([l_x, 0, l_z])

    return pose_mat

def render_mesh(mesh_fname, n_samples=1000, cam_r=1, cam_theta=np.pi/16):
    """
    Render mesh in random poses.

    Note: If getting a display error, try
    `os.environ['PYOPENGL_PLATFORM'] = 'egl'`

    Args:
        mesh_fname (string): path to mesh file
        n_samples (int, optional): Number of random poses to sample. Defaults to 1000.

    Returns:
        Tuple(np.array, np.array, np.array): Left image, right image, object pose
    """
    # -- Load mesh -- #
    mesh = trimesh.load(mesh_fname)
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(224, 224)
    baseColorFactor = [210, 210, 210, 1]
    texture = pyrender.MetallicRoughnessMaterial('mug_material', baseColorFactor=baseColorFactor)
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=texture))


    # -- Define Cameras -- #
    l_cam_pose = make_camera_pose(cam_r, -cam_theta)
    r_cam_pose = make_camera_pose(cam_r, cam_theta)

    camera = pyrender.PerspectiveCamera(yfov = np.pi / 3.0, aspectRatio=1.0)
    l_cam_node = scene.add(camera, pose=l_cam_pose)
    r_cam_node = scene.add(camera, pose=r_cam_pose)

    # -- Define Lights -- #
    light = pyrender.SpotLight(color=np.ones(3),
                               intensity=3.0,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=l_cam_pose)
    scene.add(light, pose=r_cam_pose)

    # -- Render -- #
    color_l_images = []
    color_r_images = []
    object_poses = []
    for i in range(n_samples):
        random_pose = np.eye(4)
        random_pose[:3, :3] = R.random().as_matrix()
        scene.set_pose(mesh_node, pose=random_pose)

        scene.main_camera_node = l_cam_node
        color_l, depth_l = renderer.render(scene)
        scene.main_camera_node = r_cam_node
        color_r, depth_r = renderer.render(scene)

        color_l_images.append(color_l)
        color_r_images.append(color_r)
        object_poses.append(random_pose)

    renderer.delete()
    return color_l_images, color_r_images, object_poses

def render_batch(data_path: str, shapenet_ids: List[str], save_dir: str, n_samples_per_ob=1000):
    os.makedirs(save_dir, exist_ok=True)

    # images are 224 x 224 x 3
    shapenet_ids_wrapped = tqdm.tqdm(shapenet_ids)
    for shapenet_id in shapenet_ids_wrapped:
        mesh_fname = osp.join(data_path, shapenet_id, 'models/model_128_df.obj')
        try:
            l_images, r_images, poses = render_mesh(mesh_fname, n_samples=n_samples_per_ob)
        except ValueError:
            print('Error rendering mesh: ', shapenet_id)
            continue

        id_save_dir = osp.join(save_dir, shapenet_id)
        os.makedirs(id_save_dir, exist_ok=True)
        for i, data in enumerate(zip(l_images, r_images, poses)):
            l_image, r_image, pose = data

            l_image = np.einsum('ijk->kij', l_image)
            r_image = np.einsum('ijk->kij', r_image)

            # -- Make same datatype -- #
            l_image = l_image.astype(np.float32) / 255
            r_image = r_image.astype(np.float32) / 255
            pose = pose.astype(np.float32)

            save_fname = osp.join(id_save_dir, 'pose_' + str(i))
            # You must load the occupany with the dataloader
            np.savez(save_fname,
                l_image = l_image,
                r_image = r_image,
                pose = pose,
                # shapenet_id = shapenet_id,
            )

# -- RENDER EVERYTHING -- #
render_batch(mug_data_path, mug_shapenet_ids, mug_stereo_path, n_samples_per_ob=1000)
render_batch(bowl_data_path, bowl_shapenet_ids, bowl_stereo_path, n_samples_per_ob=1000)
render_batch(bottle_data_path, bottle_shapenet_ids, bottle_stereo_path, n_samples_per_ob=1000)
