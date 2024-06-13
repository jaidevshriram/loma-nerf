import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
loma_dir = os.path.join(current, "loma_public")
sys.path.append(loma_dir)

import numpy as np
import pdb
from tqdm.auto import tqdm

import compiler
import ctypes
from ctypes import *

from mlp_utils import *
from pos_encoding import positional_encoding
from dataloader import NeRFDataset


def get_rays(height, width, normalized_K, c2w_poses):
    """
    Get the rays for a given image size and camera poses.

    Args:
        height: The height of the image.
        width: The width of the image.
        normalized_K: The normalized camera intrinsics.
        c2w_poses: The camera to world poses.

    Returns:
        ray origins, ray directions
    """

    range_coord = np.linspace(0, 1, width)
    i, j = np.meshgrid(range_coord, range_coord, indexing="xy")

    i, j = i.flatten(), j.flatten()

    points = np.stack([i, j], axis=0)
    points_homo = np.stack([i, j, np.ones_like(i)], axis=0)

    points_cam = np.linalg.inv(normalized_K) @ points_homo

    R = c2w_poses[:3, :3]
    T = c2w_poses[:3, 3]

    ray_origins = R @ points_cam + T[:, None]
    ray_directions = R @ points_cam

    # Normalzie the ray directions
    return ray_origins.T, ray_directions.T


def ray_march(rays_o, rays_d, net, near, far, num_samples):
    """
    Perform ray marching to render an image from rays using the MLP.

    Args:
        rays_o (np.array): Origin of rays.
        rays_d (np.array): Direction of rays.
        net (callable): Neural network function for evaluating RGB and density.
        near (float): Near bound for ray marching.
        far (float): Far bound for ray marching.
        num_samples (int): Number of samples per ray.

    Returns:
        np.array: Accumulated color for each ray.
    """
    t_vals = np.linspace(near, far, num_samples)
    points = (
        rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]
    )  # Shape: [num_rays, num_samples, 3]

    colors, densities = net(points.reshape(-1, 3))
    colors = colors.reshape(-1, num_samples, 3)
    densities = densities.reshape(-1, num_samples)

    # Calculate weights for RGBA accumulation
    alpha = 1.0 - np.exp(-densities * (t_vals[1] - t_vals[0]))
    weights = alpha * np.cumprod(
        1.0 - alpha + 1e-10, axis=1
    )  # Cumulative product along each ray
    weights[:, 1:] = weights[:, :-1]
    weights[:, 0] = alpha[:, 0]

    # Accumulate color along the ray
    rgb = (weights[:, :, None] * colors).sum(axis=1)

    return rgb


if __name__ == "__main__":

    # Config
    img_size = 128
    num_iterations = 50000
    num_samples_along_ray = 32
    mlp_max_size = 256  # This is the maximum size that our MLP can process in Loma
    max_img_chunk_size = 256 / num_samples_along_ray
    chunk_size_x = np.floor(max_img_chunk_size**0.5)
    chunk_size = chunk_size_x**2
    num_encoding_functions = 5
    num_layers_mlp = 3
    mlp_filter_size = 16
    out_channels = 4
    near = 2.0
    far = 6.0

    # Create the dataset
    dataset = NeRFDataset("data/lego", img_size=img_size, phase="train")

    # Compile the nerf code and obtain the code to evaluate the nerf and compute the gradients
    with open("scripts/nerf.py") as f:
        _, lib = compiler.compile(f.read(), target="c", output_filename="_code/nerf")

    nerf_evaluate_and_march = lib.nerf_evaluate_and_march
    grad_nerf_evaluate_and_march = lib.grad_nerf_evaluate_and_march

    # Create the MLP
    in_channels = 3 + (
        2 * 3 * num_encoding_functions
    )  # for every point, we have sin/cos encoding, repeated for num_encoding_functions
    ws, bs = get_sample_mlp(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers_mlp,
        filter_size=mlp_filter_size,
    )
    ws_shape = np.array([np.array(w).shape for w in ws], dtype=np.int32)
    bs_shape = np.array([[len(b), 1] for b in bs], dtype=np.int32)

    ws_padded = pad_array(ws)
    bs_padded = pad_array(bs)

    # Main training loop
    for i in tqdm(range(num_iterations)):

        # Choose a random pose and image
        train_idx = np.random.randint(len(dataset))
        train_data = dataset[train_idx]
        train_img = train_data["image"]
        train_pose = train_data["pose"]
        focal_length = train_data["focal_length"]

        K = np.array([[focal_length, 0, 0], [0, focal_length, 0], [0, 0, 1]]).astype(
            np.float32
        )

        # Sample a bunch of rays
        # TODO: Sample the rays here - this will obtain ray origins and ray directions
        ray_origins, ray_directions = get_rays(img_size, img_size, K, train_pose)
        target_gt = train_img.reshape(-1, 3)

        # Compute chunks
        num_chunks = int(np.ceil(img_size / chunk_size))

        for chunk_idx in range(num_chunks):
            chunk_start = int(chunk_idx * chunk_size)
            chunk_end = int(min((chunk_idx + 1) * chunk_size, img_size))

            # Get the subset of the chunk rays and gt
            chunk_ray_origins = ray_origins[chunk_start:chunk_end]  # N x 3
            chunk_ray_directions = ray_directions[chunk_start:chunk_end]  # N x 3
            chunk_target_gt = target_gt[chunk_start:chunk_end]  # N x 3

            # Sample a bunch of points along the ray - by assuming a near and far threshold
            depth_values = np.linspace(near, far, num_samples_along_ray)
            depth_values += (
                np.random.randn(num_samples_along_ray)
                * (far - near)
                / num_samples_along_ray
            )  # Perturb the depth values

            sample_points = (
                chunk_ray_origins[:, None, :]
                + chunk_ray_directions[:, None, :] * depth_values[None, :, None]
            )

            # Incorporate positional encoding into the sample points
            sample_points_encoded = positional_encoding(
                sample_points, num_functions=num_encoding_functions
            )

            # Perform ray marching and compute the rendered image
            rendered_image = ray_march(
                chunk_ray_origins,
                chunk_ray_directions,
                net=nerf_evaluate_and_march,
                near=near,
                far=far,
                num_samples=num_samples_along_ray,
            )
            # The following need to happen in Loma
            # Evaluate and obtain RGB+Density at query points along each ray
            # Render the points along each ray - this must be differentiable!
            # Compute the loss between the new image and the target image
            # TODO: Call the MLP and perform raymarching
            # TODO: Raymarch and accumulate all points - this will produce a new image
            # TODO: Compute the loss
            loss = nerf_evaluate_and_march()

            # Backpropagate the loss

    pass
