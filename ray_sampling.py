import numpy as np


def generate_rays(image_size, num_rays, cameras):
    rays_o = []
    rays_d = []

    for transform_matrix in cameras:
        # Sample random pixels on the image plane
        pixels_x = np.random.randint(0, image_size[0], size=num_rays)
        pixels_y = np.random.randint(0, image_size[1], size=num_rays)

        # Convert pixel coordinates to camera space coordinates
        rays_dir = np.stack(
            [
                (pixels_x - image_size[0] / 2) / image_size[0],
                (pixels_y - image_size[1] / 2) / image_size[1],
                -np.ones(num_rays),
            ],
            axis=-1,
        )

        # Normalize ray directions
        rays_dir = rays_dir / np.linalg.norm(rays_dir, axis=1, keepdims=True)

        # Apply rotation from the transform matrix
        rotation_matrix = np.array(transform_matrix[:3, :3])
        rays_dir = np.dot(rays_dir, rotation_matrix.T)

        # Camera position (translation part of the transform)
        camera_pos = np.array(transform_matrix[:3, 3]).reshape(1, 3)
        camera_pos = np.repeat(camera_pos, num_rays, axis=0)

        rays_o.append(camera_pos)
        rays_d.append(rays_dir)

    # Concatenate rays from all cameras
    rays_o = np.concatenate(rays_o, axis=0)
    rays_d = np.concatenate(rays_d, axis=0)

    return rays_o, rays_d


# Example usage:
# cameras = [
#     np.array(
#         [
#             [-0.999902, 0.004192, -0.013346, -0.053798],
#             [-0.013989, -0.299659, 0.953944, 3.845470],
#             [-0.000001, 0.954037, 0.299688, 1.208082],
#             [0.0, 0.0, 0.0, 1.0],
#         ]
#     ),
#     np.array(
#         [
#             [0.442964, 0.313777, -0.839837, -3.385493],
#             [-0.896540, 0.155031, -0.414948, -1.672709],
#             [0.0, 0.936755, 0.349987, 1.410843],
#             [0.0, 0.0, 0.0, 1.0],
#         ]
#     ),
# ]
# image_size = (800, 600)
# num_rays = 100

# rays_o, rays_d = generate_rays(image_size, num_rays, cameras)

# print(rays_o.shape)
# print(rays_d.shape)

# print(rays_o)
# print(rays_d)
# print(cameras)