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
from pos_encoding import positional_encoding_3d
from dataloader import NeRFDataset

# Set a seed for reproducibility
np.random.seed(215)

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

    directions = np.stack(
        [
            (i - normalized_K[0, 2]) / normalized_K[0, 0],
            -(j - normalized_K[1, 2]) / normalized_K[1, 1],
            -np.ones_like(i),
        ],
        axis=-1,
    )

    points = np.stack([i, j], axis=0)
    points_homo = np.stack([i, j, np.ones_like(i)], axis=0)

    points_cam = np.linalg.inv(normalized_K) @ points_homo

    R = c2w_poses[:3, :3]
    T = c2w_poses[:3, 3]

    ray_origins = T[None, :].repeat(directions.shape[0], 0)
    ray_directions = directions @ R

    # Normalzie the ray directions
    return ray_origins, ray_directions

# def ray_march(rays_o, rays_d, net, near, far, num_samples):
#     """
#     Perform ray marching to render an image from rays using the MLP.

#     Args:
#         rays_o (np.array): Origin of rays.
#         rays_d (np.array): Direction of rays.
#         net (callable): Neural network function for evaluating RGB and density.
#         near (float): Near bound for ray marching.
#         far (float): Far bound for ray marching.
#         num_samples (int): Number of samples per ray.

#     Returns:
#         np.array: Accumulated color for each ray.
#     """
#     t_vals = np.linspace(near, far, num_samples)
#     points = (
#         rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]
#     )  # Shape: [num_rays, num_samples, 3]

#     colors, densities = net(points.reshape(-1, 3))
#     colors = colors.reshape(-1, num_samples, 3)
#     densities = densities.reshape(-1, num_samples)

#     # Calculate weights for RGBA accumulation
#     alpha = 1.0 - np.exp(-densities * (t_vals[1] - t_vals[0]))
#     weights = alpha * np.cumprod(
#         1.0 - alpha + 1e-10, axis=1
#     )  # Cumulative product along each ray
#     weights[:, 1:] = weights[:, :-1]
#     weights[:, 0] = alpha[:, 0]

#     # Accumulate color along the ray
#     rgb = (weights[:, :, None] * colors).sum(axis=1)

#     return rgb

def run_mlp(
    input,
    ws,
    bs,
):

    """
    Run the MLP on the input.

    Args:
        input: The input to the MLP.
        ws: The weights of the MLP.
        bs: The biases of the MLP.

    Returns:
        The output of the MLP.
    """

    num_layers = len(ws)

    for w, b in zip(ws, bs):
        input = np.matmul(input, w) + b[None, :]

        if num_layers < (len(ws) - 1):
            input = np.maximum(0, input)
        else:
            input[:, :3] = 1 / (1 + np.exp(-input[:, :3]))
            input[:, 3] = np.maximum(0, input[:, 3])

    return input


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        lr_t = self.learning_rate * (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

if __name__ == "__main__":

    wandb.init(project="loma-nerf-fixed")

    # Config
    img_size = 512
    num_iterations = 50000
    num_samples_along_ray = 4
    mlp_max_size = 256  # This is the maximum size that our MLP can process in Loma
    max_img_chunk_size = 256 / num_samples_along_ray
    chunk_size_x = np.floor(max_img_chunk_size**0.5)
    chunk_size = chunk_size_x**2
    num_encoding_functions = 5 # TODO: set this higher
    num_layers_mlp = 3
    mlp_filter_size = 30
    out_channels = 4
    near = 2.0
    far = 6.0
    step_size = 5e-4

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

    # Get shape of intermediate outputs
    # TODO: Magic number 128
    fake_input_coords = np.random.randn(256, in_channels).astype(np.float32)
    intermediate_shapes = trace_mlp_and_get_intermediate_outputs(
        fake_input_coords, list(zip(ws, bs))
    )
    intermediate_shapes = np.array(intermediate_shapes, dtype=np.int32)
    intermediate_shape_max_dims = np.max(intermediate_shapes)
    intermediate_outputs = np.zeros(
        (len(ws), intermediate_shape_max_dims, mlp_max_size), dtype=np.float32
    )

    ws_padded = pad_array(ws)
    bs_padded = pad_array(bs)

    # Keep track of the losses
    losses = []

    # Optimizer
    optimizer = AdamOptimizer(learning_rate=step_size)

    # Main training loop
    for i in tqdm(range(num_iterations)):

        # Choose a random pose and image
        # train_idx = np.random.randint(len(dataset))
        train_idx = 0

        train_data = dataset[train_idx]
        train_img = train_data["image"]
        train_pose = train_data["pose"]
        focal_length = train_data["focal_length"] # TODO: Fix this
        # focal_length = 1.0

        # train_img = np.ones_like(train_img) * 0.5

        K = np.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]]).astype(
            np.float32
        )

        # Sample a bunch of rays
        # TODO: Sample the rays here - this will obtain ray origins and ray directions
        ray_origins, ray_directions = get_rays(img_size, img_size, K, train_pose)
        target_gt = train_img.reshape(-1, 3)

        # Compute chunks
        num_chunks = int(np.ceil((img_size * img_size) / chunk_size))

        # assert num_chunks > 1, "The chunk size is too large."

        for chunk_idx in range(num_chunks):
            chunk_start = int(chunk_idx * chunk_size)
            chunk_end = int(min((chunk_idx + 1) * chunk_size, ray_origins.shape[0]))

            # Get the subset of the chunk rays and gt
            chunk_ray_origins = ray_origins[chunk_start:chunk_end]  # N x 3
            chunk_ray_directions = ray_directions[chunk_start:chunk_end]  # N x 3
            chunk_target_gt = target_gt[chunk_start:chunk_end]  # N x 3

            # Sample a bunch of points along the ray - by assuming a near and far threshold
            depth_values = np.linspace(near, far, num_samples_along_ray)
            # depth_values += (
            #     np.random.randn(num_samples_along_ray)
            #     * (far - near)
            #     / num_samples_along_ray
            # )  # TODO: Perturb the depth values

            sample_points = (
                chunk_ray_origins[:, None, :]
                + chunk_ray_directions[:, None, :] * depth_values[None, :, None]
            )

            # Incorporate positional encoding into the sample points
            sample_points_encoded = positional_encoding_3d(
                sample_points, num_functions=num_encoding_functions
            )

            dists = np.concatenate(
                (
                    depth_values[1:] - depth_values[:-1],
                    np.ones_like(depth_values[:1]) * 1e8,
                )
            )[None, :].repeat(chunk_ray_origins.shape[0], axis=0)

            sample_rgba = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray, 4))
            alpha = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray))
            cumprod_alpha = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray))
            weights_samples = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray))
            accumulated_color = np.zeros((chunk_ray_origins.shape[0], 3))

            accumulated_color_c = convert_ndim_array_to_ndim_ctypes(accumulated_color)

            # The following need to happen in Loma
            # Evaluate and obtain RGB+Density at query points along each ray
            # Render the points along each ray - this must be differentiable!
            # Compute the loss between the new image and the target image
            loss = nerf_evaluate_and_march(
                    # The input to the MLP
                    convert_ndim_array_to_ndim_ctypes(sample_points_encoded.reshape(-1, in_channels)),
                    # The height of the input
                    ctypes.c_int(sample_points_encoded.shape[0] * sample_points_encoded.shape[1]),
                    # The width of the input
                    ctypes.c_int(in_channels),
                    # The weights array of shape N x weight_shape[0] x weight_shape[1]
                    convert_ndim_array_to_ndim_ctypes(ws_padded),
                    # The bias array of shape N x bias_shape[0]
                    convert_ndim_array_to_ndim_ctypes(bs_padded),
                    # The target image tensor
                    convert_ndim_array_to_ndim_ctypes(chunk_target_gt),
                    # The height of the target image tensor
                    ctypes.c_int(chunk_target_gt.shape[0]),
                    # The width of the target image tensor
                    ctypes.c_int(chunk_target_gt.shape[1]),
                    # The number of weights
                    num_layers_mlp,
                    # The shapes of the weights [N x 2]
                    convert_ndim_array_to_ndim_ctypes(ws_shape),
                    # The shapes of the biases [N x 1]
                    convert_ndim_array_to_ndim_ctypes(bs_shape),
                    # The shapes of the intermediate outputs [N x 2]
                    convert_ndim_array_to_ndim_ctypes(intermediate_shapes),
                    # The intermediate outputs of the layers
                    convert_ndim_array_to_ndim_ctypes(intermediate_outputs),
                    # The image sample RGBA tensor
                    convert_ndim_array_to_ndim_ctypes(sample_rgba),
                    # The number of samples along the ray
                    ctypes.c_int(num_samples_along_ray),
                    # The distance between samples
                    convert_ndim_array_to_ndim_ctypes(dists),
                    # The alpha value array 
                    convert_ndim_array_to_ndim_ctypes(alpha),
                    # Cumprod alpha array
                    convert_ndim_array_to_ndim_ctypes(cumprod_alpha),
                    # The set of weights for the density
                    convert_ndim_array_to_ndim_ctypes(weights_samples),
                    # The accumulated color
                    accumulated_color_c
            )

            losses.append(loss)

            d_input = np.zeros_like(sample_points_encoded.reshape(-1, in_channels))
            d_input_height = ctypes.c_int(sample_points_encoded.shape[0] * sample_points_encoded.shape[1])
            d_input_width = ctypes.c_int(in_channels)
            d_ws = np.zeros_like(ws_padded)
            d_bs = np.zeros_like(bs_padded)
            d_target = np.zeros_like(chunk_target_gt)
            d_target_height = ctypes.c_int(chunk_target_gt.shape[0])
            d_target_width = ctypes.c_int(chunk_target_gt.shape[1])
            d_num_layers = ctypes.c_int(num_layers_mlp)
            d_ws_shape = np.zeros_like(ws_shape)
            d_bs_shape = np.zeros_like(bs_shape)
            d_intermediate_shapes = np.zeros_like(intermediate_shapes)
            d_intermediate_outputs = np.zeros_like(intermediate_outputs)
            d_sample_rgba = np.zeros_like(sample_rgba)
            d_num_samples = ctypes.c_int(num_samples_along_ray)
            d_dists = np.zeros_like(dists)
            d_alpha = np.zeros_like(alpha)
            d_cumprod_alpha = np.zeros_like(cumprod_alpha)
            d_weights_samples = np.zeros_like(weights_samples)
            d_accumulated_color = np.zeros_like(accumulated_color)

            d_ws = convert_ndim_array_to_ndim_ctypes(d_ws)
            d_bs = convert_ndim_array_to_ndim_ctypes(d_bs)

            # Compute the gradients
            grad_nerf_evaluate_and_march(
                    # The input to the MLP
                    convert_ndim_array_to_ndim_ctypes(sample_points_encoded.reshape(-1, in_channels)),
                    # THe derivative of the input
                    convert_ndim_array_to_ndim_ctypes(d_input),
                    # The height of the input
                    ctypes.c_int(sample_points_encoded.shape[0] * sample_points_encoded.shape[1]),
                    # The derivative of the loss with respect to the height
                    ctypes.byref(d_input_height),
                    # The width of the input
                    ctypes.c_int(in_channels),
                    # The derivative of the loss with respect to the width
                    ctypes.byref(d_input_width),
                    # The weights array of shape N x weight_shape[0] x weight_shape[1]
                    convert_ndim_array_to_ndim_ctypes(ws_padded),
                    # The derivative of the weights
                    d_ws,
                    # The bias array of shape N x bias_shape[0]
                    convert_ndim_array_to_ndim_ctypes(bs_padded),
                    # The derivative of the bias
                    d_bs,
                    # The target image tensor
                    convert_ndim_array_to_ndim_ctypes(chunk_target_gt),
                    # The derivative of the target
                    convert_ndim_array_to_ndim_ctypes(d_target),
                    # The height of the target image tensor
                    ctypes.c_int(chunk_target_gt.shape[0]),
                    # The derivative of the height
                    ctypes.byref(d_target_height),
                    # The width of the target image tensor
                    ctypes.c_int(chunk_target_gt.shape[1]),
                    # The derivative of the width
                    ctypes.byref(d_target_width),
                    # The number of weights
                    ctypes.c_int(num_layers_mlp),
                    # THe derivative of the number of layers
                    ctypes.byref(d_num_layers),
                    # The shapes of the weights [N x 2]
                    convert_ndim_array_to_ndim_ctypes(ws_shape),
                    # The derivative of the shapes of the weights
                    convert_ndim_array_to_ndim_ctypes(d_ws_shape),
                    # The shapes of the biases [N x 1]
                    convert_ndim_array_to_ndim_ctypes(bs_shape),
                    # The derivative of the shapes of the biases
                    convert_ndim_array_to_ndim_ctypes(d_bs_shape),
                    # The shapes of the intermediate outputs [N x 2]
                    convert_ndim_array_to_ndim_ctypes(intermediate_shapes),
                    # The derivative of the shapes of the intermediate outputs
                    convert_ndim_array_to_ndim_ctypes(d_intermediate_shapes),
                    # The intermediate outputs of the layers
                    convert_ndim_array_to_ndim_ctypes(intermediate_outputs),
                    # The derivative of the intermediate outputs
                    convert_ndim_array_to_ndim_ctypes(d_intermediate_outputs),
                    # The image sample RGBA tensor
                    convert_ndim_array_to_ndim_ctypes(sample_rgba),
                    # The derivative of the image sample RGBA tensor
                    convert_ndim_array_to_ndim_ctypes(d_sample_rgba),
                    # The number of samples along the ray
                    ctypes.c_int(num_samples_along_ray),
                    # The derivative of the number of samples
                    ctypes.byref(d_num_samples),
                    # The distance between samples
                    convert_ndim_array_to_ndim_ctypes(dists),
                    # The derivative of the distances
                    convert_ndim_array_to_ndim_ctypes(d_dists),
                    # The alpha value array
                    convert_ndim_array_to_ndim_ctypes(alpha),
                    # The derivative of the alpha value array
                    convert_ndim_array_to_ndim_ctypes(d_alpha),
                    # Cumprod alpha array
                    convert_ndim_array_to_ndim_ctypes(cumprod_alpha),
                    # The derivative of the cumprod alpha array
                    convert_ndim_array_to_ndim_ctypes(d_cumprod_alpha),
                    # The set of weights for the density
                    convert_ndim_array_to_ndim_ctypes(weights_samples),
                    # The derivative of the weights for the density
                    convert_ndim_array_to_ndim_ctypes(d_weights_samples),
                    # The accumulated color
                    convert_ndim_array_to_ndim_ctypes(accumulated_color),
                    # The derivative of the accumulated color
                    convert_ndim_array_to_ndim_ctypes(d_accumulated_color),
                    # The loss
                    losses[-1]
            )

            # Get the gradients
            d_ws_padded = lp_lp_lp_c_float_to_numpy(d_ws, ws_padded.shape)
            d_bs_padded = lp_lp_c_float_to_numpy(d_bs, bs_padded.shape)
            accumulated_color = lp_lp_c_float_to_numpy(accumulated_color_c, accumulated_color.shape)

            # Check if any nan in the gradients
            if np.isnan(d_ws_padded).any() or np.isnan(d_bs_padded).any():
                print("NaN in gradients")
                pdb.set_trace()
                break

            # Check if the gradient is zero
            if np.allclose(d_ws_padded, 0) and np.allclose(d_bs_padded, 0):
                print("The gradients are zero.")
                pass

            # Update the weights
            # ws_padded -= step_size * d_ws_padded
            # bs_padded -= step_size * d_bs_padded
            ws_padded, bs_padded = optimizer.update([ws_padded, bs_padded], [d_ws_padded, d_bs_padded])

            wandb.log({"loss": loss})

            # Convert the padded arrays to regular arrays
            old_ws = ws
            old_bs = bs

            ws = convert_padded_array_to_regular(ws_padded, [np.array(w).shape for w in ws])
            bs = convert_padded_array_to_regular(bs_padded, [np.array(b).shape for b in bs])

            # Check if the weights have changed
            is_close_ws = [np.allclose(w1, w2) for w1, w2 in zip(ws, old_ws)]
            is_close_bs = [np.allclose(b1, b2) for b1, b2 in zip(bs, old_bs)]

            # Check if any are close
            if all(is_close_ws):
                pass
                # print("The weights have not changed.")
                # pdb.set_trace()
                # break

            if all(is_close_bs):
                pass
                # print("The biases have not changed.")
                # pdb.set_trace()
                # break

            # if i % 25 == 0:
            #     predicted = accumulated_color.reshape(img_size, img_size, 3)

            #     # Recompute loss
            #     loss = (accumulated_color - chunk_target_gt) ** 2
            #     loss = loss.sum()

            #     print("Loss: ", loss)

            #     # # Save the image
            #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            #     ax[0].imshow(train_img)
            #     ax[0].set_title("Target")

            #     ax[1].imshow(predicted)
            #     ax[1].set_title("Prediction")

            #     ax[2].plot(losses)
            #     ax[2].set_title("Loss")

            #     # plt.savefig(f"logs_3d/{i}.png")

            # print("Comparing with target image...", chunk_target_gt.sum())
            # pdb.set_trace()
            # break
   
        # pdb.set_trace()

        print("Iteration: ", i, " Loss: ", sum(losses[-num_chunks:])/num_chunks, num_chunks, " chunks")
    
        if i % 25 == 0:
            # Save the model
            # np.save("models/weights.npy", ws_padded)
            # np.save("models/biases.npy", bs_padded)

            # Save the losses
            # np.save("models/losses.npy", losses)

            ray_origins, ray_directions = get_rays(img_size, img_size, K, train_pose)

            output_img = np.zeros((ray_origins.shape[0], 3))

            num_chunks = int(np.ceil((img_size * img_size) / chunk_size))

            assert num_chunks > 1, "The chunk size is too large."

            for chunk_idx in range(num_chunks):
                chunk_start = int(chunk_idx * chunk_size)
                chunk_end = int(min((chunk_idx + 1) * chunk_size, ray_origins.shape[0]))

                chunk_ray_origins = ray_origins[chunk_start:chunk_end]
                chunk_ray_directions = ray_directions[chunk_start:chunk_end]
                chunk_target_gt = target_gt[chunk_start:chunk_end]

                depth_values = np.linspace(near, far, num_samples_along_ray)

                sample_points = (
                    chunk_ray_origins[:, None, :]
                    + chunk_ray_directions[:, None, :] * depth_values[None, :, None]
                )

                sample_points_encoded = positional_encoding_3d(
                    sample_points, num_functions=num_encoding_functions
                )

                dists = np.concatenate(
                    (
                        depth_values[1:] - depth_values[:-1],
                        np.ones_like(depth_values[:1]) * 1e8,
                    )
                )[None, :].repeat(chunk_ray_origins.shape[0], axis=0)

                sample_rgba = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray, 4))
                alpha = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray))
                cumprod_alpha = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray))
                weights_samples = np.zeros((chunk_ray_origins.shape[0], num_samples_along_ray))
                accumulated_color = np.zeros((chunk_ray_origins.shape[0], 3))

                accumulated_color_c = convert_ndim_array_to_ndim_ctypes(accumulated_color)
                
                
                _ = nerf_evaluate_and_march(
                        # The input to the MLP
                        convert_ndim_array_to_ndim_ctypes(sample_points_encoded.reshape(-1, in_channels)),
                        # The height of the input
                        ctypes.c_int(sample_points_encoded.shape[0] * sample_points_encoded.shape[1]),
                        # The width of the input
                        ctypes.c_int(in_channels),
                        # The weights array of shape N x weight_shape[0] x weight_shape[1]
                        convert_ndim_array_to_ndim_ctypes(ws_padded),
                        # The bias array of shape N x bias_shape[0]
                        convert_ndim_array_to_ndim_ctypes(bs_padded),
                        # The target image tensor
                        convert_ndim_array_to_ndim_ctypes(chunk_target_gt),
                        # The height of the target image tensor
                        ctypes.c_int(chunk_target_gt.shape[0]),
                        # The width of the target image tensor
                        ctypes.c_int(chunk_target_gt.shape[1]),
                        # The number of weights
                        num_layers_mlp,
                        # The shapes of the weights [N x 2]
                        convert_ndim_array_to_ndim_ctypes(ws_shape),
                        # The shapes of the biases [N x 1]
                        convert_ndim_array_to_ndim_ctypes(bs_shape),
                        # The shapes of the intermediate outputs [N x 2]
                        convert_ndim_array_to_ndim_ctypes(intermediate_shapes),
                        # The intermediate outputs of the layers
                        convert_ndim_array_to_ndim_ctypes(intermediate_outputs),
                        # The image sample RGBA tensor
                        convert_ndim_array_to_ndim_ctypes(sample_rgba),
                        # The number of samples along the ray
                        ctypes.c_int(num_samples_along_ray),
                        # The distance between samples
                        convert_ndim_array_to_ndim_ctypes(dists),
                        # The alpha value array 
                        convert_ndim_array_to_ndim_ctypes(alpha),
                        # Cumprod alpha array
                        convert_ndim_array_to_ndim_ctypes(cumprod_alpha),
                        # The set of weights for the density
                        convert_ndim_array_to_ndim_ctypes(weights_samples),
                        # The accumulated color
                        accumulated_color_c
                )
                
                accumulated_color = lp_lp_c_float_to_numpy(accumulated_color_c, accumulated_color.shape)

                output_img[chunk_start:chunk_end] = accumulated_color

            # Volume render now
            # sigma = predictions[:, :, 3]
            # rgb = predictions[:, :, :3]
            # dists = np.concatenate(
            #     (
            #         depth_values[1:] - depth_values[:-1],
            #         np.ones_like(depth_values[:1]) * 1e8,
            #     )
            # )[None, :].repeat(ray_origins.shape[0], axis=0)

            # alpha = 1.0 - np.exp(-sigma * dists)
            # weights = alpha * np.cumprod(1.0 - alpha + 1e-10, axis=1)

            # rgb = (weights[:, :, None] * rgb).sum(axis=1).reshape(img_size, img_size, 3)
            # depth = (weights * depth_values).sum(axis=1).reshape(img_size, img_size)

            rgb = output_img.reshape(img_size, img_size, 3)

            # Save the image
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(train_img)
            ax[0].set_title("Target")

            ax[1].imshow(rgb)
            ax[1].set_title("Prediction")

            # ax[2].imshow(depth, cmap="turbo")
            # ax[2].set_title("Depth")

            ax[2].plot(losses)
            ax[2].set_title("Loss")

            plt.savefig(f"logs_3d/{i}.png")

            # Compute the loss manually
            loss = 0
            for i in range(img_size):
                for j in range(img_size):
                    # Compute the loss
                    loss += ((train_img[i, j] - rgb[i, j]) ** 2).sum()
            # print("Loss: ", loss)
            
            wandb.log({"image": [wandb.Image(plt)]})

            plt.close()
            