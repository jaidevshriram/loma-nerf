import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
loma_dir = os.path.join(current, "loma_public")
sys.path.append(loma_dir)

# import torch
import compiler
import ctypes
from ctypes import *
import numpy as np
import numpy.ctypeslib as npct
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import wandb

from pos_encoding import positional_encoding

# Set a seed for reproducibility
np.random.seed(215)

def get_ndims(arr):
    """
        @param arr: The input array, stored as a list of lists of lists of ... of floats

        @return: The number of dimensions of the array
    """

    ndims = 0
    while isinstance(arr, list):
        ndims += 1
        arr = arr[0]

    return ndims

def convert_ndim_array_to_ndim_ctypes(arr):

    if isinstance(arr, np.ndarray):
        arr = arr.tolist()

    ndims = get_ndims(arr)

    if ndims == 1:
        type_of_arr = type(arr[0])

        c_type_of_arr = {
            int: ctypes.c_int,
            float: ctypes.c_float
        }[type_of_arr]

        return (c_type_of_arr * len(arr))(*arr)
    elif ndims == 2:

        type_of_arr = type(arr[0][0])

        c_type_of_arr = {
            int: ctypes.c_int,
            float: ctypes.c_float
        }[type_of_arr]

        # Define the pointer types
        LP_c_float = POINTER(c_type_of_arr)
        LP_LP_c_float = POINTER(LP_c_float)

        # Convert the list of lists to a ctypes array
        num_rows = len(arr)
        num_cols = len(arr[0])
        
        # Create an array of LP_c_float
        row_pointers = (LP_c_float * num_rows)()

        # Fill the row pointers with the data from the original array
        for i in range(num_rows):
            row = (c_type_of_arr * num_cols)(*arr[i])
            row_pointers[i] = row

        # Create a pointer to the array of row pointers
        pointer_to_row_pointers = cast(row_pointers, LP_LP_c_float)

        return pointer_to_row_pointers

    elif ndims == 3:

        type_of_arr = type(arr[0][0][0])

        c_type_of_arr = {
            int: ctypes.c_int,
            float: ctypes.c_float
        }[type_of_arr]
        
        # Define the pointer types
        LP_c_float = POINTER(c_type_of_arr)
        LP_LP_c_float = POINTER(LP_c_float)
        LP_LP_LP_c_float = POINTER(LP_LP_c_float)

        # Convert the list of lists to a ctypes array
        num_rows = len(arr)
        num_cols = len(arr[0])
        num_depth = len(arr[0][0])

        # Create an array of LP_LP_c_float
        row_pointers = (LP_LP_c_float * num_rows)()

        # Fill the row pointers with the data from the original array
        for i in range(num_rows):
            col_pointers = (LP_c_float * num_cols)()
            for j in range(num_cols):
                col = (c_type_of_arr * num_depth)(*arr[i][j])
                col_pointers[j] = col
            row_pointers[i] = col_pointers

        # Create a pointer to the array of row pointers
        pointer_to_row_pointers = cast(row_pointers, LP_LP_LP_c_float)

        if c_type_of_arr == ctypes.c_int:
            print(f"Pointer to row pointers: {pointer_to_row_pointers}")

        return pointer_to_row_pointers

    else:
        raise ValueError("Unsupported number of dimensions")

def lp_lp_lp_c_float_to_numpy(lp_lp_lp_c_float, shape):
    """
    Convert a LP_LP_LP_c_float to a NumPy array.

        @param lp_lp_lp_c_float: The LP_LP_LP_c_float object
        @param shape: Tuple representing the shape of the 3D array (depth, rows, columns)
        
        @return: NumPy array

        # Example usage:
        # Assuming `lp_lp_lp_c_float` is a ctypes object of type LP_LP_LP_c_float
        # and the shape of the 3D array is known
        # lp_lp_lp_c_float = some_ctypes_function_that_returns_LP_LP_LP_c_float()
        # shape = (depth, rows, columns)

        # array = lp_lp_lp_c_float_to_numpy(lp_lp_lp_c_float, shape)
        # print(array)
    """
    depth, rows, columns = shape
    array_3d = np.zeros(shape, dtype=np.float32)
    
    for d in range(depth):
        for r in range(rows):
            for c in range(columns):
                array_3d[d, r, c] = lp_lp_lp_c_float[d][r][c]
    
    return array_3d

def lp_lp_c_float_to_numpy(lp_lp_c_float, shape):
    """
    Convert a LP_LP_c_float to a NumPy array.

        @param lp_lp_c_float: The LP_LP_c_float object
        @param shape: Tuple representing the shape of the 2D array (rows, columns)
        
        @return: NumPy array
    """
    rows, columns = shape
    array_2d = np.zeros(shape, dtype=np.float32)
    
    for r in range(rows):
        for c in range(columns):
            array_2d[r, c] = lp_lp_c_float[r][c]
    
    return array_2d

def get_linear_weight(in_channels, out_channels):
    """
        @param in_channels: The number of input channels
        @param out_channels: The number of output channels

        @return: Randomly initialized weights of shape (out_channels, in_channels)

    """

    return np.random.randn(in_channels, out_channels).astype(np.float32).tolist()

def get_sample_mlp(in_channels=2, out_channels=3, num_layers=2, filter_size=16):
    """
        @param in_channels: The number of input channels
        @param out_channels: The number of output channels
        @param num_layers: The number of layers

        @return: List of weights and biases
    """

    weights = []
    biases = []
    for i in range(num_layers):

        layer_out_channels = out_channels

        if i != num_layers - 1:
            layer_out_channels = filter_size

        weights.append(get_linear_weight(in_channels, layer_out_channels))
        biases.append(np.random.randn(layer_out_channels).astype(np.float32).tolist())
        in_channels = layer_out_channels

    # Convert to list of lists
    weights = [w for w in weights]
    biases = [b for b in biases]

    return weights, biases

def trace_mlp_and_get_intermediate_outputs(input, mlp):
    """

        @param input: The input to the MLP - (N, in_channels)
        @param mlp: The MLP - List of weights and biases
            example: [ 
                2D array of weights of shape (in_channels, out_channels) for layer 1,
                2D array of weights of shape (out_channels, out_channels) for layer 2,
                ...
            ]

        @return: List of shapes of the intermediate outputs
    """
    
    shapes = [] # List of shapes of the intermediate outputs

    # Initialize the output
    output = input

    # Iterate over the layers
    for i, (w, b) in enumerate(mlp):

        w = np.array(w)
        b = np.array(b)

        print(f"Layer {i} - w: {w.shape}, b: {b.shape}, input: {output.shape}")
        output = output @ w + b[None, :]
        shapes.append(output.shape)

        if i < len(mlp) - 1:
            output = np.maximum(output, 0)  # ReLU

    return shapes

def evaluate_mlp(input, mlp):
    """

        @param input: The input to the MLP - (N, in_channels)
        @param mlp: The MLP - List of weights and biases
            example: [ 
                2D array of weights of shape (in_channels, out_channels) for layer 1,
                2D array of weights of shape (out_channels, out_channels) for layer 2,
                ...
            ]

        @return: The output of the MLP
    """

    # Initialize the output
    output = input

    # Iterate over the layers
    for i, (w, b) in enumerate(mlp):

        w = np.array(w)
        b = np.array(b)

        output = output @ w + b[None, :]

        if i < len(mlp) - 1:
            output = np.maximum(output, 0)  # ReLU
        elif i == len(mlp) - 1:
            output = 1 / (1 + np.exp(-output))  # Sigmoid

    return output

def pad_array(arr):
    """

        @param arr: The input array, stored as a list of lists of lists of ... of floats

        @return: The padded array

        This function will identify the largest size of this multi-dimensional array and pad the array to that size along all dimensions
    """

    ndims = get_ndims(arr)
    max_dims = [0] * ndims

    # Collect all the arrays along each dimension and get the maximum size
    def set_max_dims(arr, max_dims, level=0):
        if level == ndims:
            return

        max_dims[level] = max(max_dims[level], len(arr))
        for a in arr:
            set_max_dims(a, max_dims, level + 1)
        
    set_max_dims(arr, max_dims)

    # Obtain padding size
    padded_array_size = [max_dims[i] for i in range(ndims)]

    # Create a new array with the padded size
    padded_array = np.zeros(padded_array_size, dtype=np.float32)

    # Copy the input array to the padded array
    def copy_to_padded(arr, padded_array, level=0, indices=[]):
        if level == ndims:
            padded_array[tuple(indices)] = arr
            return

        for i, a in enumerate(arr):
            copy_to_padded(a, padded_array, level + 1, indices + [i])

    copy_to_padded(arr, padded_array)

    return padded_array

def convert_padded_array_to_regular(padded_array, og_shapes, level=0, indices=[]):
    """
        @param padded_array: The padded array
        @param og_shape: The original shape of the array
        @param level: The current level of the recursion
        @param indices: The current indices of the recursion

        @return: The original array
    """

    # pdb.set_trace()

    arrs = []

    for i, shape in enumerate(og_shapes):

        arr = padded_array[i]

        if len(shape) == 1:
            arr = arr[:shape[0]]
        elif len(shape) == 2:
            arr = arr[:shape[0], :shape[1]]
        else:
            raise ValueError("Unsupported number of dimensions")

        arrs.append(arr)

    return arrs

if __name__ == "__main__":

    wandb.init(project="loma")

    with open("scripts/mlp_fit.py") as f:
        _, lib = compiler.compile(f.read(), target="c", output_filename="_code/mlp_fit")


    # Define the rendering function and the gradient function
    f = lib.mlp_fit
    mult_a_b = lib.mult_a_b
    grad_f = lib.grad_mlp_fit

    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32) # 3 x 2
    b = np.array([[100], [200]], dtype=np.float32) # 2 x 1
    c = np.array([[0], [0], [0]], dtype=np.float32) # 3 x 1

    c_a = convert_ndim_array_to_ndim_ctypes(a)
    c_b = convert_ndim_array_to_ndim_ctypes(b)
    c_c = convert_ndim_array_to_ndim_ctypes(c)
    mult_a_b(c_a, a.shape[0], a.shape[1], c_b, b.shape[0], b.shape[1], c_c)

    c = lp_lp_c_float_to_numpy(c_c, c.shape)

    assert np.allclose(c, np.array([[500], [1100], [1700]], dtype=np.float32))

    # exit()

    # Load a target image
    img_size = 16
    target_image = Image.open("data/warren.jpeg").resize((img_size, img_size))
    target_color_gt = (
        np.array(target_image, dtype=np.float32) / 255.0
    ).reshape(-1, 3)  # Normalize to [0, 1]
    target_color_gt = np.ascontiguousarray(target_color_gt)

    # Create the input to the MLP
    num_encoding_functions = 5

    input_coords_grid = np.meshgrid(
        np.linspace(0, 1, img_size), np.linspace(0, 1, img_size)
    )
    input_coords_og = np.stack(input_coords_grid, axis=-1).reshape(-1, 2)

    input_coords = positional_encoding(input_coords_og, num_functions=num_encoding_functions)

    # Create the MLP
    num_layers = 3
    in_channels = input_coords.shape[1]
    out_channels = 3
    ws, bs = get_sample_mlp(in_channels=in_channels, out_channels=out_channels, num_layers=num_layers)
    ws_shape = np.array([np.array(w).shape for w in ws], dtype=np.int32)
    bs_shape = np.array([[len(b), 1] for b in bs], dtype=np.int32)

    ws_padded = pad_array(ws)
    bs_padded = pad_array(bs)

    # Outputs - Initialize a tensor for the sake of it
    output_tensor = np.zeros((target_color_gt.shape[0], 3), dtype=np.float32)

    # Intermediate outputs
    intermediate_shapes = trace_mlp_and_get_intermediate_outputs(input_coords, list(zip(ws, bs)))
    intermediate_shapes = np.array(intermediate_shapes, dtype=np.int32)
    intermediate_shape_max_dims = np.max(intermediate_shapes)
    intermediate_outputs = np.zeros((num_layers, intermediate_shape_max_dims, intermediate_shape_max_dims)).astype(np.float32)

    # Gradient descent loop
    step_size = 1e-4
    loss = [
        f(
            # The input to the MLP
            convert_ndim_array_to_ndim_ctypes(input_coords),
            # The height of the input
            ctypes.c_int(input_coords.shape[0]),
            # The width of the input
            ctypes.c_int(input_coords.shape[1]),
            # The output of the layer - just initialized outside to get the shape correct
            convert_ndim_array_to_ndim_ctypes(output_tensor),
            # The weights array of shape N x weight_shape[0] x weight_shape[1]
            convert_ndim_array_to_ndim_ctypes(ws_padded),
            # The bias array of shape N x bias_shape[0]
            convert_ndim_array_to_ndim_ctypes(bs_padded),
            # The target image tensor
            convert_ndim_array_to_ndim_ctypes(target_color_gt),
            # The height of the target image tensor
            ctypes.c_int(target_color_gt.shape[0]),
            # The width of the target image tensor
            ctypes.c_int(target_color_gt.shape[1]),
            # The number of weights
            num_layers,
            # The shapes of the weights [N x 2]
            convert_ndim_array_to_ndim_ctypes(ws_shape),
            # The shapes of the biases [N x 1]
            convert_ndim_array_to_ndim_ctypes(bs_shape),
            # The shapes of the intermediate outputs [N x 2]
            convert_ndim_array_to_ndim_ctypes(intermediate_shapes),
            # The intermediate outputs of the layers
            convert_ndim_array_to_ndim_ctypes(intermediate_outputs),
        )
    ]

    NUM_STEPS = 50000

    for i in range(NUM_STEPS):

        d_input_coords = np.zeros_like(input_coords, dtype=np.float32)
        d_input_height = ctypes.c_int(input_coords.shape[0])
        d_input_width = ctypes.c_int(input_coords.shape[1])
        d_output = np.zeros_like(output_tensor, dtype=np.float32)
        d_ws = np.zeros_like(ws_padded, dtype=np.float32)
        d_bs = np.zeros_like(bs_padded, dtype=np.float32)
        d_target = np.zeros_like(target_color_gt, dtype=np.float32)
        d_target_height = ctypes.c_int(target_color_gt.shape[0])
        d_target_width = ctypes.c_int(target_color_gt.shape[1])
        d_num_layers = ctypes.c_int(num_layers)
        d_ws_shape = np.zeros_like(ws_shape, dtype=np.int32)
        d_bs_shape = np.zeros_like(bs_shape, dtype=np.int32)
        d_intermediate_shapes = np.zeros_like(intermediate_shapes, dtype=np.int32)
        d_intermediate_outputs = np.zeros_like(intermediate_outputs, dtype=np.float32)

        # Make all the non array derivatives - ctypes.byref(...)
        d_input_height = ctypes.byref(d_input_height)
        d_input_width = ctypes.byref(d_input_width)
        d_target_height = ctypes.byref(d_target_height)
        d_target_width = ctypes.byref(d_target_width)
        d_num_layers = ctypes.byref(d_num_layers)
        d_ws = convert_ndim_array_to_ndim_ctypes(d_ws)
        d_bs = convert_ndim_array_to_ndim_ctypes(d_bs)

        grad_f(
            # The input to the MLP
            convert_ndim_array_to_ndim_ctypes(input_coords),
            # The derivative of the loss w.r.t. the input
            convert_ndim_array_to_ndim_ctypes(d_input_coords),
            # The height of the input
            ctypes.c_int(input_coords.shape[0]),
            # The derivative of the loss w.r.t. the height of the input
            d_input_height,
            # The width of the input
            ctypes.c_int(input_coords.shape[1]),
            # The derivative of the loss w.r.t. the width of the input
            d_input_width,
            # The output of the layer - just initialized outside to get the shape correct
            convert_ndim_array_to_ndim_ctypes(output_tensor),
            # The derivative of the loss w.r.t. the output
            convert_ndim_array_to_ndim_ctypes(d_output),
            # The weights array of shape N x weight_shape[0] x weight_shape[1]
            convert_ndim_array_to_ndim_ctypes(ws_padded),
            # The derivative of the loss w.r.t. the weights
            d_ws,
            # The bias array of shape N x bias_shape[0]
            convert_ndim_array_to_ndim_ctypes(bs_padded),
            # The derivative of the loss w.r.t. the biases
            d_bs,
            # The target image tensor
            convert_ndim_array_to_ndim_ctypes(target_color_gt),
            # The derivative of the loss w.r.t. the target
            convert_ndim_array_to_ndim_ctypes(d_target),
            # The height of the target image tensor
            ctypes.c_int(target_color_gt.shape[0]),
            # The derivative of the loss w.r.t. the height of the target image tensor
            d_target_height,
            # The width of the target image tensor
            ctypes.c_int(target_color_gt.shape[1]),
            # The derivative of the loss w.r.t. the width of the target image tensor
            d_target_width,
            # The number of weights
            num_layers,
            # The derivative of the loss w.r.t. the number of weights
            d_num_layers,
            # The shapes of the weights [N x 2]
            convert_ndim_array_to_ndim_ctypes(ws_shape),
            # The derivative of the loss w.r.t. the shapes of the weights
            convert_ndim_array_to_ndim_ctypes(d_ws_shape),
            # The shapes of the biases [N x 1]
            convert_ndim_array_to_ndim_ctypes(bs_shape),
            # The derivative of the loss w.r.t. the shapes of the biases
            convert_ndim_array_to_ndim_ctypes(d_bs_shape),
            # The shapes of the intermediate outputs [N x 2]
            convert_ndim_array_to_ndim_ctypes(intermediate_shapes),
            # The derivative of the loss w.r.t. the shapes of the intermediate outputs
            convert_ndim_array_to_ndim_ctypes(d_intermediate_shapes),
            # The intermediate outputs of the layers
            convert_ndim_array_to_ndim_ctypes(intermediate_outputs),
            # The derivative of the loss w.r.t. the intermediate outputs
            convert_ndim_array_to_ndim_ctypes(d_intermediate_outputs),
            # The return value
            loss[-1]
        )

        # Print gradient stats - min, max, mean
        d_ws_padded = lp_lp_lp_c_float_to_numpy(d_ws, ws_padded.shape)
        d_bs_padded = lp_lp_c_float_to_numpy(d_bs, bs_padded.shape)

        # print(f"Gradient stats for all derivatives:")
        # print("Input coords: ", np.min(d_input_coords), np.max(d_input_coords), np.mean(d_input_coords))
        # print("Output tensor: ", np.min(d_output), np.max(d_output), np.mean(d_output))
        # print("Weights: ", np.min(d_ws_padded), np.max(d_ws_padded), np.mean(d_ws_padded))
        # print("Biases: ", np.min(d_bs_padded), np.max(d_bs_padded), np.mean(d_bs_padded))
        # print("Target: ", np.min(d_target), np.max(d_target), np.mean(d_target))
        # print("Intermediate outputs: ", np.min(d_intermediate_outputs), np.max(d_intermediate_outputs), np.mean(d_intermediate_outputs))

        # Check if any nan in the gradients
        if np.isnan(d_ws_padded).any() or np.isnan(d_bs_padded).any():
            print("NaN in the gradients")

            # Count the percentage of nan values in the gradients
            nan_count = np.isnan(d_ws_padded).sum() + np.isnan(d_bs_padded).sum()
            total_count = d_ws_padded.size + d_bs_padded.size
            nan_percentage = nan_count / total_count
            print(f"Percentage of NaN values in the gradients: {nan_percentage}")

            # Convert nan to zero
            # d_ws_padded = np.nan_to_num(d_ws_padded)
            # d_bs_padded = np.nan_to_num(d_bs_padded)
            pdb.set_trace()
            break

        # Take optimizer steps for weights and biases
        ws_padded -= step_size * d_ws_padded
        bs_padded -= step_size * d_bs_padded

        step_loss = f(
            # The input to the MLP
            convert_ndim_array_to_ndim_ctypes(input_coords),
            # The height of the input
            ctypes.c_int(input_coords.shape[0]),
            # The width of the input
            ctypes.c_int(input_coords.shape[1]),
            # The output of the layer - just initialized outside to get the shape correct
            convert_ndim_array_to_ndim_ctypes(output_tensor),
            # The weights array of shape N x weight_shape[0] x weight_shape[1]
            convert_ndim_array_to_ndim_ctypes(ws_padded),
            # The bias array of shape N x bias_shape[0]
            convert_ndim_array_to_ndim_ctypes(bs_padded),
            # The target image tensor
            convert_ndim_array_to_ndim_ctypes(target_color_gt),
            # The height of the target image tensor
            ctypes.c_int(target_color_gt.shape[0]),
            # The width of the target image tensor
            ctypes.c_int(target_color_gt.shape[1]),
            # The number of weights
            num_layers,
            # The shapes of the weights [N x 2]
            convert_ndim_array_to_ndim_ctypes(ws_shape),
            # The shapes of the biases [N x 1]
            convert_ndim_array_to_ndim_ctypes(bs_shape),
            # The shapes of the intermediate outputs [N x 2]
            convert_ndim_array_to_ndim_ctypes(intermediate_shapes),
            # The intermediate outputs of the layers
            convert_ndim_array_to_ndim_ctypes(intermediate_outputs),
        )

        wandb.log({"loss": step_loss}, step=i)
        loss.append(step_loss)

        if i % 250 == 0:

            print(f"Iteration {i}, loss: {loss[-1]}")

            # Obttain the weights and obtain the final prediction
            ws = convert_padded_array_to_regular(ws_padded, [np.array(w).shape for w in ws])
            bs = convert_padded_array_to_regular(bs_padded, [np.array(b).shape for b in bs])
            final_pred = evaluate_mlp(input_coords, list(zip(ws, bs)))
            final_pred_img = final_pred.reshape(img_size, img_size, 3)

            fig, ax = plt.subplots(1, 3, figsize=(10, 5))

            ax[0].plot(np.arange(len(loss)), np.array(loss))
            ax[0].set_ylabel("loss")
            ax[0].set_xlabel("iteration")

            ax[1].imshow(target_color_gt.reshape(img_size, img_size, 3))
            ax[1].set_title("Target image")

            ax[2].imshow(final_pred_img)
            ax[2].set_title("Predicted image")

            # Save the figure
            plt.savefig(f"logs/iter_{i}.png")

            # Log the figure to wandb
            wandb.log({"final_pred": [wandb.Image(plt)]}, step=i)

            plt.close()

    ws = convert_padded_array_to_regular(ws_padded, [np.array(w).shape for w in ws])
    bs = convert_padded_array_to_regular(bs_padded, [np.array(b).shape for b in bs])
    final_pred = evaluate_mlp(input_coords, list(zip(ws, bs)))
    final_pred_img = final_pred.reshape(img_size, img_size, 3)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].plot(np.arange(len(loss)), np.array(loss))
    ax[0].set_ylabel("loss")
    ax[0].set_xlabel("iteration")

    ax[1].imshow(target_color_gt.reshape(img_size, img_size, 3))
    ax[1].set_title("Target image")

    ax[2].imshow(final_pred_img)
    ax[2].set_title("Predicted image")

    plt.savefig(f"logs/iter_{NUM_STEPS}.png")
    plt.close()