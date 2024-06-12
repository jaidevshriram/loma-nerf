import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
loma_dir = os.path.join(current, "loma_public")
sys.path.append(loma_dir)

import compiler
import ctypes
from ctypes import *
import numpy as np
import numpy.ctypeslib as npct
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import wandb


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