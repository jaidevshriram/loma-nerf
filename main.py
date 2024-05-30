import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
loma_dir = os.path.join(current, 'loma_public')
sys.path.append(loma_dir)

# import torch
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Vec3:
    x: float
    y: float
    z: float

class Image:
    data: list

class Weights:
    W1: list
    b1: list
    W2: list
    b2: list

class DWeights:
    dW1: list
    db1: list
    dW2: list
    db2: list

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x: float) -> float:
    sig = sigmoid(x)
    return sig * (1 - sig)

def make_vec3(x: float, y: float, z: float) -> Vec3:
    ret = Vec3()
    ret.x = x
    ret.y = y
    ret.z = z
    return ret

def make_weights(input_size: int, hidden_size: int, output_size: int) -> Weights:
    w = Weights()
    w.W1 = []
    i = 0
    while i < hidden_size:
        w.W1.append([])
        j = 0
        while j < input_size:
            w.W1[i].append(random.uniform(-0.01, 0.01))
            j += 1
        i += 1
    w.b1 = [0 for _ in range(hidden_size)]
    w.W2 = []
    i = 0
    while i < output_size:
        w.W2.append([])
        j = 0
        while j < hidden_size:
            w.W2[i].append(random.uniform(-0.01, 0.01))
            j += 1
        i += 1
    w.b2 = [0 for _ in range(output_size)]
    return w

def matmul(A: list, B: list) -> list:
    result = []
    i = 0
    while i < len(A):
        result.append(0)
        j = 0
        while j < len(B):
            result[i] += A[i][j] * B[j]
            j += 1
        i += 1
    return result

def matadd(A: list, B: list) -> list:
    result = []
    i = 0
    while i < len(A):
        result.append(A[i] + B[i])
        i += 1
    return result

def get_mlp_weights(layer_count: int):
    
    ws = []
    bs = []
    
    for i in range(layer_count):
        ws.append(np.random.randn(16, 16).astype(np.float32))
        bs.append(np.random.randn(16).astype(np.float32))

    return ws, bs

if __name__ == '__main__':

    with open('scripts/mlp_fit.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/mlp_fit')

    # Load a target image
    target_image = Image.open('data/warren.jpeg')
    target_image_np = np.array(target_image, dtype=np.float32) / 255.0 # Normalize to [0, 1]

    # Create weights for the matrix
    ws, bs = get_mlp_weights(3)

    # Create feature grid
    feature_grid_dims = (32, 32)
    feature_grid = np.random.randn(feature_grid_dims[0], feature_grid_dims[1], 3).astype(np.float32)

    # Define the rendering function and the gradient function
    f = lib.mlp_fit
    grad_f = lib.grad_mlp_fit

    # Gradient descent loop
    step_size = 1e-2
    loss = [f(feature_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              bs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              target_image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              ctypes.c_int(target_image_np.shape[0]),
              ctypes.c_int(target_image_np.shape[1]))]

    for i in range(1000):
        gx = np.zeros_like(feature_grid, dtype=np.float32)
        grad_f(feature_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               bs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               target_image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               ctypes.c_int(target_image_np.shape[0]),
               ctypes.c_int(target_image_np.shape[1]),
               gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        feature_grid -= step_size * gx
        loss.append(f(feature_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      bs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      target_image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      ctypes.c_int(target_image_np.shape[0]),
                      ctypes.c_int(target_image_np.shape[1]))
        )

    plt.plot(np.arange(len(loss)), np.array(loss))
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.show()
    

