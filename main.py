import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
loma_dir = os.path.join(current, "loma_public")
sys.path.append(loma_dir)

# import torch
import compiler
import ctypes
import numpy as np
import numpy.ctypeslib as npct
import matplotlib.pyplot as plt
from PIL import Image
import pdb


def get_mlp_weights(layer_count: int):

    ws = []
    bs = []

    for i in range(layer_count):
        ws.append(np.random.randn(3, 3).astype(np.float32))
        bs.append(np.random.randn(3).astype(np.float32))

    ws = np.array(ws)
    bs = np.array(bs)

    return ws, bs

if __name__ == "__main__":

    with open("scripts/mlp_fit.py") as f:
        _, lib = compiler.compile(f.read(), target="c", output_filename="_code/mlp_fit")

    exit()

    # Load a target image
    target_image = Image.open("data/warren.jpeg").resize((16, 16))
    target_image_np = (
        np.array(target_image, dtype=np.float32) / 255.0
    ).reshape(-1, 3).T  # Normalize to [0, 1]

    # Create weights for the matrix
    ws, bs = get_mlp_weights(3)

    # Create feature grid
    feature_grid_dims = (16, 16)
    feature_grid = np.random.randn(
        feature_grid_dims[0], feature_grid_dims[1], 3
    ).astype(np.float32).reshape(-1, 3).T

    # Define the rendering function and the gradient function
    f = lib.mlp_fit
    grad_f = lib.grad_mlp_fit

    # Gradient descent loop
    step_size = 1e-2
    # pdb.set_trace()
    loss = [
        f(
            feature_grid.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
            ws.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))),
            bs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))),
            target_image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(target_image_np.shape[0]),
            ctypes.c_int(target_image_np.shape[1]),
        )
    ]

    for i in range(1000):
        gx = np.zeros_like(feature_grid, dtype=np.float32)
        gw = np.zeros_like(ws, dtype=np.float32)
        gb = np.zeros_like(bs, dtype=np.float32)
        gtarget = np.zeros_like(target_image_np, dtype=np.float32)
        gwidth = ctypes.c_int(target_image_np.shape[0])
        gheight = ctypes.c_int(target_image_np.shape[1])
        greturn = ctypes.c_float(0.0)

        grad_f(
            feature_grid.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
            ws.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
            bs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
            target_image_np.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
            ctypes.c_int(target_image_np.shape[0]),
            ctypes.c_int(target_image_np.shape[1]),
            gx.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
            gw.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))),
            gb.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))),
            gtarget.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gwidth,
            gheight,
            greturn
        )
        feature_grid -= step_size * gx
        loss.append(
            f(
                feature_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                target_image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(target_image_np.shape[0]),
                ctypes.c_int(target_image_np.shape[1]),
            )
        )

    plt.plot(np.arange(len(loss)), np.array(loss))
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.show()
