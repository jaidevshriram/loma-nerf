import random
from PIL import Image as PILImage
import math
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
loma_dir = os.path.join(current, 'loma_public')
sys.path.append(loma_dir)

from forward_diff import forward_diff
from reverse_diff import reverse_diff

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

# Function signatures for forward and backward diff
def get_structs():
    return {}

def get_funcs():
    return {}

def get_diff_structs():
    return {}

def get_func_to_fwd():
    return {}

def forward(x: list, w: Weights) -> list:
    z1 = matadd(matmul(w.W1, x), w.b1)
    a1 = []
    i = 0
    while i < len(z1):
        a1.append(sigmoid(z1[i]))
        i += 1
    z2 = matadd(matmul(w.W2, a1), w.b2)
    a2 = []
    i = 0
    while i < len(z2):
        a2.append(sigmoid(z2[i]))
        i += 1
    return a2

# Obtain the necessary dictionaries for differentiation
structs = get_structs()
funcs = get_funcs()
diff_structs = get_diff_structs()
func_to_fwd = get_func_to_fwd()

d_forward = forward_diff('forward_diff_func', structs, funcs, diff_structs, forward, func_to_fwd)

def loss(output: list, target: list) -> float:
    err = 0
    i = 0
    while i < len(output):
        err += (output[i] - target[i]) ** 2
        i += 1
    return err

d_loss = forward_diff('loss_diff_func', structs, funcs, diff_structs, loss, func_to_fwd)

def backprop(x: list, y: list, w: Weights, dw: DWeights, learning_rate: float) -> float:
    pred = forward(x, w)
    d_pred = d_forward(x, w)
    err = loss(pred, y)
    d_err = d_loss(pred, y)
    
    i = 0
    while i < len(dw.dW1):
        j = 0
        while j < len(dw.dW1[0]):
            dw.dW1[i][j] -= learning_rate * d_pred.dval[i] * x[j]
            j += 1
        dw.db1[i] -= learning_rate * d_pred.dval[i]
        i += 1
    
    i = 0
    while i < len(dw.dW2):
        j = 0
        while j < len(dw.dW2[0]):
            dw.dW2[i][j] -= learning_rate * d_err.dval * pred[j]
            j += 1
        dw.db2[i] -= learning_rate * d_err.dval
        i += 1
    
    return err

def train(image: Image, target: Image, w: Weights, epochs: int, learning_rate: float):
    dw = DWeights()
    dw.dW1 = [[0 for _ in range(len(w.W1[0]))] for _ in range(len(w.W1))]
    dw.db1 = [0 for _ in range(len(w.b1))]
    dw.dW2 = [[0 for _ in range(len(w.W2[0]))] for _ in range(len(w.W2))]
    dw.db2 = [0 for _ in range(len(w.b2))]
    
    epoch = 0
    while epoch < epochs:
        y = 0
        while y < len(image.data):
            x = 0
            while x < len(image.data[0]):
                input_vec = [image.data[y * len(image.data[0]) + x].x, image.data[y * len(image.data[0]) + x].y, image.data[y * len(image.data[y]) + x].z]
                target_vec = [target.data[y * len(target.data[0]) + x].x, target.data[y * len(target.data[0]) + x].y, target.data[y * len(target.data[0]) + x].z]
                err = backprop(input_vec, target_vec, w, dw, learning_rate)
                x += 1
            y += 1
        epoch += 1
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Error: {err}")

def load_image(filepath: str, target_size: tuple) -> Image:
    pil_image = PILImage.open(filepath).resize(target_size)
    pil_image = pil_image.convert("RGB")
    data = []
    y = 0
    while y < target_size[1]:
        x = 0
        while x < target_size[0]:
            r, g, b = pil_image.getpixel((x, y))
            data.append(make_vec3(r / 255.0, g / 255.0, b / 255.0))
            x += 1
        y += 1
    image = Image()
    image.data = data
    return image

def main():
    image_width = 64
    image_height = 64
    epochs = 1000
    learning_rate = 0.01
    
    input_image = load_image("/data/warren.jpeg", (image_width, image_height))
    
    # For demonstration, let's set the target image to be the same as input
    target_image = load_image("/data/warren.jpeg", (image_width, image_height))
    
    input_size = 3
    hidden_size = 4
    output_size = 3
    
    w = make_weights(input_size, hidden_size, output_size)
    
    train(input_image, target_image, w, epochs, learning_rate)
    
if __name__ == "__main__":
    main()