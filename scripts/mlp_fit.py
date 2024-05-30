# def matmult(
#     a: In[Array[Array[float]]],
#     b: In[Array[Array[float]]]
# ) -> Out[Array[Array[float]]]:

#     output: Array[Array[float, 100], 100]

#     i = 0
#     while (i < 16, max_iter := 100):
#         j = 0
#         while (j < 16, max_iter := 100):
#             k = 0
#             while (k < 16, max_iter := 100):
#                 output[i][j] = output[i][j] + a[i][k] * b[k][j]
#                 k = k + 1
#             j = j + 1
#         i = i + 1

#     return output

def matvecmult(
    a: In[Array[Array[float, 3], 3]], # The weight usually has shape (3 x 3)
    b: In[Array[Array[float, 3], 256]] # The input usually has shape (256, 3)
) -> Out[Array[Array[float, 3], 256]]:

    output: Array[Array[float, 3], 256] # The output usually has shape (256, 3)

    # Essentialyl, compute b @ a
    i: int = 0 # Row iterator
    j: int = 0 # Column iterator
    k: int = 0 # Inner iterator

    while (i < 256, max_iter := 256):

        j = 0
        while (j < 3, max_iter := 3):

            k = 0
            while (k < 3, max_iter := 3):
                output[i][j] = output[i][j] + a[j][k] * b[i][k]
                k = k + 1
            
            j = j + 1
        
        i = i + 1

    return output

def matvecadd(
    a: In[Array[Array[float, 3], 256]], # The image usually has shape (256, 3)
    b: In[Array[float, 3]]  # The bias usually has shape (3,)
) -> Out[Array[Array[float, 3], 256]]:

    output: Array[Array[float, 3], 256] # The output usually has shape (256, 3)

    i: int = 0
    j: int = 0

    while (i < 256, max_iter := 256):
        j = 0
        while (j < 3, max_iter := 3):
            output[i][j] = a[i][j] + b[j]
            j = j + 1
        i = i + 1

    return output

# def matmatadd(
#     a: In[Array[Array[float]]], # The weight usually has shape (3 x 3)
#     b: In[Array[Array[float]]] # The bias usually has shape (3,)
# ) -> Out[Array[Array[float]]]:
    
#     output: Array[Array[float, 100], 100]

#     i: int = 0
#     j: int = 0

#     while (i < 16, max_iter := 100):
#         j = 0
#         while (j < 16, max_iter := 100):
#             output[i][j] = a[i][j] + b[i][j]
#             j = j + 1
#         i = i + 1

#     return output

def vecvecadd(
    a: In[Array[float, 3]],
    b: In[Array[float, 3]]
) -> Out[Array[float, 3]]:
        
        output: Array[float, 3]
    
        i: int = 0
    
        while (i < 3, max_iter := 3):
            output[i] = a[i] + b[i]
            i = i + 1
    
        return output

def relu(
    x: In[Array[Array[float, 3], 256]]
) -> Out[Array[Array[float, 3], 256]]:

    output: Array[Array[float, 3], 256]

    i: int = 0
    j: int = 0

    while (i < 256, max_iter := 256):
        j = 0
        while (j < 3, max_iter := 3):
            if x[i][j] > 0:
                output[i][j] = x[i][j]
            else:
                output[i][j] = 0
            j = j + 1
        i = i + 1

    return output

def mse(
    x: In[Array[Array[float, 3], 256]],
    y: In[Array[Array[float, 3], 256]]
) -> Out[float]:

    output: float = 0

    i: int = 0
    j: int = 0

    while (i < 256, max_iter := 256):
        j = 0
        while (j < 3, max_iter := 3):
            output = output + (x[i][j] - y[i][j]) * (x[i][j] - y[i][j])
            j = j + 1
        i = i + 1

    return output

def mlp_fit(
    x: In[Array[Array[float, 3], 256]],
    ws: In[Array[Array[Array[float, 3], 3], 3]],
    bs: In[Array[Array[float, 3], 3]],
    target_image: In[Array[Array[float, 3], 256]],
    target_image_h: In[int],
    target_image_w: In[int]
) -> float:

    i: int = 0
    j: int = 0
    k: int = 0
    batch_num: int = 0

    layouer_counter: int = 0
    num_layers: int = 3

    layer_input: Array[Array[float, 3], 256] = x # 256 x 3
    layer_output: Array[Array[float, 3], 256]

    weight: Array[Array[float, 3], 3] # 3 x 3
    bias: Array[float, 3] # 3

    while (layouer_counter < num_layers, max_iter := 3):

        weight = ws[layouer_counter] # 3 x 3
        bias = bs[layouer_counter] # 3

        # Multiply layer input with weights and add the bias
        layer_output = matvecmult(weight, layer_input)
        layer_output = matvecadd(layer_output, bias)

        # Apply ReLU activation function
        layer_output = relu(layer_output)

        # Update layer input
        layer_input = layer_output

        # Update layer counter
        layouer_counter = layouer_counter + 1

    # Compute loss
    loss: float = mse(layer_output, target_image)

    return loss

grad_mlp_fit = rev_diff(mlp_fit)