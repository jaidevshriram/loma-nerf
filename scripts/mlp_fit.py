def matmult(
    a: In[Array[Array[float]]],
    b: In[Array[Array[float]]]
) -> Out[Array[Array[float]]]:

    output: Array[Array[float, 100], 100]

    i = 0
    while (i < 16, max_iter := 100):
        j = 0
        while (j < 16, max_iter := 100):
            k = 0
            while (k < 16, max_iter := 100):
                output[i][j] = output[i][j] + a[i][k] * b[k][j]
                k = k + 1
            j = j + 1
        i = i + 1

    return output

def matvecmult(
    a: In[Array[Array[float]]],
    b: In[Array[float]]
) -> Out[Array[float]]:
    
    output: Array[float, 100]

    i = 0
    while (i < 16, max_iter := 100):
        j = 0
        while (j < 16, max_iter := 100):
            output[i] = output[i] + a[i][j] * b[j]
            j = j + 1
        i = i + 1

    return output

def matmatadd(
    a: In[Array[Array[float]]],
    b: In[Array[Array[float]]]
) -> Out[Array[Array[float]]]:
    
    output: Array[Array[float, 100], 100]

    i = 0
    while (i < 16, max_iter := 100):
        j = 0
        while (j < 16, max_iter := 100):
            output[i][j] = a[i][j] + b[i][j]
            j = j + 1
        i = i + 1

    return output

def relu(
    x: In[Array[Array[float]]]
) -> Out[Array[Array[float]]]:

    output: Array[Array[float, 100], 100]

    i = 0
    while (i < 16, max_iter := 100):
        j = 0
        while (j < 16, max_iter := 100):
            if x[i][j] > 0:
                output[i][j] = x[i][j]
            else:
                output[i][j] = 0
            j = j + 1
        i = i + 1

    return output

def mse(
    x: In[Array[Array[float]]],
    y: In[Array[Array[float]]]
) -> Out[float]:

    output: float = 0

    i = 0
    while (i < 16, max_iter := 100):
        j = 0
        while (j < 16, max_iter := 100):
            output = output + (x[i][j] - y[i][j]) * (x[i][j] - y[i][j])
            j = j + 1
        i = i + 1

    return output

def mlp_fit(
    x: In[Array[Array[float]]],
    ws: In[Array[Array[Array[float]]]],
    bs: In[Array[Array[float]]],
    target_image: In[Array[Array[float]]],
    target_image_h: In[int],
    target_image_w: In[int]
) -> float:

    i: int = 0
    j: int = 0
    k: int = 0

    layouer_counter: int = 0
    num_layers: int = 3
    layer_input: Array[Array[float], float] = x

    layer_output: Array[Array[float, 100], 100]

    while (layouer_counter < num_layers, max_iter := num_layers):

        weight = ws[layouer_counter]
        bias = bs[layouer_counter]

        # Multiply layer input with weights and add the bias
        layer_output = matvecmult(weight, layer_input)
        layer_output = matmatadd(layer_output, bias)

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