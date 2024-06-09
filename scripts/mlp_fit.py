def mlp_fit(
    layer_input: In[Array[Array[float]]], # The input usually coordinates of a grid/3d volume
    layer_input_h: In[int], # The height of the input
    layer_input_w: In[int], # The width of the input
    layer_output: In[Array[Array[float]]], # The output of the layer - just initialized outside to get the shape correct
    ws: In[Array[Array[Array[float]]]], # The weights array of shape N x weight_shape[0] x weight_shape[1]
    bs: In[Array[Array[Array[float]]]], # The bias array of shape N x bias_shape[0]
    target_image: In[Array[Array[float]]], # The target image tensor
    target_image_h: In[int], # The height of the target image tensor
    target_image_w: In[int], # The width of the target image tensor
    num_weights: In[int], # The number of weights
    weight_shapes: In[Array[Array[int]]], # The shapes of the weights [N x 2]
    bias_shapes: In[Array[Array[int]]], # The shapes of the biases [N x 1]
    intermediate_output_shapes: In[Array[Array[int]]], # The shapes of the intermediate outputs [N x 2]
    intermediate_outputs: In[Array[Array[Array[float]]]] # The intermediate outputs of the layers
) -> float:

    i: int = 0
    j: int = 0
    k: int = 0
    batch_num: int = 0

    layer_counter: int = 0
    num_layers: int = 3

    # For matvecmult
    i_mult: int = 0
    j_mult: int = 0
    k_mult: int = 0

    i_relu: int = 0
    j_relu: int = 0

    i_mse: int = 0
    j_mse: int = 0
    
    while (layer_counter < num_layers, max_iter := 1000):

        # Multiply layer input with weights and add the bias
        # if layer_counter == 0:
        # matvecmult(ws[layer_counter], layer_input, weight_shapes[layer_counter][0], weight_shapes[layer_counter][1], layer_input_h, layer_input_w, intermediate_outputs[layer_counter])

        # ==============
        # Matrix Vector Multiplication
        # ==============

        if layer_counter == 0:

            i_mult = 0
            j_mult = 0
            k_mult = 0

            while (i_mult < weight_shapes[layer_counter][0], max_iter := 1000):
                    j_mult = 0
                    while (j_mult < weight_shapes[layer_counter][1], max_iter := 1000):
                        k_mult = 0
                        while (k_mult < layer_input_w, max_iter := 1000):
                            intermediate_outputs[layer_counter][i_mult][j_mult] = intermediate_outputs[layer_counter][i_mult][j_mult] + ws[layer_counter][i_mult][j_mult] * layer_input[i_mult][k_mult]
                            k_mult = k_mult + 1
                        j_mult = j_mult + 1
                    i_mult = i_mult + 1
        
        else: 
        # TODO: Broken - segfault atm

            i_mult = 0
            j_mult = 0
            k_mult = 0

            while (i_mult < weight_shapes[layer_counter][0], max_iter := 1000):
                    j_mult = 0
                    while (j_mult < weight_shapes[layer_counter][1], max_iter := 1000):
                        k_mult = 0
                        while (k_mult < intermediate_output_shapes[layer_counter - 1][1], max_iter := 1000):
                            intermediate_outputs[layer_counter][i_mult][j_mult] = intermediate_outputs[layer_counter][i_mult][j_mult] + ws[layer_counter][i_mult][j_mult] * intermediate_outputs[layer_counter - 1][i_mult][k_mult]
                            k_mult = k_mult + 1
                        j_mult = j_mult + 1
                    i_mult = i_mult + 1

        # ==============
        # ReLU
        # TODO: Broken - segfault atm
        # ==============
        i_relu = 0
        j_relu = 0

        while (i_relu < intermediate_output_shapes[layer_counter][0], max_iter := 1000):
            j_relu = 0
            while (j_relu < intermediate_output_shapes[layer_counter][1], max_iter := 1000):
                if intermediate_outputs[layer_counter][i_relu][j_relu] > 0:
                    intermediate_outputs[layer_counter][i_relu][j_relu] = intermediate_outputs[layer_counter][i_relu][j_relu]
                else:
                    intermediate_outputs[layer_counter][i_relu][j_relu] = 0
                j_relu = j_relu + 1
            i_relu = i_relu + 1

        # Update layer counter
        layer_counter = layer_counter + 1

    # Compute loss - MSE
    loss: float = 0

    i_mse = 0
    j_mse = 0

    while (i_mse < target_image_h, max_iter := 1000):
        j_mse = 0
        while (j_mse < target_image_w, max_iter := 1000):
            loss = loss + (layer_input[i_mse][j_mse] - target_image[i_mse][j_mse]) * (layer_input[i_mse][j_mse] - target_image[i_mse][j_mse])
            j_mse = j_mse + 1
        i_mse = i_mse + 1

    return loss

grad_mlp_fit = rev_diff(mlp_fit)
# grad_mlp_fit = fwd_diff(mlp_fit)