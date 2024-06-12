def nerf_one_iter(
    layer_input: In[Array[Array[float]]], # The input usually coordinates of a grid/3d volume
    layer_input_h: In[int], # The height of the input
    layer_input_w: In[int], # The width of the input
    layer_output: In[Array[Array[float]]], # The output of the layer - just initialized outside to get the shape correct
    ws: In[Array[Array[Array[float]]]], # The weights array of shape N x weight_shape[0] x weight_shape[1]
    bs: In[Array[Array[float]]], # The bias array of shape N x bias_shape[0]
    target_image: In[Array[Array[float]]], # The target image tensor
    target_image_h: In[int], # The height of the target image tensor
    target_image_w: In[int], # The width of the target image tensor
    num_weights: In[int], # The number of weights
    weight_shapes: In[Array[Array[int]]], # The shapes of the weights [N x 2]
    bias_shapes: In[Array[Array[int]]], # The shapes of the biases [N x 1]
    intermediate_output_shapes: In[Array[Array[int]]], # The shapes of the intermediate outputs [N x 2]
    intermediate_outputs: In[Array[Array[Array[float]]]] # The intermediate outputs of the layers
) -> float:

    # Input is of shape (num_pixels_height, num_pixels_width, num_samples, num_features)
    # so total number of pixels is num_pixels_height * num_pixels_width * num_samples. If this is meant to be 256, at 32 samples per ray, this would be 8x8 pixels.

    # We can assume that it is a pre-processed input to avoid doing reshaping operations here.

    # Then, I can use the same code as below to evaluate teh MLP. This will give me a (num_pixels_height, num_pixels_width, num_samples, 4) tensor. The first 3 channels are RGB and the last channel is density.

    # Then, I need to reshape this to (num_pixels_height, num_pixels_width, num_samples, 4)
    # THen I need to go over height, width, samples, and compute the target color, depth, and accumulation for each pixel.

    # T

    i: int = 0
    j: int = 0
    k: int = 0
    batch_num: int = 0

    layer_counter: int = 0

    # For matvecmult
    i_mult: int = 0
    j_mult: int = 0
    k_mult: int = 0

    i_relu: int = 0
    j_relu: int = 0

    i_mse: int = 0
    j_mse: int = 0
    
    # Compute loss - MSE
    loss: float = 0

    while (layer_counter < num_weights, max_iter := 3):

    #     # Multiply layer input with weights and add the bias

    #     # ==============
    #     # Matrix Vector Multiplication
          # TODO: ADD THE BIAS
    #     # ==============

        if layer_counter == 0:

            i_mult = 0
            j_mult = 0
            k_mult = 0

            while (i_mult < layer_input_h, max_iter := 256):
                j_mult = 0
                while (j_mult < weight_shapes[layer_counter][1], max_iter := 32):
                    k_mult = 0
                    while (k_mult < layer_input_w, max_iter := 32):
                        intermediate_outputs[layer_counter][i_mult][j_mult] = intermediate_outputs[layer_counter][i_mult][j_mult] + layer_input[i_mult][k_mult] * ws[layer_counter][k_mult][j_mult]
                        k_mult = k_mult + 1
                    j_mult = j_mult + 1
                i_mult = i_mult + 1

            # Add the bias
            i_mult = 0
            j_mult = 0

            while (i_mult < intermediate_output_shapes[layer_counter][0], max_iter := 256):
                j_mult = 0
                while (j_mult < intermediate_output_shapes[layer_counter][1], max_iter := 32):
                    intermediate_outputs[layer_counter][i_mult][j_mult] = intermediate_outputs[layer_counter][i_mult][j_mult] + bs[layer_counter][j_mult]
                    j_mult = j_mult + 1
                i_mult = i_mult + 1
        
        else: 

            i_mult = 0
            j_mult = 0
            k_mult = 0

            while (i_mult < intermediate_output_shapes[layer_counter - 1][0], max_iter := 256):
                j_mult = 0
                while (j_mult < weight_shapes[layer_counter][1], max_iter := 32):
                    k_mult = 0
                    while (k_mult < intermediate_output_shapes[layer_counter - 1][1], max_iter := 32):
                        intermediate_outputs[layer_counter][i_mult][j_mult] = intermediate_outputs[layer_counter][i_mult][j_mult] + intermediate_outputs[layer_counter - 1][i_mult][k_mult] * ws[layer_counter][k_mult][j_mult]
                        k_mult = k_mult + 1
                    j_mult = j_mult + 1
                i_mult = i_mult + 1

            # Add the bias
            i_mult = 0
            j_mult = 0

            while (i_mult < intermediate_output_shapes[layer_counter][0], max_iter := 256):
                j_mult = 0
                while (j_mult < intermediate_output_shapes[layer_counter][1], max_iter := 32):
                    intermediate_outputs[layer_counter][i_mult][j_mult] = intermediate_outputs[layer_counter][i_mult][j_mult] + bs[layer_counter][j_mult]
                    j_mult = j_mult + 1
                i_mult = i_mult + 1

    #     # ==============
    #     # ReLU
        # TODO: ReLU leads to segfault due to memory issue
    #     # ==============

        # Only apply RELU to the intermediate outputs
        if layer_counter < (num_weights - 1):
            i_relu = 0
            j_relu = 0

            while (i_relu < intermediate_output_shapes[layer_counter][0], max_iter := 256):
                j_relu = 0
                while (j_relu < intermediate_output_shapes[layer_counter][1], max_iter := 32):
                    if intermediate_outputs[layer_counter][i_relu][j_relu] > 0:
                        intermediate_outputs[layer_counter][i_relu][j_relu] = intermediate_outputs[layer_counter][i_relu][j_relu]
                    else:
                        intermediate_outputs[layer_counter][i_relu][j_relu] = 0
                    j_relu = j_relu + 1
                i_relu = i_relu + 1
        elif layer_counter == (num_weights - 1):

            # Apply sigmoid to the last layer
            i_relu = 0
            j_relu = 0

            while (i_relu < intermediate_output_shapes[layer_counter][0], max_iter := 256):
                j_relu = 0
                while (j_relu < intermediate_output_shapes[layer_counter][1], max_iter := 32):
                    intermediate_outputs[layer_counter][i_relu][j_relu] = 1 / (1 + exp(-intermediate_outputs[layer_counter][i_relu][j_relu]))
                    j_relu = j_relu + 1
                i_relu = i_relu + 1

        # Update layer counter
        layer_counter = layer_counter + 1

    i_mse = 0
    j_mse = 0

    while (i_mse < target_image_h, max_iter := 500):
        j_mse = 0
        while (j_mse < target_image_w, max_iter := 32):
            loss = loss + (intermediate_outputs[num_weights - 1][i_mse][j_mse] - target_image[i_mse][j_mse]) * (intermediate_outputs[num_weights - 1][i_mse][j_mse] - target_image[i_mse][j_mse])
            j_mse = j_mse + 1
        i_mse = i_mse + 1

    return loss


def mult_a_b(
    a: In[Array[Array[float]]],
    a_h: In[int],
    a_w: In[int],
    b: In[Array[Array[float]]],
    b_h: In[int],
    b_w: In[int],
    c: Out[Array[Array[float]]],
):

    i: int = 0
    j: int = 0
    k: int = 0

    while (i < a_h, max_iter := 256):
        j = 0
        while (j < b_w, max_iter := 32):
            k = 0
            while (k < a_w, max_iter := 32):
                c[i][j] = c[i][j] + a[i][k] * b[k][j]
                k = k + 1
            j = j + 1
        i = i + 1

grad_mlp_fit = rev_diff(mlp_fit)
# grad_mlp_fit = fwd_diff(mlp_fit)