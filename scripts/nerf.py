def nerf_evaluate_and_march(
    layer_input: In[Array[Array[float]]], # The input usually coordinates of a grid/3d volume
    layer_input_h: In[int], # The height of the input
    layer_input_w: In[int], # The width of the input
    ws: In[Array[Array[Array[float]]]], # The weights array of shape N x weight_shape[0] x weight_shape[1]
    bs: In[Array[Array[float]]], # The bias array of shape N x bias_shape[0]
    target_image: In[Array[Array[float]]], # The target image tensor
    target_image_h: In[int], # The height of the target image tensor
    target_image_w: In[int], # The width of the target image tensor
    num_weights: In[int], # The number of weights
    weight_shapes: In[Array[Array[int]]], # The shapes of the weights [N x 2]
    bias_shapes: In[Array[Array[int]]], # The shapes of the biases [N x 1]
    intermediate_output_shapes: In[Array[Array[int]]], # The shapes of the intermediate outputs [N x 2]
    intermediate_outputs: In[Array[Array[Array[float]]]], # The intermediate outputs of the layers
    img_sample_rgba_arr: In[Array[Array[Array[float]]]], # The image samples of shape (num_pixels_height * num_pixels_width, num_samples, 4)
    num_samples: In[int], # The number of samples
    dists: In[Array[Array[float]]], # The delta t or step size array of shape (num_pixels_height * num_pixels_width, num_samples)
    alpha: In[Array[Array[float]]], # The alpha array of shape (num_pixels_height * num_pixels_width, num_samples)
    cumprod_alpha: In[Array[Array[float]]], # The cumulative product of alpha array of shape (num_pixels_height * num_pixels_width, num_samples)
    weights_samples: In[Array[Array[float]]], # The weights array of shape (num_pixels_height * num_pixels_width, num_samples)
    accumulated_color: In[Array[Array[float]]], # The accumulated color array of shape (num_pixels_height * num_pixels_width, 3)
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

    i_copy: int = 0
    j_copy: int = 0
    k_copy: int = 0

    i_alpha: int = 0
    j_alpha: int = 0

    i_weight: int = 0
    j_weight: int = 0
    
    # Compute loss - MSE
    loss: float = 0

    while (layer_counter < num_weights, max_iter := 3):

    #     # Multiply layer input with weights and add the bias

    #     # ==============
    #     # Matrix Vector Multiplication
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

            # Apply sigmoid or Relu to the last layer
            i_relu = 0
            j_relu = 0

            while (i_relu < intermediate_output_shapes[layer_counter][0], max_iter := 256):
                j_relu = 0
                while (j_relu < intermediate_output_shapes[layer_counter][1], max_iter := 32):

                    if j_relu == 3:
                        # ReLU
                        if intermediate_outputs[layer_counter][i_relu][j_relu] > 0:
                            intermediate_outputs[layer_counter][i_relu][j_relu] = intermediate_outputs[layer_counter][i_relu][j_relu]
                        else:
                            intermediate_outputs[layer_counter][i_relu][j_relu] = 0
                    else:
                        # Sigmoid for color
                        intermediate_outputs[layer_counter][i_relu][j_relu] = 1 / (1 + exp(-intermediate_outputs[layer_counter][i_relu][j_relu]))
                    j_relu = j_relu + 1
                i_relu = i_relu + 1

        # Update layer counter
        layer_counter = layer_counter + 1

    # =================
    # Convert this to N S 4 tensor from the prev output of shape N*S 4
    # =================

    img_size: int = target_image_h # This is the number of pixels in the height*width

    i_copy = 0
    j_copy = 0
    k_copy = 0

    while (i_copy < img_size, max_iter := 256):
        j_copy = 0
        while (j_copy < num_samples, max_iter := 32):
            k_copy = 0
            while (k_copy < 4, max_iter := 5):
                # img_sample_rgba_arr[i_copy][j_copy][k_copy] = img_sample_rgba_arr[i_copy][j_copy][k_copy] + intermediate_outputs[num_weights - 1][i_copy * num_samples + j_copy][k_copy]
                img_sample_rgba_arr[i_copy][j_copy][k_copy] = intermediate_outputs[num_weights - 1][i_copy * num_samples + j_copy][k_copy]
                k_copy = k_copy + 1
            j_copy = j_copy + 1
        i_copy = i_copy + 1

    # =================
    # Accumulate/Render this to get the final color in the 0th channel
    # =================

    i_alpha = 0
    j_alpha = 0

    while (i_alpha < img_size, max_iter := 256):
        j_alpha = 0
        while (j_alpha < num_samples, max_iter := 32):
            alpha[i_alpha][j_alpha] = 1.0 - exp(-img_sample_rgba_arr[i_alpha][j_alpha][3] * (dists[i_alpha][j_alpha]))
            j_alpha = j_alpha + 1
        i_alpha = i_alpha + 1

    # # =================
    # # Compute the cumulative product of alpha
    # # =================

    i_alpha = 0
    j_alpha = 0

    # We want to cmopute cumprod(1 - alpha + 1e-10) for each pixel and sample
    while (i_alpha < img_size, max_iter := 256):
        j_alpha = 0
        while (j_alpha < num_samples, max_iter := 32):
            cumprod_alpha[i_alpha][j_alpha] = 1.0 - alpha[i_alpha][j_alpha] + 1e-10
            j_alpha = j_alpha + 1
        i_alpha = i_alpha + 1

    i_alpha = 0
    j_alpha = 0

    # Actually compute the cumulative product
    while (i_alpha < img_size, max_iter := 256):
        j_alpha = 0
        while (j_alpha < num_samples, max_iter := 32):
            if j_alpha > 0:
                cumprod_alpha[i_alpha][j_alpha] = cumprod_alpha[i_alpha][j_alpha - 1] * cumprod_alpha[i_alpha][j_alpha]
            j_alpha = j_alpha + 1
        i_alpha = i_alpha + 1

    # Shift the cumprod_alpha array by 1
    i_alpha = 0
    j_alpha = 0

    while (i_alpha < img_size, max_iter := 256):
        j_alpha = 0
        while (j_alpha < num_samples, max_iter := 32):
            if j_alpha == 0:
                weights_samples[i_alpha][j_alpha] = alpha[i_alpha][j_alpha]
            else:
                weights_samples[i_alpha][j_alpha] = cumprod_alpha[i_alpha][j_alpha - 1]
            j_alpha = j_alpha + 1
        i_alpha = i_alpha + 1

    # Set all the first element of cumprod_alpha to 1
    i_alpha = 0
    j_alpha = 0

    while (i_alpha < img_size, max_iter := 256):
        j_alpha = 0
        while (j_alpha < num_samples, max_iter := 32):
            if j_alpha == 0:
                cumprod_alpha[i_alpha][j_alpha] = 1
            j_alpha = j_alpha + 1
        i_alpha = i_alpha + 1

    # =================
    # Compute the weights
    # =================

    i_weight = 0
    j_weight = 0

    while (i_weight < img_size, max_iter := 256):
        j_weight = 0
        while (j_weight < num_samples, max_iter := 32):
            weights_samples[i_weight][j_weight] = alpha[i_weight][j_weight] * cumprod_alpha[i_weight][j_weight]
            j_weight = j_weight + 1
        i_weight = i_weight + 1

    # =================
    # Accumulate color
    # =================

    i_weight = 0
    j_weight = 0

    while (i_weight < img_size, max_iter := 256):
        j_weight = 0
        while (j_weight < num_samples, max_iter := 32):
            accumulated_color[i_weight][0] = accumulated_color[i_weight][0] + weights_samples[i_weight][j_weight] * img_sample_rgba_arr[i_weight][j_weight][0]
            accumulated_color[i_weight][1] = accumulated_color[i_weight][1] + weights_samples[i_weight][j_weight] * img_sample_rgba_arr[i_weight][j_weight][1]
            accumulated_color[i_weight][2] = accumulated_color[i_weight][2] + weights_samples[i_weight][j_weight] * img_sample_rgba_arr[i_weight][j_weight][2]
            j_weight = j_weight + 1
        i_weight = i_weight + 1

    # =================
    # Compute loss - MSE
    # =================

    i_mse = 0
    j_mse = 0

    while (i_mse < target_image_h, max_iter := 500):
        j_mse = 0
        while (j_mse < target_image_w, max_iter := 32):
            loss = loss + (accumulated_color[i_mse][j_mse] - target_image[i_mse][j_mse]) * (accumulated_color[i_mse][j_mse] - target_image[i_mse][j_mse])
            j_mse = j_mse + 1
        i_mse = i_mse + 1

    return loss

grad_nerf_evaluate_and_march = rev_diff(nerf_evaluate_and_march)