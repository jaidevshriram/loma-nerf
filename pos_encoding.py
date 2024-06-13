import numpy as np
import pdb

def positional_encoding_2d(
    input,
    num_functions=5,
):
    """

        Converts the input to a positional encoding for, where the number of encodings is num_functions.

        Args:
            input: A tensor of shape (batch_size, num_features)
            num_functions: The number of positional encodings to use.
        
        Returns:
            A tensor of shape (batch_size, num_features_w_positional_encoding)
    """

    new_input = [input]

    for i in range(num_functions):
        new_input.append(np.sin((2.0 ** i) * input))
        new_input.append(np.cos((2.0 ** i) * input))

    combined_input = np.array(new_input) # (num_functions, batch_size, num_features)

    # Reshape to (batch_size, num_features, num_functions)
    combined_input = np.transpose(combined_input, (1, 0, 2))

    # Combine the last two dimensions
    combined_input = np.reshape(combined_input, (combined_input.shape[0], -1)).astype(np.float32)

    assert np.isclose(input, combined_input[:, :input.shape[1]]).all(), "The input and output are not the same."

    return combined_input

def positional_encoding_3d(
    input,
    num_functions=5,
):

    """
            Converts the input to a positional encoding for, where the number of encodings is num_functions.
    
            Args:
                input: A tensor of shape (batch_size, num_samples, num_features)
                num_functions: The number of positional encodings to use.
            
            Returns:
                A tensor of shape (batch_size, num_samples, num_features_w_positional_encoding)
        """

    new_input = [input]

    for i in range(num_functions):
        new_input.append(np.sin((2.0 ** i) * input))
        new_input.append(np.cos((2.0 ** i) * input))

    combined_input = np.array(new_input) # (num_functions, batch_size, num_samples, num_features)

    # Reshape to (batch_size, num_samples, num_features, num_functions)
    combined_input = np.transpose(combined_input, (1, 2, 0, 3))

    # Combine the last two dimensions
    combined_input = np.reshape(combined_input, (combined_input.shape[0], combined_input.shape[1], -1)).astype(np.float32)

    assert np.isclose(input, combined_input[:, :, :input.shape[2]]).all(), "The input and output are not the same."

    return combined_input