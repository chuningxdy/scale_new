import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import a_scale.nn.run_nn_utils as utils

class TwoLayerMLP(eqx.Module):
    """
    A two-layer neural network implemented using Equinox.

    This neural network consists of a hidden layer with ReLU activation
    and an output layer. It is designed to be used with JAX for efficient
    computation.

    Attributes:
        hidden_layer (eqx.nn.Linear): The hidden layer of the network.
        output_layer (eqx.nn.Linear): The output layer of the network.

    Args:
        input_dim (int): The dimensionality of the input.
        hidden_dim (int): The number of neurons in the hidden layer.
        output_dim (int): The dimensionality of the output.
        key (jax.random.PRNGKey): A key for random number generation.
    """

    hidden_layer: eqx.nn.Linear
    output_layer: eqx.nn.Linear

    def __init__(self, key, input_dim=784, output_dim=10, requested_num_parameters=10000):
        hidden_key, output_key = jax.random.split(key)
        hidden_dim = self.get_hidden_dim(requested_num_parameters, input_dim, output_dim)
        self.hidden_layer = eqx.nn.Linear(input_dim, hidden_dim, key=hidden_key)
        self.output_layer = eqx.nn.Linear(hidden_dim, output_dim, key=output_key)
    
    def get_hidden_dim(self, requested_num_parameters, input_dim, output_dim):
        # params = (input_dim * hidden_dim + hidden_dim) + (hidden_dim * output_dim + output_dim)
        # params = (input_dim + 1) * hidden_dim + (hidden_dim + 1) * output_dim
        # params = (input_dim + 1 + output_dim) * hidden_dim + output_dim
        # hidden_dim = (requested_num_parameters - output_dim) // (input_dim + 1 + output_dim)
        return max(1, (requested_num_parameters - output_dim) // (input_dim + 1 + output_dim))

    @eqx.filter_jit
    def __call__(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (jax.numpy.ndarray): The input tensor.

        Returns:
            jax.numpy.ndarray: The output of the neural network.
        """
        x = jax.nn.relu(self.hidden_layer(x.ravel()))
        return self.output_layer(x)

class SimpleCNN(eqx.Module):
    layers: list

    def __init__(self, key, input_channels=1, input_height=28, input_width=28,
                 output_dim=10, requested_num_parameters=1000):
        key1, key2, key3 = jax.random.split(key, 3)

        ks = 3
        pad = 1
        c, h, w = utils.conv_output_dimensions(input_height, input_width, ks, pad, 1, input_channels)
        c, h, w = utils.conv_output_dimensions(h, w, 3, 0, 3, c)
        c, h, w = utils.conv_output_dimensions(h, w, ks, pad, 1, c)
        c, h, w = utils.conv_output_dimensions(h, w, 3, 0, 3, c)
        channels = self.get_channels(ks, input_channels, h, w, requested_num_parameters)
        self.layers = [
            eqx.nn.Conv2d(input_channels, channels, kernel_size=ks, padding=pad, key=key1),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=3, stride=3),
            eqx.nn.Conv2d(channels, channels, kernel_size=ks, padding=pad, key=key2),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=3, stride=3),
            jnp.ravel,
            eqx.nn.Linear(h*w*channels, output_dim, key=key3),
        ]

    def get_channels(self, ks, ic, oh, ow, params):
        """
        Calculate the number of channels for the CNN layers based on target parameter count.

        Args:
            ks (int): Kernel size of the convolutional layers
            ic (int): Number of input channels
            oh (int): Height of the output convolutional layer
            ow (int): Width of the output convolutional layer
            params (int): Target number of parameters for the model

        Returns:
            int: Number of channels to use in convolutional layers
        """
        # c = channels
        # params = (ks ** 2 * c * ic + c) + (ks ** 2 * c ** 2 + c) + oh*ow*c*10 + 10
        # params = ks ** 2 * c ** 2 + (ks ** 2 * ic + oh * ow * 10 + 2) * c + 10
        a = ks ** 2
        b = ks ** 2 * ic + 2 + oh * ow * 10
        c = 10 - params
        return int((np.sqrt(b**2 - 4 * a * c) - b)/(2 * a))

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x