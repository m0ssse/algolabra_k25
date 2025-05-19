import numpy as np

def sigmoid(x: np.ndarray) -> tuple:
    """
    The sigmoid activation function.

    Args:
        x: The points where the function is evaluated

    Output:
        A tuple containing the values of the function and its derivative at each points in the input array
    """
    val = 1./(np.exp(x)+1)
    return val, np.multiply(val, 1-val) #The derivative of sigmoid is sigmoid*(1-sigmoid)

class Network:
    def __init__(self, neuron_counts: list, activation_functions: list):
        """
        Initialize the network.

        Args:
            neuron_counts: The number of neurons in each layer of the network
        """
        self.layer_size = neuron_counts
        self.activation_functions = activation_functions
        self.weights = []
        self.biases = []
        for i in range(1, len(neuron_counts)):
            n, m = self.layer_size[i-1], self.layer_size[i]
            self.weights.append(np.random.rand(n, m))
            self.biases.append(np.random.rand(m, 1))

    def feed_forward(self, x: np.ndarray) -> tuple:
        """
        This method computes the output of the network with the given output. The method returns the output and a list
        of neuron activations in each layer.

        Args:
            x: A numpy array corresponding to the input vector¨

        """
        activations = []

