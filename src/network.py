import numpy as np

class Network:
    def __init__(self, neuron_counts: list, activation_functions: list, weights: list = None, biases: list = None, randomise_weights: bool = True, randomise_biases: bool = True):
        """
        Initialize the network.

        Args:
            neuron_counts: The number of neurons in each layer of the network
            activation_functions: The activation function for each layer
            weights: A list of arrays containing to the weight matrix between each layer of neurons. Only the first n-1 matrices will be considered where n is the number of layers
            biases: A list of bias vectors for each layer
            randomise_weights: If no weights are provided and this flag is True, then the weights will be initially randomised. Otherwise, the weights are set to zero
            randomise_bias: If not biases are provided and this flag is True, then the biases will be initially randomised. Otherwise, they will be set to zero
        """
        self.layer_size = neuron_counts
        self.activation_functions = activation_functions
        if weights is None:
            self.weights = []
            if randomise_weights:
                for i in range(1, len(neuron_counts)):
                    n, m = self.layer_size[i-1], self.layer_size[i]
                    self.weights.append(20*np.random.rand(m, n)-10) #The rand function gives uniformly distributed random values in the range 0...1. We apply a linear transformation to widen this distribution
            else:
                for i in range(1, len(neuron_counts)):
                    n, m = self.layer_size[i-1], self.layer_size[i]
                    self.weights.append(np.zeros((m, n)))
        else:
            self.weights = weights
        if biases is None:
            self.biases = []
            if randomise_biases:
                for i in range(1, len(neuron_counts)):
                    m = self.layer_size[i]
                    self.biases.append(20*np.random.rand(m, 1)-10)
            else:
                for i in range(1, len(neuron_counts)):
                    m = self.layer_size[i]
                    self.biases.append(np.zeros((m, 1)))                    


        else:
            self.biases = biases

    def feed_forward(self, x: np.ndarray) -> tuple:
        """
        This method computes the output of the network with the given output. The method returns a tuple containing the weighted sum, activation and the derivation of the activation in each layer.
        The derivative is returned as it is required for backpropagation

        Args:
            x: A numpy array corresponding to the input vector

        Output:
            A tuple whose element is a list containing the activations of each layer and the second element is a list containing the derivatives of the activation of each neuron

        """
        activations = [x]
        derivatives = []
        for w, b, a in zip(self.weights, self.biases, self.activation_functions):
            z = w@x+b
            activation, derivative = a(z)
            activations.append(activation)
            derivatives.append(derivative)
            x = activation
        return activations, derivatives
    
    @classmethod
    def sigmoid(cls, x: np.ndarray) -> tuple:
        """
        The sigmoid activation function.

        Args:
            x: The points where the function is evaluated

        Output:
            A tuple containing the values of the function and its derivative at each points in the input array
        """
        val = 1./(np.exp(-x)+1)
        return val, np.multiply(val, 1-val) #The derivative of sigmoid is sigmoid*(1-sigmoid)

    @classmethod
    def identity(cls, x: np.ndarray) -> tuple:
        return x, np.ones(x.shape)


if __name__=="__main__":
    mynw = Network([2, 3, 4], [Network.identity, Network.identity])
    a, _  = mynw.feed_forward(np.array([[1], [1]]))
    print(a[-1])