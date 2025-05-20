import numpy as np

class Network:
    def __init__(self, neuron_counts: list, activation_functions: list, weights: list = None, biases: list = None):
        """
        Initialize the network.

        Args:
            neuron_counts: The number of neurons in each layer of the network
        """
        self.layer_size = neuron_counts
        self.activation_functions = activation_functions
        if weights is None:
            self.weights = []
            for i in range(1, len(neuron_counts)):
                n, m = self.layer_size[i-1], self.layer_size[i]
                self.weights.append(np.random.rand(n, m))
        else:
            self.weights = weights
        if biases is None:
            self.biases = []
            for i in range(1, len(neuron_counts)):
                m = self.layer_size[i]
                self.biases.append(np.random.rand(m, 1))
        else:
            self.biases = biases

    def feed_forward(self, x: np.ndarray) -> tuple:
        """
        This method computes the output of the network with the given output. The method returns a tuple containing the weighted sum, activation and the derivation of the activation in each layer

        Args:
            x: A numpy array corresponding to the input vector¨

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
    w1 = np.ones((2, 2))
    w2 = np.eye(2)
    b = np.zeros((2, 1))
    my_network = Network([2, 2, 2], [Network.sigmoid, Network.identity], [w2, w1], [b, b])
    x = np.array([[3], [1]])
    a, d = my_network.feed_forward(x)
    print(a)