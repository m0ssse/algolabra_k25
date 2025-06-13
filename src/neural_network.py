import numpy as np

class Network:
    def __init__(self, size_in: int, size_hidden: int, size_out: int) -> None:
        self.w_hidden = np.random.rand(size_hidden, size_in)
        self.b_hidden = np.random.rand((size_hidden, 1))

        self.w_out = np.random.rand(size_out, size_hidden)
        self.b_out = np.random.rand((size_out, 1))

    def feed_forward(self, x: np.array):
        """
        Compute the output of the network for the given vector x

        input:
            x: The input vector

        output:
            The weighted inputs, the activations and the derivative of the hidden layer
        """

        z_hidden = self.w_hidden@x+self.b_hidden
        a_hidden, da_hidden = Network.sigmoid(z_hidden)

        z_out = self.w_out@a_hidden+self.b_out
        a_out = Network.softmax(z_out)

        return z_hidden, z_out, a_hidden, a_out, da_hidden
    
    def backpropagation(self, x: np.array, y: np.array):
        """
        Calculates the gradients with respect to weights and biases. These error function used is categorical cross-entropy. More
        information about computing the gradient of this error function with a softmax output layer can be found at
        https://mattpetersen.github.io/softmax-with-cross-entropy


        """
        z_hidden, z_out, a_hidden, a_out, da_hidden = self.feed_forward(x)
        error_out = y-a_out
    
    @classmethod
    def sigmoid(cls, x: np.array) -> np.array:
        """
        Returns the sigmoid function
        """
        x = 1/(1+np.exp(-x))
        return x, x*(1-x)
    
    @classmethod
    def softmax(cls, x: np.array) -> np.array:
        """
        Returns the value of the softmax function. Before computing the exponentials, we subtract the maximum value
        to improve numerical stability. This does not affect the output since this subtraction turns into a constant factor after exponentiation,
        which cancels out
        """
        x = np.exp(x-np.max(x))
        return x/np.sum(x)
        
