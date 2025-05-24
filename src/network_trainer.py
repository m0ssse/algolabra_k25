import numpy as np
import network as nw
#import keras

class NetworkTrainer:    
    def backpropagation(self, network: nw.Network, x: np.ndarray, y: np.ndarray, cost_function: callable) -> tuple:
        """
        This method implements the backpropagation algorithm.

        This is essentially copied exactly from Heli Tuominen's course material (https://tim.jyu.fi/view/143092#neuroverkon-opettaminen). The main difference is
        that all the sums in the material have been written as matrix-vector products. The material uses different way of writing the indices so we need to apply transposes to account for the different matrix shapes

        Args:
            network: the network to be trained
            x: the training input
            y: the expected output
            cost_function: The cost function to be minimized

        Output:
            A tuple containing the new changes to be made to each weight matrix and bias vector
        """
        activations, derivatives = network.feed_forward(x)
        output = activations[-1]
        C , dC = cost_function(output, y)
        delta_weights = []
        delta_bias = []
        db = dC*derivatives[-1]
        dW = db@np.transpose(activations[-2])
        delta_weights.append(dW)
        delta_bias.append(db)
        for i in range(-2, -len(network.weights)-1, -1):
            db = derivatives[i]*(np.transpose(network.weights[i+1])@db)
            dW = db@np.transpose(activations[i-1])
            delta_weights.append(dW)
            delta_bias.append(db)
        return C, delta_weights[::-1], delta_bias[::-1]

    @classmethod
    def quadratic(cls, x, y) -> tuple:
        d = x-y
        return 0.5*np.transpose(d)@d, d
