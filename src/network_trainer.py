import numpy as np
import network as nw
#import keras

class NetworkTrainer:    
    def backpropagation(self, network: nw.Network, x: np.ndarray, y: np.ndarray, cost_function: callable) -> tuple:
        """
        This method implements the backpropagation algorithm.

        This is essentially copied exactly from Heli Tuominen's course material (https://tim.jyu.fi/view/143092#neuroverkon-opettaminen). The main difference is
        that all the sums in the material have been written as matrix-vector products

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
        dW = activations[-2]@np.transpose(db)
        delta_weights.append(dW)
        delta_bias.append(db)
        for i in range(-2, -len(network.weights)-1, -1):
            db = derivatives[i]*(network.weights[i+1]@db)
            dW = activations[i-1]@np.transpose(db)
            delta_weights.append(dW)
            delta_bias.append(db)
        return C, delta_weights[::-1], delta_bias[::-1]

    @classmethod
    def quadratic(cls, x, y) -> tuple:
        d = x-y
        return 0.5*np.transpose(d)@d, d
    
if __name__=="__main__":
    w1 = np.array([[0.1, 0.3], [0.2, 0.4]])
    w2 = np.array([[0.5, 0.6], [0.7, 0.8]])
    b1 = 0.25*np.ones((2, 1))
    b2 = 0.35*np.ones((2, 1))
    x = np.array([[0.1], [0.5]])
    y = np.array([[0.05], [0.95]])
    net = nw.Network([2, 2, 2], [nw.Network.sigmoid, nw.Network.sigmoid], [w1, w2], [b1, b2])
    trainer = NetworkTrainer()
    rate = 0.6
    cost, weight_changes, bias_changes = trainer.backpropagation(net, x, y, NetworkTrainer.quadratic)
    for _ in range(100):
        print(cost)
        for i in range(len(net.weights)):
            net.weights[i]-=rate*weight_changes[i]
            net.biases[i]-=rate*bias_changes[i]
        cost, weight_changes, bias_changes = trainer.backpropagation(net, x, y, NetworkTrainer.quadratic)
        
    