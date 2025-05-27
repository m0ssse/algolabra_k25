import unittest
import network as nw
import numpy as np


class TestNetwork(unittest.TestCase):
    def test_predetermined_weights(self):
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        weights = [np.random.randint(0, 4, (layer_sizes[i], layer_sizes[i-1])) for i in range(1, network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)], weights)
        for w1, w2 in zip(weights, network.weights):
            self.assertTrue((w1==w2).all())
        x = np.ones((layer_sizes[0], 1))
        network.feed_forward(x) #We pass a vector through the network to verify that the resulting network has consistent sizes

    def test_randomised_weights(self):
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)])
        for w in network.weights:
            self.assertTrue((w<np.ones(w.shape)).all())
            self.assertTrue((w>np.zeros(w.shape)).all())
        x = np.ones((layer_sizes[0], 1))
        network.feed_forward(x)       

    def test_zero_weights(self):
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)], randomise_weights=False)
        for w in network.weights:
            self.assertTrue((w==np.zeros(w.shape)).all())
        x = np.ones((layer_sizes[0], 1))
        network.feed_forward(x)

    def test_predetermined_bias(self):
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        biases = [np.random.randint(0, 10, (layer_sizes[i], 1)) for i in range(1, network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)], biases=biases)
        for b1, b2 in zip(biases, network.biases):
            self.assertTrue((b1==b2).all())
        x = np.ones((layer_sizes[0], 1))
        network.feed_forward(x)

    def test_randomised_bias(self):
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)])
        for b in network.biases:
            self.assertTrue((b<np.ones(b.shape)).all())
            self.assertTrue((b>np.zeros(b.shape)).all())
        x = np.ones((layer_sizes[0], 1))
        network.feed_forward(x)

    def test_zero_bias(self):
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)], randomise_biases=False)
        for b in network.biases:
            self.assertTrue((b==np.zeros(b.shape)).all())
        x = np.ones((layer_sizes[0], 1))
        network.feed_forward(x)

    def test_output_size(self):
        """The output size should be the same as the number of neurons in the final layer"""
        network_size = np.random.randint(2, 7)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)])
        x = np.zeros((layer_sizes[0], 1))
        out, _ = network.feed_forward(x)
        self.assertEqual(out[-1].shape, (layer_sizes[-1], 1))

    def test_output_with_identity_no_bias(self):
        """
        If there are no biases and all activations are identity functions, the network should behave as a matrix product
        """
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        weights = [np.random.randint(0, 4, (layer_sizes[i], layer_sizes[i-1])) for i in range(1, network_size)]
        network = nw.Network(layer_sizes, [nw.Network.identity for _ in range(network_size-1)], weights, None, False, False)
        x = np.eye(layer_sizes[0])
        out, _ = network.feed_forward(x)
        w_prod = weights[0]
        for i in range(1, len(weights)):
            self.assertTrue((w_prod==out[i]).all())
            w_prod = weights[i]@w_prod
        self.assertTrue((w_prod==out[-1]).all())