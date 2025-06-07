import unittest
import network as nw
import network_trainer as nwt
import numpy as np

class TestNetworkTrainer(unittest.TestCase):
    def test_loss_decreases(self):
        network_size = np.random.randint(3, 10)
        layer_sizes = [np.random.randint(2, 10) for _ in range(network_size)]
        network = nw.Network(layer_sizes, [nw.Network.sigmoid for _ in range(network_size-1)])
        x = np.random.rand(layer_sizes[0], 1)
        y = np.random.rand(layer_sizes[-1], 1)
        trainer = nwt.NetworkTrainer()
        cost_prev, delta_w, delta_b = trainer.gradient(network, x, y, nwt.NetworkTrainer.quadratic)
        for _ in range(10):
            for i, (w, dw, b, db) in enumerate(zip(network.weights, delta_w, network.biases, delta_b)):
                network.weights[i] = w-0.1*dw
                network.biases[i] = b-0.1*db
            cost_new, delta_w, delta_b = trainer.gradient(network, x, y, nwt.NetworkTrainer.quadratic)
            self.assertLess(cost_new, cost_prev)
            cost_prev = cost_new
