import unittest
import neural_network as nw
import mnist_loader as loader
from pathlib import Path
from random import sample
from itertools import pairwise
import numpy as np


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.images_train, _, _, _ = loader.read_image_data(Path("data/train-images.idx3-ubyte"))
        self.labels_train, _ = loader.read_labels(Path("data/train-labels.idx1-ubyte"))

    def test_sigmoid(self):
        val0, deriv0 = nw.Network.sigmoid(0)
        val1, deriv1 = nw.Network.sigmoid(1)
        self.assertEqual(val0, .5)
        self.assertEqual(deriv0, .25)
        self.assertAlmostEqual(val1, 0.731058573) #almostEqual is used to account for rounding
        self.assertAlmostEqual(deriv1, 0.196611933)

    def test_softmax_output_sums_to_one(self):
        factor = 10*np.random.randn()
        x = np.ones(10)*factor
        self.assertEqual(np.sum(nw.Network.softmax(x)), 1)

    def test_softmax_when_all_elements_are_the_same(self):
        factor = 10*np.random.randn()
        x = np.ones(10)*factor
        np.testing.assert_almost_equal(nw.Network.softmax(x), x/(factor*10))

    def test_softmax_offset(self):
        """
        The output of the softmax should stay the same if a constant offset is applied to each element in the array
        """
        x = 10*np.random.rand(10)
        ref_value = nw.Network.softmax(x)
        tests = 10
        for _ in range(tests):
            offset = 10*np.random.randn()
            np.testing.assert_almost_equal(nw.Network.softmax(x-offset), ref_value) 

    def test_network_overfits_with_few_training_examples(self):
        train_data = [list(zip(self.images_train, self.labels_train))[:4]]
        test_images = self.images_train[:4]
        test_labels = self.labels_train[:4]
        network = nw.Network(784, 128, 10)
        learn_rate = 3
        upper_limit = 30
        for i in range(upper_limit):
            network.epoch(train_data, learn_rate)
            if network.check_accuracy(test_images, test_labels)==len(test_labels):
                break
        self.assertLess(i, upper_limit-1)

    def test_training_loss_decreases(self):
        network = nw.Network(784, 128, 10)
        data_train = sample(list(zip(self.images_train, self.labels_train)), 128)
        batch = [data_train]
        learn_rate = 3
        epochs = 10
        training_losses = []
        for _ in range(epochs):
            training_losses.append(network.epoch(batch, learn_rate))
        self.assertGreater(training_losses[0], training_losses[-1])

    def test_parameters_change(self):
        """
        To test that the network parameters change on each iteration, we check that all gradients are nonzero. To do this
        we compare each gradient component to zero using np.isclose function, which returns a boolean array where each element
        is either True or False depending on whether the corresponding elements in the parameter arrays are close up to some default
        tolerance. We want this to return an array where every element is False so to check this, we use np.logical_not to perform elementwise
        not on the resulting array and finally calling .all() to check whether each element in the resulting array is True
        """
        network = nw.Network(784, 128, 10)
        data_train = sample(list(zip(self.images_train, self.labels_train)), 128)
        batch = [data_train]
        learn_rate = 3
        epochs = 10
        for _ in range(epochs):
            delta_w_hidden, delta_b_hidden, delta_w_out, delta_b_out, _ = network.batch_average(batch[0])
            res = np.logical_not(np.isclose(delta_w_hidden, np.ones(delta_w_hidden.shape))).all() and ...
            np.logical_not(np.isclose(delta_b_hidden, np.ones(delta_b_hidden.shape))).all() and ...
            np.logical_not(np.isclose(delta_w_out, np.ones(delta_w_out.shape))).all() and ...
            np.logical_not(np.isclose(delta_b_out, np.ones(delta_b_out.shape))).all()
            self.assertTrue(res)
            network.update(delta_w_hidden, delta_b_hidden, delta_w_out, delta_b_out, learn_rate)

        
        
