import unittest
import neural_network as nw
import mnist_loader as loader
from pathlib import Path
from random import sample
import numpy as np


class TestNetwork(unittest.TestCase):
    def setUp(self):
        images_train, _, _, _ = loader.read_image_data(Path("data/train-images.idx3-ubyte"))
        labels_train, _ = loader.read_labels(Path("data/train-labels.idx1-ubyte"))

    def test_sigmoid(self):
        val0, deriv0 = nw.Network.sigmoid(0)
        val1, deriv1 = nw.Network.sigmoid(1)
        self.assertEqual(val0, .5)
        self.assertEqual(deriv0, .25)
        self.assertAlmostEqual(val1, 0.731058573) #almostEqual is used to account for rounding
        self.assertAlmostEqual(deriv1, 0.196611933)

    def softmax_output_sums_to_one(self):
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