import unittest
import mnist_loader as loader
import numpy as np

class DataLoaderTest(unittest.TestCase):
    def test_data_set_size(self):
        images, size, _, _ = loader.MNISTLoader.read_image_data("data/train-images.idx3-ubyte")
        self.assertEqual(len(images), size)

    def test_image_sizes(self):
        images, _, rows, cols = loader.MNISTLoader.read_image_data("data/train-images.idx3-ubyte")
        for im in images:
            self.assertEqual(len(im), rows*cols)

    def test_label_amount(self):
        labels, size = loader.MNISTLoader.read_labels("data/train-labels.idx1-ubyte")
        self.assertEqual(len(labels), size)

    def test_image_amount_vs_label_amount(self):
        _, image_N, _, _ = loader.MNISTLoader.read_image_data("data/train-images.idx3-ubyte")
        _, label_N = loader.MNISTLoader.read_labels("data/train-labels.idx1-ubyte")
        self.assertEqual(image_N, label_N)

    def test_label_range(self):
        labels, _ = loader.MNISTLoader.read_labels("data/train-labels.idx1-ubyte")
        for l in labels:
            self.assertGreaterEqual(l, 0)
            self.assertLessEqual(l, 9)