import unittest
import mnist_loader as loader
from pathlib import Path
import numpy as np

class DataLoaderTest(unittest.TestCase):
    def test_data_set_size(self):
        images, size, _, _ = loader.read_image_data(Path("data/train-images.idx3-ubyte"))
        self.assertEqual(len(images), size)

    def test_image_sizes(self):
        images, _, rows, cols = loader.read_image_data(Path("data/train-images.idx3-ubyte"))
        for im in images:
            self.assertEqual(len(im), rows*cols)

    def test_label_amount(self):
        labels, size = loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        self.assertEqual(len(labels), size)

    def test_image_amount_vs_label_amount(self):
        _, image_N, _, _ = loader.read_image_data(Path("data/train-images.idx3-ubyte"))
        _, label_N = loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        self.assertEqual(image_N, label_N)

    def labels_have_exactly_one_nonzero_element(self):
        labels, _ = loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        for l in labels:
            self.assertEqual(np.count_nonzero(l), 1)

    def lables_have_correct_shape(self):
        labels, _ = loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        for l in labels:
            self.assertEqual(l.shape, (10, 1))