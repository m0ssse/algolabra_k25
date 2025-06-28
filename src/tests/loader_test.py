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

    def test_labels_have_exactly_one_nonzero_element(self):
        labels, _ = loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        for l in labels:
            self.assertEqual(np.count_nonzero(l), 1)

    def test_labels_have_correct_shape(self):
        
        labels, _ = loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        for l in labels:
            self.assertEqual(l.shape, (10, 1))

    def test_batching(self):
        images, N, _, _ = loader.read_image_data(Path("data/train-images.idx3-ubyte"))
        labels, _ = loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        for batch_size in (8, 16, 32, 64, 128):
            batches = loader.make_batches(images, labels, batch_size)
            total = 0
            for i in range(len(batches)-1): #All but the last batch should have batch_size items and the total number of items across all batches should equal the number of training items
                batch = batches[i]
                self.assertEqual(len(batch), batch_size)
                total+=len(batch)
            total+=len(batches[-1])
            self.assertLessEqual(len(batches[-1]), batch_size)
            self.assertEqual(total, N)
