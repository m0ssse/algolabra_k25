import numpy as np
import neural_network
import mnist_loader
from constants import epochs, batch_size, learn_rate
from pathlib import Path

def main():
    network = neural_network.Network(784, 32, 10)

    train_images, _, _, _ = mnist_loader.read_image_data(Path("data/train-images.idx3-ubyte"))
    train_labels, _ = mnist_loader.read_labels(Path("data/train-labels.idx1-ubyte"))
    test_images, _, _, _ = mnist_loader.read_image_data(Path("data/t10k-images.idx3-ubyte"))
    test_labels, _ = mnist_loader.read_labels(Path("data/t10k-labels.idx1-ubyte"))

    batch_size = 32
    batches = mnist_loader.make_batches(train_images, train_labels, batch_size)

    epochs = 10
    learn_rate = 3

    network.train_network(epochs, batches, learn_rate, test_images, test_labels)

if __name__=="__main__":
    main()