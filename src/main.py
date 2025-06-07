import numpy as np
import network as nw
import network_trainer as nwt
import mnist_loader as loader
from constants import epochs, batch_size, learn_rate

def main():
    images_train, train_size, rows, cols = loader.MNISTLoader.read_image_data("data/train-images.idx3-ubyte")
    labels_train, _ = loader.MNISTLoader.read_labels("data/train-labels.idx1-ubyte")
    images_test, test_size, _, _ = loader.MNISTLoader.read_image_data("data/t10k-images.idx3-ubyte")
    labels_test, _ = loader.MNISTLoader.read_labels("data/t10k-labels.idx1-ubyte")

    network = nw.Network([rows*cols, 10, 10, 10], 
                         [nw.Network.sigmoid, nw.Network.sigmoid, nw.Network.sigmoid, nw.Network.sigmoid]
                         )
    trainer = nwt.NetworkTrainer()
    batches = trainer.make_batches(images_train, labels_train, batch_size)
    for i in range(epochs):
        print(f"epoch {i+1}")
        total = trainer.epoch(network, batches, nwt.NetworkTrainer.quadratic, learn_rate, True)
        correct_labels = 0
        for test_im, test_label in zip(images_test, labels_test):
            output, _ = network.feed_forward(test_im)
            output_label = np.argmax(output[-1])
            correct_labels+=test_label==output_label
        print(f"average training loss {total[0, 0]/train_size}")
        print(f"correctly labeled {correct_labels}/{test_size} test images")
        print()

if __name__=="__main__":
    main()