import numpy as np
import network as nw
import network_trainer as nwt
import mnist_loader as loader

def main():
    images_train, train_size, rows, cols = loader.MNISTLoader.read_image_data("data/train-images.idx3-ubyte")
    labels_train, _ = loader.MNISTLoader.read_labels("data/train-labels.idx1-ubyte")
    images_test, test_size, _, _ = loader.MNISTLoader.read_image_data("data/t10k-images.idx3-ubyte")
    labels_test, _ = loader.MNISTLoader.read_labels("data/t10k-labels.idx1-ubyte")

    network = nw.Network([rows*cols, 16, 16, 10], 
                         [nw.Network.sigmoid, nw.Network.sigmoid, nw.Network.sigmoid, nw.Network.sigmoid]
                         )
    
    trainer = nwt.NetworkTrainer()
    epochs = 500
    rate = 0.7
    for i in range(epochs):
        print(f"epoch {i+1}")
        total=0
        for train_im, train_label in zip(images_train, labels_train):
            label_vector = np.zeros((10, 1))
            label_vector[train_label] = 1
            C, delta_W, delta_b = trainer.backpropagation(network, train_im, label_vector, nwt.NetworkTrainer.quadratic)
            total+=C
            for i, (W, dW, b, db) in enumerate(zip(network.weights, delta_W, network.biases, delta_b)):
                network.weights[i] = W-rate*dW
                network.biases[i] = b-rate*db
        correct_labels = 0
        for test_im, test_label in zip(images_test, labels_test):
            output, _ = network.feed_forward(test_im)
            output_label = np.argmax(output[-1])
            correct_labels+=test_label==output_label
        print(f"average cost {total[0, 0]/train_size}")
        print(f"correctly labeled {correct_labels}/{test_size} test images")
        print()
        #rate = 1-correct_labels/test_size



if __name__=="__main__":
    main()