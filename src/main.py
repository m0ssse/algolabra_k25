import numpy as np
import neural_network
import mnist_loader
from pathlib import Path

def help():
    print("Commands: ")
    print("a: Create a new network")
    print("t: Train a network")
    print("s: Show networks")
    print("x: Exit")

def main():
    train_images, _, _, _ = mnist_loader.read_image_data(Path("data/train-images.idx3-ubyte"))
    train_labels, _ = mnist_loader.read_labels(Path("data/train-labels.idx1-ubyte"))
    test_images, _, _, _ = mnist_loader.read_image_data(Path("data/t10k-images.idx3-ubyte"))
    test_labels, _ = mnist_loader.read_labels(Path("data/t10k-labels.idx1-ubyte"))

    batch_size = 32
    batches = mnist_loader.make_batches(train_images, train_labels, batch_size)

    networks = []
    while True:
        help()
        command = input()
        if command=="x":
            return
        if command=="a":
            hidden_neurons = int(input("How many neurons in the hidden layer? "))
            networks.append(neural_network.Network(784, hidden_neurons, 10))
        if command=="s":
            for i, network in enumerate(networks):
                print(f"{i+1}: {network}")
        if command=="t":
            ind = int(input("Choose a network to train "))
            if ind<1 or ind>len(networks):
                continue
            epochs = int(input("Choose the number of epochs "))
            learn_rate = float(input("Choose the learn rate "))
            networks[ind-1].train_network(epochs, batches, learn_rate, test_images, test_labels)



if __name__=="__main__":
    main()