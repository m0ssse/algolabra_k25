#import numpy as np
import neural_network
import mnist_loader
from pathlib import Path

"""
To run the application, navigate to the repository root and type 'python src/main.py' to run the application.
This ensures that the MNIST data is loaded correctly
"""

class NeuralNetworkApplication:
    def __init__(self):
        self.train_images, _, _, _ = mnist_loader.read_image_data(Path("data/train-images.idx3-ubyte"))
        self.train_labels, _ = mnist_loader.read_labels(Path("data/train-labels.idx1-ubyte"))
        self.test_images, _, _, _ = mnist_loader.read_image_data(Path("data/t10k-images.idx3-ubyte"))
        self.test_labels, _ = mnist_loader.read_labels(Path("data/t10k-labels.idx1-ubyte"))

        batch_size = 32
        self.batches = mnist_loader.make_batches(self.train_images, self.train_labels, batch_size)

        self.networks = []

    def help(self):
        print("Commands: ")
        print("a: Create a new network")
        print("t: Train a network")
        print("s: Show networks")
        print("x: Exit")

    def show_networks(self):
        for i, network in enumerate(self.networks):
            print(f"{i+1}: {network}")
    
    def add_network(self):
        hidden_neurons = -1
        while hidden_neurons<0:
            try:
                hidden_neurons = int(input("Choose the number of neurons in the hidden layer or type 0 to go back: "))
                if hidden_neurons<0:
                    raise ValueError
            except ValueError:
                print("The number of neurons must be a positive integer!")
            if hidden_neurons==0:
                return
        self.networks.append(neural_network.Network(784, hidden_neurons, 10))

    def train_network(self):
        if not self.networks:
            print("No networks to train!")
            return
        print(f"There are {len(self.networks)} networks available to train.")
        ind = -1
        while ind<0:
            try:
                ind = int(input(f"Choose a network to train or type 0 to go back: "))
                if ind<0 or ind>len(self.networks):
                    raise ValueError
            except ValueError:
                print("Invalid choice!")
        if ind==0:
            return
        epochs=-1
        learn_rate = -1
        while epochs<0:
            try:
                epochs = int(input("Choose the number of epochs (suitable values should be around 10) or type 0 to cancel: "))
                if epochs<0:
                    raise ValueError
            except ValueError:
                print("The number of epochs must be a positive integer")
        if epochs==0:
            return

        while learn_rate<0:
            try:
                learn_rate = float(input("Choose the learn rate (suitable values should be around 3) or type 0 to cancel: "))
                if learn_rate<0:
                    raise ValueError
            except ValueError:
                print("The learn rate must be a positive number!")
        if learn_rate==0:
            return
        
        show_progress = bool(int(input("Would you like to display training progress? (1: show progress, 0: don't show progress) ")))
        print("Training...")
        self.networks[ind-1].train_network(epochs, self.batches, learn_rate, self.test_images, self.test_labels, show_progress)
        print("Training finished!")
        print()

    def run(self):
        while True:
            self.help()
            command = input()
            if command=="x":
                return
            if command=="a":
                self.add_network()
            if command=="s":
                self.show_networks()
            if command=="t":
                self.train_network()

if __name__=="__main__":
    NeuralNetworkApplication().run()
