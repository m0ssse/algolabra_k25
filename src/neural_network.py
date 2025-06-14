import numpy as np
import mnist_loader
from random import shuffle
from pathlib import Path


class Network:
    def __init__(self, size_in: int, size_hidden: int, size_out: int) -> None:
        self.w_hidden = np.random.randn(size_hidden, size_in)
        self.b_hidden = 0*np.random.randn(size_hidden, 1)

        self.w_out = np.random.randn(size_out, size_hidden)
        self.b_out = 0*np.random.randn(size_out, 1)
        
    def feed_forward(self, x: np.array):
        """
        Compute the output of the network for the given vector x

        input:
            x: The input vector

        output:
            The activations at both layers and the derivative of the activation at the hidden layer
        """

        z_hidden = self.w_hidden@x+self.b_hidden
        a_hidden, da_hidden = Network.sigmoid(z_hidden)

        z_out = self.w_out@a_hidden+self.b_out
        a_out = Network.softmax(z_out)

        return a_hidden, a_out, da_hidden
    
    def backpropagation(self, x: np.array, y: np.array):
        """
        Calculates the gradients with respect to weights and biases. These error function used is categorical cross-entropy. More
        information about computing the gradient of this error function with a softmax output layer can be found at
        https://mattpetersen.github.io/softmax-with-cross-entropy

        inputs:
            x: The input for the network
            y: The expected output corresponding to x
        
        outputs:
            The gradients of the cost function with respect to weights and biases


        """
        a_hidden, a_out, da_hidden = self.feed_forward(x)
        error_out = a_out-y

        dw_out = error_out@a_hidden.T
        db_out = error_out

        error_hidden = (self.w_out.T@error_out)*da_hidden

        dw_hidden = error_hidden@x.T
        db_hidden = error_hidden

        return dw_hidden, db_hidden, dw_out, db_out, error_out
    
    def batch_average(self, batch: list):
        """
        Calculates the average of the gradients of the cost function wrt the parameters of the network over a single batch of training data
        """
        delta_w_hidden, delta_b_hidden, delta_w_out, delta_b_out = [np.zeros_like(arr) for arr in (self.w_hidden, self.b_hidden, self.w_out, self.b_out)]
        for x, label in batch:
            dw_hidden, db_hidden, dw_out, db_out, _ = self.backpropagation(x, label)
            delta_w_hidden+=dw_hidden
            delta_b_hidden+=db_hidden
            delta_w_out+=dw_out
            delta_b_out+=db_out
        return delta_w_hidden/len(batch), delta_b_hidden/len(batch), delta_w_out/len(batch), delta_b_out/len(batch)
    
    def update(self, delta_w_hidden: np.array, delta_b_hidden: np.array, delta_w_out: np.array, delta_b_out: np.array, learn_rate: float) -> None:
        """
        Updates the parameters of the network

        inputs:
            The gradients of the parameters wrt the cost function and learn rate
        """
        self.w_hidden-=learn_rate*delta_w_hidden
        self.b_hidden-=learn_rate*delta_b_hidden
        self.w_out-=learn_rate*delta_w_out
        self.b_out-=learn_rate*delta_b_out

    def epoch(self, batches: list, learn_rate: float) -> None:
        """
        Performs a single epoch: i.e. processes all the batches once and updates the parameters of the network after each batch
        """
        for batch in batches:
            gradients = self.batch_average(batch)
            self.update(*gradients, learn_rate)

    def train_network(self, epochs: int, batches: list, learn_rate: float, test_images: list, test_labels: list):
        for i in range(epochs):
            shuffle(batches)
            self.epoch(batches, learn_rate)
            accuracy = self.check_accuracy(test_images, test_labels)
            print(f"epoch {i+1}: {accuracy} test images labeled correctly")
            
    def check_accuracy(self, test_images: list, test_labels: list):
        res = 0
        for x, label in zip(test_images, test_labels):
            prediction, true_label = np.argmax(self.feed_forward(x)[1]), np.argmax(label)
            res+=prediction==true_label
        return res
        
    
    @classmethod
    def sigmoid(cls, x: np.array) -> tuple:
        """
        Returns the value and derivative of the sigmoid function
        """
        x = 1/(1+np.exp(-x))
        return x, x*(1-x)
    
    @classmethod
    def softmax(cls, x: np.array) -> np.array:
        """
        Returns the value of the softmax function. Before computing the exponentials, we subtract the maximum value
        to improve numerical stability. This does not affect the output since this subtraction turns into a constant factor after exponentiation,
        which cancels out
        """
        x = np.exp(x-np.max(x))
        return x/np.sum(x)
    

if __name__=="__main__":
    print("jeejee")
    network1 = Network(784, 32, 10)

    train_images, _, _, _ = mnist_loader.read_image_data(Path("data/train-images.idx3-ubyte"))
    train_labels, _ = mnist_loader.read_labels(Path("data/train-labels.idx1-ubyte"))
    test_images, _, _, _ = mnist_loader.read_image_data(Path("data/t10k-images.idx3-ubyte"))
    test_labels, _ = mnist_loader.read_labels(Path("data/t10k-labels.idx1-ubyte"))

    batch_size = 32
    batches = mnist_loader.make_batches(train_images, train_labels, batch_size)

    epochs = 10
    learn_rate = 3

    network1.train_network(epochs, batches, learn_rate, test_images, test_labels)

