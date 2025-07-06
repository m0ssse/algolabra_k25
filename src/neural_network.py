import numpy as np
import mnist_loader
from random import shuffle
from pathlib import Path


class Network:
    def __init__(self, size_in: int, size_hidden: int, size_out: int) -> None:
        """Initializes the network containing an input layer, one hidden layer and an output layer. The weights and biases are randomised.

        input:
            The sizes of the layers.
        """
        
        self.hidden_neurons = size_hidden

        
        self.w_hidden = np.random.randn(size_hidden, size_in)
        self.b_hidden = 0*np.random.randn(size_hidden, 1)

        self.w_out = np.random.randn(size_out, size_hidden)
        self.b_out = 0*np.random.randn(size_out, 1)

        self.accuracy = 0

    def __str__(self):
        return f"{self.hidden_neurons} neurons in the hidden layer. Accuracy {(100*self.accuracy):.2f} %"
        
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
        cost = Network.crossentropy(a_out, y)
        error_out = a_out-y

        dw_out = error_out@a_hidden.T
        db_out = error_out

        error_hidden = (self.w_out.T@error_out)*da_hidden

        dw_hidden = error_hidden@x.T
        db_hidden = error_hidden

        return dw_hidden, db_hidden, dw_out, db_out, error_out, cost
    
    def batch_average(self, batch: list):
        """
        Calculates the average of the gradients of the cost function wrt the parameters of the network over a single batch of training data
        """
        delta_w_hidden, delta_b_hidden, delta_w_out, delta_b_out = [np.zeros_like(arr) for arr in (self.w_hidden, self.b_hidden, self.w_out, self.b_out)]
        total_cost = 0
        for x, label in batch:
            dw_hidden, db_hidden, dw_out, db_out, _, cost = self.backpropagation(x, label)
            delta_w_hidden+=dw_hidden
            delta_b_hidden+=db_hidden
            delta_w_out+=dw_out
            delta_b_out+=db_out
            total_cost+=cost
        return delta_w_hidden/len(batch), delta_b_hidden/len(batch), delta_w_out/len(batch), delta_b_out/len(batch), cost
    
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

    def epoch(self, batches: list, learn_rate: float, update: bool = True) -> np.array:
        """
        Performs a single epoch: i.e. processes all the batches once and updates the parameters of the network after each batch

        inputs:
            batches: A list of batches used for training
            learn_rate: The learn rate used in gradient descent.
            update: A flag that determines whether the computed gradients are used to update the parameters of the network

        outputs:
            The sum of the values of the loss functions over the training data set
            
        """
        total_cost = 0
        for batch in batches:
            *gradients, cost = self.batch_average(batch)
            total_cost+=cost[0] #Cost is a 1x1 array since the network operates on 2d-arrays so we extract the only element
            if update:
                self.update(*gradients, learn_rate)
        return total_cost

    def train_network(self, epochs: int, batches: list, learn_rate: float, test_images: list, test_labels: list, show_progress: bool = False) -> list:
        """
        This method is used to train the network by performing a specified number of epochs. 

        inputs:
            epochs: The number of epochs to perform
            batches: List of training data batches
            learn_rate: The learn rate used in gradient descent
            test_images: The images in the test data set, used to determine the accuracy of the network
            test_labels: The corresponding labels
            show_progress: A flag to determine whether to print how the training is progressing after each epoch
        output:
            a list of total training losses for each epoch
        """
        training_losses = []
        for i in range(epochs):
            shuffle(batches) #The batches are shuffled to reduce the risk of overfitting
            training_loss = self.epoch(batches, learn_rate)
            training_losses.append(training_loss)
            correct_labels = self.check_accuracy(test_images, test_labels)
            self.accuracy = correct_labels/len(test_images)
            if show_progress:
                print(f"epoch {i+1}: {correct_labels}/{len(test_images)} test images labeled correctly. Training loss: {training_loss}")
        return training_losses   

    def check_accuracy(self, test_images: list, test_labels: list):
        """
        This method checks the accuracy of the network by having the network classify each image in a given data set and counting how many are correctly labelled

        inputs:
            test_images: A list of image arrays to be classified
            test_labels: The true labels for each image

        outputs:
            The number of correctly classified images
        """
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
    
    @classmethod
    def crossentropy(cls, x: np.array, y: np.array) -> float:
        """
        Calcualtes the value of the cross-entropy loss function. For the definition, see e.g. https://en.wikipedia.org/wiki/Cross-entropy
        
        While this function is not used during training per se, it is still useful to implement in order to verify that the training loss of the network does in fact decrease over time

        inputs:
            x: The output of the network
            y: The one-hot encoded label
        outputs:
            The value of categorical cross-entropy
        """
        return -np.log(x[np.where(y)]) #Since y is one-hot encoded, we can use np.where to pick the only relevant nonzero element in the sum
    
    

if __name__=="__main__":
    network1 = Network(784, 32, 10)

    train_images, _, _, _ = mnist_loader.read_image_data(Path("data/train-images.idx3-ubyte"))
    train_labels, _ = mnist_loader.read_labels(Path("data/train-labels.idx1-ubyte"))
    test_images, _, _, _ = mnist_loader.read_image_data(Path("data/t10k-images.idx3-ubyte"))
    test_labels, _ = mnist_loader.read_labels(Path("data/t10k-labels.idx1-ubyte"))

    batch_size = 32
    batches = mnist_loader.make_batches(train_images, train_labels, batch_size)

    epochs = 10
    learn_rate = 3

    network1.train_network(epochs, batches, learn_rate, test_images, test_labels, True)

