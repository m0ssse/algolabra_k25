import numpy as np
import network as nw
from random import shuffle
#import keras

class NetworkTrainer:    
    def gradient(self, network: nw.Network, y: np.ndarray, activations: np.ndarray, derivatives: np.ndarray, cost_function: callable) -> tuple:
        """
        This method calculates the gradient of the cost function with respect to the parameters of the given network

        This is essentially copied exactly from Heli Tuominen's course material (https://tim.jyu.fi/view/143092#neuroverkon-opettaminen). The main difference is
        that all the sums in the material have been written as matrix-vector products. The material uses different way of writing the indices so we need to apply transposes to account for the different matrix shapes

        Args:
            network: the network to be trained
            y: the expected output
            activations: the activations from the corresponding forward pass
            derivatives: the derivatives of the activation functions from the forward pass
            cost_function: The cost function

        Output:
            A tuple containing the value of the cost function from the forward pass, the derivatives of the cost function wrt weights and derivatives of the cost function wrt biases
        """
        output = activations[-1]
        C , dC = cost_function(output, y)
        delta_weights = []
        delta_bias = []
        db = dC*derivatives[-1]
        dW = db@np.transpose(activations[-2])
        delta_weights.append(dW)
        delta_bias.append(db)
        for i in range(-2, -len(network.weights)-1, -1):
            db = derivatives[i]*(np.transpose(network.weights[i+1])@db)
            dW = db@np.transpose(activations[i-1])
            delta_weights.append(dW)
            delta_bias.append(db)
        return C, delta_weights[::-1], delta_bias[::-1]
    
    def make_batches(self, train_images: list, train_labels: list, batch_size: int) -> list:
        """
        Split the training data into batches where each batch contains a subset of the training data. 
        The size of each batch (except possibly the last one if the nmber of training examples is not 
        divisible by batch size) is the same

        inputs:
            train_images: a list of training images. Each training image is a numpy array
            train_labels: a list of corresponding labels
            batch size: the size of each batch
        """
        batches = []
        N = (len(train_images)-1)//batch_size+1 #Division that rounds up
        for i in range(N):
            batches.append((train_images[i*batch_size:(i+1)*batch_size], train_labels[i*batch_size:(i+1)*batch_size]))
        return batches
    
    def batch_average(self, network: nw.Network, batch: list, cost_function: callable) -> tuple:
        """
        Calculate the average cost and gradient for a given batch

        params:
            network: the network that is being trained
            batch: a batch containing a subset of training data and the corresponding labels
            cost_function: the cost function used to train the network
        """
        train_images, train_labels = batch
        N = len(train_images)
        average_cost = 0
        average_dw = [np.zeros(w.shape) for w in network.weights]
        average_db = [np.zeros(b.shape) for b in network.biases]
        #Iterate over the data in each batch and calculate the average cost and gradient
        for train_image, train_label in zip(train_images, train_labels):
            label_vector = np.zeros((10, 1)) #Use one-hot encoding for the labels
            label_vector[train_label] = 1
            activations, derivatives = network.feed_forward(train_image)
            C, dW, db = self.gradient(network, label_vector, activations, derivatives, cost_function)
            average_cost+=C
            for i, (grad_w, grad_b) in enumerate(zip(dW, db)):
                average_dw[i]+=grad_w
                average_db[i]+=grad_b
        average_cost/=N
        for i in range(len(average_dw)):
            average_dw[i]/=N
            average_db[i]/=N
        return average_cost, average_dw, average_db
    
    def epoch(self, network: nw.Network, batches: list, cost_function: callable, learning_rate: float, randomise_batches: bool = True) -> None:
        """
        This method essentially implements the backpropagation algorithm, which is a fancy name for gradient descent.
        The method iterates over the batched training data and determines the average gradients for each batch and finally updates the weights
        """
        total_cost = 0
        if randomise_batches:
            shuffle(batches)
        for batch in batches:
            average_cost, average_dw, average_db = self.batch_average(network, batch, cost_function)
            for i, (dw, db) in enumerate(zip(average_dw, average_db)):
                network.weights[i]-=learning_rate*dw
                network.biases[i]-=learning_rate*db
            total_cost+=average_cost*len(batch)
        return total_cost
            



    @classmethod
    def quadratic(cls, x, y) -> tuple:
        d = x-y
        return 0.5*np.transpose(d)@d, d
