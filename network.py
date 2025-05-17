import numpy as np

class Network:
    def __init__(self, neuron_counts: list):
        """
        Initialize the network.

        Args:
            neuron_counts: The number of neurons in each layer of the network
        """
        if len(neuron_counts)<2:
            raise ValueError("The network must have at least two layers")
        for N in neuron_counts:
            if type(N)!=int or N<=0:
                raise ValueError("Invalid neuron count")