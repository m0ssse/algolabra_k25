import unittest
import network
import numpy as np

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.network = network.Network([2, 2], [network.identity, network.identity], [])