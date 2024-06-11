"""
models package

This package contains implementations of various types of neural networks, including:

Classes:
    FeedforwardNeuralNetwork:
        Implements a feedforward neural network with methods for forward propagation, training, 
        and saving/loading models.

    ConvolutionalNeuralNetwork:
        Implements a convolutional neural network with methods for forward propagation, training, 
        and saving/loading models.

    RecurrentNeuralNetwork:
        Implements a recurrent neural network with methods for forward propagation, training, 
        and saving/loading models.

Modules:
    feedforward.py:
        Contains the FeedforwardNeuralNetwork class.

    convolutional.py:
        Contains the ConvolutionalNeuralNetwork class.

    recurrent.py:
        Contains the RecurrentNeuralNetwork class.

Usage:
    from models import FeedforwardNeuralNetwork, ConvolutionalNeuralNetwork, RecurrentNeuralNetwork

    # Example of creating a feedforward neural network
    fnn = FeedforwardNeuralNetwork([input_size, hidden_size, output_size])

    # Example of creating a convolutional neural network
    cnn = ConvolutionalNeuralNetwork()

    # Example of creating a recurrent neural network
    rnn = RecurrentNeuralNetwork(input_size, output_size)

"""

from .feedforward import FeedforwardNeuralNetwork
from .convolutional import ConvolutionalNeuralNetwork
from .recurrent import RecurrentNeuralNetwork

__all__ = [
    'FeedforwardNeuralNetwork',
    'ConvolutionalNeuralNetwork',
    'RecurrentNeuralNetwork'
]
