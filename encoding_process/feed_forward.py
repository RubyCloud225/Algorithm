import numpy as np

class FeedForward:
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        """
        Initialize the FeedForward class

        params:
            d_model (int): Dimensionality of the model
            hidden_dim (int): Dimensionality of the hidden layers
            dropout (float): Dropout rate
        """
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Initialize weights and biases for the three hidden layers
        self.W1 = np.random.rand(d_model, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.rand(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.rand(hidden_dim, d_model)
        self.b3 = np.zeros(d_model)

    def relu(self, x):
        """
        Compute Relu activation function.

        params:
        x (numpy array): Input matrix

        returns: 
            numpy array: Feed forward output.
        """

        return np.maximum(x, 0)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def feed_forward(self, x):
        """
        Compute feed forward output.

        Parameters:
        x (numpy array): Input matrix.

        Returns:
        numpy array: Feed forward output

        """
        x = np.dot(x, self.W1) + self.b1
        x = self.relu(x)
        x = np.dot(x, self.W2) + self.b2
        x = self.rely(x)
        x = np.dot(x, self.W3) + self.b3
        return x
    
    def add_norm(self, input, output):
        """
        Perform add and norm operation

        param:
            input (numpy array): Input matrix
            output (numpy array): Output matrix

        Returns:
        numpy array: Add and norm output
        """
        return input + output

"""
# example usage 
from feed_forward import FeedForward

# Sample data
x = np.random.rand(10, 128)

# Create a FeedForward instance
feed_forward = FeedForward(d_model=128, hidden_dim=256)

# Compute feed forward output
output = feed_forward.feed_forward(x)
print(output)

# Perform add and norm operation
input = np.random.rand(10, 128)
output = feed_forward.add_norm(input, output)
print(output)


"""