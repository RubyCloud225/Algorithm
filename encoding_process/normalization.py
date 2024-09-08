import numpy as np
from feed_forward import relu, relu_derivative


class LayerNormalization:
    def __init__(self, epsilon=1e-8):
        """
        Initialize the Normalization layer

        Params: 
            Epsilon (float): Small value added for numerical stability 
        """

        self.epsilon = epsilon
        self.gamma = 1.0
        self.beta = 0.0

    def forward(self, x):
        """
        Compute the Normalization output
        Param:
            x (numpy array): input matrix
        returns:
            numpy array: Normalization output
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * x_normalized + self.beta
    
    def backward(self, dout):
        """
        Compute the gradients of the Layer Normalization layer.

        Parameters:
        dout (numpy array): Output gradients.

        Returns:
        numpy array: Input gradients.
        """
        x  = self.x #Store the input for backward pass
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        dgamma = np.sum(dout * x_normalized, axis=-1, keepdims=True)
        dbeta = np.sum(dout, axis=-1, keepdims = True)

        dx_normalized = dout * self.gamma
        dvariance = np.sum(dx_normalized * (x - mean), axis =-1, keepdims=True) * -0.5 * (variance + self.epsilon) ** -1.5
        dmean = np.sum(dx_normalized * -1 / np.sqrt(variance + self.epsilon), axis=-1, keepdims=True) + dvariance * -2 * (x - mean)

        dx = dx_normalized / np.sqrt(variance + self.epsilon) + dvariance * 2 * (x - mean) / x.shape[-1] + dmean / x.shape[-1]

        self.gamma -= dgamma
        self.beta -= dbeta

        return dx

"""
Encoding the normalized layer using three layer archtecture
"""
    
class Encoder:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        Initialize the Encoder Layer

        params: 
            input_dim (int): Input dimension
            hidden_dim1 (int): First hidden layer dimension
            hidden_dim2 (int): Second hidden layer dimension
            output_dim (int): Output dimension
        """
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim

        # Initialize weights and biases for the encoder

        self.W1 = np.random.rand(input_dim, hidden_dim1)
        self.b1 = np.zeros((hidden_dim1,))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2)
        self.b2 = np.zeros((hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, output_dim)
        self.b3 = np.zeros((output_dim))

    def forward(self, x):
        """
        Compute the Encoder output.
        
        Params:
            x (numpy array): input matrix.
        
        returns:
            numpy array: Encoder output.
        """
        #Compute the first hidden layer
        hidden1 = relu(np.dot(x, self.W1) + self.b1)
        #compute the secton hidden layer
        hidden2 = relu(np.dot(hidden1, self.W2) + self.b2)
        #Compute the output layer
        output = np.dot(hidden2, self.w3) + self.b3

        return output
    
    def backward(self, dout):
        """
        Compute the gradients of the encoder layer 

        params:
            dout(numpy array): output gradients

        Returns:
            numpy array: input gradients
        """

        # Compute the gradients of the output layer
        dhidden2 = dout

        #Compute the gradients of the second hidden layer 
        dhidden1 = dhidden2 * relu_derivative(np.dot(self.hidden1, self.W2) + self.b2)
        dhidden1 = np.dot(dhidden1, self.W2.T)

        #Compute the gradients of the first hidden layer
        dx = dhidden1 * relu_derivative(np.dot(self.x, self.W1) + self.b1)
        dx = np.dot(dx, self.W1.T)

        #Compute the gradients of the weights and biases
        self.W3 -= np.dot(self.hidden2.T, dout)
        self.b3 -= dout
        self.W2 -= np.dot(self.hidden1.T, dhidden2)
        self.b2 -= dhidden2
        self.W1 -= np.dot(self.x.T, dx)
        self.b1 -= dx

        return dx
    
    def __call__(self, x):
        self.x = x
        self.hidden1 = relu(np.dot(x, self.W1) + self.b1)
        self.hidden2 = relu(np.dot(self.hidden1, self.W2) + self.b2)
        return np.dot(self.hidden2, self.W3) + self.b3