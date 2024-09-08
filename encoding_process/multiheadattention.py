import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize the multiheadAttention class

        Param:
            d_model (int): Dimensionality of the model
            num_head (int): Number of attention heads
            dropout (float): dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

    def attention(self, Q, K, V, mask=None):
        """
        Compute attention scores.

        Params:
            Q (numpy array): Query Matrix.
            K (numpy array): Key Matrix.
            V (numpy array): Value Matrix.
            mask (numpy array): Mask Matrix.

        Returns:
            Numpy array: Attention scores
        """

        scores = np.matmul(Q, K.T) / np.sqrt(self.d_model)
        if mask is not None:
            scores += mask
        weights = self.dropout(np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True))
        return weights
    
    def multi_head_attention(self, Q, K, V, mask=None):
        """
        Compute Multi-head attention.

        params:
            Q (numpy array): Query matrix.
            K (numpy array): Key matrix.
            V (numpy array): Value matrix.
            mask (numpy array): Mask matrix.

        Returns:
        numpy array: Multi-head attention output.
        """
        Q = Q.reshape(-1, self.num_heads, self.d_model // self.num_heads)
        K = K.reshape(-1, self.num_heads, self.d_model // self.num_heads)
        V = V.reshape(-1, self.num_heads, self.d_model // self.num_heads)
        attention_scores = self.attention(Q, K, V, mask)
        output = attention_scores * V
        output = output.reshape(-1, self.d_model)
        return output
    
    def add_norm(self, input, output):
        """
        Perform add and norm operation.

        Params:
            Input (numpy array): Input matrix
            output (numpy array): Output matrix
        returns:
            numpy array: add and norm output.
        """
        return input + output
    
"""
#example usage
from multi_head_attention import MultiHeadAttention

# Sample data
Q = np.random.rand(10, 128)
K = np.random.rand(10, 128)
V = np.random.rand(10, 128)
mask = np.random.rand(10, 10)

# Create a MultiHeadAttention instance
attention = MultiHeadAttention(d_model=128, num_heads=8)

# Compute multi-head attention
output = attention.multi_head_attention(Q, K, V, mask)
print(output)

# Perform add and norm operation
input = np.random.rand(10, 128)
output = attention.add_norm(input, output)
print(output)

"""
