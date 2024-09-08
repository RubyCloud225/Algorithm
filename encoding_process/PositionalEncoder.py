import numpy as np

"""
Generates a positional encoding matrix. 

Combing this with word2vec to produce a sentence 

"""
class PositionalEncoder:
    def __init__(self, max_seq_len, d_model):
        """
        Initialize the PositionalEncoder class

        params: 
            Max_seq_len (int): Maximum sequence length.
            d_model (int): Dimension of the model.
        """

        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def positional_encoding(self, sentence):
        """
        Perform positional encoding on a sentence

        params:
            Sentence (list of str): Sentence to encode.
        
        Returns: 
            Numpy Array: Positionally encoded sentence
        """
        seq_len = len(sentence)
        pos_encoding = np.zeros((seq_len, self.d_model))
        for pos in range(seq_len):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                else:
                    pos_encoding[pos, i] = np.cos(pos / (10000 ** ((i - 1) / self.d_model)))
        return pos_encoding
    
    def encode_sentence(self, sentence, embeddings):
        """
        Encode a sentence using positional encoding and word embeddings.

        Params: 
            Sentence (list of str): sentence to encode.
            embedding (dict): word embeddings.
        
        Returns:
            numpy array: Encoded sentence
        """
        seq_len = len(sentence)
        encoded_sentence = np.zeros((seq_len, self.d_model))
        pos_encoding = self.positional_encoding(sentence)
        for i, word in enumerate(sentence):
            encoded_sentence[i] = embeddings.get(word, np.zeros(self.d_model)) + pos_encoding[i]
        return encoded_sentence
"""

# example usage 
from positional_encoder import PositionalEncoder

# Sample data
sentence = ["This", "is", "a", "test", "sentence"]
embeddings = {"This": np.random.rand(128), "is": np.random.rand(128), "a": np.random.rand(128), "test": np.random.rand(128), "sentence": np.random.rand(128)}

# Create a PositionalEncoder instance
encoder = PositionalEncoder(max_seq_len=10, d_model=128)

# Encode the sentence
encoded_sentence = encoder.encode_sentence(sentence, embeddings)
print(encoded_sentence)

"""
        
