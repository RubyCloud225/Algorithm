import numpy as np
from collections import defaultdict
import random

"""
This is basic: need to establish subsampling and more research.

For the initial embedding of vectorized data before we use activation functions as hidden layers
"""

class VectorEmbedding:
    def __init__(self, sentences, vector_dim=100, window=5, subsample_threshold=1e-3, min_count=1, learning_rate=0.01):
        """
        Initialize the word2vec class

        params:
            sentences (list of list of str): Sentences to train the word2vec model on.
            vector_dim (int): Dimensionality of the vectors.
            window (int): Window size for the word2vec model
            min_count (int): Minimum count for words to be included in the model
            learning_rat (float): Learning rate for the training algorithm
            subsample_threshold (float): Threshold for subsample frequent words.
        """

        self.vector_dim = vector_dim
        self.window = window
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        self.learning_rate = learning_rate
        self.vocab = self.build_vocab(sentences)
        self.w2v_model = self.train(sentences)
    
    def build_vocab(self, sentences):
        """
        Build the vocabulary from the sentences

        param: sentences (list of list of str): Sentences to build the vocabulary from 

        Returns: 
            dict: Vocabulary with word frequences
        """
        vocab = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                vocab[word] += 1
        vocab = {word: freq for word, freq in vocab.items() if freq >= self.min_count}
        return vocab
    
    def subsample(self, sentence):
        """
        Subsample frequent words in a sentence.

        Params:
            Sentence (list of str): Sentence to subsample
        
        Returns: list of str: subsampled sentence.
        """
        subsampled_sentence = []
        for word in sentence:
            freq = self.vocab[word]
            prob = (np.sqrt(freq / self.subsample_threshold) + 1) * (self.subsample_threshold / freq)
            if random.random() < prob:
                subsampled_sentence.append(word)
        return subsampled_sentence
    
    def train(self, sentences):
        """
        Train the word2Vec model

        param:
        sentences (list of list of str): Sentences to train the model on

        returns:
            dict: word2vec model with word vectors
        """
        w2v_model = {}
        for sentence in sentences:
            subsampled_sentence = self.subsample(sentence)
            for i, word in enumerate(subsampled_sentence):
                context_words = subsampled_sentence[max(0, i - self.window):i] + subsampled_sentence[i + 1:min(len(subsampled_sentence), i + self.window + 1)]
                for context_word in context_words:
                    if word not in w2v_model:
                        w2v_model[word] = np.random.rand(self.vector_dim)
                    if context_word not in w2v_model:
                        w2v_model[context_word] = np.random.rand(self.vector_dim)
                    error = self.learning_rate * (w2v_model[context_word] - w2v_model[word])
                    w2v_model[word] -= error
                    w2v_model[context_word] += error
        return w2v_model
    
    def get_vector(self, word):
        """
        Get the vector for a specific word

        Params: word(str): Word to get the vector for.

        Returns: numpy array: Vector for the specified word.
        """
        return self.w2v_model.get(word, np.zeros(self.vector_dim))
    
    def similarity(self, word1, word2):
        """
        Get the similarity between two words.

        Params:
            Word1 (str): First Word.
            Word2 (str): Second Word.
        
        Returns:
            float: similarity between the two words.
        """

        vector1 = self.get_vector(word1)
        vector2 = self.get_vector(word2)
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

"""
#example usage

from word2vec import Vectorembedding

# Sample data
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"], ["cat", "and", "dog", "are", "friends"]]

# Create a Word2Vec model
w2v = Word2Vec(sentences, vector_dim=100, subsample_threshold=1e-3)

# Get the vector for a word
vector_cat = w2v.get_vector("cat")
print(vector_cat)

# Get the similarity between two words
similarity = w2v.similarity("cat", "dog")
print

"""
