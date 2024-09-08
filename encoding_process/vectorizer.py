import re
from collections import Counter
import math
import os
import pickle
from functools import lru_cache

"""
First input encoding

params: Data from sql database, pdfs training data and csv file for finance. 

This class vectorizes the words to saves it cache on each stage of the initial encoding. 

"""

class Vectorizer:
    def __init__(self, method='bow', max_features=None, cache_dir=None):
        """
        Initialize the vectorizer class.

        params:
        Method: bag of words (bow)
        max_features (int): maximum number of features to extract
        cache_dir (str): Directory to store cache files
        """

        self.method = method
        self.max_featurs = max_features
        self.cache_dir = cache_dir
        self.vocabulary = None
        self. idf_values = None
    
    def _tokenize(self, text):
        """
        
        Tokenize the text into words

        Params:
        text(str): Text to tokenize
        
        returns:
        list of STR: Tokenized words
        """
        return re.findall(r'\b\w+\b', text.lower())
    
    @lru_cache(maxsize=128)
    def _build_vocabulary(self, X):
        """
        Build the vocabulary from the text date

        param: X(list of str): list of text documents

        returns: list of str: vocabulary

        """
        vocabulary = set()
        for text in X:
            words = self._tokenize(text)
            vocabulary.update(words)
        return list(vocabulary)
    
    @lru_cache(maxsize=128)
    def _calculate_idf(self, X):
        """
        Calculate the inverse document frequency (idf) for each word in the vocabulary

        Params: X (list of str): List of text documents

        returns: dict: IDF values
        
        """
        idf_values = {}
        for word in self.vocabulary:
            doc_count = sum(1 for text in X if word in self._tokenize(text))
            idf_values[word] = math.log(len(X) / doc_count)
        return idf_values
    
    def _save_cache(self, filename, data):
        """
        Save data to a cache file

        Params: filename(str): Cache file name
        data: Data to save
        """
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_cache(self, filename):
        """
        Load data from cache file

        Param: Cache file name 

        returns: Data loaded from cache
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def fit(self, X):
        """
        Fit the vectorizer to the data 

        Params: X(list of str): List of text documents
        """
        cache_filename = f"{self.cache_dir}/vocabulary.pkl"
        if os.path.exists(cache_filename):
            self.vocabulary = self._build_vocabulary(tuple(X))
        else:
            self.vocabulary = self._calculate_idf(tuple(X))
            self._save_cache(cache_filename, self.idf_values)
    
    def transform(self, X):
        """
        Transform the data into vectors

        params: X(list of str): List of text documents

        returns: list of list of float: vectorized data
        """
        vectorized_data = []
        for text in X:
            words = self._tokenize(text)
            word_counts = Counter(words)
            vector = []
            for word in self.vocabulary:
                if self.method == 'bow':
                    vector.append(word_counts[word])
                elif self.method == 'tfidf':
                    tf = word_counts[word] / len(words)
                    idf = self.idf_values[word]
                    vector.append(tf * idf)
            vectorized_data.append(vector)
        return vectorized_data
    
    def fit_transform(self, X):
        """
        Fit the vectorizer to the data and its transform it into vectors

        params: X (list of str): List of text documents

        returns: list of list of float: vectorized data.
        """
        self.fit(X)
        return self.transform(X)
