import feed_forward
import numpy as np
from multiheadattention import MultiHeadAttention
import normalization
import vectorizer
from word2vec import VectorEmbedding
from PositionalEncoder import PositionalEncoder
from data_preprocess import DataProcessor

class neuralnetworkencoder:
    def __init__(self, data, word, number_of_instances, max_seq_len, d_model):
        self.vectorizer = vectorizer.vectorizer()
        self.embedding = VectorEmbedding()
        self.model = self.build_model()
        self.multiheadattention = MultiHeadAttention
        self.feedforward = feed_forward.FeedForward()
        self.normalization = normalization.LayerNormalization()
        self.data = data
        self.number_of_instances = number_of_instances
        self.word = word
        self.max_seq_len = max_seq_len
        self.d_model = d_model # calculate this independently
    
    def obtaining_data(self):
        extractor = DataProcessor(self.data)
        server_data = extractor.get_data_from_sql_server()
        pdf_data = extractor.read_pdf_file()
        csv_data = extractor.get_row_data()
        data = server_data + pdf_data + csv_data
        return data

    def vectorize(self, sentences):
        vectorized_data = self.vectorizer.vectorize(self.data)
        vector_data = vectorized_data(sentences, vector_dim=100, subsample_threshold=1e-3)
        vector = sum(vector_data.get_vector(self.word) + vector_data.similarity(self.word))
        return vector

    def fit_transform(self):
        self.model.fit(self.data)

    def create_multi_head_attention(self):
        attention = self.multiheadattention(num_heads=8, key_dim=128)
        output = attention.multi_head_attention(self.data)
        input = np.random.rand(10, 128)
        output = self.multiheadattention(input, input)
        return output

    def position_encoding(self):
        sentence = [self.data]
        embeddings = {sentence : np.random.rand(self.number_of_instances)} #need a def to calculate the number of instances of a word
        encoder = PositionalEncoder(self.max_seq_len, self.d_model)
        encoded_sentence = encoder.encode_sentence(sentence, embeddings)
        return encoded_sentence

    def feed_forward(self, encoded_sentence):
        encoded_sentence = np.random.rand(10, 128)
        output = self.feed_forward(encoded_sentence)
        return output

    def normalize(self, output):
        normalize = self.normalization(output)
        dx = normalize.forward(output)
        dx2 = normalize.backward(dx)
        return dx2

    def encode(self):
        encoded_sentence = self.position_encoding()
        encoded_sentence = self.feed_forward(encoded_sentence)
        encoded_sentence = self.normalize(encoded_sentence)
        return encoded_sentence
