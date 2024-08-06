from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer


class Tokenizing(BaseEstimator, TransformerMixin):
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.tokenizer = None

    def fit(self, x, y=None):
        self.tokenizer = Tokenizer(num_words=self.num_words)
        self.tokenizer.fit_on_texts(x)
        return self

    def transform(self, x, y=None):
        sequences = self.tokenizer.texts_to_sequences(x)
        return sequences

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)
