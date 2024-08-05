__author__: str = 'Pouya "Adrian" Firouzmakan'

from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config.config import config


class Padding(BaseEstimator, TransformerMixin):

    def __init__(self, maxlen=None):
        self.maxlen = maxlen

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        data = x
        data_pad = pad_sequences(data, maxlen=self.maxlen)

        return data_pad
