__author__: str = 'Pouya "Adrian" Firouzmakan'


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from config.config import config

class DefModel():
    def __init__(self, embed_input_dim=None, embed_input_length=None, embed_output_dim=None,
                 n_lstm_layers=None, n_units=None, drop_rate=None):
        self.n_lstm_layers = n_lstm_layers
        self.n_units = n_units
        self.embed_input_dim = embed_input_dim
        self.embed_input_length = embed_input_length
        self.embed_output_dim = embed_output_dim
        self.drop_rate = drop_rate

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        data = x
        model = Sequential()
        model.add(Embedding(input_dim=self.embed_input_dim,
                            input_length=self.embed_input_length,
                            output_dim=self.embed_output_dim))
        for i in range(self.n_lstm_layers):
            model.add(LSTM(units=self.n_units, return_sequences=True))
            model.add(Dropout(rate=self.drop_rate))

