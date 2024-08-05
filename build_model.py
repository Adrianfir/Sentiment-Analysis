__author__: str = 'Pouya "Adrian" Firouzmakan'


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from config.config import config

class DefModel():
    def __init__(self):
        self.n_lstm_layers = config['model']['n_lstm_layers']
        self.units = config['model']['units']
        self.embed_input_length = config['model']['embed_input_length']
        self.embed_output_dim = config['model']['embed_out_put_dim']