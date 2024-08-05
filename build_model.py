__author__: str = 'Pouya "Adrian" Firouzmakan'


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from config.config import config

class DefModel():
    def __init__(self):
        self.n_lstm_layers = config['model']['n_lstm_layers']
        self.units = config['model']['units']
        self.embedding_