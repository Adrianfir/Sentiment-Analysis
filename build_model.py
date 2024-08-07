from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from sklearn.base import BaseEstimator


class DefModel(BaseEstimator):
    def __init__(self, embed_input_dim, embed_input_length, embed_output_dim,
                 n_lstm_layers, n_units, drop_rate, val_split):
        self.embed_input_dim = embed_input_dim
        self.embed_input_length = embed_input_length
        self.embed_output_dim = embed_output_dim
        self.n_lstm_layers = n_lstm_layers
        self.n_units = n_units
        self.drop_rate = drop_rate
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.embed_input_dim,
                            input_length=self.embed_input_length,
                            output_dim=self.embed_output_dim))

        for i in range(self.n_lstm_layers):
            model.add(LSTM(units=self.n_units, return_sequences=(i < self.n_lstm_layers - 1)))
            model.add(Dropout(rate=self.drop_rate))

        # Final output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, x, y, epochs=5, batch_size=64, validation_split=0.2):
        self.model = self.build_model()
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return self

    def predict(self, x):
        return (self.model.predict(x)).astype("int32")

    def score(self, x, y):
        loss, accuracy = self.model.evaluate(x, y)
        return accuracy
