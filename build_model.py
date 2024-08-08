import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from sklearn.base import BaseEstimator


class DefModel(BaseEstimator):
    def __init__(self, embed_input_dim, embed_input_length, embed_output_dim,
                 n_units, drop_rate):
        self.embed_input_dim = embed_input_dim
        self.embed_input_length = embed_input_length
        self.embed_output_dim = embed_output_dim
        self.n_units = n_units
        self.drop_rate = drop_rate
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.embed_input_dim,
                            input_length=self.embed_input_length,
                            output_dim=self.embed_output_dim))

        for i in range(len(self.n_units)):
            model.add(LSTM(units=self.n_units[i], return_sequences=(i < len(self.n_units) - 1)))
            model.add(Dropout(rate=self.drop_rate))

        # Final output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        return model

    def fit(self, x, y, epochs, batch_size, validation_split):
        self.model = self.build_model()
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split, verbose=1)
        return self

    def predict(self, x):
        return (self.model.predict(x)).astype("int32")

    def score(self, x, y):
        loss, accuracy = self.model.evaluate(x, y)
        return accuracy
