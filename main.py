__author__ = 'Pouya "Adrian" Firouzmakan'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import joblib

from prep_text import PrepText
from padding import Padding
from build_model import DefModel  # Import the wrapper class
from tokenizing import Tokenizing  # Import the custom tokenizer
from config.config import config

if __name__ == "__main__":
    df = pd.read_csv(config['data']['path'], encoding='latin1')
    df.columns = config['data']['col_names']
    x = df[config['data']['text_col']]
    y = df[config['data']['label_col']]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=config['data']['test_size'],
                                                    random_state=config['rest']['rand_state'])

    re_text = PrepText()
    tokenizer = Tokenizing(num_words=config['data']['tokenizer_n_words'])  # Use custom wrapper
    padding = Padding(maxlen=config['data']['max_seq_length'])

    lstm_model = DefModel(embed_input_dim=config['data']['tokenizer_n_words'],
                          embed_input_length=config['data']['max_seq_length'],
                          embed_output_dim=config['model']['emb_output_dim'],
                          n_lstm_layers=config['model']['n_lstm_layers'],
                          n_units=config['model']['n_units'],
                          drop_rate=config['model']['drop_rate'],
                          val_split=config['model']['val_split'])

    pipeline = Pipeline([
        ('re_text', re_text),
        ('tokenizer', tokenizer),
        ('padding', padding),
        ('lstm_model', lstm_model)
    ])

    pipeline.fit(xtrain, ytrain, lstm_model__epochs=config['model']['epochs'],
                 lstm_model__batch_size=config['model']['batch_size'],
                 lstm_model__validation_split=config['model']['val_split'])

    accuracy = pipeline.score(xtest, ytest)
    print(f"Test Accuracy: {accuracy:.4f}")
    joblib.dump(pipeline, 'saved_pipeline.pkl')
