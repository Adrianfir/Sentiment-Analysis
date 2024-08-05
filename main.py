"""
This is the amin.py
"""
__author__: str= 'Pouya "Adrian" Firouzmakan'

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from prep_text import PrepText
from padding import Padding
from build_model import DefModel
from config.config import config


if __name__ == "__main.py__":
    df = pd.read_csv(config['data']['path'], encoding='latin1')
    df.columns = config['data']['col_names']
    x = df['comment']
    y = df['target']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=config['data']['test_size'])

    re_text = PrepText()
    tokenizer = Tokenizer(num_words=config['data']['tokenizer_n_words'])
    padding = Padding(maxlen=config['data']['max_seq_length'])
    xtrain_pad = pad_sequences(xtrain, maxlen=config['data'][''])

    operations = [('re_text', re_text),
                  ('tokenizer', tokenizer),
                  ('padding', padding),
                  ('lstm_model', lstm_model)]

    pipe_line = Pipeline(steps=operations)


