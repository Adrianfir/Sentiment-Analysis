"""
This is the amin.py
"""
__author__: str= 'Pouya "Adrian" Firouzmakan'

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import prep_text
from config.config import config


if __name__ == "__main.py__":
    df = pd.read_csv(config['data']['path'], encoding='latin1')
    df.columns = config['data']['col_names']
    x = df['comment']
    y = df['target']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=config['data']['test_size'])
    operations = [('prep_text', prep_text),
                  ('lstm_model', lstm_model)]
    pipe_line = Pipeline(steps=operations)


