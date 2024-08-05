__author__: str = 'Pouya "Adrian" Firouzmakan'

import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from config.config import config
import util.util as util


class PrepText(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        data = x
        vectorizer = np.vectorize(util.text_manipulation)
        data = vectorizer(data)
        return data
