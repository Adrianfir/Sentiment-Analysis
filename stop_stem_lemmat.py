__author__: str = 'Pouya "Adrian" Firouzmakan'

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class StopStemLemmat(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.remove('not')
        self.lemmatizer = None
        self.stemmer = None

    def fit(self, x, y=None):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        return self

    def transform(self, x, y=None):
        data = []

        for word in x:
            if word not in x:
                word = self.lemmatizer(word, "v")
                word = self.stemmer(word)
                data.append(word)
        return data




