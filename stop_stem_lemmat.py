__author__: str = 'Pouya "Adrian" Firouzmakan'

from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')


class StopStemLemmat():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
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




