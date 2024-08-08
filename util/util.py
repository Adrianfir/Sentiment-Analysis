__author__: str = 'Pouya "Adrian" Firouzmakan'
__all__ = []

import re
import nltk
from nltk.corpus import stopwords


def text_manipulation(text):
    """

    :param text:
    :return:
    """
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'httop\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    text = text.lower()
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')

    text = [word for word in text if not word in stop_words]

    return text


def data_label(data):
    """

    :param data:
    :return:
    """

