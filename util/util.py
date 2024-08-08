__author__: str = 'Pouya "Adrian" Firouzmakan'
__all__ = []

import re


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
    return text


def data_label(data):
    """

    :param data:
    :return:
    """

