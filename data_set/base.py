__author__ = 'jeff'

from tensorflow_datasets import show_examples


class Data(object):

    def __init__(self, *args, **kwargs):
        self.data = None

    def shuffle(self):
        """打乱顺序"""
        pass

    def translate(self):
        """"""
        pass

    def __iter__(self):
        return iter(self.data)

    def tow_split(self):
        """"""
        pass

    def three_split(self):
        """"""

    def __repr__(self):
        return str(self.data)


class DataSet(object):
    """"""

    def __init__(self):
        pass


