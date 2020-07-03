__author__ = 'jeff'

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_addons as tfa


class BertCrf(keras.models):

    def __init__(self, max_lenth=100, tag_lenth=4, dropout=0.2, bert_dir=None, dense_h=100):
        """"""
        self.bert_dir = bert_dir
        self.max_lenth = max_lenth
        self.tag_lenth = tag_lenth
        self.drop = dropout
        self.dense_h = dense_h
        if bert_dir:
            self.bert = hub.KerasLayer(self.bert_dir)
        self.dense = keras.layers.Dense(self.tag_lenth)
        self.dropout = keras.layers.Dropout(self.drop)
        self.translation = tf.Variable(tf.random.normal((self.tag_lenth, self.tag_lenth)))

    def call(self, input_x):
        """"""
        bert_embeding = self.bert(input_x)
        drop_embeding = self.dropout(bert_embeding)
        out = self.dense(drop_embeding)
        return out

