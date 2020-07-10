__author__ = 'jeff'

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_addons as tfa
import numpy as np


class BertCrf(keras.Model):

    def __init__(self, max_lenth=100, tag_lenth=4, dropout=0.2, bert_dir=None, dense_h=100):
        """"""
        super(BertCrf, self).__init__()
        self.bert_dir = bert_dir
        self.max_lenth = max_lenth
        self.tag_lenth = tag_lenth
        self.drop = dropout
        self.dense_h = dense_h
        if bert_dir:
            self.bert = hub.KerasLayer(self.bert_dir, trainable=False)
        self.dense = keras.layers.Dense(self.tag_lenth)
        self.dropout = keras.layers.Dropout(self.drop)
        self.transition_params = tf.Variable(tf.random.normal((self.tag_lenth, self.tag_lenth)))

        # max_seq_length = 128  # Your choice here.
        # input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
        #                                        name="input_word_ids")
        # input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
        #                                    name="input_mask")
        # segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
        #                                     name="segment_ids")

    # @tf.function
    # def __call__(self, *args, **kwargs):
    #     return super(BertCrf, self).__call__(*args, **kwargs)

    # @tf.function
    def call(self, padded_inputs):
        """"""
        # padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        #     input_x, padding="post", maxlen=self.max_lenth
        # )
        mask = tf.ones_like(padded_inputs)
        segment_ids = tf.zeros_like(padded_inputs)
        pooled_output, sequence_output = self.bert([padded_inputs, mask, segment_ids])
        drop_embeding = self.dropout(sequence_output)
        out = self.dense(drop_embeding)
        return out

    def caculate_loss(self, lable, logit, sequence_lengths):
        """"""
        # lable = tf.Tensor(lable)
        # sequence_lengths = tf.Tensor(sequence_lengths)
        lable = tf.keras.preprocessing.sequence.pad_sequences(
            lable, padding="post", maxlen=self.max_lenth
        )
        # lable = np.array(lable)

        # lable, sequence_lengths = y_true[:, :-1], y_true[:, -1]
        log_likelihood, transition_params = tfa.text.crf_log_likelihood(
            logit, lable, sequence_lengths, self.transition_params)
        # mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        # reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        loss = - tf.reduce_mean(log_likelihood)
        return loss


