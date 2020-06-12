__author__ = 'jeff'

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from common.msr_data_deal import Data


# def crf_loss(logit, lables):
#     return tfa.text.crf_log_likelihood(logit)


class LstmCrfModel(tf.keras.Model):

    def __init__(self, hidden_num, vocab, embedding_size, training=False, dp=0.2):
        super(LstmCrfModel, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab
        self.training = training
        self.dp = dp
        # self.batch = batch
        # self.s_lenth = s_lenth

        self.embedding = tf.keras.layers.Embedding(len(vocab), embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=False, dropout=self.dp))
        self.vectorizer = TextVectorization(output_mode="int", output_sequence_length=6, split=None)

    def call(self, inputs):
        inputs = self.embedding(inputs)
        logits = self.biLSTM(inputs, training=self.training)

        # if self.training:
        #     # label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
        #     log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits,
        #                                                                          labels,
        #                                                                          text_lens,
        #                                                                          transition_params=self.transition_params)

        return logits
    #     label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
    #     log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,
    #                                                                            label_sequences,
    #                                                                            text_lens,
    #                                                                            transition_params=self.transition_params)
    #     return logits, text_lens, log_likelihood
    # else:
    #     return logits, text_lens

    def caculate_loss(self, logit, y):
        loss = tfa.text.crf_log_likelihood(logit, y, 10)
        return loss

    def data_deal(self):
        """"""

    # def fit(self,
    #       x=None,
    #       y=None,
    #       batch_size=None,
    #       epochs=1,
    #       verbose=1,
    #       callbacks=None,
    #       validation_split=0.,
    #       validation_data=None,
    #       shuffle=True,
    #       class_weight=None,
    #       sample_weight=None,
    #       initial_epoch=0,
    #       steps_per_epoch=None,
    #       validation_steps=None,
    #       validation_freq=1,
    #       max_queue_size=10,
    #       workers=1,
    #       use_multiprocessing=False,
    #       **kwargs):


    # def train_step(self, data):
    #     """"""
    #     print('======')
    #     print(x, y)
    #
    # def train_on_batch(self,
    #                  x,
    #                  y=None,
    #                  sample_weight=None,
    #                  class_weight=None,
    #                  reset_metrics=True):
    #     print('=====')


# def crf_loss(y_true, y_pred):
#     """"""
#     return tfa.text.crf_log_likelihood(y_pred, y_true, 10)


def get_lstm_crf(hidden_num, embedding_size, dp=0.2, vacab=[]):
    """"""
    embed = tf.keras.layers.Embedding(len(vacab), embedding_size)
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_num, return_sequences=True, dropout=dp))

    x = tf.keras.Input(shape=(100))
    ex = embed(x)
    logit = bilstm(ex, training=True)
    # logit = tf.reduce_mean(logit, axis=0)
    # model = tf.keras.Model(x, logit)
    # model.compile(loss=crf_loss, optimizer='adam')
    # model.summary()
    # return model
    return logit


def train():
    pass



if __name__ == '__main__':
    # training_data = np.array([["我 喜 欢 你"], ["你"]])
    # training_data = np.array([['我', '喜', '欢', '你'], ['你', '爱', '我', '么']])
    # # ['我', '喜', '欢', '你']e
    #
    # vectorizer = TextVectorization(output_mode="int", output_sequence_length=6, split=None)
    #
    d = Data()
    vab = np.array(d.get_dictionary())
    #
    # vectorizer.set_vocabulary(vab)
    # # vectorizer.adapt(training_data)
    #
    # integer_data = vectorizer(training_data)
    # print(integer_data)

    # train_data = d.get_train()
    # x,  y = train_data[:, 0], train_data[:, 1]
    model = LstmCrfModel(
        hidden_num=128,
        embedding_size=128,
        vocab=vab,
        training=True,
        dp=0.2,
    )
    # model.compile(run_eagerly=True)
    model.fit(x, y, batch_size=256, epochs=10)

    m = get_lstm_crf(128, 128, vacab=vab)



