__author__ = 'jeff'

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from common.msr_data_deal import Data


# def crf_loss(logit, lables):
#     return tfa.text.crf_log_likelihood(logit)

def sentence_to_index(vacab_dt, sentences, max_lenth=100):
    """"""
    lenth_vab = len(vacab_dt)
    res = []
    for s in sentences:
        r = [vacab_dt.get(i, lenth_vab) for i in s]
        r.extend([0] * max(max_lenth-len(s), 0))
        res.append(r[:max_lenth])
    return np.array(res)


class CrfLoss(tf.keras.losses.Loss):

    def __init__(self, name="custom_mse"):
        super(CrfLoss, self).__init__(name=name)
        # self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        lable, sequence_lengths = y_true[:, :-1], y_true[:, -1]
        log_likelihood, transition_params = tfa.text.crf_log_likelihood(
            y_pred, lable, sequence_lengths)
        # mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        # reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        loss = - tf.reduce_mean(log_likelihood)
        return loss


class LstmCrfModel(tf.keras.Model):

    def __init__(self, hidden_num, vocab, embedding_size, training=False, dp=0.2, sentence_lenth=100):
        super(LstmCrfModel, self).__init__()
        self.num_hidden = hidden_num
        self.vocab = vocab
        self.vacab_dt = {v: index for index, v in enumerate(vocab)}
        self.vacab_size = len(vab) + 1
        self.training = training
        self.dp = dp
        self.sentence_lenth = sentence_lenth
        self.label_dict = {"B": 0, "E": 1, "S": 2, "M": 3}

        self.embedding = tf.keras.layers.Embedding(self.vacab_size, embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=True, dropout=self.dp))
        # self.vectorizer = TextVectorization(output_mode="int", output_sequence_length=6, split=None)

    def call(self, inputs):
        inputs = self.embedding(inputs)
        logits = self.biLSTM(inputs, training=self.training)
        return logits
    #     label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
    #     log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,
    #                                                                            label_sequences,
    #                                                                            text_lens,
    #                                                                            transition_params=self.transition_params)
    #     return logits, text_lens, log_likelihood
    # else:
    #     return logits, text_lens

    # def caculate_loss(self, logit, y):
    #     loss = tfa.text.crf_log_likelihood(logit, y, 10)
    #     return loss

    def fit(self, x=None, y=None, batch_size=None,**kwargs):
        _x = self.deal_x(x)
        _y = self.deal_y(y)
        super(LstmCrfModel, self).fit(_x, _y, **kwargs)

    def deal_x(self, sentence_list):
        """"""
        return sentence_to_index(self.vacab_dt, sentence_list, max_lenth=self.sentence_lenth)

    def deal_y(self, sentence_list):
        """"""
        res = sentence_to_index(self.label_dict, sentence_list, max_lenth=self.sentence_lenth)
        l = np.array([[min(len(s), self.sentence_lenth)] for s in sentence_list])
        y = np.column_stack([res, l])
        return y


def crf_loss(y_true, y_pred):
    """"""
    return tfa.text.crf_log_likelihood(y_pred, y_true, 10)


def get_lstm_crf(hidden_num, embedding_size, dp=0.2, vacab=[]):
    """"""
    embed = tf.keras.layers.Embedding(len(vacab), embedding_size)
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_num, return_sequences=True, dropout=dp))

    x = tf.keras.Input(shape=(100))
    ex = embed(x)
    logit = bilstm(ex, training=True)
    # logit = tf.reduce_mean(logit, axis=0)
    model = tf.keras.Model(x, logit)
    model.compile(loss=CrfLoss(), optimizer='adam')
    model.summary()
    return model
    # return logit


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

    train_data = d.get_train()
    x_train,  y_train = train_data[:, 0], train_data[:, 1]
    model = LstmCrfModel(
        hidden_num=128,
        embedding_size=128,
        vocab=vab,
        training=True,
        dp=0.2,
    )
    model.compile(loss=CrfLoss(), run_eagerly=True)
    # model.compile(loss=CrfLoss())
    # model.compile(loss=crf_loss, run_eagerly=True)
    # model.build((10, 100))
    # model.summary()
    model.fit(x_train, y_train, batch_size=1024, epochs=10)

    # m = get_lstm_crf(128, 128, vacab=vab)



