__author__ = 'jeff'

import os
import sys
import random
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
p_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
b_path = os.sep.join(p_list)
sys.path.append(b_path)

import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np

from data_set.msr import msr
from settings import base_dir
from core.bert import BertCrf


def split_to_index(tag_list, tag_dt=None):
    """"""
    res = []
    index_lst = []
    for i in range(len(tag_list)):
        if not tag_dt:
            tag = tag_list[i]
        else:
            dt = {v: k for k, v in tag_dt.items()}
            tag = dt[tag_list[i]]
        if tag == "B":
            index_lst = [str(i)]
        elif tag == 'S':
            res.append(str(i))
        else:
            index_lst.append(str(i))
            if tag == 'E':
                res.append(','.join(index_lst))
    return res


class BertCrfPredictor(object):

    def __init__(self, model_dir):
        # self.model = tf.saved_model.load(model_dir)
        self.model = tf.keras.models.load_model(model_dir)
        self.max_lenth = 100
        print('====')

        # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=self.model)
        # manager = tf.train.CheckpointManager(ckpt, self.check_point_dir, max_to_keep=3)
        # ckpt.restore(manager.latest_checkpoint)

    def predict(self, inputs):
        """"""
        # x_batch = self.sentence_to_index(x, dt=self.vacab_dt, default=0)
        # sequence_lengths = np.array([len(s) for s in x_batch])
        x, lenths = inputs[0], inputs[1]
        lenths = pd.Series(lenths)
        lenths = np.array(list(lenths.where(lenths <= self.max_lenth, self.max_lenth).values))

        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            x, padding="post", maxlen=self.max_lenth
        )

        logit = self.model(padded_inputs)
        tags, _ = tfa.text.crf_decode(logit, self.model.transition_params, lenths)
        # tfa.text.viterbi_decode(logit, self.transition_params)
        print(tags)
        return tags

    def evaluation(self, percent=0.2):
        """"""
        batch_size = 16
        msr.test.shuffle()
        batch_num = len(msr.test.math_data) // batch_size
        pre_res = []
        normal_res = []
        for i in random.choices(range(batch_num), k=math.floor(batch_num*percent)):
            test = msr.test.math_data[i*batch_size: (i+1)*batch_size]
            x, y, lenths = test[:, 0], test[:, 1], test[:, 2]
            y_pre = self.predict([x, lenths])
            pre_words = [split_to_index(np.array(y_pre[i][:len(x[i])]), msr.label_dt) for i in range(len(y_pre))]
            normal_words = [split_to_index(t, msr.label_dt) for t in y]
            pre_res.extend(pre_words)
            normal_res.extend(normal_words)
        self.caculate_index(normal_res, pre_res)
        # correct_word_number = 0
        # pre_word_number = 0
        # normal_word_number = 0
        # for i in range(len(y)):
        #     correct_word_number += len(set(normal_words[i]) & set(pre_words[i]))
        #     normal_word_number += len(normal_words[i])
        #     pre_word_number += len(pre_words[i])
        #
        # precision = correct_word_number / pre_word_number
        # recall = correct_word_number / normal_word_number
        # f1 = 2 * (precision * recall) / (precision + recall)
        #
        # print('评估结果， 准确率： %s,  召回率：%s,  F1: %s' % (precision, recall, f1))

    def caculate_index(self, normal_words, pre_words):
        """"""
        correct_word_number = 0
        pre_word_number = 0
        normal_word_number = 0
        for i in range(len(pre_words)):
            correct_word_number += len(set(normal_words[i]) & set(pre_words[i]))
            normal_word_number += len(normal_words[i])
            pre_word_number += len(pre_words[i])

        precision = correct_word_number / pre_word_number
        recall = correct_word_number / normal_word_number
        f1 = 2 * (precision * recall) / (precision + recall)

        print('评估结果， 准确率： %s,  召回率：%s,  F1: %s' % (precision, recall, f1))


if __name__ == '__main__':
    model_dir = os.sep.join([base_dir, 'models', 'bert_crf'])
    obj = BertCrfPredictor(model_dir)
    obj.evaluation(0.05)


