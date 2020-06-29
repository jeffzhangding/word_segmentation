__author__ = 'jeff'

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tqdm
import tensorflow_addons as tfa
import sys
import logging

p_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
b_path = os.sep.join(p_list)
sys.path.append(b_path)

from common.msr_data_deal import Data
from core.lstm_crf import LstmCrfModel

_logger = logging.getLogger()


def sentence_to_index(sentences, dt, default=None):
    """"""
    res = []
    for s in sentences:
        r = [dt.get(i, default) for i in s]
        res.append(r)
    return res


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


class LstmFenciTrainer(object):

    def __init__(self):
        path_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
        path_list.extend(['models', 'fenci_1'])
        base_dir = os.sep.join(path_list)
        self.check_point_dir = os.path.join(base_dir, 'check_point')
        self.model_dir = os.path.join(base_dir, 'model_dir')

        self.d = Data()
        self.vocab = np.array(self.d.get_dictionary())
        self.vacab_dt = {v: index+1 for index, v in enumerate(self.vocab)}

        self.label_dict = {"B": 0, "E": 1, "S": 2, "M": 3}
        # S表示单字为词，B表示词的首字，M表示词的中间字，E表示词的结尾字

        self.model = LstmCrfModel(
            hidden_num=128,
            embedding_size=128,
            vocab_size=len(self.vocab) + 1,
            dp=0.2,
        )

    # @tf.function
    def train(self, batch_size=100, epochs=10):
        """"""
        train_data = self.d.get_train()
        x, y = train_data[:, 0], train_data[:, 1]

        optimizer = tf.keras.optimizers.Adam()

        # opt = tf.keras.optimizers.Adam(0.1)
        # dataset = toy_dataset()
        # iterator = iter(dataset)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.check_point_dir, max_to_keep=3)

        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            batch = len(x) // batch_size
            # Iterate over the batches of the dataset.
            for step in tqdm.tqdm(range(batch)):

                x_batch_train = sentence_to_index(x[batch_size*step: batch_size*(step+1)],
                                                       dt=self.vacab_dt, default=0)
                y_batch_train = sentence_to_index(y[batch_size * step: batch_size * (step + 1)],
                                                       dt=self.label_dict, default=0)
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                sequence_lengths = np.array([len(s) for s in x_batch_train])

                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = self.model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = self.model.caculate_loss(y_batch_train, logits, sequence_lengths)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log every 200 batches.
                # if step % 200 == 0:
                ckpt.step.assign_add(1)
                # _logger.info('======%s', str(loss_value))
                if int(ckpt.step) % 10 == 0:
                    save_path = manager.save()
                    print('')
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    print("epoch: {}, loss {:1.2f}".format(epoch, loss_value.numpy()))
                    # print(
                    #     "Training loss (for one batch) at step %d: %.4f"
                    #     % (step, float(loss_value))
                    # )
                    # print("Seen so far: %s samples" % ((step + 1) * 64))
        tf.saved_model.save(self.model, self.model_dir)

    def pred(self, x):
        """"""
        x_batch = sentence_to_index(x, dt=self.vacab_dt, default=0)
        sequence_lengths = np.array([len(s) for s in x_batch])
        logit = self.model(x_batch, training=False)
        tags, _ = tfa.text.crf_decode(logit, self.model.transition_params, sequence_lengths)
        # tfa.text.viterbi_decode(logit, self.transition_params)
        print(tags)
        return tags

    def evaluation(self):
        """"""
        test_data = self.d.get_test()
        x, y = test_data[:, 0][:100], test_data[:, 1][:100]

        y_pre = self.pred(x)
        pre_words = [split_to_index(np.array(y_pre[i][:len(x[i])]), self.label_dict) for i in range(len(y_pre))]
        normal_words = [split_to_index(t) for t in y]
        correct_word_number = 0
        pre_word_number = 0
        normal_word_number = 0
        for i in range(len(y)):
            correct_word_number += len(set(normal_words[i]) & set(pre_words[i]))
            normal_word_number += len(normal_words[i])
            pre_word_number += len(pre_words[i])

        precision = correct_word_number / pre_word_number
        recall = correct_word_number/normal_word_number
        f1 = 2 * (precision * recall) / (precision + recall)

        print('评估结果， 准确率： %s,  召回率：%s,  F1: %s' % (precision, recall, f1))


if __name__ == '__main__':
    train_obj = LstmFenciTrainer()
    train_obj.train(batch_size=512, epochs=100)
    train_obj.evaluation()
