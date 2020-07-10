__author__ = 'jeff'

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
p_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
b_path = os.sep.join(p_list)
sys.path.append(b_path)

from core.bert import BertCrf
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import numpy as np
import tqdm

from data_set.msr import msr
from settings import base_dir


class BertCrfTrainer(object):
    """"""

    def __init__(self, max_lenth, dropout=0.2, dense_h=100, batch_size=5, bert_dir=None):
        if not bert_dir:
            self.bert_dir = "E:\\模型文件\\bert\\5a9399249ae4b8d5dad3da5dac3112b72e13a2e1"
        else:
            self.bert_dir = bert_dir
        self.max_lenth = max_lenth
        self.model = BertCrf(max_lenth=max_lenth, tag_lenth=4, dropout=dropout, bert_dir=self.bert_dir, dense_h=dense_h)
        self.check_point_dir = os.path.sep.join([base_dir, 'models', 'bert_crf', 'check_point'])
        self.model_dir = os.path.sep.join([base_dir, 'models', 'bert_crf'])
        self.batch_size = batch_size
        # self.model.build((None, self.max_lenth))

    def train(self, epochs):
        """"""
        optimizer = tf.keras.optimizers.Adam()

        # opt = tf.keras.optimizers.Adam(0.1)
        # dataset = toy_dataset()
        # iterator = iter(dataset)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.check_point_dir, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        # tf.saved_model.save(self.model, self.model_dir)
        # tf.keras.models.save_model(self.model, self.model_dir)
        # return
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        batch_number = len(msr.train.math_data) // self.batch_size

        for epoch in range(epochs):
            msr.train.shuffle()
            x, y, lenths = msr.train.math_data[:, 0], msr.train.math_data[:, 1], msr.train.math_data[:, 2]
            lenths = pd.Series(lenths)
            lenths = lenths.where(lenths <= self.max_lenth, self.max_lenth).values
            for i in range(batch_number):
                x_batch = list(x[i*self.batch_size:(i+1)*self.batch_size])
                y_batch = list(y[i*self.batch_size:(i+1)*self.batch_size])
                sequence_lengths = np.array(list(lenths[i*self.batch_size:(i+1)*self.batch_size]))
                x_batch = tf.keras.preprocessing.sequence.pad_sequences(
                    x_batch, padding="post", maxlen=self.max_lenth
                )
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = self.model(x_batch)  # Logits for this minibatch

                    break
                    # Compute the loss value for this minibatch.
                    loss_value = self.model.caculate_loss(y_batch, logits, sequence_lengths)

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
                    print("Saved checkpoint for step {}: {}, total step:{}".format(int(ckpt.step), save_path,
                                                                                   batch_number))
                    print("epoch: {}, loss {:1.2f}".format(epoch, loss_value.numpy()))
                    # print(
                    #     "Training loss (for one batch) at step %d: %.4f"
                    #     % (step, float(loss_value))
                    # )
                    # print("Seen so far: %s samples" % ((step + 1) * 64))
        # self.model.save(self.model_dir)
        tf.keras.models.save_model(self.model, self.model_dir, include_optimizer=False)
        # tf.saved_model.save(self.model, self.model_dir)
        return


if __name__ == '__main__':
    # trainer = BertCrfTrainer(
    #     max_lenth=500, batch_size=32, dense_h=128, bert_dir='/home/robot/jeff/trans_models/bert'
    # )
    trainer = BertCrfTrainer(
        max_lenth=100, batch_size=2, dense_h=128)
    trainer.train(1)

