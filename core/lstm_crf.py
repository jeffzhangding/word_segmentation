__author__ = 'jeff'

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tqdm

from common.msr_data_deal import Data


path_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
path_list.extend(['models', 'fenci_1'])
model_dir = os.sep.join(path_list)
check_point_dir = os.path.join(model_dir, 'check_point')
save_dir = os.path.join(model_dir, 'save_model')

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


class CrfLoss(tf.keras.losses.Loss):

    def __init__(self, name="custom_mse"):
        super(CrfLoss, self).__init__(name=name)
        # self.regularization_factor = regularization_factor
        self.transition_params = None

    def call(self, y_true, y_pred):
        lable, sequence_lengths = y_true[:, :-1], y_true[:, -1]
        log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(
            y_pred, lable, sequence_lengths, self.transition_params)
        # mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        # reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        loss = - tf.reduce_mean(log_likelihood)
        return loss


class LstmCrfModel(tf.keras.Model):

    def __init__(self, hidden_num, vocab_size, embedding_size, tag_lenth=4, dp=0.2):
        super(LstmCrfModel, self).__init__()
        self.num_hidden = hidden_num
        # self.vocab = vocab
        # self.vacab_dt = {v: index+1 for index, v in enumerate(vocab)}
        self.vacab_size = vocab_size
        self.dp = dp
        # self.sentence_lenth = sentence_lenth
        # self.label_dict = {"B": 0, "E": 1, "S": 2, "M": 3}
        # S表示单字为词，B表示词的首字，M表示词的中间字，E表示词的结尾字
        self.tag_lenth = tag_lenth

        self.embedding = tf.keras.layers.Embedding(self.vacab_size, embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=True, dropout=self.dp))
        self.transition_params = tf.Variable(tf.random.normal(shape=(self.tag_lenth, self.tag_lenth)))
        # self.softmax = tf.keras.layers.Softmax()
        self.dense = tf.keras.layers.Dense(self.tag_lenth)

    def caculate_loss(self, lable, logit, sequence_lengths):
        """"""
        # lable = tf.Tensor(lable)
        # sequence_lengths = tf.Tensor(sequence_lengths)
        lable = tf.keras.preprocessing.sequence.pad_sequences(
            lable, padding="post"
        )
        # lable = np.array(lable)

        # lable, sequence_lengths = y_true[:, :-1], y_true[:, -1]
        log_likelihood, transition_params = tfa.text.crf_log_likelihood(
            logit, lable, sequence_lengths, self.transition_params)
        # mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        # reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        loss = - tf.reduce_mean(log_likelihood)
        return loss

    def call(self, inputs, training=False):
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            inputs, padding="post"
        )
        inputs = self.embedding(padded_inputs)
        lstm_vector = self.biLSTM(inputs, training=training)
        logits = self.dense(lstm_vector)
        return logits

    def sentence_to_index(self, sentences, dt, default=None):
        """"""
        res = []
        for s in sentences:
            r = [dt.get(i, default) for i in s]
            res.append(r)
        return res

    # @tf.function
    def train(self, x, y, batch_size=100, epochs=10):
        """"""
        optimizer = tf.keras.optimizers.Adam()

        # opt = tf.keras.optimizers.Adam(0.1)
        # dataset = toy_dataset()
        # iterator = iter(dataset)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=self)
        manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

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

                x_batch_train = self.sentence_to_index(x[batch_size*step: batch_size*(step+1)],
                                                       dt=self.vacab_dt, default=0)
                y_batch_train = self.sentence_to_index(y[batch_size * step: batch_size * (step + 1)],
                                                       dt=self.label_dict, default=0)
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                sequence_lengths = np.array([len(s) for s in x_batch_train])

                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = self(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = self.caculate_loss(y_batch_train, logits, sequence_lengths)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                # Log every 200 batches.
                # if step % 200 == 0:
                ckpt.step.assign_add(1)
                print('======%s', str(loss_value))
                # break
                if int(ckpt.step) % 2 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()))
                    # print(
                    #     "Training loss (for one batch) at step %d: %.4f"
                    #     % (step, float(loss_value))
                    # )
                    # print("Seen so far: %s samples" % ((step + 1) * 64))
        tf.saved_model.save(self, save_dir)

    def pred(self, x):
        """"""
        x_batch = self.sentence_to_index(x, dt=self.vacab_dt, default=0)
        sequence_lengths = np.array([len(s) for s in x_batch])
        logit = self(x_batch, training=False)
        tags, _ = tfa.text.crf_decode(logit, self.transition_params, sequence_lengths)
        # tfa.text.viterbi_decode(logit, self.transition_params)
        print(tags)
        return tags

    def evl(self, x, y):
        """"""
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


def save_model_test(m):
    """"""
    path_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
    path_list.extend(['models', 'fenci_1'])
    model_dir = os.sep.join(path_list)
    m.save(model_dir)


def load_model_test():
    path_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
    path_list.extend(['models', 'fenci_1'])
    model_dir = os.sep.join(path_list)
    m = tf.keras.models.load_model(model_dir)


def test():
    """"""
    raw_inputs = [
        [711, 632, 71],
        [73, 8, 3215, 55, 927],
        [83, 91, 1, 645, 1253, 927],
    ]

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        raw_inputs, padding="post"
    )

    embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)

    masked_output = embedding(padded_inputs)
    print(padded_inputs)


def test_fenci():
    """"""
    d = Data()
    vab = np.array(d.get_dictionary())
    train_data = d.get_train()
    x_train,  y_train = train_data[:, 0], train_data[:, 1]
    model = LstmCrfModel(
        hidden_num=128,
        embedding_size=128,
        vocab=vab,
        dp=0.2,
    )
    # model.summary()
    model.train(x_train, y_train, batch_size=512, epochs=1)
    # model.pred()
    test_data = d.get_test()
    x_test, y_test = test_data[:, 0], test_data[:, 1]
    model.evl(x_test[:1000], y_test[:1000])
    # save_model_test(model)



if __name__ == '__main__':
    # training_data = np.array([["我 喜 欢 你"], ["你"]])
    # training_data = np.array([['我', '喜', '欢', '你'], ['你', '爱', '我', '么']])
    # # ['我', '喜', '欢', '你']e
    #
    # vectorizer = TextVectorization(output_mode="int", output_sequence_length=6, split=None)
    #

    # test()

    test_fenci()





