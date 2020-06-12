"""
分词

"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np

from common.data_deal import Data

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)


class FenciNerCore(object):

    def __init__(self, io_sequence_size, vocab_size, class_size=6, keep_prob=0.5, learning_rate=0.001, trainable=False):
        self.is_training = trainable
        self.vocab_size = vocab_size
        self.io_sequence_size = io_sequence_size
        self.learning_rate = learning_rate
        self.embedding_size = 200
        self.hidden_size = 256
        self.output_class_size = class_size
        self.keep_prob = keep_prob
        self.num_layers = 1
        with tf.name_scope("ner_declare"):
            self.inputs = tf.placeholder(tf.int32, [None, self.io_sequence_size], name="char_inputs")
            # self.word_vec = tf.placeholder(dtype=tf.float32, shape=(None, self.io_sequence_size, self.embedding_size),
            #                                name='p_word')
            self.targets = tf.placeholder(tf.int32, [None, self.io_sequence_size], name="targets")  # [None,num_classes]
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.create_declare()
        self.build()
        self.create_loss()

    def create_declare(self):
        with tf.name_scope("ner_declare"):
            self.embedding_variable = tf.get_variable("embedding_variable",
                                                      shape=[self.vocab_size, self.embedding_size],
                                                      initializer=tf.random_normal_initializer(stddev=0.1))
            self.weight_variable = tf.get_variable("weight_variable",
                                                   shape=[self.hidden_size * 2, self.output_class_size],
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
            self.bias_variable = tf.get_variable("bias_variable", shape=[self.output_class_size])

    def LSTM(self, x):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    def build(self):
        self.embedded_layer = tf.nn.embedding_lookup(self.embedding_variable, self.inputs, name="embedding_layer")  # shape:[None,sentence_length,embed_size]
        # with tf.variable_scope("lstm_p", reuse=None):
        #     p_output, _ = self.LSTM(self.embedded_layer)
        # p_embedding = tf.concat((p_output, self.word_vec), axis=-1)
        # p_embedding = p_output

        p_embedding = self.embedded_layer

        with tf.name_scope("ner_layer"):
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size), output_keep_prob=self.keep_prob)
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size), output_keep_prob=self.keep_prob)
            lstm_cell_fws = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
            lstm_cell_bws = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fws,
                                                                        lstm_cell_bws,
                                                                        p_embedding,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            self.outputs = tf.concat([output_fw, output_bw], axis=-1)
            # print(self.outputs.shape)
            self.outputs = tf.nn.dropout(self.outputs, self.keep_prob)
            self.outputs = tf.reshape(self.outputs, [-1, 2 * self.hidden_size])
            # print(self.outputs.shape)
            self.logits = tf.matmul(self.outputs, self.weight_variable) + self.bias_variable
            # print(self.logits.shape)
            self.logits = tf.reshape(self.logits, [-1, self.io_sequence_size, self.output_class_size])
            # print(self.logits.shape)

    def create_loss(self):
        with tf.name_scope("ner_loss"):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                       tag_indices=self.targets,
                                                                                       sequence_lengths=self.sequence_lengths)
            if self.is_training == True:
                self.cost_func = -tf.reduce_mean(log_likelihood)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_func)

    def decode(self, terms, taggs):
        char_item = []
        tag_item = []
        raw_content = {}
        for i in range(len(terms)):
            if taggs[i][0] == 'B':
                if len(char_item) > 0 and len(tag_item) > 0:
                    content = ''.join(char_item)
                    key = tag_item[0][2:]
                    position = (i - len(content), len(content))
                    if key in raw_content.keys():
                        raw_content[key].append((content, position))
                    else:
                        raw_content[key] = [(content, position)]
                    char_item = []
                    tag_item = []
                char_item.append(terms[i])
                tag_item.append(taggs[i])
            elif taggs[i][0] == 'O':
                if len(char_item) > 0 and len(tag_item) > 0:
                    content = ''.join(char_item)
                    position = (i-len(content), len(content))
                    key = tag_item[0][2:]
                    if key in raw_content.keys():
                        raw_content[key].append((content,position))
                    else:
                        raw_content[key] = [(content,position)]
                    char_item = []
                    tag_item = []
            else:
                char_item.append(terms[i])
                tag_item.append(taggs[i])
        if len(char_item) > 0 and len(tag_item) > 0:  # 超出后循环外处理
            content = ''.join(char_item)
            key = tag_item[0][2:]
            position = (len(terms) - len(content), len(content))
            if key in raw_content.keys():
                raw_content[key].append((content,position))
            else:
                raw_content[key] = [(content, position)]
        return raw_content

    def train(self):
        """"""


class NerTrainner:
    def __init__(self):
        self.model_dir = "./models//cws_2"
        self.data = Data()
        self.char_index = {k: v for v, k in enumerate(self.data.get_dictionary())}

        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 130
        vocab_size = len(self.char_index)+1

        self.classnames = {'O': 0, 'B': 1, 'I': 2, 'S': 3}  # B:开始， I：中间词汇， S：单个字的词
        class_size = len(self.classnames)  # brand,price,position,item,unknow
        keep_prob = 0.5
        learning_rate = 0.0005
        trainable = True
        self.batch_size = 128

        with tf.variable_scope('ner_query'):
            self.model = FenciNerCore(self.io_sequence_size, vocab_size,
                                      class_size, keep_prob, learning_rate, trainable)

    def train(self, epochs=30):
        records, lables = self.data.get_train()
        print(len(records))
        batch_count = int(len(records) / self.batch_size)
        print("prepare data success ...")
        initer = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(initer)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            saver = tf.train.Saver()
            if ckpt is not None and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            for epoch in range(epochs):
                train_loss_value = 0.
                # train_accuracy_value = 0.
                for i in range(batch_count):
                    batch_records = records[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_labels = lables[i * self.batch_size:(i + 1) * self.batch_size]
                    xrows, xlens, yrows = self.convert_batch(batch_records, batch_labels)
                    feed_dict = {self.model.inputs: xrows, self.model.targets: yrows, self.model.sequence_lengths: xlens}
                    batch_loss_value, _ = session.run([self.model.cost_func, self.model.optimizer], feed_dict)
                    train_loss_value += batch_loss_value / batch_count
                    if i % 100 == 0:
                        batch_buffer = "Progress {0}/{1} , cost : {2}".format(i + 1, batch_count, batch_loss_value)
                        # print(batch_buffer, end="\r", flush=True)
                        print(batch_buffer)
                print("Epoch: %d/%d , train cost=%f " % ((epoch + 1), epochs, train_loss_value))
                saver.save(session, os.path.join(self.model_dir, "ner.dat"))

    def convert_batch(self, sentense_list, label_list):
        xrows = np.zeros((self.batch_size, self.io_sequence_size), dtype=np.float32)
        xlens = np.zeros((self.batch_size), dtype=np.int32)
        yrows = np.zeros((self.batch_size, self.io_sequence_size), dtype=np.int32)
        count = len(sentense_list)
        for i in range(count):
            sent_text, tags = sentense_list[i][:-1], label_list[i][:-1]
            xlen = len(sent_text)
            if xlen > self.io_sequence_size:
                # print(xlen)
                xlen = self.io_sequence_size
            xlens[i] = xlen
            xrows[i] = self.convert_xrow(sent_text)
            yrows[i] = self.convert_classids(tags)
        return xrows, xlens, yrows

    def convert_classids(self,tags):
        yrow = np.zeros(self.io_sequence_size, dtype=np.int32)
        for i in range(len(tags[:self.io_sequence_size])):
            yrow[i] = self.classnames[tags[i]]
        return yrow

    def convert_xrow(self, input_text):
        char_vector = np.zeros((self.io_sequence_size), dtype=np.int32)
        for i in range(len(input_text[: self.io_sequence_size])):
            char_value = input_text[i]
            if char_value in self.char_index.keys():
                char_vector[i] = self.char_index[char_value]
        return char_vector


class NerPredicter:
    def __init__(self):
        self.model_dir = "./models//ner_1"
        self.data = Data()
        self.char_index = {k: v for v, k in enumerate(self.data.get_dictionary())}

        self.keep_prob = 1.0
        self.is_training = False
        # self.char_index = {' ': 0}
        # self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 150  # 70
        self.vocab_size = len(self.char_index) + 1
        self.batch_size = 64
        self.ner_index_class = {'O': 0, 'B': 1, 'I': 2, 'S': 3}  # B:开始， I：中间词汇， S：单个字的词
        self.class_size = len(self.ner_index_class)
        self.classids = {}
        for key in self.ner_index_class.keys():
            self.classids[self.ner_index_class[key]] = key

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('ner_query'):
                self.model = FenciNerCore(io_sequence_size=self.io_sequence_size, vocab_size=self.vocab_size,
                                          class_size=self.class_size, keep_prob=self.keep_prob,
                                          trainable=self.is_training)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        self.session = tf.Session(graph=self.graph, config=config)
        with self.session.as_default():
            self.load()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt is not None and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("ner load {} success...".format(str(ckpt.model_checkpoint_path)))

    def convert_row(self, input_text):
        char_vector = np.zeros((self.io_sequence_size), dtype=np.int32)
        for i in range(len(input_text[: self.io_sequence_size])):
        # for i in range(len(input_text)):
            char_value = input_text[i]
            if char_value in self.char_index.keys():
                char_vector[i] = self.char_index[char_value]
        return char_vector

    def predict(self, input_text):
        input_text = input_text.strip().lower()
        char_vector = self.convert_row(input_text.strip().lower())
        seq_len_list = np.array([len(input_text)], dtype=np.int32)
        feed_dict = {self.model.inputs: np.array([char_vector], dtype=np.float32), self.model.sequence_lengths: seq_len_list}
        logits, transition_params = self.session.run([self.model.logits, self.model.transition_params], feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        if label_list is not None and len(label_list) > 0:
            # taggs = []
            # for i in range(seq_len_list[0]):
            #     taggs.append(self.classids[label_list[0][i]])
            # output_labels = self.model.decode(list(input_text), taggs)
            # data_items = {}
            # for key in output_labels.keys():
            #     value = output_labels[key]
            #     #去除下标偏移
            #     data_items[key] = [vitem [0] for vitem in value]
            # return data_items
            return label_list
        else:
            return None

    def to_char(self, label_list):
        """"""
        res = []
        for i in label_list:
            res.append([self.classids[w] for w in i])
        return res

    def test(self):
        """"""
        sentences, labels = self.data.get_test()
        res = []
        for i in range(len(sentences)):
        # for i in range(100):
            p = self.predict(sentences[i])
            p = self.to_char(p)
            res.append(int(labels[i] == ''.join(p[0])))
        print(sum(res) / len(sentences))

    def test2(self):
        sentences, labels = self.data.get_test()
        res = []
        sum = 0
        t_res = 0
        # for i in range(len(sentences)):
        for i in range(3000):
            p = self.predict(sentences[i])
            p = self.to_char(p)
            for j in range(len(p[0])):
                sum += 1
                if labels[i][j] == p[0][j]:
                    t_res += 1
        print(t_res / sum)


if __name__ == '__main__':
    NerTrainner().train(30)
    # NerPredicter().test()
    NerPredicter().test2()

