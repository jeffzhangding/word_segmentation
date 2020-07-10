__author__ = 'jeff'

import os
import numpy as np
import pickle
from random import shuffle
import json

from .base import Data, DataSet
from common.common import four_tag


def sentence_to_index(sentences, dt, default=None):
    """"""
    res = []
    for s in sentences:
        r = [dt.get(i, default) for i in s]
        res.append(r)
    return res


class MsrData(Data):

    def __init__(self, data_file, dumps_file=None):
        """"""
        super(MsrData, self).__init__()
        self.data_file = data_file
        self.dumps_file = dumps_file
        self.load_data()
        self._math_data = None

    @property
    def math_data(self):
        """"""
        if self._math_data is None:
            return self.data
        else:
            return self._math_data

    def load_data(self):
        """"""
        if os.path.exists(self.dumps_file):
            with open(self.dumps_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = self.load_by_file()
            self.dumps()

    def dumps(self):
        """"""
        with open(self.dumps_file, 'wb') as f:
            pickle.dump(self.data, f)

    def math_indicate(self, vab_dt, label_dt, *args, **kwargs):
        """"""
        res = []
        for s, lable in self.data:
            r, m_lable = [vab_dt['[CLS]']], [label_dt['S']]
            r.extend([vab_dt.get(i, vab_dt['[UNK]']) for i in s])
            m_lable.extend([label_dt[i] for i in lable])
            res.append([r, m_lable, len(m_lable)])
        res = np.array(res)
        self._math_data = res
        return res

    def result_to_char(self):
        """"""

    def load_by_file(self):
        """"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            res = self.to_bio(data)
            return res

    @staticmethod
    def to_bio(data_list):
        """转换成分词的数据格式"""
        res = []
        for l in data_list:
            l = l.replace('“', '')
            words_list = l[:-1].split('  ')
            # input_s, tag_s = tow_tag(words_list)
            input_s, tag_s = four_tag(words_list)
            if input_s:
                res.append([input_s, tag_s])
            assert len(input_s) == len(tag_s)
        return res

    def shuffle(self):
        shuffle(self.data)
        shuffle(self.math_data)


class MsrDataSet(DataSet):

    def __init__(self, train_file=None, dumps_train=None, test_file=None, dump_test=None, vab_file=None):
        super(MsrDataSet, self).__init__()
        self.train = MsrData(train_file, dumps_train)
        self.test = MsrData(test_file, dump_test)
        self.vab_file = vab_file
        self.vab = None
        self.vab_dt = None
        self.label_dt = {"B": 0, "E": 1, "S": 2, "M": 3}
        self.load_vab()
        self.init_data()

    def load_vab(self):
        with open(self.vab_file, 'rb') as f:
            self.vab = json.load(f)
        self.vab_dt = {k: v for v, k in enumerate(self.vab)}

    def init_data(self):
        self.train.math_indicate(self.vab_dt, self.label_dt)
        self.test.math_indicate(self.vab_dt, self.label_dt)


path_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
path_list.extend(['data', 'msr'])
data_dir = os.sep.join(path_list)
msr = MsrDataSet(
    train_file=os.path.join(data_dir, 'msr_training.utf8'),
    test_file=os.path.join(data_dir, 'msr_test_gold.utf8'),
    dumps_train=os.path.join(data_dir, 'train.pkl'),
    dump_test=os.path.join(data_dir, 'test.pkl'),
    vab_file=os.path.join(data_dir, 'bert_vab.json')
)

if __name__ == '__main__':
    pass


