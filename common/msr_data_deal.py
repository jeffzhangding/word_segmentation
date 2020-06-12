__author__ = 'jeff'

import os
import pickle
import numpy as np
import json

from common.common import tow_tag, four_tag

path_list = str(os.path.abspath(__file__)).split(os.sep)[:-2]
path_list.extend(['data', 'msr'])
data_dir = os.sep.join(path_list)


class Data(object):
    """"""

    def __init__(self):
        self.train_file = os.path.join(data_dir, 'msr_training.utf8')
        self.test_file = os.path.join(data_dir, 'msr_test_gold.utf8')
        self.deal_train = os.path.join(data_dir, 'train.pkl')
        self.deal_test = os.path.join(data_dir, 'test.pkl')
        self.dictionary = os.path.join(data_dir, 'dictionary.json')

    def to_bio(self, file_name):
        """转换成分词的数据格式"""
        res = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                l = l.replace('“', '')
                words_list = l[:-1].split('  ')
                # input_s, tag_s = tow_tag(words_list)
                input_s, tag_s = four_tag(words_list)
                if input_s:
                    res.append([input_s, tag_s])
                assert len(input_s) == len(tag_s)
        return np.array(res)

    def get_data(self, deal_file_name, file_name):
        if os.path.exists(deal_file_name):
            with open(deal_file_name, 'rb') as f:
                res = pickle.load(f)
        else:
            res = self.to_bio(file_name)
            with open(deal_file_name, 'wb') as f:
                pickle.dump(res, f)
        return res

    def get_test(self):
        """"""
        return self.get_data(self.deal_test, self.test_file)

    def get_train(self):
        """"""
        return self.get_data(self.deal_train, self.train_file)

    def get_dictionary(self):
        with open(self.dictionary, 'rb') as f:
            res = json.load(f)
            return res


if __name__ == '__main__':
    # r = Data().get_train()
    r = Data().get_test()
    print(r)


