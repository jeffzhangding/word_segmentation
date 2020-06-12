__author__ = 'jeff'

import os
import json


class Data(object):
    """"""

    def __init__(self):
        self.train_file = '../data/CTBtrainingset.txt'
        self.test_file = '../data/output.txt'
        self.deal_train = './data/train.txt'
        self.deal_test = './data/test.txt'
        self.dictionary = './data/dictionary.json'

    def to_bio(self, file_name):
        """转换成分词的数据格式"""
        res = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                line_list = l[:-1].split(' ')
                biaozhu = []
                for word in line_list:
                    if len(word) == 1:
                        biaozhu.append('S')
                    elif len(word) > 1:
                        biaozhu.append('B'+'I'*(len(word)-1))
                sentence = ''.join(line_list) + '\n'
                biaozhu_s = ''.join(biaozhu) + '\n'
                res.append(sentence)
                res.append(biaozhu_s)
                assert len(sentence) == len(biaozhu_s)
        return res

    def get_data(self, deal_file_name, file_name):
        word, lable = [], []
        if os.path.exists(deal_file_name):
            with open(deal_file_name, 'r', encoding='utf-8') as f:
                while True:
                    w = f.readline()
                    a = f.readline()
                    if w == '' or a == '':
                        break
                    word.append(w[:-1])
                    lable.append(a[:-1])
            return word, lable
        else:
            res = self.to_bio(file_name)
            for i in range(len(res) // 2):
                word.append(res[2 * i])
                lable.append(res[2 * i + 1])
            with open(deal_file_name, 'w', encoding='utf-8') as f:
                f.writelines(res)
        return word, lable

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
    r = Data().get_train()
    # r = Data().get_dictionary()
    print(r)

